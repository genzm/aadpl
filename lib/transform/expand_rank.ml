(* Phase 5(b): ⎉ rank expansion — AST-to-AST transform.
   Rank(k, prim, args) applies prim at cell rank k.
   expand eliminates all Rank nodes by:
     1. computing each arg's frame (leading axes above cell rank k)
     2. checking leading agreement (shorter frame is prefix of longest)
     3. inserting Broadcast for missing frame axes
     4. shifting axis parameters past the frame *)

open Ast.Types

(* cell_rank: intrinsic minimum rank per argument.
   k in Rank(k, p, args) must be >= max(cell_rank p). *)
let cell_rank : prim -> int list = function
  | Neg | Exp | Log | Sqrt | Relu | Step -> [0]
  | Add | Sub | Mul | Div | Max2  -> [0; 0]
  | Sum_axis _ | Max_axis _ | Argmax_axis _ -> [1]
  | Select_axis _                -> [1; 0]  (* data rank >= 1; index is scalar-per-cell *)
  | Gather _                      -> [1]
  | Matmul                        -> [2; 2]
  | Scatter_add (_, _, _)         -> [1]
  | Scatter_select_add _          -> [0; 0]
  | Apply_view spec ->
    [match spec with
     | [] -> 0
     | op :: _ -> (match op with
       | Vtranspose p -> Array.length p
       | Vslice r -> Array.length r
       | Vbroadcast _ -> 0
       | Vreshape _ -> 1)]
  | Adjoint_view (spec, target_shape) ->
    [Array.length (viewspec_output_shape spec target_shape)]

(* shift_viewop/shift_viewspec: shift viewspec axis parameters by frame *)
let shift_viewop (frame : int array) (op : viewop) : viewop =
  let f = Array.length frame in
  match op with
  | Vtranspose p ->
    let fr = Array.init f (fun i -> i) in
    let cr = Array.map (fun i -> i + f) p in
    Vtranspose (Array.append fr cr)
  | Vslice r ->
    let frame_ranges = Array.map (fun sz -> (0, sz, 1)) frame in
    Vslice (Array.append frame_ranges r)
  | Vbroadcast (a, s) -> Vbroadcast (a + f, s)
  | Vreshape s -> Vreshape (Array.append frame s)

let shift_viewspec (frame : int array) (spec : viewspec) : viewspec =
  List.map (shift_viewop frame) spec

(* shift_prim: shift axis parameters by frame (full shape, not just rank).
   Slice needs frame sizes for the prepended full-range slices.
   Reshape needs frame sizes prepended to the target shape. *)
let shift_prim (frame : int array) (p : prim) : prim =
  let f = Array.length frame in
  if f = 0 then p
  else match p with
  | Sum_axis a              -> Sum_axis (a + f)
  | Max_axis a              -> Max_axis (a + f)
  | Argmax_axis a           -> Argmax_axis (a + f)
  | Select_axis a           -> Select_axis (a + f)
  | Scatter_add (axis, indices, target_shape) ->
    Scatter_add (axis + f, indices, Array.append frame target_shape)
  | Scatter_select_add (axis, sz) ->
    Scatter_select_add (axis + f, sz)
  | Gather (axis, indices)  -> Gather (axis + f, indices)
  | Apply_view spec         -> Apply_view (shift_viewspec frame spec)
  | Adjoint_view (spec, ts) ->
    Adjoint_view (shift_viewspec frame spec, Array.append frame ts)
  | _                       -> p  (* elementwise, matmul: no axis params *)

(* --- shape inference --- *)

type shape_env = (string * int array) list

let rec infer_shape (senv : shape_env) (e : expr) : int array =
  match e with
  | Const (_, v) -> v.View.Tensor.view.View.Ndview.shape
  | Var (_, s) ->
    (match List.assoc_opt s senv with
     | Some sh -> sh
     | None -> failwith ("expand_rank: unbound variable: " ^ s))
  | Prim (_, p, args) ->
    let shapes = List.map (infer_shape senv) args in
    output_shape p shapes
  | Let (_, s, e1, body) ->
    let sh1 = infer_shape senv e1 in
    infer_shape ((s, sh1) :: senv) body
  | Rank _ ->
    failwith "infer_shape: nested Rank not supported (expand inner Rank first)"

and output_shape (p : prim) (shapes : int array list) : int array =
  match p, shapes with
  | (Neg | Exp | Log | Sqrt | Relu | Step), [s] -> s
  | (Add | Sub | Mul | Div | Max2), [s; _] -> s
  | (Sum_axis axis | Max_axis axis | Argmax_axis axis), [s] ->
    let r = Array.length s in
    Array.init (r - 1) (fun i -> if i < axis then s.(i) else s.(i + 1))
  | Select_axis axis, [s; _] ->
    let r = Array.length s in
    Array.init (r - 1) (fun i -> if i < axis then s.(i) else s.(i + 1))
  | Gather (axis, indices), [s] ->
    Array.init (Array.length s) (fun k ->
      if k = axis then Array.length indices else s.(k))
  | Scatter_add (_, _, target_shape), [_] -> target_shape
  | Scatter_select_add (axis, target_axis_size), [s; _] ->
    let r = Array.length s in
    Array.init (r + 1) (fun i ->
      if i < axis then s.(i)
      else if i = axis then target_axis_size
      else s.(i - 1))
  | Apply_view spec, [s] -> viewspec_output_shape spec s
  | Adjoint_view (_, target_shape), [_] -> target_shape
  | Matmul, [sa; sb] ->
    (* batched matmul: frame ++ [m; n] *)
    let ra = Array.length sa in
    let m = sa.(ra - 2) in
    let n = sb.(Array.length sb - 1) in
    let nframe = ra - 2 in
    Array.init (nframe + 2) (fun i ->
      if i < nframe then sa.(i)
      else if i = nframe then m
      else n)
  | _ -> assert false

(* --- expand --- *)

exception Expand_error of loc * string

let expand (e : expr) : expr =
  let rec go senv e =
    match e with
    | Const _ | Var _ -> e
    | Let (loc, s, e1, e2) ->
      let e1' = go senv e1 in
      let sh1 = infer_shape senv e1' in
      Let (loc, s, e1', go ((s, sh1) :: senv) e2)
    | Prim (loc, p, args) ->
      Prim (loc, p, List.map (go senv) args)
    | Rank (loc, k, p, args) ->
      let args = List.map (go senv) args in
      let crs = cell_rank p in
      (* validate k >= intrinsic cell rank *)
      List.iter (fun cr ->
        if k < cr then
          raise (Expand_error (loc,
            Printf.sprintf "cell rank %d < intrinsic minimum %d" k cr)))
        crs;
      (* compute each arg's frame *)
      let shapes = List.map (infer_shape senv) args in
      let frames = List.mapi (fun i s ->
        let r = Array.length s in
        if r < k then
          raise (Expand_error (loc,
            Printf.sprintf "arg %d rank %d < cell rank %d" i r k));
        Array.sub s 0 (r - k)) shapes in
      (* find longest frame F *)
      let max_frame = List.fold_left (fun acc f ->
        if Array.length f > Array.length acc then f else acc) [||] frames in
      let nf = Array.length max_frame in
      (* check leading agreement: each frame_i must be a prefix of F *)
      List.iteri (fun i fi ->
        let ni = Array.length fi in
        for j = 0 to ni - 1 do
          if fi.(j) <> max_frame.(j) then
            raise (Expand_error (loc,
              Printf.sprintf "arg %d frame mismatch at axis %d: %d vs %d"
                i j fi.(j) max_frame.(j)))
        done) frames;
      (* insert Vbroadcast for missing frame axes *)
      let args_bc = List.map2 (fun arg fi ->
        let ni = Array.length fi in
        let a = ref arg in
        for j = ni to nf - 1 do
          a := Prim (loc, Apply_view [Vbroadcast (j, max_frame.(j))], [!a])
        done;
        !a) args frames in
      Prim (loc, shift_prim max_frame p, args_bc)
  in
  go [] e

(* is_expanded: true if no Rank nodes remain *)
let rec is_expanded (e : expr) : bool =
  match e with
  | Const _ | Var _ -> true
  | Let (_, _, e1, e2) -> is_expanded e1 && is_expanded e2
  | Prim (_, _, args) -> List.for_all is_expanded args
  | Rank _ -> false
