(* Reference interpreter for the deterministic array language.
   Invariant: writes go to fresh buffers only. Prim allocates a new dst
   for every computation. Values from Const and Var may be shared (read-only).
   Structural ops (transpose, broadcast, slice) return views into the same
   buf via map_view, which sets shared = true. This prevents the buf from
   being used as dst by any kernel. *)

open View

type env = (string * Types.value) list

exception Shape_mismatch of Types.loc * string
exception Eval_error of Types.loc * string

let shape_of (t : Tensor.t) = t.view.Ndview.shape
let rank_of (t : Tensor.t) = Ndview.rank t.view

let assert_shape loc msg cond =
  if not cond then raise (Shape_mismatch (loc, msg))

(* --- output shape validation per primitive --- *)

let validate loc (p : Types.prim) (args : Tensor.t list) =
  match p, args with
  | (Types.Neg | Exp | Log | Sqrt | Relu | Step), [_] -> ()
  | (Types.Add | Sub | Mul | Div | Max2), [x; y] ->
    assert_shape loc "map2: shape mismatch" (shape_of x = shape_of y)
  | Sum_axis axis, [x] ->
    assert_shape loc "sum_axis: axis out of range"
      (0 <= axis && axis < Array.length (shape_of x))
  | Max_axis axis, [x] ->
    assert_shape loc "max_axis: axis out of range"
      (0 <= axis && axis < Array.length (shape_of x))
  | Transpose perm, [x] ->
    assert_shape loc "transpose: perm length"
      (Array.length perm = Array.length (shape_of x))
  | Reshape new_shape, [x] ->
    assert_shape loc "reshape: numel mismatch"
      (Ndview.numel x.view = Array.fold_left ( * ) 1 new_shape)
  | Broadcast (axis, _size), [x] ->
    assert_shape loc "broadcast: axis out of range"
      (0 <= axis && axis <= Array.length (shape_of x))
  | Slice ranges, [x] ->
    let s = shape_of x in
    let r = Array.length s in
    assert_shape loc "slice: ranges length" (Array.length ranges = r);
    Array.iteri (fun k (start, stop, step) ->
      assert_shape loc "slice: step must be nonzero" (step <> 0);
      if step > 0 then begin
        assert_shape loc "slice: start out of range"
          (0 <= start && start < s.(k));
        assert_shape loc "slice: stop out of range"
          (start < stop && stop <= s.(k))
      end else begin
        assert_shape loc "slice: start out of range"
          (0 <= start && start < s.(k));
        assert_shape loc "slice: stop out of range"
          (-1 <= stop && stop < start)
      end) ranges
  | Gather (axis, indices), [x] ->
    let s = shape_of x in
    assert_shape loc "gather: axis out of range"
      (0 <= axis && axis < Array.length s);
    Array.iter (fun j ->
      assert_shape loc "gather: index out of range"
        (0 <= j && j < s.(axis))) indices
  | Matmul, [a; b] ->
    assert_shape loc "matmul: a must be rank >= 2" (rank_of a >= 2);
    assert_shape loc "matmul: b must be rank >= 2" (rank_of b >= 2);
    assert_shape loc "matmul: a and b must have same rank"
      (rank_of a = rank_of b);
    let ra = rank_of a in
    let nframe = ra - 2 in
    let sa = shape_of a in
    let sb = shape_of b in
    assert_shape loc "matmul: inner dim mismatch"
      (sa.(nframe + 1) = sb.(nframe));
    for i = 0 to nframe - 1 do
      assert_shape loc "matmul: frame shape mismatch"
        (sa.(i) = sb.(i))
    done
  | _, _ ->
    raise (Eval_error (loc, "wrong number of arguments"))

(* --- map1/map2 function dispatch --- *)

let map1_f = function
  | Types.Neg  -> fun x -> -.x
  | Exp  -> exp
  | Log  -> log
  | Sqrt -> sqrt
  | Relu -> fun x -> if x > 0.0 then x else 0.0
  | Step -> fun x -> if x > 0.0 then 1.0 else 0.0
  | _ -> assert false

let map2_f = function
  | Types.Add  -> ( +. )
  | Sub  -> ( -. )
  | Mul  -> ( *. )
  | Div  -> ( /. )
  | Max2 -> max
  | _ -> assert false

(* --- output shape for allocating ops --- *)

let alloc_shape (p : Types.prim) (args : Tensor.t list) : int array =
  match p, args with
  | (Types.Neg | Exp | Log | Sqrt | Relu | Step), [x] -> shape_of x
  | (Types.Add | Sub | Mul | Div | Max2), [x; _] -> shape_of x
  | (Sum_axis axis | Max_axis axis), [x] ->
    let s = shape_of x in
    let r = Array.length s in
    Array.init (r - 1) (fun i -> if i < axis then s.(i) else s.(i + 1))
  | Gather (axis, indices), [x] ->
    let s = shape_of x in
    Array.init (Array.length s) (fun k ->
      if k = axis then Array.length indices else s.(k))
  | Matmul, [a; b] ->
    let sa = shape_of a in
    let sb = shape_of b in
    let ra = Array.length sa in
    let nframe = ra - 2 in
    Array.init (nframe + 2) (fun i ->
      if i < nframe then sa.(i)
      else if i = nframe then sa.(nframe)
      else sb.(Array.length sb - 1))
  (* view ops (Transpose/Broadcast/Slice/Reshape) are handled before
     alloc_shape is called; they never reach here. *)
  | _ -> assert false

(* --- eval --- *)

let rec eval (env : env) (e : Types.expr) : Types.value =
  match e with
  | Const (_, v) -> v
  | Var (loc, s) ->
    (match List.assoc_opt s env with
     | Some v -> v
     | None -> raise (Eval_error (loc, "unbound variable: " ^ s)))
  | Let (_, s, e1, e2) ->
    let v1 = eval env e1 in
    eval ((s, v1) :: env) e2
  | Rank (loc, _, _, _) ->
    raise (Eval_error (loc, "Rank node must be expanded before eval"))
  | Prim (loc, p, args) ->
    let vs = List.map (eval env) args in
    validate loc p vs;
    (match p, vs with
     (* --- view-only ops: no allocation, shared mark set --- *)
     | Transpose perm, [x] ->
       Tensor.transpose x ~perm
     | Broadcast (axis, size), [x] ->
       Tensor.broadcast x ~axis ~size
     | Slice ranges, [x] ->
       Tensor.slice x ~ranges
     | Reshape new_shape, [x] ->
       (* View ops always return shared = true for predictability.
          Non-contiguous: materialize first, then mark shared. *)
       (match Tensor.reshape x ~shape:new_shape with
        | Some t -> t
        | None ->
          let n = Array.fold_left ( * ) 1 new_shape in
          let dst = Buf.create n in
          Tensor.read_view ~src:x.buf ~view:x.view ~dst;
          let t = Tensor.of_buf dst (Ndview.contiguous new_shape) in
          t.buf.Buf.shared <- true; t)

     (* --- allocating ops: fresh dst --- *)
     | _ ->
       let os = alloc_shape p vs in
       let on = Array.fold_left ( * ) 1 os in
       let dst = Buf.create on in
       (match p, vs with
        | (Neg | Exp | Log | Sqrt | Relu | Step), [x] ->
          Kernel.Naive.map1 ~f:(map1_f p) ~src:x.buf ~view:x.view ~dst;
          Tensor.of_buf dst (Ndview.contiguous os)
        | (Add | Sub | Mul | Div | Max2), [x; y] ->
          Kernel.Naive.map2 ~f:(map2_f p) ~src1:x.buf ~view1:x.view
            ~src2:y.buf ~view2:y.view ~dst;
          Tensor.of_buf dst (Ndview.contiguous os)
        | Sum_axis axis, [x] ->
          Kernel.Naive.sum_axis ~src:x.buf ~view:x.view ~axis ~dst;
          Tensor.of_buf dst (Ndview.contiguous os)
        | Max_axis axis, [x] ->
          let argmax = Array.make on 0 in
          Kernel.Naive.max_axis ~src:x.buf ~view:x.view ~axis ~dst
            ~dst_argmax:argmax;
          Tensor.of_buf dst (Ndview.contiguous os)
        | Gather (axis, indices), [x] ->
          Kernel.Naive.gather ~src:x.buf ~view:x.view ~axis ~indices ~dst;
          Tensor.of_buf dst (Ndview.contiguous os)
        | Matmul, [a; b] ->
          let nframe = rank_of a - 2 in
          Kernel.Naive.matmul ~a:a.buf ~view_a:a.view ~b:b.buf ~view_b:b.view
            ~dst ~nframe;
          Tensor.of_buf dst (Ndview.contiguous os)
        | _ -> raise (Eval_error (loc, "wrong number of arguments"))))
