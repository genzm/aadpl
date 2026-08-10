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

(* --- viewspec validation --- *)

(* Validate each viewop's preconditions against the current shape,
   raising Shape_mismatch on violation. *)
let validate_viewspec loc (spec : Types.viewspec) (in_shape : int array) : unit =
  let sh = ref in_shape in
  List.iter (fun (op : Types.viewop) ->
    let s = !sh in
    let r = Array.length s in
    (match op with
     | Vtranspose p ->
       assert_shape loc "viewspec: transpose perm length mismatch"
         (Array.length p = r);
       let seen = Array.make r false in
       Array.iter (fun i ->
         assert_shape loc "viewspec: transpose perm index out of range"
           (0 <= i && i < r);
         assert_shape loc "viewspec: transpose perm not a permutation"
           (not seen.(i));
         seen.(i) <- true) p
     | Vslice ranges ->
       assert_shape loc "viewspec: slice ranges length mismatch"
         (Array.length ranges = r);
       Array.iteri (fun k (start, stop, step) ->
         assert_shape loc "viewspec: slice step must be nonzero" (step <> 0);
         if step > 0 then begin
           assert_shape loc "viewspec: slice start out of range"
             (0 <= start && start < s.(k));
           assert_shape loc "viewspec: slice stop out of range"
             (start < stop && stop <= s.(k))
         end else begin
           assert_shape loc "viewspec: slice start out of range"
             (0 <= start && start < s.(k));
           assert_shape loc "viewspec: slice stop out of range"
             (-1 <= stop && stop < start)
         end) ranges
     | Vbroadcast (axis, size) ->
       assert_shape loc "viewspec: broadcast axis out of range"
         (0 <= axis && axis <= r);
       assert_shape loc "viewspec: broadcast size must be positive"
         (size > 0)
     | Vreshape new_shape ->
       let old_numel = Array.fold_left ( * ) 1 s in
       let new_numel = Array.fold_left ( * ) 1 new_shape in
       assert_shape loc "viewspec: reshape numel mismatch"
         (old_numel = new_numel));
    sh := Types.viewop_output_shape op s
  ) spec

(* --- output shape validation per primitive --- *)

(* --- viewspec realization --- *)

let realize_viewop (v : Ndview.view) (op : Types.viewop) : Ndview.view option =
  match op with
  | Vtranspose p -> Some (Ndview.transpose v ~perm:p)
  | Vslice r -> Some (Ndview.slice v ~ranges:r)
  | Vbroadcast (a, s) -> Some (Ndview.broadcast v ~axis:a ~size:s)
  | Vreshape s -> Ndview.reshape v ~shape:s

(* Realize a viewspec starting from contiguous in_shape.
   Returns None if reshape hits a non-contiguous intermediate. *)
let realize_spec (spec : Types.viewspec) (in_shape : int array) : Ndview.view option =
  let v = ref (Ndview.contiguous in_shape) in
  let ok = ref true in
  List.iter (fun op ->
    if !ok then
      match realize_viewop !v op with
      | Some v' -> v := v'
      | None -> ok := false
  ) spec;
  if !ok then Some !v else None

(* Apply a single viewop's adjoint: scatter-add x through the viewop's view *)
let adjoint_viewop_eval (op : Types.viewop) (target_shape : int array) (x : Tensor.t)
    : Tensor.t =
  let on = Array.fold_left ( * ) 1 target_shape in
  let dst = Buf.create on in
  for i = 0 to on - 1 do Buf.set dst i 0.0 done;
  (* From contiguous target_shape, realize one viewop — always succeeds *)
  let v = match realize_viewop (Ndview.contiguous target_shape) op with
    | Some v -> v
    | None -> failwith "adjoint_viewop_eval: unexpected reshape failure on contiguous" in
  let xn = Ndview.numel x.view in
  let x_buf =
    if Ndview.is_contiguous x.view then x.buf
    else begin
      let tmp = Buf.create xn in
      Tensor.read_view ~src:x.buf ~view:x.view ~dst:tmp; tmp
    end in
  Tensor.add_view ~src:x_buf ~view:v ~acc:dst;
  Tensor.of_buf dst (Ndview.contiguous target_shape)

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
  | Gather (axis, indices), [x] ->
    let s = shape_of x in
    assert_shape loc "gather: axis out of range"
      (0 <= axis && axis < Array.length s);
    Array.iter (fun j ->
      assert_shape loc "gather: index out of range"
        (0 <= j && j < s.(axis))) indices
  | Argmax_axis axis, [x] ->
    assert_shape loc "argmax_axis: axis out of range"
      (0 <= axis && axis < Array.length (shape_of x))
  | Select_axis axis, [x; idx] ->
    assert_shape loc "select_axis: axis out of range"
      (0 <= axis && axis < Array.length (shape_of x));
    (* idx shape = x shape with axis removed *)
    let sx = shape_of x in
    let r = Array.length sx in
    let expected = Array.init (r - 1) (fun i ->
      if i < axis then sx.(i) else sx.(i + 1)) in
    assert_shape loc "select_axis: index shape mismatch"
      (shape_of idx = expected)
  | Scatter_add (_axis, _indices, _target_shape), [_x] -> ()
  | Scatter_select_add (axis, _target_axis_size), [_x; idx] ->
    let si = shape_of idx in
    let sx = shape_of _x in
    assert_shape loc "scatter_select_add: index shape mismatch"
      (si = sx);
    assert_shape loc "scatter_select_add: axis out of range"
      (0 <= axis && axis < Array.length sx + 1)
  | Apply_view spec, [x] ->
    validate_viewspec loc spec (shape_of x)
  | Adjoint_view (spec, target_shape), [x] ->
    validate_viewspec loc spec target_shape;
    let expected = Types.viewspec_output_shape spec target_shape in
    assert_shape loc "adjoint_view: input shape mismatch"
      (shape_of x = expected)
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
  | (Sum_axis axis | Max_axis axis | Argmax_axis axis), [x] ->
    let s = shape_of x in
    let r = Array.length s in
    Array.init (r - 1) (fun i -> if i < axis then s.(i) else s.(i + 1))
  | Select_axis axis, [x; _idx] ->
    let s = shape_of x in
    let r = Array.length s in
    Array.init (r - 1) (fun i -> if i < axis then s.(i) else s.(i + 1))
  | Scatter_add (_, _, target_shape), [_] -> target_shape
  | Scatter_select_add (axis, target_axis_size), [x; _] ->
    let sx = shape_of x in
    let r = Array.length sx in
    Array.init (r + 1) (fun i ->
      if i < axis then sx.(i)
      else if i = axis then target_axis_size
      else sx.(i - 1))
  | Gather (axis, indices), [x] ->
    let s = shape_of x in
    Array.init (Array.length s) (fun k ->
      if k = axis then Array.length indices else s.(k))
  | Adjoint_view (_, target_shape), [_] -> target_shape
  | Matmul, [a; b] ->
    let sa = shape_of a in
    let sb = shape_of b in
    let ra = Array.length sa in
    let nframe = ra - 2 in
    Array.init (nframe + 2) (fun i ->
      if i < nframe then sa.(i)
      else if i = nframe then sa.(nframe)
      else sb.(Array.length sb - 1))
  (* view ops (Transpose/Broadcast/Slice/Reshape/Apply_view) are handled before
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
     | Apply_view spec, [x] ->
       (* Chain view ops on input tensor *)
       List.fold_left (fun t (op : Types.viewop) -> match op with
         | Vtranspose p -> Tensor.transpose t ~perm:p
         | Vslice r -> Tensor.slice t ~ranges:r
         | Vbroadcast (a, s) -> Tensor.broadcast t ~axis:a ~size:s
         | Vreshape s ->
           (match Tensor.reshape t ~shape:s with
            | Some r -> r
            | None ->
              let n = Array.fold_left ( * ) 1 s in
              let dst = Buf.create n in
              Tensor.read_view ~src:t.buf ~view:t.view ~dst;
              let r = Tensor.of_buf dst (Ndview.contiguous s) in
              r.buf.Buf.shared <- true; r)
       ) x spec

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
        | Argmax_axis axis, [x] ->
          let argmax = Array.make on 0 in
          (* reuse max_axis kernel; discard max values, keep argmax *)
          let dst_max = Buf.create on in
          Kernel.Naive.max_axis ~src:x.buf ~view:x.view ~axis ~dst:dst_max
            ~dst_argmax:argmax;
          for i = 0 to on - 1 do
            Buf.set dst i (float_of_int argmax.(i))
          done;
          Tensor.of_buf dst (Ndview.contiguous os)
        | Select_axis axis, [x; idx] ->
          (* select x[..., idx[out_idx], ...] along axis *)
          let s = shape_of x in
          let r = Array.length s in
          Ndview.iter_indices os (fun oi out_idx ->
            let j = int_of_float (Buf.get idx.buf
              (Ndview.index_of idx.view out_idx)) in
            let full_idx = Array.init r (fun k ->
              if k < axis then out_idx.(k)
              else if k = axis then j
              else out_idx.(k - 1)) in
            Buf.set dst oi
              (Buf.get x.buf (Ndview.index_of x.view full_idx)));
          Tensor.of_buf dst (Ndview.contiguous os)
        | Gather (axis, indices), [x] ->
          Kernel.Naive.gather ~src:x.buf ~view:x.view ~axis ~indices ~dst;
          Tensor.of_buf dst (Ndview.contiguous os)
        | Scatter_add (axis, indices, target_shape), [x] ->
          (* Materialize x to contiguous (scatter_add reads src flatly) *)
          let sx = shape_of x in
          let sn = Array.fold_left ( * ) 1 sx in
          let x_buf = if Ndview.is_contiguous x.view then x.buf
            else begin
              let tmp = Buf.create sn in
              Tensor.read_view ~src:x.buf ~view:x.view ~dst:tmp; tmp
            end in
          for i = 0 to on - 1 do Buf.set dst i 0.0 done;
          let target_view = Ndview.contiguous target_shape in
          Kernel.Naive.scatter_add ~src:x_buf ~view:target_view
            ~axis ~indices ~acc:dst;
          Tensor.of_buf dst (Ndview.contiguous target_shape)
        | Scatter_select_add (axis, target_axis_size), [x; idx] ->
          (* Zero-initialize target, scatter x back using idx *)
          for i = 0 to on - 1 do Buf.set dst i 0.0 done;
          let sx = shape_of x in
          let r = Array.length sx in
          let target_shape = Array.init (r + 1) (fun i ->
            if i < axis then sx.(i)
            else if i = axis then target_axis_size
            else sx.(i - 1)) in
          let target_view = Ndview.contiguous target_shape in
          Ndview.iter_indices sx (fun _si src_idx ->
            let j = int_of_float (Buf.get idx.buf
              (Ndview.index_of idx.view src_idx)) in
            let full_idx = Array.init (r + 1) (fun k ->
              if k < axis then src_idx.(k)
              else if k = axis then j
              else src_idx.(k - 1)) in
            let pos = Ndview.index_of target_view full_idx in
            Buf.set dst pos
              (Buf.get dst pos +.
               Buf.get x.buf (Ndview.index_of x.view src_idx)));
          Tensor.of_buf dst (Ndview.contiguous target_shape)
        | Adjoint_view (spec, target_shape), [x] ->
          (* Decompose: walk spec in reverse, applying single-op adjoints *)
          let shapes = Types.viewspec_intermediate_shapes spec target_shape in
          let ops = Array.of_list spec in
          let current = ref x in
          for i = Array.length ops - 1 downto 0 do
            current := adjoint_viewop_eval ops.(i) shapes.(i) !current
          done;
          !current
        | Matmul, [a; b] ->
          let nframe = rank_of a - 2 in
          Kernel.Naive.matmul ~a:a.buf ~view_a:a.view ~b:b.buf ~view_b:b.view
            ~dst ~nframe;
          Tensor.of_buf dst (Ndview.contiguous os)
        | _ -> raise (Eval_error (loc, "wrong number of arguments"))))
