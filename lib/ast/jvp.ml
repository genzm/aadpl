(* Forward-mode AD: reference JVP interpreter.
   jvp_eval env expr = (primal, tangent).
   Each variable in env maps to a (primal, tangent) pair.
   Const has zero tangent. Structural ops are linear: tangent = same op on dt.
   §3.1: Gather indices are integer — no tangent contribution from indices. *)

open View

type dual = Types.value * Types.value   (* primal, tangent *)
type dual_env = (string * dual) list

let shape_of (t : Tensor.t) = t.view.Ndview.shape
let rank_of (t : Tensor.t) = Ndview.rank t.view

let zeros_like (t : Tensor.t) : Tensor.t =
  Tensor.make (shape_of t)

(* --- tensor arithmetic helpers --- *)

let t_map2 f (x : Tensor.t) (y : Tensor.t) : Tensor.t =
  let os = shape_of x in
  let on = Array.fold_left ( * ) 1 os in
  let dst = Buf.create on in
  Kernel.Naive.map2 ~f ~src1:x.buf ~view1:x.view
    ~src2:y.buf ~view2:y.view ~dst;
  Tensor.of_buf dst (Ndview.contiguous os)

let t_add = t_map2 ( +. )
let t_mul = t_map2 ( *. )

(* --- JVP interpreter --- *)

let rec jvp_eval (env : dual_env) (e : Types.expr) : dual =
  match e with
  | Const (_, v) -> (v, zeros_like v)
  | Var (loc, s) ->
    (match List.assoc_opt s env with
     | Some d -> d
     | None -> raise (Eval.Eval_error (loc, "unbound variable: " ^ s)))
  | Let (_, s, e1, e2) ->
    let d1 = jvp_eval env e1 in
    jvp_eval ((s, d1) :: env) e2
  | Rank (loc, _, _, _) ->
    raise (Eval.Eval_error (loc, "Rank node must be expanded before jvp_eval"))
  | Prim (loc, p, args) ->
    let ds = List.map (jvp_eval env) args in
    let vs = List.map fst ds in
    Eval.validate loc p vs;
    jvp_prim loc p ds

and jvp_prim _loc (p : Types.prim) (ds : dual list) : dual =
  match p, ds with

  (* --- map1: tangent = f'(x) * dx --- *)
  | Types.Neg, [(x, dx)] ->
    let os = shape_of x in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.map1 ~f:(fun v -> -.v) ~src:x.buf ~view:x.view ~dst:dst_p;
    let dst_t = Buf.create on in
    Kernel.Naive.map1 ~f:(fun v -> -.v) ~src:dx.buf ~view:dx.view ~dst:dst_t;
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     Tensor.of_buf dst_t (Ndview.contiguous os))

  | Exp, [(x, dx)] ->
    let os = shape_of x in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.map1 ~f:exp ~src:x.buf ~view:x.view ~dst:dst_p;
    let primal = Tensor.of_buf dst_p (Ndview.contiguous os) in
    (primal, t_mul primal dx)   (* exp'(x) = exp(x) *)

  | Log, [(x, dx)] ->
    let os = shape_of x in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.map1 ~f:log ~src:x.buf ~view:x.view ~dst:dst_p;
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     t_map2 ( /. ) dx x)       (* log'(x) = 1/x *)

  | Sqrt, [(x, dx)] ->
    let os = shape_of x in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.map1 ~f:sqrt ~src:x.buf ~view:x.view ~dst:dst_p;
    let primal = Tensor.of_buf dst_p (Ndview.contiguous os) in
    (* sqrt'(x) = 1/(2*sqrt(x)) = dx / (2*primal) *)
    let two_p = Tensor.make os in
    Kernel.Naive.map1 ~f:(fun v -> 2.0 *. v) ~src:primal.buf
      ~view:primal.view ~dst:two_p.buf;
    (primal, t_map2 ( /. ) dx two_p)

  | Relu, [(x, dx)] ->
    let os = shape_of x in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.map1 ~f:(fun v -> if v > 0.0 then v else 0.0)
      ~src:x.buf ~view:x.view ~dst:dst_p;
    (* relu'(x) = step(x), tangent = step(x) * dx *)
    let dst_s = Buf.create on in
    Kernel.Naive.map1 ~f:(fun v -> if v > 0.0 then 1.0 else 0.0)
      ~src:x.buf ~view:x.view ~dst:dst_s;
    let step_x = Tensor.of_buf dst_s (Ndview.contiguous os) in
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     t_mul step_x dx)

  | Step, [(x, _dx)] ->
    let os = shape_of x in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.map1 ~f:(fun v -> if v > 0.0 then 1.0 else 0.0)
      ~src:x.buf ~view:x.view ~dst:dst_p;
    (* step is piecewise constant — tangent is zero *)
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     zeros_like x)

  (* --- map2 --- *)
  | Add, [(_, dx); (_, dy)] ->
    let primal = Eval.eval [] (Types.prim p [Types.const (fst (List.hd ds));
                                             Types.const (fst (List.nth ds 1))]) in
    (primal, t_add dx dy)

  | Sub, [(_, dx); (_, dy)] ->
    let primal = Eval.eval [] (Types.prim p [Types.const (fst (List.hd ds));
                                             Types.const (fst (List.nth ds 1))]) in
    (primal, t_map2 ( -. ) dx dy)

  | Mul, [(x, dx); (y, dy)] ->
    let primal = Eval.eval [] (Types.prim p [Types.const x; Types.const y]) in
    (primal, t_add (t_mul dx y) (t_mul x dy))

  | Div, [(x, dx); (y, dy)] ->
    let primal = Eval.eval [] (Types.prim p [Types.const x; Types.const y]) in
    (* d(x/y) = (dx*y - x*dy) / y^2 *)
    let num = t_map2 ( -. ) (t_mul dx y) (t_mul x dy) in
    (primal, t_map2 ( /. ) num (t_mul y y))

  | Max2, [(x, dx); (y, dy)] ->
    let os = shape_of x in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.map2 ~f:max ~src1:x.buf ~view1:x.view
      ~src2:y.buf ~view2:y.view ~dst:dst_p;
    (* tangent = (1 - step(y-x)) * dx + step(y-x) * dy
       step(y-x) = 1 where y > x, 0 where y <= x
       So: where x >= y, pick dx; where y > x, pick dy.
       Tie (x = y) → step(0) = 0 → pick dx (left wins). *)
    let dst_yx = Buf.create on in
    Kernel.Naive.map2 ~f:( -. ) ~src1:y.buf ~view1:y.view
      ~src2:x.buf ~view2:x.view ~dst:dst_yx;
    let yx = Tensor.of_buf dst_yx (Ndview.contiguous os) in
    let dst_s = Buf.create on in
    Kernel.Naive.map1 ~f:(fun v -> if v > 0.0 then 1.0 else 0.0)
      ~src:yx.buf ~view:yx.view ~dst:dst_s;
    let s = Tensor.of_buf dst_s (Ndview.contiguous os) in
    (* (1 - s) * dx + s * dy *)
    let dst_oms = Buf.create on in
    Kernel.Naive.map1 ~f:(fun v -> 1.0 -. v) ~src:s.buf ~view:s.view ~dst:dst_oms;
    let one_minus_s = Tensor.of_buf dst_oms (Ndview.contiguous os) in
    let tangent = t_add (t_mul one_minus_s dx) (t_mul s dy) in
    (Tensor.of_buf dst_p (Ndview.contiguous os), tangent)

  (* --- reduce: sum_axis is linear --- *)
  | Sum_axis axis, [(x, dx)] ->
    let os = Eval.alloc_shape p [x] in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.sum_axis ~src:x.buf ~view:x.view ~axis ~dst:dst_p;
    let dst_t = Buf.create on in
    Kernel.Naive.sum_axis ~src:dx.buf ~view:dx.view ~axis ~dst:dst_t;
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     Tensor.of_buf dst_t (Ndview.contiguous os))

  (* --- reduce: max_axis — tangent = gather dx at argmax --- *)
  | Max_axis axis, [(x, dx)] ->
    let os = Eval.alloc_shape p [x] in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    let argmax = Array.make on 0 in
    Kernel.Naive.max_axis ~src:x.buf ~view:x.view ~axis ~dst:dst_p
      ~dst_argmax:argmax;
    let s = shape_of x in
    let r = Array.length s in
    let dst_t = Buf.create on in
    Ndview.iter_indices os (fun oi out_idx ->
      let full_idx = Array.init r (fun k ->
        if k < axis then out_idx.(k)
        else if k = axis then argmax.(oi)
        else out_idx.(k - 1)) in
      Buf.set dst_t oi
        (Buf.get dx.buf (Ndview.index_of dx.view full_idx)));
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     Tensor.of_buf dst_t (Ndview.contiguous os))

  (* --- structural ops: all linear — same op on tangent --- *)
  | Transpose perm, [(x, dx)] ->
    (Tensor.transpose x ~perm,
     Tensor.transpose dx ~perm)

  | Broadcast (axis, size), [(x, dx)] ->
    (Tensor.broadcast x ~axis ~size,
     Tensor.broadcast dx ~axis ~size)

  | Slice ranges, [(x, dx)] ->
    (Tensor.slice x ~ranges,
     Tensor.slice dx ~ranges)

  | Reshape new_shape, [(x, dx)] ->
    let reshape_one t =
      match Tensor.reshape t ~shape:new_shape with
      | Some r -> r
      | None ->
        let n = Array.fold_left ( * ) 1 new_shape in
        let dst = Buf.create n in
        Tensor.read_view ~src:t.buf ~view:t.view ~dst;
        let r = Tensor.of_buf dst (Ndview.contiguous new_shape) in
        r.buf.Buf.shared <- true; r
    in
    (reshape_one x, reshape_one dx)

  (* --- gather: linear in values (§3.1) --- *)
  | Gather (axis, indices), [(x, dx)] ->
    let os = Eval.alloc_shape p [x] in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.gather ~src:x.buf ~view:x.view ~axis ~indices ~dst:dst_p;
    let dst_t = Buf.create on in
    Kernel.Naive.gather ~src:dx.buf ~view:dx.view ~axis ~indices ~dst:dst_t;
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     Tensor.of_buf dst_t (Ndview.contiguous os))

  (* --- matmul: d(AB) = dA@B + A@dB --- *)
  | Matmul, [(a, da); (b, db)] ->
    let nframe = rank_of a - 2 in
    let os = Eval.alloc_shape p [a; b] in
    let on = Array.fold_left ( * ) 1 os in
    let dst_p = Buf.create on in
    Kernel.Naive.matmul ~a:a.buf ~view_a:a.view ~b:b.buf ~view_b:b.view
      ~dst:dst_p ~nframe;
    let dst_da_b = Buf.create on in
    Kernel.Naive.matmul ~a:da.buf ~view_a:da.view ~b:b.buf ~view_b:b.view
      ~dst:dst_da_b ~nframe;
    let dst_a_db = Buf.create on in
    Kernel.Naive.matmul ~a:a.buf ~view_a:a.view ~b:db.buf ~view_b:db.view
      ~dst:dst_a_db ~nframe;
    let t_da_b = Tensor.of_buf dst_da_b (Ndview.contiguous os) in
    let t_a_db = Tensor.of_buf dst_a_db (Ndview.contiguous os) in
    (Tensor.of_buf dst_p (Ndview.contiguous os),
     t_add t_da_b t_a_db)

  | _ -> assert false
