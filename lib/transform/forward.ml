(* Forward-mode AD: AST-to-AST transform.
   forward expr = (bindings, primal_expr, tangent_expr)
   where bindings is a list of (name, expr) pairs shared by both.
   Generated variable names use '%' prefix to avoid user-variable collision. *)

open Ast.Types

type bindings = (string * expr) list

(* --- gensym --- *)

let counter = ref 0

let gensym prefix =
  let n = !counter in
  incr counter;
  Printf.sprintf "%%%s%d" prefix n

let reset_gensym () = counter := 0

(* tangent variable for user variable s *)
let tangent_name s = "%" ^ s ^ ".t"

(* --- helpers --- *)

(* Ensure expr is a Var; if not, bind it and return (bindings, var_expr). *)
let ensure_var (prefix : string) (e : expr) : bindings * expr =
  match e with
  | Var _ -> ([], e)
  | _ ->
    let name = gensym prefix in
    ([(name, e)], var name)

(* Wrap bindings around an expression as nested Lets. *)
let wrap_bindings (bs : bindings) (body : expr) : expr =
  List.fold_right (fun (s, e) acc -> let_ s e acc) bs body

(* --- forward transform --- *)

(* Returns (bindings, primal_expr, tangent_expr).
   bindings contain both primal intermediates and residuals.
   Both primal_expr and tangent_expr may reference bindings variables. *)

let rec forward (e : expr) : bindings * expr * expr =
  match e with
  | Const (_, v) ->
    let shape = v.View.Tensor.view.View.Ndview.shape in
    let zero = View.Tensor.make shape in
    ([], e, const zero)

  | Var (_, s) ->
    ([], e, var (tangent_name s))

  | Let (_, s, e1, e2) ->
    let (bs1, p1, t1) = forward e1 in
    let ts = tangent_name s in
    let (bs2, p2, t2) = forward e2 in
    (* bindings: bs1, then bind s=p1 and %s.t=t1, then bs2 *)
    (bs1 @ [(s, p1); (ts, t1)] @ bs2, p2, t2)

  | Rank (loc, _, _, _) ->
    raise (Ast.Eval.Eval_error (loc, "Rank must be expanded before forward"))

  | Prim (_loc, p, args) ->
    let fwd_args = List.map forward args in
    forward_prim p fwd_args

and forward_prim (p : prim) (fwd_args : (bindings * expr * expr) list)
    : bindings * expr * expr =
  match p, fwd_args with

  (* === map1 linear: tangent = same op === *)
  | Neg, [(bs, px, tx)] ->
    (bs, prim Neg [px], prim Neg [tx])

  (* === map1 nonlinear === *)
  | Exp, [(bs, px, tx)] ->
    (* residual: exp(x); tangent = exp(x) * tx *)
    let (bs_v, v) = ensure_var "p" px in
    let r = gensym "r" in
    let all_bs = bs @ bs_v @ [(r, prim Exp [v])] in
    (all_bs, var r, prim Mul [var r; tx])

  | Log, [(bs, px, tx)] ->
    (* tangent = tx / x *)
    let (bs_v, v) = ensure_var "p" px in
    (bs @ bs_v, prim Log [v], prim Div [tx; v])

  | Sqrt, [(bs, px, tx)] ->
    (* residual: sqrt(x); tangent = tx / (sqrt(x) + sqrt(x)) *)
    let (bs_v, v) = ensure_var "p" px in
    let r = gensym "r" in
    let all_bs = bs @ bs_v @ [(r, prim Sqrt [v])] in
    (all_bs, var r, prim Div [tx; prim Add [var r; var r]])

  | Relu, [(bs, px, tx)] ->
    (* tangent = step(x) * tx *)
    let (bs_v, v) = ensure_var "p" px in
    (bs @ bs_v, prim Relu [v], prim Mul [prim Step [v]; tx])

  | Step, [(bs, px, _tx)] ->
    (* piecewise constant: tangent = zero = step(x) - step(x) *)
    let (bs_v, v) = ensure_var "p" px in
    let r = gensym "r" in
    let all_bs = bs @ bs_v @ [(r, prim Step [v])] in
    (all_bs, var r, prim Sub [var r; var r])

  (* === map2 linear === *)
  | Add, [(bs1, p1, t1); (bs2, p2, t2)] ->
    (bs1 @ bs2, prim Add [p1; p2], prim Add [t1; t2])

  | Sub, [(bs1, p1, t1); (bs2, p2, t2)] ->
    (bs1 @ bs2, prim Sub [p1; p2], prim Sub [t1; t2])

  (* === map2 nonlinear === *)
  | Mul, [(bs1, p1, t1); (bs2, p2, t2)] ->
    (* d(x*y) = dx*y + x*dy — need residuals x and y *)
    let (bv1, v1) = ensure_var "p" p1 in
    let (bv2, v2) = ensure_var "p" p2 in
    (bs1 @ bs2 @ bv1 @ bv2,
     prim Mul [v1; v2],
     prim Add [prim Mul [t1; v2]; prim Mul [v1; t2]])

  | Div, [(bs1, p1, t1); (bs2, p2, t2)] ->
    (* d(x/y) = (dx*y - x*dy) / y^2 *)
    let (bv1, v1) = ensure_var "p" p1 in
    let (bv2, v2) = ensure_var "p" p2 in
    (bs1 @ bs2 @ bv1 @ bv2,
     prim Div [v1; v2],
     prim Div [prim Sub [prim Mul [t1; v2]; prim Mul [v1; t2]];
               prim Mul [v2; v2]])

  | Max2, [(bs1, p1, t1); (bs2, p2, t2)] ->
    (* tangent = t1 + step(p2-p1) * (t2-t1) *)
    let (bv1, v1) = ensure_var "p" p1 in
    let (bv2, v2) = ensure_var "p" p2 in
    (bs1 @ bs2 @ bv1 @ bv2,
     prim Max2 [v1; v2],
     prim Add [t1; prim Mul [prim Step [prim Sub [v2; v1]];
                             prim Sub [t2; t1]]])

  (* === reduce: linear === *)
  | Sum_axis _ as p, [(bs, px, tx)] ->
    (bs, prim p [px], prim p [tx])

  (* === structural: linear — tangent = same op === *)
  | (Gather _ | Scatter_add _
    | Apply_view _ | Adjoint_view _) as p,
    [(bs, px, tx)] ->
    (bs, prim p [px], prim p [tx])

  (* scatter_select_add: linear in first arg *)
  | Scatter_select_add _ as p, [(bs1, p1, t1); (bs2, p2, _t2)] ->
    (bs1 @ bs2, prim p [p1; p2], prim p [t1; p2])

  (* === matmul: d(AB) = dA@B + A@dB === *)
  | Matmul, [(bs1, p1, t1); (bs2, p2, t2)] ->
    let (bv1, v1) = ensure_var "p" p1 in
    let (bv2, v2) = ensure_var "p" p2 in
    (bs1 @ bs2 @ bv1 @ bv2,
     prim Matmul [v1; v2],
     prim Add [prim Matmul [t1; v2]; prim Matmul [v1; t2]])

  (* === max_axis: tangent = select_axis(dx, argmax(x)) === *)
  | Max_axis axis, [(bs, px, tx)] ->
    let (bs_v, v) = ensure_var "p" px in
    let am = gensym "am" in
    let all_bs = bs @ bs_v @ [(am, prim (Argmax_axis axis) [v])] in
    (all_bs,
     prim (Max_axis axis) [v],
     prim (Select_axis axis) [tx; var am])

  (* === argmax: integer-valued, tangent is zero (T(Z) = 0) === *)
  | Argmax_axis _ as p, [(bs, px, _tx)] ->
    let r = gensym "r" in
    let all_bs = bs @ [(r, prim p [px])] in
    (all_bs, var r, prim Sub [var r; var r])

  (* === select_axis: linear in first arg, index has no tangent === *)
  | Select_axis _ as p, [(bs1, p1, t1); (bs2, p2, _t2)] ->
    (bs1 @ bs2,
     prim p [p1; p2],
     prim p [t1; p2])

  | _ -> assert false
