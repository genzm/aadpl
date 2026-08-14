(* Forward-mode AD: AST-to-AST transform.
   forward expr = (bindings, primal_expr, tangent_expr)
   where bindings are let-like scopes shared by both.
   Generated variable names use '%' prefix to avoid user-variable collision. *)

open Ast.Types

type bindings = binding list

(* --- gensym --- *)

let counter = ref 0
let ns = ref ""

let gensym prefix =
  let n = !counter in
  incr counter;
  Printf.sprintf "%%%s%s%d" !ns prefix n

let reset_gensym () = counter := 0

let with_ns new_ns f =
  let old = !ns in
  ns := new_ns;
  let result = f () in
  ns := old;
  result

(* tangent variable for user variable s *)
let tangent_name s = "%" ^ s ^ ".t"

(* --- helpers --- *)

(* Ensure expr is a Var; if not, bind it and return (bindings, var_expr). *)
let ensure_var (prefix : string) (e : expr) : bindings * expr =
  match e with
  | Var _ -> ([], e)
  | _ ->
      let name = gensym prefix in
      ([ Let_binding (name, e) ], var name)

(* Wrap bindings around an expression as nested Lets. *)
let wrap_bindings (bs : bindings) (body : expr) : expr =
  List.fold_right (fun binding acc ->
    match binding with
    | Let_binding (name, expression) -> let_ name expression acc
    | Scan_binding (loc, scan) -> Scan (loc, scan, acc)) bs body

let of_let_bindings bindings =
  List.map (fun (name, expression) -> Let_binding (name, expression)) bindings

let wrap_let_bindings bindings body =
  wrap_bindings (of_let_bindings bindings) body

module SS = Set.Make(String)

let rec free_vars = function
  | Const _ -> SS.empty
  | Var (_, name) -> SS.singleton name
  | Prim (_, _, arguments) -> List.fold_left (fun variables argument ->
      SS.union variables (free_vars argument)) SS.empty arguments
  | Let (_, name, bound, body) ->
      SS.union (free_vars bound) (SS.remove name (free_vars body))
  | Scan _ -> failwith "forward free_vars: nested Scan not supported yet"
  | Rank _ -> failwith "forward free_vars: Rank must be expanded first"
  | Sample _ | Score _ -> failwith "forward free_vars: effect in Scan body"

let rec subst_many substitutions expression =
  match expression with
  | Var (_, name) ->
      (match List.assoc_opt name substitutions with
       | Some replacement -> replacement
       | None -> expression)
  | Const _ -> expression
  | Prim (loc, primitive, arguments) ->
      Prim (loc, primitive, List.map (subst_many substitutions) arguments)
  | Let (loc, name, bound, body) ->
      Let (loc, name, subst_many substitutions bound,
        subst_many (List.remove_assoc name substitutions) body)
  | Scan _ -> failwith "forward subst: nested Scan not supported yet"
  | Rank _ -> failwith "forward subst: Rank must be expanded first"
  | Sample _ | Score _ -> failwith "forward subst: effect in Scan body"

let select_index index expression =
  prim (Sum_axis 0) [prim (Gather (0, [|index|])) [expression]]

let select_step steps expression = select_index (steps - 1) expression

let reverse_steps steps expression =
  prim (Gather (0, Array.init steps (fun index -> steps - 1 - index))) [expression]

let classify_local bindings seeds =
  let dependencies = ref seeds in
  let primal = ref [] and tangent = ref [] in
  List.iter (function
    | Scan_binding _ -> failwith "forward: nested Scan binding in body"
    | Let_binding (name, rhs) as binding ->
      if SS.is_empty (SS.inter (free_vars rhs) !dependencies) then
        primal := binding :: !primal
      else begin
        tangent := binding :: !tangent;
        dependencies := SS.add name !dependencies
      end) bindings;
  List.rev !primal, List.rev !tangent

let inline_primal_bindings bindings =
  let substitutions = ref [] in
  List.map (function
    | Scan_binding _ -> failwith "forward: nested Scan binding in body"
    | Let_binding (name, rhs) ->
      let rhs = subst_many !substitutions rhs in
      substitutions := (name, rhs) :: !substitutions;
      name, rhs) bindings

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
  | Var (_, s) -> ([], e, var (tangent_name s))
  | Let (_, s, e1, e2) ->
      let bs1, p1, t1 = forward e1 in
      let ts = tangent_name s in
      let bs2, p2, t2 = forward e2 in
      (* bindings: bs1, then bind s=p1 and %s.t=t1, then bs2 *)
      (bs1 @ [ Let_binding (s, p1); Let_binding (ts, t1) ] @ bs2, p2, t2)
  | Scan (loc, scan, continuation) -> forward_scan loc scan continuation
  | Rank (loc, _, _, _) ->
      raise (Ast.Eval.Eval_error (loc, "Rank must be expanded before forward"))
  | Sample _ -> failwith "forward: Sample not supported"
  | Score _ -> failwith "forward: Score not supported"
  | Prim (_loc, p, args) ->
      let fwd_args = List.map forward args in
      forward_prim p fwd_args

and forward_scan loc scan continuation =
  let carry_names = List.map (fun (name, _, _) -> name) scan.carries in
  let input_names = List.map fst scan.inputs in
  let primal_carries = List.map (fun name -> name, gensym "scan.p.") carry_names in
  let primal_inputs = List.map (fun name -> name, gensym "scan.x.") input_names in
  let rename =
    List.map (fun (name, generated) -> name, var generated)
      (primal_carries @ primal_inputs) in
  let forward_initials = List.map (fun (name, init, _) ->
    let bindings, primal, tangent = forward init in
    name, bindings, primal, tangent) scan.carries in
  let forward_inputs = List.map (fun (name, input) ->
    let bindings, primal, tangent = forward input in
    name, bindings, primal, tangent) scan.inputs in
  let outer_bindings =
    List.concat_map (fun (_, bindings, _, _) -> bindings) forward_initials
    @ List.concat_map (fun (_, bindings, _, _) -> bindings) forward_inputs in
  let body_results = List.map (fun (name, _, next) ->
    let renamed = subst_many rename next in
    let bindings, primal, tangent = forward renamed in
    name, bindings, primal, tangent) scan.carries in
  let body_free = List.fold_left (fun variables (_, _, next) ->
    SS.union variables (free_vars (subst_many rename next))) SS.empty scan.carries in
  let seeds = SS.fold (fun name seeds -> SS.add (tangent_name name) seeds)
    body_free SS.empty in
  let body_bindings =
    List.concat_map (fun (_, bindings, _, _) -> bindings) body_results in
  let primal_bindings, tangent_bindings = classify_local body_bindings seeds in
  let tangent_uses = List.fold_left (fun variables binding -> match binding with
    | Scan_binding _ -> failwith "forward: nested Scan binding in body"
    | Let_binding (_, rhs) -> SS.union variables (free_vars rhs)) SS.empty
      tangent_bindings
    |> fun variables -> List.fold_left (fun variables (_, _, _, tangent) ->
      SS.union variables (free_vars tangent)) variables body_results in
  let inlined_primal = inline_primal_bindings primal_bindings
    |> List.filter (fun (name, _) -> SS.mem name tangent_uses) in
  let primal_next name =
    let _, _, primal, _ = List.find (fun (candidate, _, _, _) ->
      candidate = name) body_results in
    wrap_bindings primal_bindings primal in
  let tangent_next name =
    let _, _, _, tangent = List.find (fun (candidate, _, _, _) ->
      candidate = name) body_results in
    wrap_bindings tangent_bindings tangent in
  let pre_names = List.map (fun name -> name, gensym "scan.pre.") carry_names in
  let residual_names = List.map (fun (name, _) -> name, gensym "scan.r.")
    inlined_primal in
  let first_index = if scan.reverse then scan.steps - 1 else 0 in
  let initial_substitutions =
    List.map (fun (name, generated) ->
      let _, _, primal, _ = List.find (fun (candidate, _, _, _) ->
        candidate = name) forward_initials in
      generated, primal) primal_carries
    @ List.map (fun (name, generated) ->
      let _, _, primal, _ = List.find (fun (candidate, _, _, _) ->
        candidate = name) forward_inputs in
      generated, select_index first_index primal)
        primal_inputs in
  let primal_scan_carries =
    List.map (fun (name, generated) ->
      let _, _, initial, _ = List.find (fun (candidate, _, _, _) ->
        candidate = name) forward_initials in
      generated, initial, primal_next name) primal_carries
    @ List.map (fun (name, generated) ->
      let _, _, initial, _ = List.find (fun (candidate, _, _, _) ->
        candidate = name) forward_initials in
      let pre = List.assoc name pre_names in
      pre, initial, var generated) primal_carries
    @ List.map2 (fun (_, expression) (_, trajectory) ->
      trajectory, subst_many initial_substitutions expression, expression)
        inlined_primal residual_names in
  let primal_scan_inputs = List.map (fun (name, generated) ->
    let _, _, primal, _ = List.find (fun (candidate, _, _, _) ->
      candidate = name) forward_inputs in
    generated, primal) primal_inputs in
  let primal_scan = {
    steps = scan.steps; carries = primal_scan_carries;
    inputs = primal_scan_inputs; collect = true; reverse = scan.reverse;
  } in
  let primal_aliases = List.map (fun (name, generated) ->
    let output = if scan.collect then var generated
      else select_step scan.steps (var generated) in
    Let_binding (name, output)) primal_carries in
  let tangent_scan_carries = List.map (fun (name, generated) ->
    let _, _, _, initial = List.find (fun (candidate, _, _, _) ->
      candidate = name) forward_initials in
    tangent_name generated, initial, tangent_next name) primal_carries in
  let tangent_scan_inputs =
    List.map (fun (name, generated) ->
      let trajectory = var (List.assoc name pre_names) in
      generated, if scan.reverse then reverse_steps scan.steps trajectory
        else trajectory)
      primal_carries
    @ List.map (fun (name, generated) ->
      let _, _, primal, _ = List.find (fun (candidate, _, _, _) ->
        candidate = name) forward_inputs in
      generated, primal) primal_inputs
    @ List.map (fun (name, generated) ->
      tangent_name generated,
      let original = List.find (fun (candidate, _, _, _) ->
        candidate = name) forward_inputs in
      let _, _, _, tangent = original in tangent) primal_inputs
    @ List.map2 (fun (local, _) (_, trajectory) ->
        let trajectory = var trajectory in
        local, if scan.reverse then reverse_steps scan.steps trajectory
          else trajectory)
        inlined_primal residual_names in
  let tangent_scan = {
    steps = scan.steps; carries = tangent_scan_carries;
    inputs = tangent_scan_inputs; collect = scan.collect; reverse = scan.reverse;
  } in
  let tangent_aliases = List.map (fun (name, generated) ->
    Let_binding (tangent_name name, var (tangent_name generated))) primal_carries in
  let continuation_bindings, primal_out, tangent_out = forward continuation in
  (outer_bindings
   @ [Scan_binding (loc, primal_scan)] @ primal_aliases
   @ [Scan_binding (loc, tangent_scan)] @ tangent_aliases
   @ continuation_bindings,
   primal_out, tangent_out)

and forward_prim (p : prim) (fwd_args : (bindings * expr * expr) list) :
    bindings * expr * expr =
  match (p, fwd_args) with
  (* === map1 linear: tangent = same op === *)
  | Neg, [ (bs, px, tx) ] -> (bs, prim Neg [ px ], prim Neg [ tx ])
  (* === map1 nonlinear === *)
  | Exp, [ (bs, px, tx) ] ->
      (* residual: exp(x); tangent = exp(x) * tx *)
      let bs_v, v = ensure_var "p" px in
      let r = gensym "r" in
      let all_bs = bs @ bs_v @ [ Let_binding (r, prim Exp [ v ]) ] in
      (all_bs, var r, prim Mul [ var r; tx ])
  | Log, [ (bs, px, tx) ] ->
      (* tangent = tx / x *)
      let bs_v, v = ensure_var "p" px in
      (bs @ bs_v, prim Log [ v ], prim Div [ tx; v ])
  | Logsigmoid, [ (bs, px, tx) ] ->
      (* d logsigmoid(x) = sigmoid(-x) = exp(logsigmoid(-x)) *)
      let bs_v, v = ensure_var "p" px in
      ( bs @ bs_v,
        prim Logsigmoid [ v ],
        prim Mul [ prim Exp [ prim Logsigmoid [ prim Neg [ v ] ] ]; tx ] )
  | Log_unit_density, [ (bs, px, tx) ] ->
      let bs_v, v = ensure_var "p" px in
      (* tx - tx is the shape-polymorphic zero available in the current IR. *)
      (bs @ bs_v, prim Log_unit_density [v], prim Sub [tx; tx])
  | (Log_support_density _ as p), [ (bs, px, tx) ] ->
      let bs_v, v = ensure_var "p" px in
      (bs @ bs_v, prim p [v], prim Sub [tx; tx])
  | Sqrt, [ (bs, px, tx) ] ->
      (* residual: sqrt(x); tangent = tx / (sqrt(x) + sqrt(x)) *)
      let bs_v, v = ensure_var "p" px in
      let r = gensym "r" in
      let all_bs = bs @ bs_v @ [ Let_binding (r, prim Sqrt [ v ]) ] in
      (all_bs, var r, prim Div [ tx; prim Add [ var r; var r ] ])
  | Relu, [ (bs, px, tx) ] ->
      (* tangent = step(x) * tx *)
      let bs_v, v = ensure_var "p" px in
      (bs @ bs_v, prim Relu [ v ], prim Mul [ prim Step [ v ]; tx ])
  | Step, [ (bs, px, _tx) ] ->
      (* piecewise constant: tangent = zero = step(x) - step(x) *)
      let bs_v, v = ensure_var "p" px in
      let r = gensym "r" in
      let all_bs = bs @ bs_v @ [ Let_binding (r, prim Step [ v ]) ] in
      (all_bs, var r, prim Sub [ var r; var r ])
  | Erf, [ (bs, px, tx) ] ->
      (* erf'(x) = (2/√π) exp(-x²); tangent = (2/√π) exp(-x²) * tx *)
      let bs_v, v = ensure_var "p" px in
      let mk_scalar f =
        let t = View.Tensor.make [||] in
        View.Buf.set t.buf 0 f;
        const t
      in
      let two_over_sqrtpi = mk_scalar 1.1283791670955126 in
      ( bs @ bs_v,
        prim Erf [ v ],
        rank 0 Mul
          [
            rank 0 Mul
              [ two_over_sqrtpi; prim Exp [ prim Neg [ prim Mul [ v; v ] ] ] ];
            tx;
          ] )
  | Erfinv, [ (bs, px, tx) ] ->
      (* erfinv'(x) = (√π/2) exp(erfinv(x)²); residual = erfinv(x) *)
      let bs_v, v = ensure_var "p" px in
      let r = gensym "r" in
      let mk_scalar f =
        let t = View.Tensor.make [||] in
        View.Buf.set t.buf 0 f;
        const t
      in
      let sqrtpi_over_2 = mk_scalar 0.88622692545275801 in
      let all_bs = bs @ bs_v @ [ Let_binding (r, prim Erfinv [ v ]) ] in
      ( all_bs,
        var r,
        rank 0 Mul
          [
            rank 0 Mul [ sqrtpi_over_2; prim Exp [ prim Mul [ var r; var r ] ] ];
            tx;
          ] )
  (* === map2 linear === *)
  | Add, [ (bs1, p1, t1); (bs2, p2, t2) ] ->
      (bs1 @ bs2, prim Add [ p1; p2 ], prim Add [ t1; t2 ])
  | Sub, [ (bs1, p1, t1); (bs2, p2, t2) ] ->
      (bs1 @ bs2, prim Sub [ p1; p2 ], prim Sub [ t1; t2 ])
  | Mask, [ (bs1, p1, t1); (bs2, p2, _t2) ] ->
      (bs1 @ bs2, prim Mask [p1; p2], prim Mask [t1; p2])
  (* === map2 nonlinear === *)
  | Mul, [ (bs1, p1, t1); (bs2, p2, t2) ] ->
      (* d(x*y) = dx*y + x*dy — need residuals x and y *)
      let bv1, v1 = ensure_var "p" p1 in
      let bv2, v2 = ensure_var "p" p2 in
      ( bs1 @ bs2 @ bv1 @ bv2,
        prim Mul [ v1; v2 ],
        prim Add [ prim Mul [ t1; v2 ]; prim Mul [ v1; t2 ] ] )
  | Div, [ (bs1, p1, t1); (bs2, p2, t2) ] ->
      (* d(x/y) = (dx*y - x*dy) / y^2 *)
      let bv1, v1 = ensure_var "p" p1 in
      let bv2, v2 = ensure_var "p" p2 in
      ( bs1 @ bs2 @ bv1 @ bv2,
        prim Div [ v1; v2 ],
        prim Div
          [
            prim Sub [ prim Mul [ t1; v2 ]; prim Mul [ v1; t2 ] ];
            prim Mul [ v2; v2 ];
          ] )
  | Max2, [ (bs1, p1, t1); (bs2, p2, t2) ] ->
      (* tangent = t1 + step(p2-p1) * (t2-t1) *)
      let bv1, v1 = ensure_var "p" p1 in
      let bv2, v2 = ensure_var "p" p2 in
      ( bs1 @ bs2 @ bv1 @ bv2,
        prim Max2 [ v1; v2 ],
        prim Add
          [
            t1;
            prim Mul [ prim Step [ prim Sub [ v2; v1 ] ]; prim Sub [ t2; t1 ] ];
          ] )
  (* === reduce: linear === *)
  | (Sum_axis _ as p), [ (bs, px, tx) ] -> (bs, prim p [ px ], prim p [ tx ])
  (* === structural: linear — tangent = same op === *)
  | ( ((Gather _ | Scatter_add _ | Apply_view _ | Adjoint_view _) as p),
      [ (bs, px, tx) ] ) ->
      (bs, prim p [ px ], prim p [ tx ])
  (* scatter_select_add: linear in first arg *)
  | (Scatter_select_add _ as p), [ (bs1, p1, t1); (bs2, p2, _t2) ] ->
      (bs1 @ bs2, prim p [ p1; p2 ], prim p [ t1; p2 ])
  (* === matmul: d(AB) = dA@B + A@dB === *)
  | Matmul, [ (bs1, p1, t1); (bs2, p2, t2) ] ->
      let bv1, v1 = ensure_var "p" p1 in
      let bv2, v2 = ensure_var "p" p2 in
      ( bs1 @ bs2 @ bv1 @ bv2,
        prim Matmul [ v1; v2 ],
        prim Add [ prim Matmul [ t1; v2 ]; prim Matmul [ v1; t2 ] ] )
  (* === max_axis: tangent = select_axis(dx, argmax(x)) === *)
  | Max_axis axis, [ (bs, px, tx) ] ->
      let bs_v, v = ensure_var "p" px in
      let am = gensym "am" in
      let all_bs = bs @ bs_v
        @ [ Let_binding (am, prim (Argmax_axis axis) [ v ]) ] in
      ( all_bs,
        prim (Max_axis axis) [ v ],
        prim (Select_axis axis) [ tx; var am ] )
  (* === argmax: integer-valued, tangent is zero (T(Z) = 0) === *)
  | (Argmax_axis _ as p), [ (bs, px, _tx) ] ->
      let r = gensym "r" in
      let all_bs = bs @ [ Let_binding (r, prim p [ px ]) ] in
      (all_bs, var r, prim Sub [ var r; var r ])
  (* === select_axis: linear in first arg, index has no tangent === *)
  | (Select_axis _ as p), [ (bs1, p1, t1); (bs2, p2, _t2) ] ->
      (bs1 @ bs2, prim p [ p1; p2 ], prim p [ t1; p2 ])
  | _ -> assert false
