(* Unzip: split forward's (bindings, primal, tangent) into
   primal-only bindings and tangent-linear bindings.

   Classification: a binding depends on tangent seeds iff its RHS
   references a seed or a previously-classified tangent variable.
   Forward's construction guarantees tangent-dependent operations
   are linear in the tangent — check_linearity verifies this. *)

open Ast.Types

type unzipped = {
  primal_bindings  : binding list;
  primal_out       : expr;
  tangent_bindings : binding list;
  tangent_out      : expr;
}

(* --- free variables of an expression --- *)

module SS = Set.Make(String)

let rec free_vars (e : expr) : SS.t =
  match e with
  | Const _ -> SS.empty
  | Var (_, s) -> SS.singleton s
  | Prim (_, _, args) ->
    List.fold_left (fun acc a -> SS.union acc (free_vars a)) SS.empty args
  | Let (_, s, e1, e2) ->
    SS.union (free_vars e1) (SS.remove s (free_vars e2))
  | Scan _ -> failwith "free_vars: Scan must be handled at its boundary"
  | Rank _ -> failwith "free_vars: Rank in forward output (expand first)"
  | Sample _ -> failwith "free_vars: Sample not supported"
  | Score _ -> failwith "free_vars: Score not supported"

let free_vars_scan (scan : scan) : SS.t =
  let names =
    List.fold_left (fun names (name, _, _) -> SS.add name names) SS.empty
      scan.carries
    |> fun names ->
      List.fold_left (fun names (name, _) -> SS.add name names) names scan.inputs
  in
  let variables = List.fold_left (fun variables (_, init, _) ->
    SS.union variables (free_vars init)) SS.empty scan.carries in
  let variables = List.fold_left (fun variables (_, input) ->
    SS.union variables (free_vars input)) variables scan.inputs in
  List.fold_left (fun variables (_, _, next) ->
    SS.union variables (SS.diff (free_vars next) names)) variables scan.carries

let binding_names = function
  | Let_binding (name, _) -> [name]
  | Scan_binding (_, scan) -> List.map (fun (name, _, _) -> name) scan.carries

let binding_free_vars = function
  | Let_binding (_, rhs) -> free_vars rhs
  | Scan_binding (_, scan) -> free_vars_scan scan

(* --- tangent dependency propagation --- *)

(* Walk bindings left-to-right. A binding is tangent-dependent if any
   free variable in its RHS is in the tangent_deps set. *)
let classify (bs : binding list) ~(seeds : SS.t)
    : binding list * binding list * SS.t =
  let tangent_deps = ref seeds in
  let primal = ref [] in
  let tangent = ref [] in
  List.iter (fun binding ->
    let fv = binding_free_vars binding in
    if SS.is_empty (SS.inter fv !tangent_deps) then
      primal := binding :: !primal
    else begin
      tangent := binding :: !tangent;
      List.iter (fun name -> tangent_deps := SS.add name !tangent_deps)
        (binding_names binding)
    end
  ) bs;
  (List.rev !primal, List.rev !tangent, !tangent_deps)

(* --- linearity check --- *)

(* Check that in each tangent binding, tangent-dependent arguments appear
   only in linear positions. Returns None if OK, Some msg if violation. *)
let check_linearity (tangent_bs : binding list) (deps : SS.t)
    : string option =
  let is_dep e = not (SS.is_empty (SS.inter (free_vars e) deps)) in
  let rec check_expr (e : expr) : string option =
    match e with
    | Const _ | Var _ -> None
    | Let (_, _, e1, e2) ->
      (match check_expr e1 with Some m -> Some m | None -> check_expr e2)
    | Scan _ -> Some "Scan in tangent bindings"
    | Rank _ -> Some "Rank in tangent bindings"
    | Sample _ -> Some "Sample in tangent bindings"
    | Score _ -> Some "Score in tangent bindings"
    | Prim (_, p, args) -> check_prim p args
  and check_prim p args =
    match p, args with
    (* linear unconditionally *)
    | (Neg | Add | Sub), _ -> check_args args
    | Mask, [a; mask] ->
      if is_dep mask then Some "Mask: mask is tangent-dependent"
      else check_args [a]
    | (Sum_axis _ | Gather _ | Scatter_add _
      | Apply_view _ | Adjoint_view _), _ ->
      check_args args
    (* Mul: at most one side tangent-dependent *)
    | Mul, [a; b] ->
      if is_dep a && is_dep b then
        Some "Mul: both arguments are tangent-dependent (quadratic)"
      else check_args args
    (* Div: only numerator may be tangent-dependent *)
    | Div, [a; b] ->
      if is_dep b then
        Some "Div: denominator is tangent-dependent (nonlinear)"
      else check_args [a; b]
    (* Matmul: at most one side *)
    | Matmul, [a; b] ->
      if is_dep a && is_dep b then
        Some "Matmul: both arguments are tangent-dependent (quadratic)"
      else check_args args
    (* Select_axis: index (2nd arg) must not be tangent-dependent *)
    | Select_axis _, [a; idx] ->
      if is_dep idx then
        Some "Select_axis: index is tangent-dependent"
      else check_args [a]
    (* Scatter_select_add: index (2nd arg) must not be tangent-dependent *)
    | Scatter_select_add _, [a; idx] ->
      if is_dep idx then
        Some "Scatter_select_add: index is tangent-dependent"
      else check_args [a]
    (* nonlinear map1/map2 must not have tangent-dependent args *)
    | (Exp | Log | Logsigmoid | Log_unit_density | Log_support_density _
      | Sqrt | Relu | Step | Erf | Erfinv | Stop_gradient), [a] ->
      if is_dep a then
        Some (Format.asprintf "%a: argument is tangent-dependent (nonlinear)"
                pp_prim p)
      else None
    | Max2, [a; b] ->
      if is_dep a || is_dep b then
        Some "Max2: argument is tangent-dependent (nonlinear)"
      else None
    | (Max_axis _ | Argmax_axis _), [a] ->
      if is_dep a then
        Some (Format.asprintf "%a: argument is tangent-dependent (nonlinear)"
                pp_prim p)
      else None
    | _ -> Some "unknown prim in tangent bindings"
  and check_args args =
    List.fold_left (fun acc a ->
      match acc with Some _ -> acc | None -> check_expr a) None args
  in
  List.fold_left (fun acc binding ->
    match acc with
    | Some _ -> acc
    | None -> (match binding with
      | Scan_binding _ -> None
      | Let_binding (name, rhs) ->
      match check_expr rhs with
      | Some msg -> Some (Printf.sprintf "binding %s: %s" name msg)
      | None -> None)
  ) None tangent_bs

(* --- unzip --- *)

let unzip ((bs, p_out, t_out) : Forward.bindings * expr * expr)
    ~(seeds : string list) : unzipped =
  (* Detect binding name shadowing (silent misclassification otherwise) *)
  let seen = Hashtbl.create (List.length bs) in
  List.iter (fun binding ->
    List.iter (fun name ->
      if Hashtbl.mem seen name then
        failwith ("unzip: duplicate binding name (shadowing): " ^ name)
      else Hashtbl.replace seen name ()) (binding_names binding)
  ) bs;
  let seed_set = SS.of_list seeds in
  let (primal_bs, tangent_bs, deps) = classify bs ~seeds:seed_set in
  (* linearity check *)
  (match check_linearity tangent_bs deps with
   | None -> ()
   | Some msg -> failwith ("unzip linearity violation: " ^ msg));
  (* also check tangent_out *)
  (match check_linearity [Let_binding ("__out__", t_out)] deps with
   | None -> ()
   | Some msg -> failwith ("unzip linearity violation in tangent_out: " ^ msg));
  { primal_bindings = primal_bs;
    primal_out = p_out;
    tangent_bindings = tangent_bs;
    tangent_out = t_out }
