open Ast.Types

exception Quadrature_error of string

type program = {
  log_marginal : expr;
  log_terms : expr;
  sites : Ast.Sites.site list;
  node_count : int;
}

let gauss_hermite node_count =
  if node_count <= 0 then invalid_arg "gauss_hermite: positive node count";
  let nodes = Array.make node_count 0.0 and weights = Array.make node_count 0.0 in
  let half = (node_count + 1) / 2 in
  let previous = ref 0.0 in
  for i = 0 to half - 1 do
    let n = float_of_int node_count in
    let z = ref (if i = 0 then
      sqrt (2.0 *. n +. 1.0) -. 1.85575 *. (2.0 *. n +. 1.0) ** (-1.0 /. 6.0)
    else if i = 1 then !previous -. 1.14 *. n ** 0.426 /. !previous
    else if i = 2 then 1.86 *. !previous -. 0.86 *. nodes.(0)
    else if i = 3 then 1.91 *. !previous -. 0.91 *. nodes.(1)
    else 2.0 *. !previous -. nodes.(i - 2)) in
    let derivative = ref 0.0 in
    let iteration = ref 0 and converged = ref false in
    while !iteration < 20 && not !converged do
      incr iteration;
      let p1 = ref (Float.pi ** (-0.25)) and p2 = ref 0.0 in
      for degree = 1 to node_count do
        let p3 = !p2 in
        p2 := !p1;
        let d = float_of_int degree in
        p1 := !z *. sqrt (2.0 /. d) *. !p2
          -. sqrt ((d -. 1.0) /. d) *. p3
      done;
      derivative := sqrt (2.0 *. n) *. !p2;
      let next = !z -. !p1 /. !derivative in
      converged := Float.abs (next -. !z) < 1e-15;
      z := next
    done;
    previous := !z;
    nodes.(i) <- !z;
    nodes.(node_count - 1 - i) <- -. !z;
    let weight = 2.0 /. (!derivative *. !derivative) in
    weights.(i) <- weight;
    weights.(node_count - 1 - i) <- weight
  done;
  nodes, weights

let tensor values =
  let result = View.Tensor.make [|Array.length values|] in
  Array.iteri (View.Buf.set result.buf) values;
  result

let gauss_hermite_lebesgue ?(half_line = false) ~scale node_count =
  if scale <= 0.0 then
    invalid_arg "gauss_hermite_lebesgue: positive scale required";
  if half_line && node_count mod 2 <> 0 then
    invalid_arg "gauss_hermite_lebesgue: half-line rule requires even K";
  let nodes, weights = gauss_hermite node_count in
  let jacobian = if half_line then scale /. sqrt 2.0
    else sqrt 2.0 *. scale in
  let values = Array.map (fun node ->
    sqrt 2.0 *. scale *. (if half_line then Float.abs node else node)) nodes in
  let log_weights = Array.mapi (fun index node ->
    log weights.(index) +. node *. node +. log jacobian) nodes in
  values, log_weights

let rec prefix_sample_frames node_count = function
  | Const _ | Var _ as expression -> expression
  | Prim (loc, primitive, arguments) ->
      Prim (loc, primitive, List.map (prefix_sample_frames node_count) arguments)
  | Let (loc, name, rhs, body) ->
      Let (loc, name, prefix_sample_frames node_count rhs,
        prefix_sample_frames node_count body)
  | Rank (loc, rank, primitive, arguments) ->
      Rank (loc, rank, primitive,
        List.map (prefix_sample_frames node_count) arguments)
  | Sample (loc, name, frame, distribution) ->
      Sample (loc, name, Array.append [|node_count|] frame, distribution)
  | Score (loc, expression) -> Score (loc, prefix_sample_frames node_count expression)

let logsumexp_axis0 node_count expression =
  let terms = Forward.gensym "terms" in
  let maximum = Forward.gensym "max" in
  let maximum_cells =
    prim (Apply_view [Vbroadcast (0, node_count)]) [var maximum] in
  let shifted = prim Sub [var terms; maximum_cells] in
  let total = prim (Sum_axis 0) [prim Exp [shifted]] in
  let_ terms expression
    (let_ maximum (prim (Max_axis 0) [var terms])
      (prim Add [var maximum; prim Log [total]]))

let sum_axes ~axis count expression =
  let rec go remaining result =
    if remaining = 0 then result
    else go (remaining - 1) (prim (Sum_axis axis) [result]) in
  go count expression

let quadrature ~site ~values ~(log_weights : View.Tensor.t)
    ~include_target_density ~preserve_frame
    ~(slots : Ast.Sites.slot list) ~model
    ~env_shapes =
  if Array.length log_weights.view.shape <> 1 then
    raise (Quadrature_error "log_weights must have shape [K]");
  let node_count = log_weights.View.Tensor.view.View.Ndview.shape.(0) in
  if node_count = 0 then
    raise (Quadrature_error "log_weights must contain at least one node");
  let model_sites = Ast.Sites.collect_sites model in
  let target = try Ast.Sites.find site model_sites with Not_found ->
    raise (Quadrature_error ("unknown site '" ^ site ^ "'")) in
  if target.kind <> `Cont then
    raise (Quadrature_error "target site must be continuous");
  if preserve_frame < 0 || preserve_frame > Array.length target.frame then
    raise (Quadrature_error "preserve_frame exceeds target site frame rank");
  let hoisted_conditions = List.filter (fun (name, role, _) ->
    role = `Condition &&
    let candidate = Ast.Sites.find name model_sites in
    let probe = Sample (candidate.loc, candidate.name, candidate.frame,
      candidate.dist) in
    Array.length candidate.frame = preserve_frame
    && Array.sub candidate.frame 0 preserve_frame
       = Array.sub target.frame 0 preserve_frame
    && not (List.mem site (Assess_expr.free_vars probe))) slots in
  let hoisted_names = List.map (fun (name, _, _) -> name) hoisted_conditions in
  let slot_names = List.map (fun (name, _, _) -> name) slots in
  let latent = List.filter
    (fun candidate -> not (List.mem candidate.Ast.Sites.name slot_names))
    model_sites in
  if List.map (fun candidate -> candidate.Ast.Sites.name) latent <> [site] then
    raise (Quadrature_error
      "Phase 13-2 quadrature requires exactly one latent target site");
  let expected_shape = Array.append [|node_count|] target.frame in
  if Expand_rank.infer_shape env_shapes values <> expected_shape then
    raise (Quadrature_error "node values must have shape [K] + site.frame");
  let prefix expression =
    prim (Apply_view [Vbroadcast (0, node_count)]) [expression] in
  let target_slot = "%q.slot." ^ site in
  let slot_bindings = List.map (fun (name, _, expression) ->
    let local = "%q.slot." ^ name in
    local, prefix expression) slots in
  let model_free = Assess_expr.free_vars model in
  let external_shapes = List.filter
    (fun (name, _) -> List.mem name model_free) env_shapes in
  let external_bindings = List.map (fun (name, _) ->
    "%q.env." ^ name, prefix (var name)) external_shapes in
  let model = List.fold_left2 (fun expression (name, _) (local, _) ->
    Assess_expr.subst ~from:name ~to_:(var local) expression)
    model external_shapes external_bindings in
  let bindings = (target_slot, values) :: slot_bindings @ external_bindings in
  let binding_shapes = List.map (fun (name, expression) ->
    name, Expand_rank.infer_shape env_shapes expression) bindings in
  let density_slots = (site, var target_slot) :: List.map2
    (fun (name, _, _) (local, _) -> name, var local) slots slot_bindings in
  let prefixed = prefix_sample_frames node_count model
    |> Expand_rank.expand ~senv:(binding_shapes @ env_shapes) in
  let log_integrand = Assess_expr.assess_expr ~ns:"q."
    ~preserve_shape:expected_shape
    ~exclude_density:((if include_target_density then [] else [site])
      @ List.filter_map (function
      | name, `Maximize, _ -> Some name | _, `Condition, _ -> None) slots
      @ hoisted_names)
    ~env_shapes:(binding_shapes @ env_shapes) prefixed density_slots in
  let weighted = rank 0 Add [const log_weights; log_integrand]
    |> Expand_rank.expand ~senv:(binding_shapes @ env_shapes) in
  let reduced = logsumexp_axis0 node_count weighted
    |> sum_axes ~axis:preserve_frame
         (Array.length target.frame - preserve_frame) in
  let outer_density = List.fold_left (fun total (name, _, value) ->
    let candidate = Ast.Sites.find name model_sites in
    let dist = List.fold_left (fun dist (slot_name, _, slot_value) ->
      Assess_expr.subst_dist ~from:slot_name ~to_:slot_value dist)
      candidate.dist slots in
    match Assess_expr.log_density_expr ~env_shapes ~loc:candidate.loc
      candidate.frame dist value with
    | None -> total
    | Some density ->
        let density = Assess_expr.sum_frame ~preserve:preserve_frame
          candidate.frame density in
        prim Add [total; density]) (Assess_expr.mk_zero
      (Array.sub target.frame 0 preserve_frame)) hoisted_conditions in
  let reduced = prim Add [reduced; outer_density] in
  let log_terms = Forward.wrap_bindings bindings weighted
    |> Expand_rank.expand ~senv:env_shapes in
  let log_marginal = Forward.wrap_bindings bindings reduced
    |> Expand_rank.expand ~senv:env_shapes in
  { log_marginal; log_terms; sites = []; node_count }
