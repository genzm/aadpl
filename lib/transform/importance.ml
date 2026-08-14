open Ast.Types

exception Importance_error of string

type program = {
  log_weights : expr;
  sites : Ast.Sites.site list;
  noise : (string * int array) list;
}

let noise_env program ~run_key =
  Ast.Sites.draw_noise ~namespace:Prng.Threefry.ns_guide ~run_key program.sites

let importance ~particles ~(slots : Ast.Sites.slot list) ~model ~guide
    ~env_shapes =
  if particles <= 0 then
    raise (Importance_error "particle count must be positive");
  Reparam.check_guide guide;
  let model_sites = Ast.Sites.collect_sites model
  and guide_sites = Ast.Sites.collect_sites guide in
  Reparam.check_trace_compat_sites ~slots ~model_sites ~guide_sites ();
  Reparam.check_support_compat ~model_sites ~guide_sites;
  let prefix expression =
    prim (Apply_view [Vbroadcast (0, particles)]) [expression] in
  let model = Quadrature.prefix_sample_frames particles model
  and guide = Quadrature.prefix_sample_frames particles guide in
  let free = List.sort_uniq String.compare
    (Assess_expr.free_vars model @ Assess_expr.free_vars guide) in
  let external_shapes = List.filter
    (fun (name, _) -> List.mem name free) env_shapes in
  let external_bindings = List.map (fun (name, _) ->
    "%i.env." ^ name, prefix (var name)) external_shapes in
  let substitute_external expression =
    List.fold_left2 (fun expression (name, _) (local, _) ->
      Assess_expr.subst ~from:name ~to_:(var local) expression)
      expression external_shapes external_bindings in
  let model = substitute_external model
  and guide = substitute_external guide in
  let slot_bindings = List.map (fun (name, _, expression) ->
    "%i.slot." ^ name, prefix expression) slots in
  let bindings = external_bindings @ slot_bindings in
  let binding_shapes = List.map (fun (name, expression) ->
    name, Expand_rank.infer_shape env_shapes expression) bindings in
  let sites = Ast.Sites.collect_sites guide in
  let density_slots = List.map (fun site ->
    site.Ast.Sites.name, var (Ast.Sites.trace_name site)) sites
    @ List.map2 (fun (name, _, _) (local, _) -> name, var local)
        slots slot_bindings in
  let guide_r = Reparam.reparam ~sites guide in
  let sample_bindings, _ = Reparam.elim_samples ~sites guide_r in
  let noise = List.map (fun site ->
    Ast.Sites.noise_name site, site.Ast.Sites.frame) sites in
  let shapes = binding_shapes @ env_shapes in
  let model = Expand_rank.expand ~senv:shapes model
  and guide = Expand_rank.expand ~senv:shapes guide in
  let trace_shapes = List.map (fun site ->
    Ast.Sites.trace_name site, site.Ast.Sites.frame) sites in
  let assess_shapes = trace_shapes @ shapes in
  let excluded = List.filter_map (function
    | name, `Maximize, _ -> Some name | _, `Condition, _ -> None) slots in
  let model_ld = Assess_expr.assess_expr ~ns:"im."
    ~preserve_shape:[|particles|] ~exclude_density:excluded
    ~env_shapes:assess_shapes model density_slots in
  let guide_ld = Assess_expr.assess_expr ~ns:"ig."
    ~preserve_shape:[|particles|] ~env_shapes:assess_shapes guide density_slots in
  let log_weights = prim Sub [model_ld; guide_ld]
    |> Forward.wrap_bindings sample_bindings
    |> Forward.wrap_bindings bindings
    |> Expand_rank.expand ~senv:(noise @ env_shapes)
    |> Desugar.fuse_views in
  {log_weights; sites; noise}
