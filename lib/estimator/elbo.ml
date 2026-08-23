(* The evidence lower bound as an Estimator IR term:

     E_{z ~ q} [ log p(x, z) - log q(z) ]

   Everything here is model semantics -- site compatibility, which densities are
   scored, and the integrand.  No reparameterization, no noise variables, and no
   commitment to an estimator appear; those belong to a lowering.

   Split out of the former Transform.build_elbo.  The steps below are that
   function's first half, verbatim and in the same order. *)

open Ast.Types

let elbo ~(slots : Ast.Sites.slot list) ~model ~guide
    ~(env_shapes : (string * int array) list) : Types.t =
  Transform.Reparam.check_guide guide;
  let model_sites = Ast.Sites.collect_sites model in
  let sites = Ast.Sites.collect_sites guide in
  Transform.Reparam.check_trace_compat_sites ~slots ~model_sites
    ~guide_sites:sites ();
  Transform.Reparam.check_support_compat ~model_sites ~guide_sites:sites;
  List.iter
    (fun (name, _, value) ->
      let site = Ast.Sites.find name model_sites in
      let shape = Transform.Expand_rank.infer_shape env_shapes value in
      if shape <> site.frame then
        raise
          (Transform.Reparam.Trace_mismatch
             (Printf.sprintf "slotted site '%s' shape does not match frame"
                name)))
    slots;
  let density_slots =
    List.map
      (fun (site : Ast.Sites.site) ->
        (site.name, var (Ast.Sites.trace_name site)))
      sites
    @ List.map (fun (name, _, value) -> (name, value)) slots
  in
  let model_expanded = Transform.Expand_rank.expand ~senv:env_shapes model in
  let guide_expanded = Transform.Expand_rank.expand ~senv:env_shapes guide in
  let trace_shapes =
    List.map
      (fun (site : Ast.Sites.site) -> (Ast.Sites.trace_name site, site.frame))
      sites
  in
  let assess_shapes = trace_shapes @ env_shapes in
  let model_ld =
    let excluded =
      List.filter_map
        (function name, `Maximize, _ -> Some name | _, `Condition, _ -> None)
        slots
    in
    Transform.Assess_expr.assess_expr ~ns:"m." ~exclude_density:excluded
      ~env_shapes:assess_shapes model_expanded density_slots
  in
  let guide_ld =
    Transform.Assess_expr.assess_expr ~ns:"g." ~env_shapes:assess_shapes
      guide_expanded density_slots
  in
  let expectation : Types.expectation =
    {
      sites;
      (* reparam runs on the unexpanded guide, as it did in build_elbo *)
      proposal = guide;
      body = prim Sub [ model_ld; guide_ld ];
      env_shapes;
    }
  in
  Types.Expect expectation
