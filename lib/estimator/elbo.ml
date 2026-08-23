(* The evidence lower bound as an Estimator IR term:

     E_{z ~ q} [ log p(x, z) - log q(z) ]

   Everything here is model semantics -- site compatibility, which densities are
   scored, and the integrand.  No reparameterization, no noise variables, and no
   commitment to an estimator appear; those belong to a lowering.

   Split out of the former Transform.build_elbo.  The steps below are that
   function's first half, verbatim and in the same order. *)

open Ast.Types

(* A proposal has to be a distribution: nothing may score inside it, and its
   sites must be distinct (collect_sites checks that).  Whether those sites can
   be ESTIMATED -- reparameterized, enumerated -- is a different question that
   belongs to the lowering, so it is deliberately not asked here.
   Transform.Reparam.check_guide answers both at once; only its objective-level
   half is used, and Estimator.lower_pathwise asks the other half. *)
let check_proposal (guide : expr) : unit =
  let rec walk = function
    | Score (loc, _) ->
        raise
          (Transform.Reparam.Guide_error (loc, "guide must not contain Score"))
    | Const _ | Var _ | Sample _ -> ()
    | Prim (_, _, args) | Rank (_, _, _, args) -> List.iter walk args
    | Let (_, _, rhs, body) ->
        walk rhs;
        walk body
    | Scan (_, scan, continuation) ->
        List.iter
          (fun (_, init, next) ->
            walk init;
            walk next)
          scan.carries;
        List.iter (fun (_, input) -> walk input) scan.inputs;
        walk continuation
  in
  walk guide

let elbo ~(slots : Ast.Sites.slot list) ~model ~guide
    ~(env_shapes : (string * int array) list) : Types.t =
  check_proposal guide;
  (* Expand first: a distribution's parameters have to be Rank-free before their
     shapes -- and so a categorical's support size -- can be inferred. *)
  let model_expanded = Transform.Expand_rank.expand ~senv:env_shapes model in
  let guide_expanded = Transform.Expand_rank.expand ~senv:env_shapes guide in
  let model_shapes = Transform.Reparam.local_shapes env_shapes model_expanded in
  let guide_shapes = Transform.Reparam.local_shapes env_shapes guide_expanded in
  let model_sites = Ast.Sites.collect_sites model_expanded in
  let sites = Ast.Sites.collect_sites guide_expanded in
  Transform.Reparam.check_trace_compat_sites ~slots ~model_sites
    ~guide_sites:sites ();
  Transform.Reparam.check_support_compat
    ~model_categorical_size:(Transform.Reparam.categorical_size model_shapes)
    ~guide_categorical_size:(Transform.Reparam.categorical_size guide_shapes)
    ~model_sites ~guide_sites:sites ();
  let supports =
    List.map
      (fun (site : Ast.Sites.site) ->
        ( site.name,
          Ast.Sites.dist_support
            ~categorical_size:
              (Transform.Reparam.categorical_size guide_shapes)
            site.dist ))
      sites
  in
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
      log_density = guide_ld;
      supports;
      env_shapes;
    }
  in
  Types.Expect expectation
