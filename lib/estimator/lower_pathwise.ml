(* Pathwise (reparameterization) lowering.

     E_{z ~ q_psi} [ f(z) ]   with   z = g_psi(u),  u ~ base

   Reparam.reparam rewrites each pushforward Sample into a base Sample plus the
   deterministic forward map; elim_samples then lifts every base Sample out as a
   free noise variable %u.<site>.  What remains is an ordinary tensor program
   whose only stochastic inputs are those free variables, so differentiating it
   with the noise passed as ~data_shapes -- which zeroes their tangents -- is
   exactly the pathwise gradient.

   The site name is preserved on the base Sample, so the Threefry counter is
   unchanged by the rewrite and a lowered program couples bit-exactly with
   simulate on a shared run_key.

   Split out of the former Transform.build_elbo: the steps below are that
   function's second half, verbatim. *)

let noise_env (program : Types.program) ~run_key =
  Ast.Sites.draw_noise ~namespace:Prng.Threefry.ns_guide ~run_key
    program.Types.sites

let lower (objective : Types.t) : Types.program =
  match objective with
  | Types.Deterministic loss -> { Types.loss; sites = []; noise = [] }
  | Types.Expect { sites; proposal; body; env_shapes; _ } ->
      (* Every site must be reparameterizable.  This is a precondition of the
         ESTIMATOR, not of the objective, which is why it is asked here and not
         in Estimator.elbo: a discrete site is a perfectly good objective, it
         just has to be lowered by enumeration instead.  Strategy.of_site says
         the same thing in the vocabulary the other lowerings use; check_guide
         is kept as the authority so the error text does not move. *)
      Transform.Reparam.check_guide proposal;
      let noise =
        List.map
          (fun (site : Ast.Sites.site) ->
            (Ast.Sites.noise_name site, site.frame))
          sites
      in
      let reparameterized = Transform.Reparam.reparam ~sites proposal in
      let bindings, _ = Transform.Reparam.elim_samples ~sites reparameterized in
      let loss =
        Transform.Forward.wrap_let_bindings bindings body
        |> Transform.Expand_rank.expand ~senv:(noise @ env_shapes)
        |> Transform.Desugar.fuse_views
      in
      { Types.loss; sites; noise }
