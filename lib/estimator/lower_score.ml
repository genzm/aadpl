(* Score-function lowering (REINFORCE).

     grad E_{z ~ q_theta} [ f_theta(z) ]
       = E [ grad f_theta(z) + sum_i f_theta(z) grad log q_theta,i(z_i | z_<i) ]

   Rather than emit that gradient directly, emit a surrogate whose ordinary
   derivative IS that gradient:

     L(z) = f(z) + sum_i stopgrad(m_i - b) * ( log q_i(z) - stopgrad(log q_i(z)) )

   The subtracted stopgrad is what keeps the VALUE honest.  Each bracket
   evaluates to zero, so L(z) = f(z) -- the program still reports the objective
   it claims to estimate -- while its derivative picks up m_i * grad log q_i,
   because stopgrad contributes no tangent and the bracket contributes no value.
   A bare surrogate f + stopgrad(f) log q would differentiate correctly and then
   report a number nobody asked for.

   [m_i] is the cost attributed to site i and [b] a baseline.  Both change the
   variance, neither changes the expectation:

   - loss-to-go drops the cost terms that cannot depend on z_i or on anything
     drawn after it.  Such a term c is measurable in z_<i, so
     E[c grad log q_i | z_<i] = c * E[grad log q_i | z_<i] = 0 and removing it
     is free.  The test for this is exact, not statistical: on a finite support
     both the expectation and the variance are sums.
   - a constant baseline is the classic control variate, unbiased for the same
     reason: E[b grad log q_i] = b * 0.

   Unlike a pathwise lowering there is no reparameterization: z is drawn from the
   proposal outside the program and supplied as %tr.<site>.  [draw] does that,
   through Simulate, which already samples every distribution in the algebra --
   including the discrete ones a pathwise lowering cannot touch. *)

open Ast.Types

exception Score_error of string

type options = {
  loss_to_go : bool;
      (* attribute to each site only the cost drawn at or after it *)
  baseline : float;  (* control variate subtracted from every multiplier *)
}

let plain = { loss_to_go = false; baseline = 0.0 }

let lower ?(options = plain) (objective : Types.t) : Types.program =
  match objective with
  | Types.Deterministic loss -> { Types.loss; sites = []; noise = [] }
  | Types.Expect { sites; body; log_density; env_shapes; _ } ->
      if sites = [] then
        raise (Score_error "a score-function estimator needs a sampled site");
      let traces = List.map Ast.Sites.trace_name sites in
      let position trace =
        let rec go index = function
          | [] -> raise (Score_error ("not a sampled site: " ^ trace))
          | candidate :: rest ->
              if candidate = trace then index else go (index + 1) rest
        in
        go 0 traces
      in
      (* Sampling order is the order sites were collected in, so "drawn at or
         after site i" is just an index comparison. *)
      let latest reached =
        List.fold_left (fun acc trace -> max acc (position trace)) (-1) reached
      in
      let draws =
        List.map
          (fun (site : Ast.Sites.site) ->
            (Ast.Sites.trace_name site, site.frame))
          sites
      in
      let cost = Decompose.split body in
      (* body already contains a copy of log q; renaming keeps the binders of the
         second copy distinct, as Unzip requires. *)
      let density = Decompose.split (Rename.binders ~tag:"s" log_density) in
      let cost_reach = Decompose.reach cost ~traces in
      let density_reach = Decompose.reach density ~traces in
      (* log q_i is the sum of the density terms whose latest site is i.  A
         proposal factorizes in sampling order, so every part of its i-th factor
         mentions z_i and possibly earlier draws, never later ones -- which makes
         the attribution a partition.  One factor is several terms: a categorical
         contributes its support indicator alongside its log-probability, a
         pushforward its base density alongside its log-Jacobian. *)
      let own_density index =
        match
          List.filter
            (fun term -> latest (density_reach term) = index)
            density.terms
        with
        | [] ->
            raise
              (Score_error
                 (Printf.sprintf "no proposal density term for site '%s'"
                    (List.nth traces index)))
        | terms -> Lit.sum terms
      in
      (* Bind each cost term once: a multiplier may reference the same term as
         another one, and a density term carries its own Let bindings. *)
      let cost_var index = Printf.sprintf "%%s.cost%d" index in
      let density_var index = Printf.sprintf "%%s.logq%d" index in
      let objective_var = "%s.f" in
      let cost_vars = List.mapi (fun index _ -> cost_var index) cost.terms in
      let multiplier index =
        if not options.loss_to_go then var objective_var
        else
          Lit.sum
            (List.filteri
               (fun term_index _ ->
                 latest (cost_reach (List.nth cost.terms term_index)) >= index)
               cost_vars
            |> List.map var)
      in
      let contribution index =
        let density = var (density_var index) in
        let multiplier =
          if options.baseline = 0.0 then multiplier index
          else rank 0 Sub [ multiplier index; Lit.scalar options.baseline ]
        in
        rank 0 Mul
          [
            prim Stop_gradient [ multiplier ];
            rank 0 Sub [ density; prim Stop_gradient [ density ] ];
          ]
      in
      let bindings =
        cost.bindings @ density.bindings
        @ List.mapi (fun index term -> (cost_var index, term)) cost.terms
        @ [ (objective_var, Lit.sum (List.map var cost_vars)) ]
        @ List.mapi (fun index _ -> (density_var index, own_density index)) sites
      in
      let surrogate =
        Lit.sum
          (var objective_var
          :: List.mapi (fun index _ -> contribution index) sites)
      in
      let loss =
        Transform.Forward.wrap_let_bindings bindings surrogate
        |> Transform.Expand_rank.expand ~senv:(draws @ env_shapes)
        |> Transform.Desugar.fuse_views
      in
      { Types.loss; sites; noise = draws }

(* Draw z ~ proposal and bind it to the trace variables the lowered program
   expects.  The guide namespace matches Lower_pathwise.noise_env, so the two
   estimators consume the same stream for a given run_key. *)
let draw (objective : Types.t) ~(env : Ast.Eval.env) ~run_key :
    (string * Ast.Types.value) list =
  match objective with
  | Types.Deterministic _ -> []
  | Types.Expect { sites; proposal; env_shapes; _ } ->
      let proposal = Transform.Expand_rank.expand ~senv:env_shapes proposal in
      let _, trace, _ =
        Ast.Simulate.simulate ~namespace:Prng.Threefry.ns_guide ~run_key env
          proposal
      in
      List.map
        (fun (site : Ast.Sites.site) ->
          (Ast.Sites.trace_name site, List.assoc site.name trace))
        sites
