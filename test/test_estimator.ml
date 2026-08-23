(* The Estimator IR boundary.

   An objective says WHICH statistical quantity is wanted; a lowering picks the
   estimator.  These tests pin the invariant that separates the two, so that a
   later lowering (enumeration, score function) can be added without disturbing
   the objective:

     - the objective's integrand is already a pure tensor expression;
       only the proposal carries Sample nodes,
     - a lowering removes every Sample and leaves the drawn randomness as free
       noise variables,
     - lowering carries no hidden state, so it is reproducible. *)

open Alcotest
open Ast.Types

let scalar value =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 value;
  t

let log_normal ~x ~mu ~sigma =
  let half = const (scalar 0.5) in
  let log_two_pi = const (scalar (log (2.0 *. Float.pi))) in
  let standardized = rank 0 Div [ rank 0 Sub [ x; mu ]; sigma ] in
  let normalizer =
    rank 0 Add [ rank 0 Mul [ half; log_two_pi ]; prim Log [ sigma ] ]
  in
  rank 0 Neg
    [
      rank 0 Add
        [
          normalizer;
          rank 0 Mul [ half; rank 0 Mul [ standardized; standardized ] ];
        ];
    ]

(* The conjugate Gaussian program of test_vi, kept small on purpose. *)
let conjugate samples =
  let model =
    let_ "z"
      (sample "z" [| samples |]
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_sigma"))
      (score
         (log_normal ~x:(var "x_obs") ~mu:(var "z") ~sigma:(var "obs_sigma")))
  in
  let guide =
    let_ "sigma_q"
      (prim Exp [ var "log_sigma_q" ])
      (sample "z" [| samples |] (Ast.Normal.normal ~mu:"mu_q" ~sigma:"sigma_q"))
  in
  let env_shapes =
    [
      ("prior_mu", [||]); ("prior_sigma", [||]); ("x_obs", [||]);
      ("obs_sigma", [||]); ("mu_q", [||]); ("log_sigma_q", [||]);
    ]
  in
  (model, guide, env_shapes)

let rec has_sample = function
  | Sample _ | Score _ -> true
  | Const _ | Var _ -> false
  | Prim (_, _, args) | Rank (_, _, _, args) -> List.exists has_sample args
  | Let (_, _, rhs, body) -> has_sample rhs || has_sample body
  | Scan (_, scan, continuation) ->
      List.exists
        (fun (_, init, next) -> has_sample init || has_sample next)
        scan.carries
      || List.exists (fun (_, input) -> has_sample input) scan.inputs
      || has_sample continuation

let contains haystack needle =
  let n = String.length needle and h = String.length haystack in
  let rec go i = i + n <= h && (String.sub haystack i n = needle || go (i + 1)) in
  go 0

let site_names sites =
  List.map (fun (site : Ast.Sites.site) -> site.name) sites

(* The objective is model semantics only: densities are already symbolic, and
   the randomness still lives in the proposal where a lowering can find it. *)
let test_objective_separates_integrand_from_proposal () =
  let model, guide, env_shapes = conjugate 3 in
  match Estimator.elbo ~slots:[] ~model ~guide ~env_shapes with
  | Estimator.Types.Deterministic _ -> fail "ELBO must be an expectation"
  | Estimator.Types.Expect expectation ->
      check (list string) "proposal sites" [ "z" ]
        (site_names expectation.sites);
      check bool "integrand is a pure tensor expression" false
        (has_sample expectation.body);
      check bool "proposal still carries the Sample" true
        (has_sample expectation.proposal)

(* Pathwise lowering turns the expectation into a deterministic program whose
   only stochastic inputs are the free noise variables. *)
let test_pathwise_lowering_eliminates_samples () =
  let model, guide, env_shapes = conjugate 3 in
  let program =
    Estimator.lower_pathwise
    @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes
  in
  check bool "lowered program has no Sample/Score" false
    (has_sample program.loss);
  check (list string) "noise variables" [ "%u.z" ] (List.map fst program.noise);
  check (list string) "sites survive lowering" [ "z" ]
    (site_names program.sites);
  let free = Transform.Assess_expr.free_vars program.loss in
  List.iter
    (fun (name, _) ->
      check bool (name ^ " is free in the lowered program") true
        (List.mem name free))
    program.noise

(* Lowering keeps no gensym or other hidden state: the same objective lowers to
   the same program every time.  This is what makes the objective safe to reuse
   across several lowerings. *)
let test_lowering_is_reproducible () =
  let model, guide, env_shapes = conjugate 3 in
  let objective = Estimator.elbo ~slots:[] ~model ~guide ~env_shapes in
  let first = Estimator.lower_pathwise objective in
  let second = Estimator.lower_pathwise objective in
  check string "lowering twice gives the same program"
    (Format.asprintf "%a" Ast.Types.pp first.loss)
    (Format.asprintf "%a" Ast.Types.pp second.loss)

let test_deterministic_objective_passes_through () =
  let expression = prim Add [ const (scalar 1.0); const (scalar 2.0) ] in
  let program =
    Estimator.lower_pathwise (Estimator.Types.Deterministic expression)
  in
  check (list string) "no sites" [] (site_names program.sites);
  check (list string) "no noise" [] (List.map fst program.noise);
  check (float 1e-12) "value preserved" 3.0
    (View.Buf.get (Ast.Eval.eval [] program.loss).View.Tensor.buf 0)

(* ── Step 5-6: strategy classification and exact enumeration ── *)

let tensor shape values =
  let result = View.Tensor.make shape in
  Array.iteri (View.Buf.set result.buf) values;
  result

let value tensor = View.Buf.get tensor.View.Tensor.buf 0

(* p(k) proportional to model_weights, q(k) to guide_weights, and the model
   scores k * theta so that the objective has something to differentiate. *)
let model_weights = [| 1.0; 2.0; 1.0 |]
let guide_weights = [| 3.0; 1.0; 4.0 |]

let discrete_program () =
  let model =
    let_ "k"
      (sample "k" [||] (D_categorical (const (tensor [| 3 |] model_weights))))
      (score (rank 0 Mul [ var "k"; var "theta" ]))
  in
  let guide =
    sample "k" [||] (D_categorical (const (tensor [| 3 |] guide_weights)))
  in
  (model, guide, [ ("theta", [||]) ])

let normalize weights =
  let total = Array.fold_left ( +. ) 0.0 weights in
  Array.map (fun w -> w /. total) weights

(* ELBO = sum_k q_k (log p_k + k*theta - log q_k), by hand. *)
let closed_form_elbo theta =
  let p = normalize model_weights and q = normalize guide_weights in
  let total = ref 0.0 in
  Array.iteri
    (fun k qk ->
      total :=
        !total +. (qk *. (log p.(k) +. (float_of_int k *. theta) -. log qk)))
    q;
  !total

(* d/dtheta of the above is sum_k q_k * k, with no sampling error at all. *)
let closed_form_gradient () =
  let q = normalize guide_weights in
  let total = ref 0.0 in
  Array.iteri (fun k qk -> total := !total +. (qk *. float_of_int k)) q;
  !total

let test_strategy_follows_the_algebra () =
  let weights = const (tensor [| 2 |] [| 1.0; 1.0 |]) in
  let cases =
    [
      ("uniform", D_uniform, Estimator.Strategy.Pathwise);
      ( "normal",
        Ast.Normal.normal ~mu:"mu" ~sigma:"sigma",
        Estimator.Strategy.Pathwise );
      ( "half_normal",
        Ast.Half_normal.half_normal ~sigma:"sigma",
        Estimator.Strategy.Pathwise );
      ( "log_normal",
        Ast.Log_normal.log_normal ~mu:"mu" ~sigma:"sigma",
        Estimator.Strategy.Pathwise );
      ("categorical", D_categorical weights, Estimator.Strategy.Enumerate);
      ( "product",
        D_product (D_uniform, D_uniform),
        Estimator.Strategy.Unsupported );
    ]
  in
  List.iter
    (fun (name, dist, expected) ->
      check string (name ^ " strategy")
        (Estimator.Strategy.to_string expected)
        (Estimator.Strategy.to_string (Estimator.Strategy.of_dist dist));
      (* The table must agree with the checker that already existed: a site is
         Pathwise exactly when check_guide accepts it as reparameterizable. *)
      let reparameterizable =
        try
          Transform.Reparam.check_guide (sample "z" [||] dist);
          true
        with Transform.Reparam.Guide_error _ -> false
      in
      check bool
        (name ^ ": Pathwise iff check_guide accepts")
        (expected = Estimator.Strategy.Pathwise)
        reparameterizable)
    cases

(* A discrete site is a perfectly good objective; it just cannot be lowered
   pathwise.  The objective must therefore build, and only the lowering refuse. *)
let test_discrete_objective_builds_but_is_not_pathwise () =
  let model, guide, env_shapes = discrete_program () in
  let objective = Estimator.elbo ~slots:[] ~model ~guide ~env_shapes in
  check string "objective admits enumeration" "enumerate"
    (Estimator.Strategy.to_string (Estimator.strategy objective));
  let raised =
    try
      ignore (Estimator.lower_pathwise objective);
      false
    with Transform.Reparam.Guide_error (_, message) ->
      check bool "explains why" true
        (String.length message > 0
        && String.index_opt message 'd' <> None);
      true
  in
  check bool "pathwise lowering refuses a discrete site" true raised

let test_enumeration_is_exact () =
  let model, guide, env_shapes = discrete_program () in
  let program =
    Estimator.lower_enumerate @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes
  in
  (* No noise variables: this is the expectation, not an estimate of it. *)
  check (list string) "enumeration draws nothing" [] (List.map fst program.noise);
  check bool "no Sample survives" false (has_sample program.loss);
  List.iter
    (fun theta ->
      let actual =
        value (Ast.Eval.eval [ ("theta", scalar theta) ] program.loss)
      in
      check (float 1e-12)
        (Printf.sprintf "ELBO at theta=%.2f" theta)
        (closed_form_elbo theta) actual)
    [ -1.5; 0.0; 0.75 ]

(* The point of enumerating: an EXACT gradient of an expectation, produced by
   the ordinary AD with no estimator-specific machinery. *)
let test_enumerated_gradient_is_exact () =
  let model, guide, env_shapes = discrete_program () in
  let program =
    Estimator.lower_enumerate @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes
  in
  let gp = Transform.grad ~param_shapes:[ ("theta", [||]) ] program.loss in
  let at theta = [ ("theta", scalar theta) ] in
  let gradient theta =
    value (Ast.Eval.eval (at theta) (List.assoc "theta" gp.grads))
  in
  let loss theta = value (Ast.Eval.eval (at theta) gp.loss) in
  check (float 1e-12) "gradient equals the closed form"
    (closed_form_gradient ()) (gradient 0.4);
  let epsilon = 1e-5 in
  let finite_difference =
    (loss (0.4 +. epsilon) -. loss (0.4 -. epsilon)) /. (2.0 *. epsilon)
  in
  check (float 1e-6) "gradient equals finite differences" finite_difference
    (gradient 0.4)

let test_enumeration_refuses_a_continuous_site () =
  let model, guide, env_shapes = conjugate 3 in
  let objective = Estimator.elbo ~slots:[] ~model ~guide ~env_shapes in
  check string "objective admits the pathwise estimator" "pathwise"
    (Estimator.Strategy.to_string (Estimator.strategy objective));
  let raised =
    try
      ignore (Estimator.lower_enumerate objective);
      false
    with Estimator.Lower_enumerate.Enumerate_error _ -> true
  in
  check bool "enumeration refuses a continuous site" true raised

(* Learned weights.  q(k) = softmax(logits), so the measure itself depends on a
   parameter -- which is the case a score-function estimator exists for, and the
   case enumeration answers exactly.  Both the value and the gradient have a
   closed form, so this is a golden reference with no sampling error in it. *)
let logits = [| 0.3; -0.7; 1.1 |]

let learned_program () =
  let model =
    let_ "k" (sample "k" [||] (D_categorical (const (tensor [| 3 |] model_weights))))
      (score (rank 0 Mul [ var "k"; var "theta" ]))
  in
  let guide = sample "k" [||] (D_categorical (prim Exp [ var "logits" ])) in
  (model, guide, [ ("theta", [||]); ("logits", [| 3 |]) ])

let softmax values =
  let weights = Array.map exp values in
  let total = Array.fold_left ( +. ) 0.0 weights in
  Array.map (fun w -> w /. total) weights

(* f_k = log p_k + k*theta - log q_k, and ELBO = sum_k q_k f_k. *)
let learned_terms theta =
  let p = normalize model_weights and q = softmax logits in
  (q, Array.mapi (fun k qk -> log p.(k) +. (float_of_int k *. theta) -. log qk) q)

let learned_elbo theta =
  let q, f = learned_terms theta in
  let total = ref 0.0 in
  Array.iteri (fun k qk -> total := !total +. (qk *. f.(k))) q;
  !total

(* d/dlogits_j sum_k q_k f_k  =  q_j (f_j - ELBO), since dq_k/dl_j = q_k(d_kj - q_j)
   and dlog q_k/dl_j = d_kj - q_j.  The two -q_j terms cancel. *)
let learned_logit_gradient theta =
  let q, f = learned_terms theta in
  let elbo = learned_elbo theta in
  Array.mapi (fun j qj -> qj *. (f.(j) -. elbo)) q

(* d/dtheta sum_k q_k f_k = sum_k q_k * k, under this program's own q. *)
let learned_theta_gradient () =
  let q = softmax logits in
  let total = ref 0.0 in
  Array.iteri (fun k qk -> total := !total +. (qk *. float_of_int k)) q;
  !total

let test_learned_categorical_weights_are_exact () =
  let model, guide, env_shapes = learned_program () in
  let program =
    Estimator.lower_enumerate @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes
  in
  check (list string) "enumeration draws nothing" [] (List.map fst program.noise);
  let theta = 0.4 in
  let env = [ ("theta", scalar theta); ("logits", tensor [| 3 |] logits) ] in
  check (float 1e-12) "ELBO with learned weights"
    (learned_elbo theta)
    (value (Ast.Eval.eval env program.loss));
  let gp =
    Transform.grad
      ~param_shapes:[ ("theta", [||]); ("logits", [| 3 |]) ]
      program.loss
  in
  let gradient name = Ast.Eval.eval env (List.assoc name gp.grads) in
  check (float 1e-12) "d/dtheta" (learned_theta_gradient ())
    (value (gradient "theta"));
  let expected = learned_logit_gradient theta in
  let actual = gradient "logits" in
  Array.iteri
    (fun j want ->
      check (float 1e-12)
        (Printf.sprintf "d/dlogits[%d]" j)
        want
        (View.Buf.get actual.View.Tensor.buf j))
    expected

(* Weights bound in a Let resolve too: local_shapes walks the proposal's spine,
   so a proposal that computes its parameters in stages is sized the same as one
   that inlines them.  The two spellings must agree exactly. *)
let test_let_bound_categorical_weights_resolve () =
  let model, inline_guide, env_shapes = learned_program () in
  let staged_guide =
    let_ "w" (prim Exp [ var "logits" ])
      (sample "k" [||] (D_categorical (var "w")))
  in
  let evaluate guide =
    let program =
      Estimator.lower_enumerate
      @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes
    in
    let env = [ ("theta", scalar 0.4); ("logits", tensor [| 3 |] logits) ] in
    value (Ast.Eval.eval env program.loss)
  in
  check (float 1e-12) "staged weights agree with inline weights"
    (evaluate inline_guide) (evaluate staged_guide)

(* The honest edge: a size the shapes do not determine is refused rather than
   guessed.  Here the weights name a variable no one declared a shape for. *)
let test_undeclared_categorical_size_is_refused () =
  let undeclared () =
    sample "k" [||] (D_categorical (prim Exp [ var "undeclared" ]))
  in
  let raised =
    try
      ignore
        (Estimator.elbo ~slots:[] ~model:(undeclared ()) ~guide:(undeclared ())
           ~env_shapes:[]);
      false
    with Failure message ->
      check bool "explains that the size must be static" true
        (contains message "statically known");
      true
  in
  check bool "an unresolvable categorical size is refused" true raised

(* ── Step 7-9: score function ── *)

let element tensor index = View.Buf.get tensor.View.Tensor.buf index

let learned_objective () =
  let model, guide, env_shapes = learned_program () in
  (Estimator.elbo ~slots:[] ~model ~guide ~env_shapes, env_shapes)

let learned_env theta =
  [ ("theta", scalar theta); ("logits", tensor [| 3 |] logits) ]

(* The subtracted stopgrad keeps the value honest: the surrogate still reports
   the objective, it just differentiates like the score-function estimator. *)
let test_surrogate_reports_the_objective () =
  let objective, _ = learned_objective () in
  let score = Estimator.lower_score objective in
  let enumerated = Estimator.lower_enumerate objective in
  check (list string) "draws are supplied from outside" [ "%tr.k" ]
    (List.map fst score.noise);
  let theta = 0.4 in
  let _, f = learned_terms theta in
  Array.iteri
    (fun k want ->
      let env =
        ("%tr.k", scalar (float_of_int k)) :: learned_env theta
      in
      check (float 1e-12)
        (Printf.sprintf "surrogate value at k=%d" k)
        want
        (value (Ast.Eval.eval env score.loss)))
    f;
  (* and the enumeration of the same objective is its expectation *)
  check (float 1e-12) "enumeration is the expectation of the surrogate"
    (learned_elbo theta)
    (value (Ast.Eval.eval (learned_env theta) enumerated.loss))

(* Unbiasedness, checked exactly rather than statistically.  The support is
   finite, so E[g_score] is a weighted sum over the three possible draws, and it
   must equal the gradient enumeration computes directly. *)
let test_score_estimator_is_unbiased () =
  let objective, _ = learned_objective () in
  let score = Estimator.lower_score objective in
  let enumerated = Estimator.lower_enumerate objective in
  let param_shapes = [ ("theta", [||]); ("logits", [| 3 |]) ] in
  let score_grad =
    Transform.grad ~param_shapes ~data_shapes:[ ("%tr.k", [||]) ] score.loss
  in
  let exact_grad = Transform.grad ~param_shapes enumerated.loss in
  let theta = 0.4 in
  let q = softmax logits in
  let expectation name width =
    let total = Array.make width 0.0 in
    Array.iteri
      (fun k qk ->
        let env = ("%tr.k", scalar (float_of_int k)) :: learned_env theta in
        let g = Ast.Eval.eval env (List.assoc name score_grad.grads) in
        for i = 0 to width - 1 do
          total.(i) <- total.(i) +. (qk *. element g i)
        done)
      q;
    total
  in
  let exact name = Ast.Eval.eval (learned_env theta) (List.assoc name exact_grad.grads) in
  check (float 1e-12) "E[score gradient] wrt theta"
    (element (exact "theta") 0)
    (expectation "theta" 1).(0);
  let expected_logits = exact "logits" in
  let averaged = expectation "logits" 3 in
  Array.iteri
    (fun j got ->
      check (float 1e-12)
        (Printf.sprintf "E[score gradient] wrt logits[%d]" j)
        (element expected_logits j) got)
    averaged

(* The statistical layer, and the only thing it is for: that the draws actually
   come out of Simulate with the right names and the right distribution.  The
   estimator's correctness is settled deterministically above. *)
let test_score_estimator_converges () =
  let objective, _ = learned_objective () in
  let score = Estimator.lower_score objective in
  let score_grad =
    Transform.grad
      ~param_shapes:[ ("theta", [||]); ("logits", [| 3 |]) ]
      ~data_shapes:[ ("%tr.k", [||]) ]
      score.loss
  in
  let theta = 0.4 in
  let env = learned_env theta in
  let draws = 20000 in
  let total = ref 0.0 in
  for run = 1 to draws do
    let z = Estimator.draw objective ~env ~run_key:(Int64.of_int run) in
    let g =
      Ast.Eval.eval (z @ env) (List.assoc "theta" score_grad.grads)
    in
    total := !total +. element g 0
  done;
  let mean = !total /. float_of_int draws in
  let expected = learned_theta_gradient () in
  Printf.printf "\n=== score function, %d draws: %.6f (exact %.6f) ===\n" draws
    mean expected;
  check bool "Monte Carlo mean is near the exact gradient" true
    (Float.abs (mean -. expected) < 0.03)

(* ── Step 10-11: several sites, loss-to-go, baseline ── *)

(* x is drawn first, then y conditioned on x, and the model scores both.  The
   guide's second factor depends on the first draw, which is what makes the
   sampling ORDER -- and so loss-to-go -- mean anything. *)
let chain_program () =
  let model =
    let_ "x"
      (sample "x" [||] (D_categorical (const (tensor [| 2 |] [| 1.0; 3.0 |]))))
      (let_ "y"
         (sample "y" [||] (D_categorical (const (tensor [| 2 |] [| 2.0; 1.0 |]))))
         (score
            (rank 0 Add
               [
                 rank 0 Mul [ var "x"; var "theta" ];
                 rank 0 Mul [ var "y"; rank 0 Add [ var "x"; var "theta" ] ];
               ])))
  in
  let guide =
    let_ "x" (sample "x" [||] (D_categorical (prim Exp [ var "phi_x" ])))
      (sample "y" [||]
         (D_categorical (prim Exp [ rank 0 Mul [ var "x"; var "phi_y" ] ])))
  in
  (model, guide,
   [ ("theta", [||]); ("phi_x", [| 2 |]); ("phi_y", [| 2 |]) ])

let chain_env =
  [
    ("theta", scalar 0.6);
    ("phi_x", tensor [| 2 |] [| 0.4; -0.3 |]);
    ("phi_y", tensor [| 2 |] [| 0.2; 0.9 |]);
  ]

let chain_params = [ ("theta", [||]); ("phi_x", [| 2 |]); ("phi_y", [| 2 |]) ]
let chain_draws = [ ("%tr.x", [||]); ("%tr.y", [||]) ]
let chain_assignments = [ (0, 0); (0, 1); (1, 0); (1, 1) ]

let trace_env x y =
  [ ("%tr.x", scalar (float_of_int x)); ("%tr.y", scalar (float_of_int y)) ]

let chain_objective () =
  let model, guide, env_shapes = chain_program () in
  Estimator.elbo ~slots:[] ~model ~guide ~env_shapes

(* q(x, y) for each assignment, from the value-level interpreter rather than by
   hand, so the reference is Ast.Assess and not a second transcription. *)
let chain_weights () =
  let _, guide, env_shapes = chain_program () in
  let guide = Transform.Expand_rank.expand ~senv:env_shapes guide in
  List.map
    (fun (x, y) ->
      let trace =
        [ ("x", scalar (float_of_int x)); ("y", scalar (float_of_int y)) ]
      in
      let _, log_q = Ast.Assess.assess chain_env guide trace in
      ((x, y), exp (value log_q)))
    chain_assignments

(* Exact mean and variance of an estimator, over the joint support.  Both are
   sums, so neither needs sampling. *)
let moments expression width =
  let mean = Array.make width 0.0 and square = Array.make width 0.0 in
  List.iter
    (fun ((x, y), q) ->
      let g = Ast.Eval.eval (trace_env x y @ chain_env) expression in
      for i = 0 to width - 1 do
        let v = element g i in
        mean.(i) <- mean.(i) +. (q *. v);
        square.(i) <- square.(i) +. (q *. v *. v)
      done)
    (chain_weights ());
  (mean, Array.mapi (fun i s -> s -. (mean.(i) *. mean.(i))) square)

let score_gradient options param =
  let objective = chain_objective () in
  let program = Estimator.lower_score ~options objective in
  let gp =
    Transform.grad ~param_shapes:chain_params ~data_shapes:chain_draws
      program.loss
  in
  List.assoc param gp.grads

let exact_gradient param =
  let program = Estimator.lower_enumerate (chain_objective ()) in
  let gp = Transform.grad ~param_shapes:chain_params program.loss in
  Ast.Eval.eval chain_env (List.assoc param gp.grads)

(* Decompose.split is used to attribute cost to sites; if it did not add back up
   to what it split, every estimator built on it would be quietly wrong. *)
let test_decomposition_is_faithful () =
  let _, _, env_shapes = chain_program () in
  match chain_objective () with
  | Estimator.Types.Deterministic _ -> fail "expected an expectation"
  | Estimator.Types.Expect expectation ->
      let split = Estimator.Decompose.split expectation.body in
      check bool "the body really does decompose" true
        (List.length split.Estimator.Decompose.terms > 1);
      let rebuilt =
        Estimator.Decompose.recombine split
        |> Transform.Expand_rank.expand ~senv:(chain_draws @ env_shapes)
      in
      List.iter
        (fun (x, y) ->
          let env = trace_env x y @ chain_env in
          check (float 1e-12)
            (Printf.sprintf "sum of terms at (%d,%d)" x y)
            (value (Ast.Eval.eval env expectation.body))
            (value (Ast.Eval.eval env rebuilt)))
        chain_assignments

let test_joint_enumeration_matches_assess () =
  let model, _, env_shapes = chain_program () in
  let model = Transform.Expand_rank.expand ~senv:env_shapes model in
  let _, guide, _ = chain_program () in
  let guide = Transform.Expand_rank.expand ~senv:env_shapes guide in
  let expected =
    List.fold_left
      (fun total ((x, y), q) ->
        let trace =
          [ ("x", scalar (float_of_int x)); ("y", scalar (float_of_int y)) ]
        in
        let _, log_p = Ast.Assess.assess chain_env model trace in
        let _, log_q = Ast.Assess.assess chain_env guide trace in
        total +. (q *. (value log_p -. value log_q)))
      0.0 (chain_weights ())
  in
  let program = Estimator.lower_enumerate (chain_objective ()) in
  check (list string) "enumeration draws nothing" [] (List.map fst program.noise);
  check (float 1e-12) "joint enumeration equals the assessed expectation" expected
    (value (Ast.Eval.eval chain_env program.loss))

(* Every variant must have the SAME expectation as the exact gradient.  On a
   finite support that is a sum, so bias is a deterministic question. *)
let test_variants_are_unbiased () =
  let variants =
    [
      ("plain", Estimator.Lower_score.plain);
      ("loss-to-go", { Estimator.Lower_score.plain with loss_to_go = true });
      ("baseline", { Estimator.Lower_score.plain with baseline = 1.5 });
      ("both", { Estimator.Lower_score.loss_to_go = true; baseline = 1.5 });
    ]
  in
  List.iter
    (fun (param, width) ->
      let exact = exact_gradient param in
      List.iter
        (fun (name, options) ->
          let mean, _ = moments (score_gradient options param) width in
          Array.iteri
            (fun i got ->
              check (float 1e-12)
                (Printf.sprintf "%s: E[g] for %s[%d]" name param i)
                (element exact i) got)
            mean)
        variants)
    [ ("theta", 1); ("phi_x", 2); ("phi_y", 2) ]

(* Loss-to-go helps the LAST site, whose score term can shed the cost that was
   already settled before it was drawn.  It can do nothing for the first site,
   which is downstream of everything. *)
let test_loss_to_go_reduces_variance () =
  let plain = Estimator.Lower_score.plain in
  let to_go = { plain with Estimator.Lower_score.loss_to_go = true } in
  let _, plain_var = moments (score_gradient plain "phi_y") 2 in
  let _, to_go_var = moments (score_gradient to_go "phi_y") 2 in
  Printf.printf "\n=== phi_y variance: plain %.6f %.6f -> loss-to-go %.6f %.6f ===\n"
    plain_var.(0) plain_var.(1) to_go_var.(0) to_go_var.(1);
  Array.iteri
    (fun i before ->
      check bool
        (Printf.sprintf "loss-to-go lowers the variance of phi_y[%d]" i)
        true
        (to_go_var.(i) < before))
    plain_var

(* A baseline near E[f] centres the multiplier, which is what a control variate
   is for.  It helps the first site, where loss-to-go cannot. *)
let test_baseline_reduces_variance () =
  let program = Estimator.lower_enumerate (chain_objective ()) in
  let elbo = value (Ast.Eval.eval chain_env program.loss) in
  let plain = Estimator.Lower_score.plain in
  let centred = { plain with Estimator.Lower_score.baseline = elbo } in
  let _, plain_var = moments (score_gradient plain "phi_x") 2 in
  let _, centred_var = moments (score_gradient centred "phi_x") 2 in
  Printf.printf "\n=== phi_x variance: plain %.6f %.6f -> baseline %.6f %.6f ===\n"
    plain_var.(0) plain_var.(1) centred_var.(0) centred_var.(1);
  Array.iteri
    (fun i before ->
      check bool
        (Printf.sprintf "the baseline lowers the variance of phi_x[%d]" i)
        true
        (centred_var.(i) < before))
    plain_var

let () =
  run "Estimator"
    [
      ( "boundary",
        [
          test_case "objective separates integrand from proposal" `Quick
            test_objective_separates_integrand_from_proposal;
          test_case "deterministic objective passes through" `Quick
            test_deterministic_objective_passes_through;
        ] );
      ( "lower_pathwise",
        [
          test_case "eliminates samples" `Quick
            test_pathwise_lowering_eliminates_samples;
          test_case "is reproducible" `Quick test_lowering_is_reproducible;
          test_case "refuses a discrete site" `Quick
            test_discrete_objective_builds_but_is_not_pathwise;
        ] );
      ( "strategy",
        [
          test_case "follows the distribution algebra" `Quick
            test_strategy_follows_the_algebra;
        ] );
      ( "lower_enumerate",
        [
          test_case "is exact" `Quick test_enumeration_is_exact;
          test_case "gradient is exact" `Quick
            test_enumerated_gradient_is_exact;
          test_case "refuses a continuous site" `Quick
            test_enumeration_refuses_a_continuous_site;
          test_case "learned weights are exact" `Quick
            test_learned_categorical_weights_are_exact;
          test_case "Let-bound weights resolve" `Quick
            test_let_bound_categorical_weights_resolve;
          test_case "undeclared size is refused" `Quick
            test_undeclared_categorical_size_is_refused;
        ] );
      ( "lower_score",
        [
          test_case "surrogate reports the objective" `Quick
            test_surrogate_reports_the_objective;
          test_case "is unbiased (exact)" `Quick
            test_score_estimator_is_unbiased;
          test_case "converges" `Slow test_score_estimator_converges;
        ] );
      ( "several sites",
        [
          test_case "decomposition is faithful" `Quick
            test_decomposition_is_faithful;
          test_case "joint enumeration matches assess" `Quick
            test_joint_enumeration_matches_assess;
          test_case "every variant is unbiased" `Quick
            test_variants_are_unbiased;
          test_case "loss-to-go reduces variance" `Quick
            test_loss_to_go_reduces_variance;
          test_case "baseline reduces variance" `Quick
            test_baseline_reduces_variance;
        ] );
    ]
