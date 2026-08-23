(* Phase 11-5a: conjugate Gaussian variational inference.
   The loop only draws guide noise, evaluates the transformed ELBO/gradient,
   and applies an Adam ascent update. *)

open Alcotest
open Ast.Types

let scalar value =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 value;
  t

let scalar_value tensor = View.Buf.get tensor.View.Tensor.buf 0

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

type adam = { mutable m : float; mutable v : float }

let adam_ascent ~step ~learning_rate state parameter gradient =
  let beta1 = 0.9 and beta2 = 0.999 and epsilon = 1e-8 in
  state.m <- (beta1 *. state.m) +. ((1.0 -. beta1) *. gradient);
  state.v <- (beta2 *. state.v) +. ((1.0 -. beta2) *. gradient *. gradient);
  let correction1 = 1.0 -. (beta1 ** float_of_int step) in
  let correction2 = 1.0 -. (beta2 ** float_of_int step) in
  parameter
  +. learning_rate *. (state.m /. correction1)
     /. (sqrt (state.v /. correction2) +. epsilon)

let conjugate_program samples =
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
  model, guide, env_shapes

let test_frame_sample_score_assess () =
  let model, _, env_shapes = conjugate_program 3 in
  let z = View.Tensor.make [|3|] in
  List.iteri (View.Buf.set z.buf) [-0.2; 0.4; 1.1];
  let env =
    [
      ("prior_mu", scalar 0.0); ("prior_sigma", scalar 1.0);
      ("x_obs", scalar 1.2); ("obs_sigma", scalar 0.7);
      ("mu_q", scalar 0.3); ("log_sigma_q", scalar (log 0.8));
    ]
  in
  let model = Transform.Expand_rank.expand ~senv:env_shapes model in
  let _, value_ld = Ast.Assess.assess env model [("z", z)] in
  let symbolic_ld = Transform.Assess_expr.assess_expr ~env_shapes model
    [("z", const z)] |> Ast.Eval.eval env in
  check (float 1e-12) "frame Sample + Score" (scalar_value value_ld)
    (scalar_value symbolic_ld)

let test_conjugate_elbo_fd () =
  let samples = 7 in
  let model, guide, env_shapes = conjugate_program samples in
  let program = Estimator.lower_pathwise @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes in
  let average_elbo =
    prim Mul [ const (scalar (1.0 /. float_of_int samples)); program.loss ]
  in
  let gp = Transform.grad
    ~param_shapes:[("mu_q", [||]); ("log_sigma_q", [||])]
    ~data_shapes:
      (program.noise @ [
        ("prior_mu", [||]); ("prior_sigma", [||]);
        ("x_obs", [||]); ("obs_sigma", [||]);
      ])
    average_elbo
  in
  let fixed_env =
    Estimator.noise_env program ~run_key:42L @ [
      ("prior_mu", scalar 0.0); ("prior_sigma", scalar 1.0);
      ("x_obs", scalar 1.2); ("obs_sigma", scalar 0.7);
    ]
  in
  let make_env mu rho =
    ("mu_q", scalar mu) :: ("log_sigma_q", scalar rho) :: fixed_env
  in
  let mu = 0.3 and rho = log 0.8 and epsilon = 1e-5 in
  let env = make_env mu rho in
  let ad name = scalar_value (Ast.Eval.eval env (List.assoc name gp.grads)) in
  let eval mu rho = scalar_value (Ast.Eval.eval (make_env mu rho) gp.loss) in
  let fd_mu = (eval (mu +. epsilon) rho -. eval (mu -. epsilon) rho)
              /. (2.0 *. epsilon) in
  let fd_rho = (eval mu (rho +. epsilon) -. eval mu (rho -. epsilon))
               /. (2.0 *. epsilon) in
  check (float 1e-6) "mu_q FD" fd_mu (ad "mu_q");
  check (float 1e-6) "rho FD" fd_rho (ad "log_sigma_q")

let test_conjugate_gaussian_vi () =
  let samples = 1000 in
  let observed = 1.2 in
  let observation_sigma = 0.7 in
  let model, guide, env_shapes = conjugate_program samples in
  let program = Estimator.lower_pathwise @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes in
  let average_elbo =
    prim Mul [ const (scalar (1.0 /. float_of_int samples)); program.loss ]
  in
  let param_shapes = [ ("mu_q", [||]); ("log_sigma_q", [||]) ] in
  let fixed_shapes =
    [
      ("prior_mu", [||]);
      ("prior_sigma", [||]);
      ("x_obs", [||]);
      ("obs_sigma", [||]);
    ]
  in
  let differentiated =
    Transform.grad ~param_shapes
      ~data_shapes:(program.noise @ fixed_shapes)
      average_elbo
  in
  let fixed_env =
    [
      ("prior_mu", scalar 0.0);
      ("prior_sigma", scalar 1.0);
      ("x_obs", scalar observed);
      ("obs_sigma", scalar observation_sigma);
    ]
  in
  let mu_q = ref (-0.5) in
  let log_sigma_q = ref (log 1.5) in
  let mu_state = { m = 0.0; v = 0.0 } in
  let sigma_state = { m = 0.0; v = 0.0 } in
  let evaluate step =
    (* SGD convention: redraw u from the deterministic Threefry stream each
       step.  MNIST uses the same policy; fixed run_key would instead be SAA. *)
    let env =
      Estimator.noise_env program ~run_key:(Int64.of_int step)
      @ [ ("mu_q", scalar !mu_q); ("log_sigma_q", scalar !log_sigma_q) ]
      @ fixed_env
    in
    Ast.Eval.eval_grad env ~primal_bindings:differentiated.primal_bindings
      ~loss_body:differentiated.loss_body
      ~grad_bindings:differentiated.grad_bindings
      ~grad_bodies:differentiated.grad_bodies
  in
  let initial_elbo, _ = evaluate 0 in
  for step = 1 to 500 do
    let _, gradients = evaluate step in
    (* the objective is an ELBO, so Adam deliberately updates in ascent direction. *)
    mu_q :=
      adam_ascent ~step ~learning_rate:0.03 mu_state !mu_q
        (scalar_value (List.assoc "mu_q" gradients));
    log_sigma_q :=
      adam_ascent ~step ~learning_rate:0.03 sigma_state !log_sigma_q
        (scalar_value (List.assoc "log_sigma_q" gradients))
  done;
  let final_elbo, _ = evaluate 501 in
  let variance = observation_sigma *. observation_sigma in
  let expected_mu = observed /. (1.0 +. variance) in
  let expected_sigma = observation_sigma /. sqrt (1.0 +. variance) in
  let expected_elbo =
    (-0.5 *. log (2.0 *. Float.pi *. (1.0 +. variance)))
    -. (observed *. observed /. (2.0 *. (1.0 +. variance)))
  in
  let actual_sigma = exp !log_sigma_q in
  let actual_elbo = scalar_value final_elbo in
  Printf.printf
    "\n\
     === 11-5a conjugate Gaussian VI (K=%d) ===\n\
    \  mu: %.6f (exact %.6f)\n\
    \  sigma: %.6f (exact %.6f)\n\
    \  ELBO: %.6f (exact %.6f)\n"
    samples !mu_q expected_mu actual_sigma expected_sigma actual_elbo
    expected_elbo;
  check bool "ELBO improves" true (actual_elbo > scalar_value initial_elbo);
  check bool "posterior mean relerr < 1e-2" true
    (Float.abs (!mu_q -. expected_mu) /. Float.abs expected_mu < 1e-2);
  check bool "posterior sigma relerr < 1e-2" true
    (Float.abs (actual_sigma -. expected_sigma) /. expected_sigma < 1e-2);
  check bool "optimal ELBO abs err < 1e-2" true
    (Float.abs (actual_elbo -. expected_elbo) < 1e-2)

let () =
  run "VI"
    [
      ("conjugate", [ test_case "closed form" `Slow test_conjugate_gaussian_vi ]);
      ( "preflight",
        [
          test_case "frame Sample + Score assess" `Quick
            test_frame_sample_score_assess;
          test_case "ELBO FD" `Quick test_conjugate_elbo_fd;
        ] );
    ]
