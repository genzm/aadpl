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

let test_conjugate_gaussian_vi () =
  let samples = 1000 in
  let observed = 1.2 in
  let observation_sigma = 0.7 in
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
      ("prior_mu", [||]);
      ("prior_sigma", [||]);
      ("x_obs", [||]);
      ("obs_sigma", [||]);
      ("mu_q", [||]);
      ("log_sigma_q", [||]);
    ]
  in
  let program = Transform.build_elbo ~model ~guide ~env_shapes in
  let average_elbo =
    prim Mul [ const (scalar (1.0 /. float_of_int samples)); program.elbo ]
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
    let env =
      Transform.noise_env program ~run_key:(Int64.of_int step)
      @ [ ("mu_q", scalar !mu_q); ("log_sigma_q", scalar !log_sigma_q) ]
      @ fixed_env
    in
    Ast.Eval.eval_grad env ~primal_bindings:differentiated.primal_bindings
      ~loss_body:differentiated.loss_body
      ~grad_bodies:differentiated.grad_bodies
  in
  let initial_elbo, _ = evaluate 0 in
  for step = 1 to 500 do
    let _, gradients = evaluate step in
    (* build_elbo returns ELBO, so Adam deliberately updates in ascent direction. *)
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
    ]
