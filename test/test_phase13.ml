open Alcotest
open Ast.Types

let scalar value =
  let result = View.Tensor.make [||] in
  View.Buf.set result.buf 0 value;
  result

let tensor shape values =
  let result = View.Tensor.make shape in
  Array.iteri (View.Buf.set result.buf) values;
  result

let value tensor index = View.Buf.get tensor.View.Tensor.buf index

let test_regularized_gamma_closed_forms () =
  List.iter (fun x ->
    check (float 2e-14) (Printf.sprintf "P(1,%g)" x)
      (1.0 -. exp (-.x)) (Transform.Special.regularized_gamma_p 1.0 x);
    check (float 2e-14) (Printf.sprintf "Q(1,%g)" x)
      (exp (-.x)) (Transform.Special.regularized_gamma_q 1.0 x))
    [0.01; 0.5; 2.0; 20.0]

let test_chi_square_one () =
  List.iter (fun (statistic, expected) ->
    check (float 2e-14) (Printf.sprintf "chi-square(1) cdf %g" statistic)
      expected (Transform.Special.chi_square_1_cdf statistic))
    [(0.0, 0.0); (0.01, 0.07965567455405795);
     (1.0, 0.6826894921370861); (3.841458820694124, 0.95);
     (10.0, 0.9984345977419975)];
  check (float 1e-35) "chi-square(1) survival at 100"
    1.5239706048320995e-23 (Transform.Special.chi_square_1_sf 100.0);
  check (float 0.0) "chi-square(1) cdf at infinity" 1.0
    (Transform.Special.chi_square_1_cdf infinity);
  check (float 0.0) "chi-square(1) survival at infinity" 0.0
    (Transform.Special.chi_square_1_sf infinity)

let test_boundary_mixture () =
  check (float 0.0) "atom at zero" 1.0
    (Transform.Special.boundary_variance_component_p_value 0.0);
  check (float 1e-14) "positive statistic halves chi-square tail" 0.025
    (Transform.Special.boundary_variance_component_p_value
       3.841458820694124);
  check (float 0.0) "optimization-scale statistic is atom" 1.0
    (Transform.Special.boundary_variance_component_p_value
       ~statistic_tolerance:1e-10 1e-12);
  check (float 0.0) "boundary estimate is atom" 1.0
    (Transform.Special.boundary_variance_component_p_value
       ~boundary_estimate:1e-9 ~boundary_tolerance:1e-8 0.1)

let test_frame_categorical_observation () =
  let frame = [|2; 2|] in
  let shared = sample "shared" frame
    (D_categorical (const (tensor [|2|] [|0.0; 1.0|]))) in
  let _, shared_trace, _ = Ast.Simulate.simulate ~run_key:70L [] shared in
  let shared_values = List.assoc "shared" shared_trace in
  for cell = 0 to 3 do
    check (float 0.0) (Printf.sprintf "shared weights cell %d" cell)
      1.0 (value shared_values cell)
  done;
  let weights = tensor [|2; 2; 2|]
    [|1.0; 3.0; 4.0; 1.0; 2.0; 2.0; 1.0; 9.0|] in
  let model = sample "y" frame (D_categorical (const weights)) in
  let _, trace, _ = Ast.Simulate.simulate ~run_key:71L [] model in
  let y = List.assoc "y" trace in
  let key = Prng.Threefry.make_key ~run_key:71L
    ~namespace:Prng.Threefry.ns_model in
  for cell = 0 to 3 do
    let ctr = Prng.Threefry.make_ctr ~site_id:0 ~component:1
      ~frame_index:cell in
    let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
    let u = Prng.Threefry.to_open_unit bits in
    let w0 = value weights (2 * cell) and w1 = value weights (2 * cell + 1) in
    let expected = if u *. (w0 +. w1) <= w0 then 0.0 else 1.0 in
    check (float 0.0) (Printf.sprintf "cell %d counter" cell)
      expected (value y cell)
  done;
  let _, assessed = Ast.Assess.assess [] model trace in
  let symbolic = Transform.Assess_expr.assess_expr
    ~env_shapes:[("y_obs", frame)] model [("y", var "y_obs")]
    |> Ast.Eval.eval [("y_obs", y)] in
  check (float 1e-12) "frame categorical assess equals expression"
    (value assessed 0) (value symbolic 0);
  let program = Transform.build_elbo
    ~slots:[("y", `Condition, var "y_obs")]
    ~model ~guide:(const (scalar 0.0)) ~env_shapes:[("y_obs", frame)] in
  let elbo = Ast.Eval.eval [("y_obs", y)] program.elbo in
  check (float 1e-12) "observed categorical enters ELBO"
    (value assessed 0) (value elbo 0)

let test_slot_roles () =
  let model = sample "theta" [||]
    (Ast.Normal.normal ~mu:"zero" ~sigma:"one") in
  let env_shapes = [("zero", [||]); ("one", [||]); ("theta_value", [||])] in
  let make role = Transform.build_elbo
    ~slots:[("theta", role, var "theta_value")]
    ~model ~guide:(const (scalar 0.0)) ~env_shapes in
  let env = [("zero", scalar 0.0); ("one", scalar 1.0);
    ("theta_value", scalar 2.0)] in
  let condition = Ast.Eval.eval env (make `Condition).elbo |> fun result -> value result 0
  and maximize = Ast.Eval.eval env (make `Maximize).elbo |> fun result -> value result 0 in
  check (float 1e-12) "Condition includes site density"
    (-.0.5 *. log (2.0 *. Float.pi) -. 2.0) condition;
  check (float 0.0) "Maximize excludes site density" 0.0 maximize

let test_gauss_hermite_rule () =
  List.iter (fun node_count ->
    let nodes, weights = Transform.Quadrature.gauss_hermite node_count in
    for degree = 0 to node_count - 1 do
      let actual = ref 0.0 in
      Array.iteri (fun index node ->
        actual := !actual +. weights.(index) *. node ** float_of_int (2 * degree))
        nodes;
      let expected = exp (Transform.Special.log_gamma
        (float_of_int degree +. 0.5)) in
      let relative = Float.abs (!actual -. expected) /. expected in
      check bool (Printf.sprintf "K=%d moment degree=%d" node_count (2 * degree))
        true (relative < 2e-12)
    done) [5; 10; 20]

let gaussian_hierarchical_model groups observations =
  let model =
    let_ "a" (sample "a" [|groups|]
      (Ast.Normal.normal ~mu:"zero" ~sigma:"tau"))
      (sample "y" [|groups; observations|]
        (Ast.Normal.normal ~mu:"a" ~sigma:"obs_scale"))
  in
  model

let gaussian_closed_log_marginal ~groups ~observations ~tau ~obs_scale y =
  let obs_variance = obs_scale *. obs_scale in
  let total = ref 0.0 in
  for group = 0 to groups - 1 do
    let mean = ref 0.0 in
    for observation = 0 to observations - 1 do
      mean := !mean +. value y (group * observations + observation)
    done;
    mean := !mean /. float_of_int observations;
    let centered = ref 0.0 in
    for observation = 0 to observations - 1 do
      let residual = value y (group * observations + observation) -. !mean in
      centered := !centered +. residual *. residual
    done;
    let n = float_of_int observations in
    let log_determinant = float_of_int (observations - 1) *. log obs_variance
      +. log (obs_variance +. n *. tau *. tau) in
    let quadratic = !centered /. obs_variance
      +. n *. !mean *. !mean /. (obs_variance +. n *. tau *. tau) in
    total := !total -. 0.5 *. (n *. log (2.0 *. Float.pi)
      +. log_determinant +. quadratic)
  done;
  !total

let test_quadrature_gaussian_closed_form () =
  let groups = 3 and observations = 3 in
  let tau = 0.5 and obs_scale = 1.0 in
  let y = tensor [|groups; observations|]
    [|-1.0; -0.2; 0.4; 0.1; 0.5; 1.2; -0.7; 0.3; 0.9|] in
  let model = gaussian_hierarchical_model groups observations in
  let env_shapes = [("zero", [||]); ("tau", [||]);
                    ("obs_scale", [||]); ("y_obs", [|groups; observations|])] in
  let expected = gaussian_closed_log_marginal ~groups ~observations
    ~tau ~obs_scale y in
  let make_program node_count =
    let nodes, weights = Transform.Quadrature.gauss_hermite node_count in
    let standardized_nodes = Array.init (node_count * groups) (fun index ->
      sqrt 2.0 *. nodes.(index / groups)) in
    let values = rank 0 Mul
      [const (tensor [|node_count; groups|] standardized_nodes); var "tau"]
      |> Transform.Expand_rank.expand ~senv:env_shapes in
    let log_weights = Array.map
      (fun weight -> log weight -. 0.5 *. log Float.pi) weights
      |> Transform.Quadrature.tensor in
    Transform.Quadrature.quadrature ~site:"a" ~values ~log_weights
      ~slots:[("y", `Condition, var "y_obs")] ~model ~env_shapes
  in
  let errors = List.map (fun node_count ->
    let program = make_program node_count in
    check int "target removed from site table" 0 (List.length program.sites);
    check int "leading node axis recorded" node_count program.node_count;
    check (array int) "log terms keep [K]+site.frame"
      [|node_count; groups|]
      (Transform.Expand_rank.infer_shape env_shapes program.log_terms);
    let actual = Ast.Eval.eval [("zero", scalar 0.0); ("tau", scalar tau);
      ("obs_scale", scalar obs_scale); ("y_obs", y)] program.log_marginal
      |> fun result -> value result 0 in
    Printf.printf "K=%d quadrature %.12f closed %.12f\n"
      node_count actual expected;
    Float.abs (actual -. expected)) [5; 10; 20] in
  match errors with
  | [e5; e10; e20] ->
      Printf.printf "\nGaussian quadrature errors K=5/10/20: %.3e %.3e %.3e\n"
        e5 e10 e20;
      check bool "K=5,10,20 error decreases" true (e10 < e5 && e20 < e10);
      check bool "K=20 error below 1e-8" true (e20 < 1e-8);
      let program = make_program 20 in
      Transform.check_no_samples program.log_marginal;
      let gradient = Transform.grad ~param_shapes:[("tau", [||])]
        ~data_shapes:[("zero", [||]); ("obs_scale", [||]);
          ("y_obs", [|groups; observations|])] program.log_marginal in
      let eval tau expression = Ast.Eval.eval
        [("zero", scalar 0.0); ("tau", scalar tau);
         ("obs_scale", scalar obs_scale); ("y_obs", y)] expression
        |> fun result -> value result 0 in
      let epsilon = 1e-5 in
      let fd = (eval (tau +. epsilon) gradient.loss
        -. eval (tau -. epsilon) gradient.loss) /. (2.0 *. epsilon) in
      let ad = eval tau (List.assoc "tau" gradient.grads) in
      check (float 1e-7) "quadrature tau FD" fd ad
  | _ -> assert false

let logistic_glmm_model groups =
  let logits = rank 0 Add
    [rank 0 Mul [var "gamma"; var "x_obs"]; var "a"] in
  let likelihood = rank 0 Add [
    rank 0 Mul [var "y_obs"; prim Logsigmoid [logits]];
    rank 0 Mul
      [rank 0 Sub [const (scalar 1.0); var "y_obs"];
       prim Logsigmoid [rank 0 Neg [logits]]];
  ] in
  let_ "gamma" (sample "gamma" [||]
    (Ast.Normal.normal ~mu:"zero" ~sigma:"prior_gamma_scale"))
    (let_ "tau" (sample "tau" [||]
      (Ast.Half_normal.half_normal ~sigma:"prior_tau_scale"))
      (let_ "a" (sample "a" [|groups|]
        (Ast.Normal.normal ~mu:"zero" ~sigma:"tau"))
        (score likelihood)))

let glmm_quadrature ~node_count ~groups ~tau_slot ~model
    ~env_shapes =
  let nodes, weights = Transform.Quadrature.gauss_hermite node_count in
  let node_cells = Array.init (node_count * groups)
    (fun index -> sqrt 2.0 *. nodes.(index / groups)) in
  let values = rank 0 Mul
    [const (tensor [|node_count; groups|] node_cells); tau_slot]
    |> Transform.Expand_rank.expand ~senv:env_shapes in
  let log_weights = Array.map
    (fun weight -> log weight -. 0.5 *. log Float.pi) weights
    |> Transform.Quadrature.tensor in
  Transform.Quadrature.quadrature ~site:"a" ~values ~log_weights
    ~slots:[("gamma", `Maximize, var "gamma_param");
            ("tau", `Maximize, tau_slot)]
    ~model ~env_shapes

type adam = { mutable mean : float; mutable variance : float }

let optimize_marginal ~steps ~learning_rate ~parameters ~fixed expression =
  let param_shapes = List.map (fun (name, _) -> name, [||]) parameters in
  let gradient = Transform.grad ~param_shapes
    ~data_shapes:(List.map (fun (name, value) ->
      name, value.View.Tensor.view.View.Ndview.shape) fixed) expression in
  let states = List.map (fun (name, _) -> name, {mean = 0.0; variance = 0.0})
    parameters in
  let evaluate () =
    let env = parameters @ fixed in
    let loss, gradients = Ast.Eval.eval_grad env
      ~primal_bindings:gradient.primal_bindings ~loss_body:gradient.loss_body
      ~grad_bindings:gradient.grad_bindings ~grad_bodies:gradient.grad_bodies in
    value loss 0, gradients in
  let initial, _ = evaluate () in
  for step = 1 to steps do
    let _, gradients = evaluate () in
    List.iter (fun (name, parameter) ->
      let state = List.assoc name states and g = value (List.assoc name gradients) 0 in
      state.mean <- 0.9 *. state.mean +. 0.1 *. g;
      state.variance <- 0.999 *. state.variance +. 0.001 *. g *. g;
      let m = state.mean /. (1.0 -. 0.9 ** float_of_int step)
      and v = state.variance /. (1.0 -. 0.999 ** float_of_int step) in
      View.Buf.set parameter.View.Tensor.buf 0
        (value parameter 0 +. learning_rate *. m /. (sqrt v +. 1e-8)))
      parameters
  done;
  let final, _ = evaluate () in
  initial, final

let test_logistic_glmm_mle () =
  let groups = 64 and observations = 20 in
  let true_gamma = 1.2 and true_tau = 0.7 in
  let model = logistic_glmm_model groups in
  let frame = [|groups; observations|] in
  let x = View.Tensor.make frame and y = View.Tensor.make frame in
  let latent = sample "a" [|groups|]
    (Ast.Normal.normal ~mu:"zero" ~sigma:"true_tau")
    |> Transform.Expand_rank.expand
         ~senv:[("zero", [||]); ("true_tau", [||])] in
  let _, latent_trace, _ = Ast.Simulate.simulate ~run_key:180L
    [("zero", scalar 0.0); ("true_tau", scalar true_tau)] latent in
  let a = List.assoc "a" latent_trace in
  let key = Prng.Threefry.make_key ~run_key:181L
    ~namespace:Prng.Threefry.ns_data in
  for group = 0 to groups - 1 do
    for observation = 0 to observations - 1 do
      let index = group * observations + observation in
      let xv = (float_of_int observation -. 9.5) /. 5.0 in
      let eta = true_gamma *. xv +. value a group in
      let probability = if eta >= 0.0 then 1.0 /. (1.0 +. exp (-.eta))
        else let e = exp eta in e /. (1.0 +. e) in
      let ctr = Prng.Threefry.make_ctr ~site_id:2 ~component:1
        ~frame_index:index in
      let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
      View.Buf.set x.buf index xv;
      View.Buf.set y.buf index
        (if Prng.Threefry.to_open_unit bits < probability then 1.0 else 0.0)
    done
  done;
  let common_shapes = [("zero", [||]); ("prior_gamma_scale", [||]);
    ("prior_tau_scale", [||]); ("gamma_param", [||]);
    ("x_obs", frame); ("y_obs", frame)] in
  let fixed = [("zero", scalar 0.0); ("prior_gamma_scale", scalar 5.0);
    ("prior_tau_scale", scalar 1.0); ("x_obs", x); ("y_obs", y)] in
  let h0 = glmm_quadrature ~node_count:20 ~groups
    ~tau_slot:(const (scalar 0.0)) ~model ~env_shapes:common_shapes in
  let h0_parameters = [("gamma_param", scalar 0.0)] in
  let h0_initial, h0_value = optimize_marginal ~steps:500 ~learning_rate:0.03
    ~parameters:h0_parameters ~fixed h0.log_marginal in
  let h1_shapes = ("rho_param", [||]) :: common_shapes in
  let tau_slot = prim Exp [var "rho_param"] in
  let h1 = glmm_quadrature ~node_count:20 ~groups ~tau_slot
    ~model ~env_shapes:h1_shapes in
  let h1_parameters = [("gamma_param", scalar 0.0);
    ("rho_param", scalar (log 0.5))] in
  let h1_initial, h1_value = optimize_marginal ~steps:700 ~learning_rate:0.025
    ~parameters:h1_parameters ~fixed h1.log_marginal in
  let gamma = value (List.assoc "gamma_param" h1_parameters) 0
  and tau = exp (value (List.assoc "rho_param" h1_parameters) 0) in
  let at_k node_count =
    let program = glmm_quadrature ~node_count ~groups
      ~tau_slot ~model ~env_shapes:h1_shapes in
    Ast.Eval.eval (h1_parameters @ fixed) program.log_marginal |> fun result ->
      value result 0 in
  let k10 = at_k 10 and k15 = at_k 15 and k20 = at_k 20
  and k30 = at_k 30 in
  Printf.printf "\nLogistic marginal MLE: H0=%.6f H1=%.6f gamma=%.4f tau=%.4f; K10/15/20/30=%.6f/%.6f/%.6f/%.6f\n"
    h0_value h1_value gamma tau k10 k15 k20 k30;
  check bool "H0 optimization improves" true (h0_value > h0_initial);
  check bool "H1 optimization improves" true (h1_value > h1_initial);
  check bool "nested-model likelihood monotonicity" true (h1_value >= h0_value);
  check bool "fixed effect agrees with generating value" true
    (Float.abs (gamma -. true_gamma) < 0.3);
  check bool "random-effect scale agrees with generating value" true
    (Float.abs (tau -. true_tau) < 0.35);
  check bool "MLE agrees with Phase 12 VI estimate" true
    (Float.abs (gamma -. 1.3387) < 0.25 && Float.abs (tau -. 0.7810) < 0.1);
  check bool "K=15 and K=20 agree" true (Float.abs (k20 -. k15) < 1e-3);
  check bool "K=20 and K=30 agree" true (Float.abs (k30 -. k20) < 1e-3)

let () =
  run "Phase 13" [
    "13-0 discrete observations", [
      test_case "frame categorical sampling and density" `Quick
        test_frame_categorical_observation;
      test_case "Condition and Maximize slots" `Quick test_slot_roles;
    ];
    "13-1 special functions", [
      test_case "regularized gamma closed forms" `Quick
        test_regularized_gamma_closed_forms;
      test_case "chi-square one and tail" `Quick test_chi_square_one;
      test_case "boundary mixture" `Quick test_boundary_mixture;
    ];
    "13-2 quadrature", [
      test_case "Gauss-Hermite moments" `Quick test_gauss_hermite_rule;
      test_case "Gaussian closed marginal" `Quick
        test_quadrature_gaussian_closed_form;
    ];
    "13-3 logistic marginal MLE", [
      test_case "H0 and H1 fixed-iteration MLE" `Slow test_logistic_glmm_mle;
    ];
  ]
