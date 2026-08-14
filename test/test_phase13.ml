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
let numel tensor = View.Ndview.numel tensor.View.Tensor.view

let filled shape value =
  let result = View.Tensor.make shape in
  for index = 0 to numel result - 1 do View.Buf.set result.buf index value done;
  result

let copy_tensor source = tensor source.View.Tensor.view.View.Ndview.shape
  (Array.init (numel source) (value source))

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

let test_importance_log_weights () =
  let particles = 5 in
  let model = sample "z" [||]
    (Ast.Normal.normal ~mu:"model_mu" ~sigma:"model_sigma") in
  let guide = sample "z" [||]
    (Ast.Normal.normal ~mu:"guide_mu" ~sigma:"guide_sigma") in
  let env_shapes = [("model_mu", [||]); ("model_sigma", [||]);
    ("guide_mu", [||]); ("guide_sigma", [||])] in
  let program = Transform.Importance.importance ~particles ~slots:[]
    ~model ~guide ~env_shapes in
  check (list (pair string (array int))) "noise keeps particle axis"
    [("%u.z", [|particles|])] program.noise;
  let uniforms = tensor [|particles|] [|0.1; 0.25; 0.5; 0.7; 0.9|] in
  let env = [("model_mu", scalar 0.0); ("model_sigma", scalar 1.0);
    ("guide_mu", scalar 0.5); ("guide_sigma", scalar 1.5);
    ("%u.z", uniforms)] in
  let actual = Ast.Eval.eval env program.log_weights in
  check (array int) "one log weight per particle" [|particles|]
    actual.View.Tensor.view.View.Ndview.shape;
  let log_normal x mu sigma =
    -.0.5 *. log (2.0 *. Float.pi) -. log sigma
    -. 0.5 *. ((x -. mu) /. sigma) ** 2.0 in
  for index = 0 to particles - 1 do
    let u = value uniforms index in
    let z = 0.5 +. 1.5 *. sqrt 2.0 *. Ast.Eval.erfinv_impl (2.0 *. u -. 1.0) in
    let expected = log_normal z 0.0 1.0 -. log_normal z 0.5 1.5 in
    check (float 1e-11) (Printf.sprintf "particle %d log p-log q" index)
      expected (value actual index)
  done;
  Transform.check_no_samples program.log_weights

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
      ~include_target_density:false
      ~preserve_frame:0
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

let test_gaussian_bayes_factor_closed_form () =
  let model = let_ "theta" (sample "theta" [||]
    (Ast.Normal.normal ~mu:"zero" ~sigma:"prior_scale"))
    (sample "y" [||] (Ast.Normal.normal ~mu:"theta" ~sigma:"obs_scale")) in
  let y = 1.3 and prior_scale = 0.8 and obs_scale = 0.6 in
  let env_shapes = [("zero", [||]); ("prior_scale", [||]);
    ("obs_scale", [||]); ("y_obs", [||]); ("theta_zero", [||])] in
  let env = [("zero", scalar 0.0); ("prior_scale", scalar prior_scale);
    ("obs_scale", scalar obs_scale); ("y_obs", scalar y);
    ("theta_zero", scalar 0.0)] in
  let h0 = Transform.build_elbo
    ~slots:[("theta", `Maximize, var "theta_zero");
            ("y", `Condition, var "y_obs")]
    ~model ~guide:(const (scalar 0.0)) ~env_shapes in
  let log_h0 = Ast.Eval.eval env h0.elbo |> fun result -> value result 0 in
  let log_normal x sigma =
    -.0.5 *. log (2.0 *. Float.pi) -. log sigma
    -. 0.5 *. (x /. sigma) ** 2.0 in
  let expected = log_normal y (sqrt (prior_scale *. prior_scale
    +. obs_scale *. obs_scale)) -. log_normal y obs_scale in
  let errors = List.map (fun node_count ->
    let nodes, log_weights = Transform.Quadrature.gauss_hermite_lebesgue
      ~scale:prior_scale node_count in
    let h1 = Transform.Quadrature.quadrature ~site:"theta"
      ~values:(const (tensor [|node_count|] nodes))
      ~log_weights:(Transform.Quadrature.tensor log_weights)
      ~include_target_density:true ~preserve_frame:0
      ~slots:[("y", `Condition, var "y_obs")] ~model ~env_shapes in
    let log_h1 = Ast.Eval.eval env h1.log_marginal
      |> fun result -> value result 0 in
    Float.abs ((log_h1 -. log_h0) -. expected)) [8; 16; 32] in
  match errors with
  | [e8; e16; e32] ->
      Printf.printf "\nGaussian log BF errors K=8/16/32: %.3e %.3e %.3e\n"
        e8 e16 e32;
      check bool "Gaussian BF converges with K" true (e16 < e8 && e32 < e16);
      check bool "Gaussian BF matches closed form" true (e32 < 1e-10)
  | _ -> assert false

let test_quadrature_hoists_global_condition () =
  let groups = 5 and node_count = 5 in
  let model = let_ "theta" (sample "theta" [||]
    (Ast.Normal.normal ~mu:"zero" ~sigma:"one"))
    (sample "a" [|groups|] (Ast.Normal.normal ~mu:"theta" ~sigma:"one")) in
  let nodes, weights = Transform.Quadrature.gauss_hermite node_count in
  let cells = Array.init (node_count * groups) (fun index ->
    0.4 +. sqrt 2.0 *. nodes.(index / groups)) in
  let log_weights = Array.map
    (fun weight -> log weight -. 0.5 *. log Float.pi) weights
    |> Transform.Quadrature.tensor in
  let env_shapes = [("zero", [||]); ("one", [||]); ("theta_value", [||])] in
  let program = Transform.Quadrature.quadrature ~site:"a"
    ~values:(const (tensor [|node_count; groups|] cells)) ~log_weights
    ~include_target_density:false ~preserve_frame:0
    ~slots:[("theta", `Condition, var "theta_value")] ~model ~env_shapes in
  let actual = Ast.Eval.eval [("zero", scalar 0.0); ("one", scalar 1.0);
    ("theta_value", scalar 0.4)] program.log_marginal |> fun result -> value result 0 in
  let expected = -.0.5 *. log (2.0 *. Float.pi) -. 0.5 *. 0.4 *. 0.4 in
  check (float 1e-12) "global prior is counted once, not once per group"
    expected actual

let logistic_glmm_model ?replications groups =
  let parameter_frame = match replications with None -> [||] | Some n -> [|n|] in
  let group_frame = match replications with
    | None -> [|groups|] | Some n -> [|n; groups|] in
  let logits = rank 0 Add
    [rank 0 Mul [var "gamma"; var "x_obs"]; var "a"] in
  let likelihood = rank 0 Add [
    rank 0 Mul [var "y_obs"; prim Logsigmoid [logits]];
    rank 0 Mul
      [rank 0 Sub [const (scalar 1.0); var "y_obs"];
       prim Logsigmoid [rank 0 Neg [logits]]];
  ] in
  let_ "gamma" (sample "gamma" parameter_frame
    (Ast.Normal.normal ~mu:"zero" ~sigma:"prior_gamma_scale"))
    (let_ "tau" (sample "tau" parameter_frame
      (Ast.Half_normal.half_normal ~sigma:"prior_tau_scale"))
      (let_ "a" (sample "a" group_frame
        (Ast.Normal.normal ~mu:"zero" ~sigma:"tau"))
        (score likelihood)))

let glmm_quadrature ~replications ~node_count ~groups ~gamma_role ~tau_role
    ~tau_slot ~model ~env_shapes =
  let nodes, weights = Transform.Quadrature.gauss_hermite node_count in
  let parameter_rank, site_shape, site_cells = match replications with
    | None -> 0, [|node_count; groups|], groups
    | Some n -> 1, [|node_count; n; groups|], n * groups in
  let node_cells = Array.init (node_count * site_cells)
    (fun index -> sqrt 2.0 *. nodes.(index / site_cells)) in
  let tau_cells = prim (Apply_view
    [Vbroadcast (0, node_count); Vbroadcast (parameter_rank + 1, groups)])
    [tau_slot] in
  let values = prim Mul [const (tensor site_shape node_cells); tau_cells] in
  let log_weights = Array.map
    (fun weight -> log weight -. 0.5 *. log Float.pi) weights
    |> Transform.Quadrature.tensor in
  Transform.Quadrature.quadrature ~site:"a" ~values ~log_weights
    ~include_target_density:false
    ~preserve_frame:parameter_rank
    ~slots:[("gamma", gamma_role, var "gamma_param");
            ("tau", tau_role, tau_slot)]
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

type tensor_adam = { m : View.Tensor.t; v : View.Tensor.t }

let optimize_batched ~steps ~learning_rate ?(nonnegative = []) ~parameters
    ~fixed expression =
  let parameter_shapes = List.map (fun (name, parameter) ->
    name, parameter.View.Tensor.view.View.Ndview.shape) parameters in
  let data_shapes = List.map (fun (name, datum) ->
    name, datum.View.Tensor.view.View.Ndview.shape) fixed in
  let gradient = Transform.grad ~param_shapes:parameter_shapes ~data_shapes expression in
  let states = List.map (fun (name, parameter) ->
    let shape = parameter.View.Tensor.view.View.Ndview.shape in
    name, {m = filled shape 0.0; v = filled shape 0.0}) parameters in
  let evaluate () =
    Ast.Eval.eval_grad (parameters @ fixed)
      ~primal_bindings:gradient.primal_bindings ~loss_body:gradient.loss_body
      ~grad_bindings:gradient.grad_bindings ~grad_bodies:gradient.grad_bodies in
  for step = 1 to steps do
    let _, gradients = evaluate () in
    let c1 = 1.0 -. 0.9 ** float_of_int step
    and c2 = 1.0 -. 0.999 ** float_of_int step in
    List.iter (fun (name, parameter) ->
      let state = List.assoc name states and gradient = List.assoc name gradients in
      for index = 0 to numel parameter - 1 do
        let g = value gradient index in
        let m = 0.9 *. value state.m index +. 0.1 *. g
        and v = 0.999 *. value state.v index +. 0.001 *. g *. g in
        View.Buf.set state.m.buf index m;
        View.Buf.set state.v.buf index v;
        let next = value parameter index +. learning_rate *. (m /. c1)
          /. (sqrt (v /. c2) +. 1e-8) in
        View.Buf.set parameter.buf index
          (if List.mem name nonnegative then max 0.0 next else next)
      done) parameters
  done;
  evaluate ()

let generate_null_bootstrap ~replications ~groups ~observations ~gamma =
  let frame = [|replications; groups; observations|] in
  let x = View.Tensor.make frame in
  let weights = View.Tensor.make [|replications; groups; observations; 2|] in
  for replication = 0 to replications - 1 do
    for group = 0 to groups - 1 do
      for observation = 0 to observations - 1 do
        let index = (replication * groups + group) * observations + observation in
        let xv = (float_of_int observation
          -. 0.5 *. float_of_int (observations - 1))
          /. (0.25 *. float_of_int observations) in
        let eta = gamma *. xv in
        let probability = if eta >= 0.0 then 1.0 /. (1.0 +. exp (-.eta))
          else let e = exp eta in e /. (1.0 +. e) in
        View.Buf.set x.buf index xv;
        View.Buf.set weights.buf (2 * index) (1.0 -. probability);
        View.Buf.set weights.buf (2 * index + 1) probability
      done
    done
  done;
  let generator = sample "y" frame (D_categorical (const weights)) in
  let sites = match Ast.Sites.collect_sites generator with
    | [site] -> [{site with id = 16}]
    | _ -> assert false in
  let _, trace, _ = Ast.Simulate.simulate ~sites
    ~namespace:Prng.Threefry.ns_data ~run_key:401L [] generator in
  x, List.assoc "y" trace

let chi_square_ks statistics =
  let sample = Array.copy statistics in
  Array.sort Float.compare sample;
  let n = Array.length sample in
  let distance = ref 0.0 in
  Array.iteri (fun index statistic ->
    let cdf = Transform.Special.chi_square_1_cdf statistic in
    let below = float_of_int index /. float_of_int n
    and through = float_of_int (index + 1) /. float_of_int n in
    distance := max !distance (max (Float.abs (cdf -. below))
      (Float.abs (through -. cdf)))) sample;
  !distance

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
  let h0 = glmm_quadrature ~replications:None ~node_count:20 ~groups
    ~gamma_role:`Maximize ~tau_role:`Maximize
    ~tau_slot:(const (scalar 0.0)) ~model ~env_shapes:common_shapes in
  let h0_parameters = [("gamma_param", scalar 0.0)] in
  let h0_initial, h0_value = optimize_marginal ~steps:500 ~learning_rate:0.03
    ~parameters:h0_parameters ~fixed h0.log_marginal in
  let h1_shapes = ("rho_param", [||]) :: common_shapes in
  let tau_slot = prim Exp [var "rho_param"] in
  let h1 = glmm_quadrature ~replications:None ~node_count:20 ~groups ~tau_slot
    ~gamma_role:`Maximize ~tau_role:`Maximize
    ~model ~env_shapes:h1_shapes in
  let h1_parameters = [("gamma_param", scalar 0.0);
    ("rho_param", scalar (log 0.5))] in
  let h1_initial, h1_value = optimize_marginal ~steps:700 ~learning_rate:0.025
    ~parameters:h1_parameters ~fixed h1.log_marginal in
  let gamma = value (List.assoc "gamma_param" h1_parameters) 0
  and tau = exp (value (List.assoc "rho_param" h1_parameters) 0) in
  let at_k node_count =
    let program = glmm_quadrature ~replications:None ~node_count ~groups
      ~gamma_role:`Maximize ~tau_role:`Maximize
      ~tau_slot ~model ~env_shapes:h1_shapes in
    Ast.Eval.eval (h1_parameters @ fixed) program.log_marginal |> fun result ->
      value result 0 in
  let k10 = at_k 10 and k15 = at_k 15 and k20 = at_k 20
  and k30 = at_k 30 in
  if Sys.getenv_opt "VAE_PROFILE" = Some "1" then begin
    Ast.Eval.reset_stats ();
    Ast.Eval.enable_stats ();
    ignore (at_k 20);
    Ast.Eval.disable_stats ();
    Ast.Eval.report ()
  end;
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

let test_boundary_lrt_bootstrap () =
  let full = Sys.getenv_opt "PHASE13_BOOTSTRAP" = Some "1" in
  let full_replications = match Sys.getenv_opt "PHASE13_BOOTSTRAP_B" with
    | Some value -> int_of_string value | None -> 200 in
  let replications, groups, observations, node_count, h0_steps, h1_steps =
    if full then full_replications, 64, 20, 20, 100, 140
    else 12, 16, 8, 10, 35, 50 in
  let parameter_frame = [|replications|]
  and data_frame = [|replications; groups; observations|] in
  let x, y = generate_null_bootstrap ~replications ~groups ~observations
    ~gamma:1.2 in
  let model = logistic_glmm_model ~replications groups in
  let fixed = [("zero", scalar 0.0); ("prior_gamma_scale", scalar 5.0);
    ("prior_tau_scale", scalar 1.0); ("x_obs", x); ("y_obs", y)] in
  let common_shapes = [("zero", [||]); ("prior_gamma_scale", [||]);
    ("prior_tau_scale", [||]); ("gamma_param", parameter_frame);
    ("x_obs", data_frame); ("y_obs", data_frame)] in
  let zeros = const (filled parameter_frame 0.0) in
  let h0 = glmm_quadrature ~replications:(Some replications) ~node_count ~groups ~tau_slot:zeros
    ~gamma_role:`Maximize ~tau_role:`Maximize
    ~model ~env_shapes:common_shapes in
  let h0_gamma = filled parameter_frame 0.0 in
  let h0_loss, _ = optimize_batched ~steps:h0_steps ~learning_rate:0.05
    ~parameters:[("gamma_param", h0_gamma)] ~fixed h0.log_marginal in
  let probe_tau = const (filled parameter_frame 0.01) in
  let probe = glmm_quadrature ~replications:(Some replications)
    ~gamma_role:`Maximize ~tau_role:`Maximize
    ~node_count ~groups ~tau_slot:probe_tau ~model ~env_shapes:common_shapes in
  let probe_loss = Ast.Eval.eval (("gamma_param", h0_gamma) :: fixed)
    probe.log_marginal in
  let boundary_directions = ref 0 in
  for index = 0 to replications - 1 do
    if value probe_loss index <= value h0_loss index then
      incr boundary_directions
  done;
  let h1_shapes = ("rho_param", parameter_frame) :: common_shapes in
  let tau_slot = prim Exp [var "rho_param"] in
  let h1 = glmm_quadrature ~replications:(Some replications) ~node_count ~groups
    ~gamma_role:`Maximize ~tau_role:`Maximize
    ~tau_slot ~model ~env_shapes:h1_shapes in
  let h1_gamma = copy_tensor h0_gamma
  and h1_rho = filled parameter_frame (log 0.3) in
  let h1_parameters = [("gamma_param", h1_gamma); ("rho_param", h1_rho)] in
  let h1_loss, _ = optimize_batched ~steps:h1_steps ~learning_rate:0.03
    ~parameters:h1_parameters ~fixed h1.log_marginal in
  let boundary_candidates = ref 0 in
  for index = 0 to replications - 1 do
    if value h1_loss index <= value h0_loss index +. 1e-6 then begin
      incr boundary_candidates;
      View.Buf.set h1_loss.buf index (value h0_loss index);
      View.Buf.set h1_gamma.buf index (value h0_gamma index);
      View.Buf.set h1_rho.buf index neg_infinity
    end
  done;
  let statistics = Array.init replications (fun index ->
    max 0.0 (2.0 *. (value h1_loss index -. value h0_loss index))) in
  let monotonic_violations = ref 0 in
  for index = 0 to replications - 1 do
    if value h1_loss index +. 1e-4 < value h0_loss index then
      incr monotonic_violations
  done;
  let boundary_tolerance = 0.03 in
  let atom_count = ref 0 and positive = ref [] in
  Array.iteri (fun index statistic ->
    if exp (value h1_rho index) <= boundary_tolerance then incr atom_count
    else positive := statistic :: !positive) statistics;
  let positives = Array.of_list !positive in
  let atom_mass = float_of_int !atom_count /. float_of_int replications in
  let ks = if Array.length positives = 0 then infinity else chi_square_ks positives in
  let tail_at_one = Array.fold_left (fun count statistic ->
    if statistic > 1.0 then count + 1 else count) 0 statistics
    |> fun count -> float_of_int count /. float_of_int replications in
  let asymptotic_tail_at_one =
    Transform.Special.boundary_variance_component_p_value 1.0 in
  let k_check = if full then 15 else max 5 (node_count - 5) in
  let h0_check = glmm_quadrature ~replications:(Some replications) ~node_count:k_check ~groups
    ~gamma_role:`Maximize ~tau_role:`Maximize
    ~tau_slot:zeros ~model ~env_shapes:common_shapes in
  let h1_check = glmm_quadrature ~replications:(Some replications) ~node_count:k_check ~groups
    ~gamma_role:`Maximize ~tau_role:`Maximize
    ~tau_slot ~model ~env_shapes:h1_shapes in
  let h0_check_values = Ast.Eval.eval (("gamma_param", h0_gamma) :: fixed)
    h0_check.log_marginal
  and h1_check_values = Ast.Eval.eval (h1_parameters @ fixed)
    h1_check.log_marginal in
  let statistic_error = ref 0.0 in
  for index = 0 to replications - 1 do
    let alternate = max 0.0 (2.0 *. (value h1_check_values index
      -. value h0_check_values index)) in
    statistic_error := max !statistic_error
      (Float.abs (statistics.(index) -. alternate))
  done;
  Printf.printf "\nBoundary LRT B=%d G=%d K=%d: atom=%.3f, positive=%d, KS=%.3f, tail(t>1)=%.3f vs %.3f, boundary directions/candidates=%d/%d, monotonic violations=%d, K statistic delta=%.3g\n"
    replications groups node_count atom_mass (Array.length positives) ks
    tail_at_one asymptotic_tail_at_one !boundary_directions !boundary_candidates
    !monotonic_violations !statistic_error;
  check int "all H1 likelihoods dominate H0" 0 !monotonic_violations;
  check bool "all LRT statistics finite" true
    (Array.for_all Float.is_finite statistics);
  if full then begin
    check bool "boundary atom mass near one half" true
      (atom_mass > 0.3 && atom_mass < 0.7);
    check bool "positive component follows chi-square one" true (ks < 0.3);
    check bool "bootstrap and asymptotic tails agree" true
      (Float.abs (tail_at_one -. asymptotic_tail_at_one) < 0.08)
  end

let test_logistic_glmm_bayes_factor () =
  let full = Sys.getenv_opt "PHASE13_BF" = Some "1" in
  let groups, observations, inner_nodes, outer_nodes =
    if full then 64, 20, 15, [24; 32; 40]
    else 16, 8, 10, [8; 12; 16] in
  let true_gamma = 1.2 and true_tau = 0.7 in
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
      let xv = (float_of_int observation
        -. 0.5 *. float_of_int (observations - 1))
        /. (0.25 *. float_of_int observations) in
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
  let evidence ~node_count ~alternative =
    let gamma_nodes, gamma_log_weights =
      Transform.Quadrature.gauss_hermite_lebesgue ~scale:0.3 node_count in
    let gamma_nodes = Array.map (fun node -> node +. 1.1769) gamma_nodes in
    let rho_nodes, tau_log_weights =
      Transform.Quadrature.gauss_hermite_lebesgue ~scale:0.5 node_count in
    let rho_nodes = Array.map (fun node -> node +. log 0.7810) rho_nodes in
    let tau_nodes = Array.map exp rho_nodes in
    let tau_log_weights = Array.mapi (fun index weight ->
      weight +. rho_nodes.(index)) tau_log_weights in
    let points = if alternative then node_count * node_count else node_count in
    let gamma = View.Tensor.make [|points|]
    and tau = View.Tensor.make [|points|]
    and outer_log_weights = View.Tensor.make [|points|] in
    for point = 0 to points - 1 do
      let gi = if alternative then point / node_count else point
      and ti = if alternative then point mod node_count else 0 in
      View.Buf.set gamma.buf point gamma_nodes.(gi);
      View.Buf.set tau.buf point (if alternative then tau_nodes.(ti) else 0.0);
      View.Buf.set outer_log_weights.buf point
        (gamma_log_weights.(gi)
         +. if alternative then tau_log_weights.(ti) else 0.0)
    done;
    let parameter_frame = [|points|]
    and data_frame = [|points; groups; observations|] in
    let x_cells = View.Tensor.broadcast x ~axis:0 ~size:points
    and y_cells = View.Tensor.broadcast y ~axis:0 ~size:points in
    let model = logistic_glmm_model ~replications:points groups in
    let env_shapes = [("zero", [||]); ("prior_gamma_scale", [||]);
      ("prior_tau_scale", [||]); ("gamma_param", parameter_frame);
      ("tau_param", parameter_frame); ("x_obs", data_frame);
      ("y_obs", data_frame)] in
    let program = glmm_quadrature ~replications:(Some points)
      ~node_count:inner_nodes
      ~groups ~gamma_role:`Condition
      ~tau_role:(if alternative then `Condition else `Maximize)
      ~tau_slot:(var "tau_param") ~model ~env_shapes in
    let weighted = prim Add [const outer_log_weights; program.log_marginal] in
    let expression = Transform.Quadrature.logsumexp_axis0 points weighted in
    let env = [("zero", scalar 0.0); ("prior_gamma_scale", scalar 5.0);
      ("prior_tau_scale", scalar 1.0); ("gamma_param", gamma);
      ("tau_param", tau); ("x_obs", x_cells); ("y_obs", y_cells)] in
    let started = Unix.gettimeofday () in
    let result = Ast.Eval.eval env expression |> fun result -> value result 0 in
    result, Unix.gettimeofday () -. started
  in
  let results = List.map (fun node_count ->
    let h0, h0_seconds = evidence ~node_count ~alternative:false
    and h1, h1_seconds = evidence ~node_count ~alternative:true in
    node_count, h1 -. h0, h0_seconds +. h1_seconds) outer_nodes in
  match results with
  | [(k4, bf4, t4); (k6, bf6, t6); (k8, bf8, t8)] ->
      Printf.printf "\nLogistic GLMM log BF K=%d/%d/%d: %.6f %.6f %.6f; seconds %.3f %.3f %.3f\n"
        k4 k6 k8 bf4 bf6 bf8 t4 t6 t8;
      if full then check bool "same GLMM favors random effects" true (bf8 > 0.0);
      check bool "log BF converges with K" true
        (Float.abs (bf8 -. bf6) < Float.abs (bf6 -. bf4));
      if full then check bool "last two log BF values agree" true
        (Float.abs (bf8 -. bf6) < 0.2)
  | _ -> assert false

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
      test_case "hoists global Condition density" `Quick
        test_quadrature_hoists_global_condition;
    ];
    "13-3 logistic marginal MLE", [
      test_case "H0 and H1 fixed-iteration MLE" `Slow test_logistic_glmm_mle;
    ];
    "13-4 boundary LRT", [
      test_case "vectorized null bootstrap" `Slow test_boundary_lrt_bootstrap;
    ];
    "13-5 importance", [
      test_case "preserves particle log weights" `Quick
        test_importance_log_weights;
    ];
    "13-6 Bayes factor", [
      test_case "Gaussian closed form" `Quick
        test_gaussian_bayes_factor_closed_form;
      test_case "same logistic GLMM, Condition slots" `Slow
        test_logistic_glmm_bayes_factor;
    ];
  ]
