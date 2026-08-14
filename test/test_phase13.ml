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
       3.841458820694124)

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
  let program = Transform.build_elbo ~observed:[("y", var "y_obs")]
    ~model ~guide:(const (scalar 0.0)) ~env_shapes:[("y_obs", frame)] in
  let elbo = Ast.Eval.eval [("y_obs", y)] program.elbo in
  check (float 1e-12) "observed categorical enters ELBO"
    (value assessed 0) (value elbo 0)

let test_gauss_hermite_rule () =
  List.iter (fun node_count ->
    let nodes, weights = Transform.Quadrature.gauss_hermite node_count in
    let total = Array.fold_left ( +. ) 0.0 weights in
    let second = ref 0.0 in
    Array.iteri (fun index node ->
      second := !second +. weights.(index) *. node *. node) nodes;
    check (float 2e-14) (Printf.sprintf "K=%d mass" node_count)
      (sqrt Float.pi) total;
    check (float 2e-14) (Printf.sprintf "K=%d second moment" node_count)
      (0.5 *. sqrt Float.pi) !second) [5; 10; 20]

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
      ~observed:[("y", var "y_obs")] ~model ~env_shapes
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

let () =
  run "Phase 13" [
    "13-0 discrete observations", [
      test_case "frame categorical sampling and density" `Quick
        test_frame_categorical_observation;
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
  ]
