(* Phase 12-0..2: HalfNormal and hierarchical site dependencies.
   Every test in this file is an implementation invariant that must pass;
   none measures mean-field approximation quality. *)

open Alcotest
open Ast.Types

let scalar value =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 value;
  t

let tensor shape values =
  let t = View.Tensor.make shape in
  Array.iteri (View.Buf.set t.buf) values;
  t

let value t i = View.Buf.get t.View.Tensor.buf i
let numel t = View.Ndview.numel t.View.Tensor.view

let filled shape x =
  let t = View.Tensor.make shape in
  for i = 0 to numel t - 1 do View.Buf.set t.buf i x done;
  t

let test_half_normal_inverse () =
  let dist = Ast.Half_normal.half_normal ~sigma:"sigma" in
  match dist with
  | D_pushforward { fwd_var; fwd; inv_var; inv; _ } ->
      List.iter
        (fun u ->
          let x = Ast.Eval.eval [("sigma", scalar 1.7); (fwd_var, scalar u)] fwd in
          let actual = Ast.Eval.eval
            [("sigma", scalar 1.7); (inv_var, x)] inv |> fun t -> value t 0 in
          check (float 1e-12) (Printf.sprintf "inv(fwd(%g))" u) u actual)
        [0.01; 0.2; 0.5; 0.9; 0.99]
  | _ -> fail "HalfNormal must be a pushforward"

let half_normal_density sigma x =
  let program = sample "tau" [||] (Ast.Half_normal.half_normal ~sigma:"sigma") in
  let _, ld = Ast.Assess.assess [("sigma", scalar sigma)] program
    [("tau", scalar x)] in
  exp (value ld 0)

let test_half_normal_density () =
  let sigma = 1.7 in
  List.iter
    (fun x ->
      let expected = sqrt (2.0 /. Float.pi) /. sigma
        *. exp (-.x *. x /. (2.0 *. sigma *. sigma)) in
      check (float 1e-12) (Printf.sprintf "density(%g)" x) expected
        (half_normal_density sigma x))
    [0.1; 0.7; 2.0; 5.0]

let test_half_normal_normalizes () =
  let sigma = 1.3 in
  let steps = 20_000 in
  let upper = 9.0 *. sigma in
  let width = upper /. float_of_int steps in
  let total = ref 0.0 in
  for i = 0 to steps - 1 do
    let x = (float_of_int i +. 0.5) *. width in
    total := !total +. half_normal_density sigma x *. width
  done;
  check (float 1e-8) "integral" 1.0 !total

let test_half_normal_sigma_fd () =
  let program = sample "tau" [||]
    (Ast.Half_normal.half_normal ~sigma:"sigma") in
  let density = Transform.Assess_expr.assess_expr
    ~env_shapes:[("sigma", [||])] program [("tau", const (scalar 0.8))] in
  let gradient = Transform.grad ~param_shapes:[("sigma", [||])] density in
  let sigma = 1.4 and epsilon = 1e-5 in
  let eval at = value (Ast.Eval.eval [("sigma", scalar at)] gradient.loss) 0 in
  let fd = (eval (sigma +. epsilon) -. eval (sigma -. epsilon))
    /. (2.0 *. epsilon) in
  let ad = Ast.Eval.eval [("sigma", scalar sigma)]
    (List.assoc "sigma" gradient.grads) |> fun t -> value t 0 in
  check (float 1e-7) "sigma FD" fd ad

let test_log_normal_density () =
  let mu = 0.3 and sigma = 0.7 and x = 1.4 in
  let dist = Ast.Log_normal.log_normal ~mu:"mu" ~sigma:"sigma" in
  let actual = Ast.Assess.log_density dist (scalar x)
    [("mu", scalar mu); ("sigma", scalar sigma)] in
  let z = (log x -. mu) /. sigma in
  let expected = -.log x -. log sigma -. 0.5 *. log (2.0 *. Float.pi)
    -. 0.5 *. z *. z in
  check (float 1e-12) "nested pushforward density" expected actual;
  check bool "positive declared support" true
    (Ast.Sites.dist_support dist = S_positive)

let test_support_runtime () =
  let loc = {file = "support"; line = 12; col = 3} in
  let program = Sample (loc, "tau", [||],
    Ast.Half_normal.half_normal ~sigma:"sigma") in
  check_raises "assess rejects value outside declared support"
    (Ast.Assess.Support_error
       (loc, "value outside support at site 'tau'"))
    (fun () -> ignore (Ast.Assess.assess [("sigma", scalar 1.0)] program
      [("tau", scalar (-0.5))]));
  let symbolic = Transform.Assess_expr.assess_expr
    ~env_shapes:[("sigma", [||])] program [("tau", const (scalar (-0.5)))] in
  check_raises "assess_expr rejects value outside declared support"
    (Ast.Eval.Eval_error (loc, "value outside declared support"))
    (fun () -> ignore (Ast.Eval.eval [("sigma", scalar 1.0)] symbolic));
  let normal = Sample (loc, "x", [||],
    Ast.Normal.normal ~mu:"mu" ~sigma:"sigma") in
  let env = [("mu", scalar 0.0); ("sigma", scalar 1.0)] in
  let _, tail_ld = Ast.Assess.assess env normal [("x", scalar 12.0)] in
  let tail_expr = Transform.Assess_expr.assess_expr
    ~env_shapes:[("mu", [||]); ("sigma", [||])] normal
    [("x", const (scalar 12.0))] |> Ast.Eval.eval env in
  check bool "far Normal tail remains finite (value)" true
    (Float.is_finite (value tail_ld 0));
  check bool "far Normal tail remains finite (expr)" true
    (Float.is_finite (value tail_expr 0))

let test_symbolic_declared_support () =
  let loc = {file = "support"; line = 20; col = 1} in
  let uniform = Sample (loc, "u", [||], D_uniform) in
  List.iter (fun endpoint ->
    check_raises "assess rejects Uniform endpoint"
      (Ast.Assess.Support_error
         (loc, "value outside support at site 'u'"))
      (fun () -> ignore (Ast.Assess.assess [] uniform
        [("u", scalar endpoint)]));
    let symbolic = Transform.Assess_expr.assess_expr ~env_shapes:[] uniform
      [("u", const (scalar endpoint))] in
    check_raises "assess_expr rejects Uniform endpoint"
      (Ast.Eval.Eval_error (loc, "value outside declared support"))
    (fun () -> ignore (Ast.Eval.eval [] symbolic)))
    [0.0; 1.0];
  let guarded = Prim (loc, Log_support_density S_positive,
    [const (scalar (-1.0))]) in
  check_raises "JVP preserves support location"
    (Ast.Eval.Eval_error (loc, "value outside declared support"))
    (fun () -> ignore (Ast.Jvp.jvp_eval [] guarded))

let test_support_declarations () =
  let distributions =
    [
      (Ast.Normal.normal ~mu:"mu" ~sigma:"sigma",
       [("mu", scalar 0.4); ("sigma", scalar 1.3)]);
      (Ast.Half_normal.half_normal ~sigma:"sigma",
       [("sigma", scalar 1.3)]);
    ]
  in
  List.iteri
    (fun dist_index (dist, env) ->
      let support = Ast.Sites.dist_support dist in
      let program = sample "x" [||] dist in
      for draw = 0 to 255 do
        let _, trace, _ = Ast.Simulate.simulate
          ~run_key:(Int64.of_int draw) env program in
        check bool (Printf.sprintf "dist %d draw %d in declared support"
          dist_index draw) true
          (Ast.Sites.support_contains support (value (List.assoc "x" trace) 0))
      done)
    distributions

let test_support_partial_order () =
  let supports =
    [S_real; S_positive; S_unit_interval; S_finite 2;
     S_product (S_positive, S_real)] in
  List.iter (fun a ->
    check bool "support reflexive" true (Ast.Sites.support_subset a a)) supports;
  List.iter (fun a -> List.iter (fun b -> List.iter (fun c ->
    if Ast.Sites.support_subset a b && Ast.Sites.support_subset b c then
      check bool "support transitive" true (Ast.Sites.support_subset a c))
    supports) supports) supports;
  check bool "finite lower bound" false
    (Ast.Sites.support_contains (S_finite 2) (-1.0));
  check bool "finite upper bound" false
    (Ast.Sites.support_contains (S_finite 2) 2.0);
  check_raises "product containment is not implemented"
    (Failure "support_contains: D_product not supported (Phase 13)")
    (fun () -> ignore (Ast.Sites.support_contains
      (S_product (S_real, S_real)) 0.0))

let hierarchical_program group_count =
  let model =
    let_ "mu"
      (sample "mu" [||]
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_mu_scale"))
      (let_ "tau"
         (sample "tau" [||]
            (Ast.Half_normal.half_normal ~sigma:"prior_tau_scale"))
         (sample "a" [|group_count|]
            (Ast.Normal.normal ~mu:"mu" ~sigma:"tau")))
  in
  let guide =
    let_ "mu"
      (sample "mu" [||]
         (Ast.Normal.normal ~mu:"q_mu" ~sigma:"q_mu_scale"))
      (let_ "tau"
         (sample "tau" [||]
            (Ast.Half_normal.half_normal ~sigma:"q_tau_scale"))
         (sample "a" [|group_count|]
            (Ast.Normal.normal ~mu:"q_a" ~sigma:"q_a_scale")))
  in
  model, guide

let hierarchical_shapes group_count =
  [
    ("prior_mu", [||]); ("prior_mu_scale", [||]);
    ("prior_tau_scale", [||]); ("q_mu", [||]);
    ("q_mu_scale", [||]); ("q_tau_scale", [||]);
    ("q_a", [|group_count|]); ("q_a_scale", [|group_count|]);
  ]

let hierarchical_env () =
  [
    ("prior_mu", scalar 0.0); ("prior_mu_scale", scalar 5.0);
    ("prior_tau_scale", scalar 1.0); ("q_mu", scalar 0.3);
    ("q_mu_scale", scalar 0.8); ("q_tau_scale", scalar 1.2);
    ("q_a", tensor [|3|] [|0.2; -0.4; 0.7|]);
    ("q_a_scale", tensor [|3|] [|0.9; 1.1; 0.75|]);
  ]

let test_hierarchical_assess_expr () =
  let model, _ = hierarchical_program 3 in
  let shapes = hierarchical_shapes 3 in
  let env = hierarchical_env () in
  let model = Transform.Expand_rank.expand ~senv:shapes model in
  let _, trace, _ = Ast.Simulate.simulate ~run_key:71L env model in
  let _, assessed = Ast.Assess.assess env model trace in
  let slots = List.map (fun (name, sample) -> name, const sample) trace in
  let symbolic = Transform.Assess_expr.assess_expr
    ~env_shapes:shapes model slots |> Ast.Eval.eval env in
  check (float 1e-12) "hierarchical assess" (value assessed 0)
    (value symbolic 0)

let test_hierarchical_coupling () =
  let model, _ = hierarchical_program 3 in
  let shapes = hierarchical_shapes 3 in
  let env = hierarchical_env () in
  let sites = Ast.Sites.collect_sites model in
  let expanded = Transform.Expand_rank.expand ~senv:shapes model in
  let _, trace, _ = Ast.Simulate.simulate ~sites ~run_key:83L env expanded in
  let reparammed = Transform.Reparam.reparam ~sites model in
  let bindings, _ = Transform.Reparam.elim_samples ~sites reparammed in
  let noise = Ast.Sites.draw_noise ~run_key:83L sites in
  List.iter
    (fun site ->
      let expression = Transform.Forward.wrap_bindings bindings
        (var (Ast.Sites.trace_name site))
        |> Transform.Expand_rank.expand ~senv:(shapes @ List.map
             (fun (name, sample) -> name, sample.View.Tensor.view.View.Ndview.shape)
             noise) in
      let actual = Ast.Eval.eval (noise @ env) expression in
      let expected = List.assoc site.Ast.Sites.name trace in
      check int (site.name ^ " size") (numel expected) (numel actual);
      for i = 0 to numel expected - 1 do
        check bool (Printf.sprintf "%s[%d] bit coupling" site.name i)
          true (value expected i = value actual i)
      done)
    sites

let test_hierarchical_all_parameter_fd () =
  let model, guide = hierarchical_program 3 in
  let shapes = hierarchical_shapes 3 in
  let param_shapes =
    [
      ("q_mu", [||]); ("q_mu_scale", [||]); ("q_tau_scale", [||]);
      ("q_a", [|3|]); ("q_a_scale", [|3|]);
    ]
  in
  let data_shapes = List.filter
    (fun (name, _) -> not (List.mem_assoc name param_shapes)) shapes in
  let program = Transform.build_elbo ~observed:[] ~model ~guide
    ~env_shapes:shapes in
  let gradient = Transform.grad ~param_shapes
    ~data_shapes:(program.noise @ data_shapes) program.elbo in
  let params = List.filter
    (fun (name, _) -> List.mem_assoc name param_shapes) (hierarchical_env ()) in
  let fixed = Transform.noise_env program ~run_key:97L
    @ List.filter (fun (name, _) -> List.mem_assoc name data_shapes)
        (hierarchical_env ()) in
  let base_env = params @ fixed in
  let gradients = List.map (fun (name, expression) ->
    name, Ast.Eval.eval base_env expression) gradient.grads in
  let epsilon = 1e-5 in
  let eval ps = value (Ast.Eval.eval (ps @ fixed) gradient.loss) 0 in
  List.iter
    (fun (name, parameter) ->
      let ad = List.assoc name gradients in
      for i = 0 to numel parameter - 1 do
        let perturb delta = List.map (fun (n, t) ->
          if n <> name then n, t else
          let copy = tensor t.View.Tensor.view.View.Ndview.shape
            (Array.init (numel t) (value t)) in
          View.Buf.set copy.buf i (value copy i +. delta);
          n, copy) params in
        let fd = (eval (perturb epsilon) -. eval (perturb (-.epsilon)))
          /. (2.0 *. epsilon) in
        let scale = max 1.0 (max (Float.abs fd) (Float.abs (value ad i))) in
        check bool (Printf.sprintf "%s[%d] FD" name i) true
          (Float.abs (fd -. value ad i) /. scale < 1e-6)
      done)
    params

let test_support_check () =
  let model = sample "tau" [||]
    (Ast.Half_normal.half_normal ~sigma:"model_scale") in
  let bad_guide = sample "tau" [||]
    (Ast.Normal.normal ~mu:"guide_mu" ~sigma:"guide_scale") in
  check_raises "Normal guide is not contained in HalfNormal model"
    (Transform.Reparam.Support_mismatch
       (dummy_loc, "guide support is not contained in model support at site 'tau'"))
    (fun () -> ignore (Transform.build_elbo ~observed:[] ~model ~guide:bad_guide
      ~env_shapes:[
        ("model_scale", [||]); ("guide_mu", [||]); ("guide_scale", [||]);
      ]));
  let broad_model = sample "tau" [||]
    (Ast.Normal.normal ~mu:"model_mu" ~sigma:"model_scale") in
  let narrow_guide = sample "tau" [||]
    (Ast.Half_normal.half_normal ~sigma:"guide_scale") in
  ignore (Transform.build_elbo ~observed:[] ~model:broad_model ~guide:narrow_guide
    ~env_shapes:[
      ("model_mu", [||]); ("model_scale", [||]); ("guide_scale", [||]);
    ])

let observed_program group_count =
  let model =
    let_ "z"
      (sample "z" [|group_count|]
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_scale"))
      (sample "y" [|group_count|]
         (Ast.Normal.normal ~mu:"z" ~sigma:"obs_scale"))
  in
  let guide = sample "z" [|group_count|]
    (Ast.Normal.normal ~mu:"q_mu" ~sigma:"q_scale") in
  model, guide

let observed_vi_program frame =
  let model =
    let_ "z"
      (sample "z" frame
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_scale"))
      (sample "y" frame (Ast.Normal.normal ~mu:"z" ~sigma:"obs_scale"))
  in
  let guide =
    let_ "q_scale" (prim Exp [var "q_rho"])
      (sample "z" frame (Ast.Normal.normal ~mu:"q_mu" ~sigma:"q_scale"))
  in
  model, guide

let gaussian_log_density ~x ~mu ~sigma =
  let d = (x -. mu) /. sigma in
  -.0.5 *. log (2.0 *. Float.pi) -. log sigma -. 0.5 *. d *. d

let test_observed_optimal_elbo () =
  let group_count = 5 and obs_scale = 0.6 in
  let frame = [|group_count|] in
  let model, guide = observed_vi_program frame in
  let y_values = [|-1.2; -0.3; 0.1; 0.8; 1.7|] in
  let posterior_variance = obs_scale *. obs_scale /. (1.0 +. obs_scale *. obs_scale) in
  let posterior_scale = sqrt posterior_variance in
  let posterior_means = Array.map (fun y -> y /. (1.0 +. obs_scale *. obs_scale)) y_values in
  let shapes = [
    ("prior_mu", [||]); ("prior_scale", [||]); ("obs_scale", [||]);
    ("q_mu", frame); ("q_rho", frame); ("y_obs", frame);
  ] in
  let program = Transform.build_elbo ~observed:[("y", var "y_obs")]
    ~model ~guide ~env_shapes:shapes in
  let env = Transform.noise_env program ~run_key:140L @ [
    ("prior_mu", scalar 0.0); ("prior_scale", scalar 1.0);
    ("obs_scale", scalar obs_scale); ("q_mu", tensor frame posterior_means);
    ("q_rho", filled frame (log posterior_scale));
    ("y_obs", tensor frame y_values);
  ] in
  let actual = Ast.Eval.eval env program.elbo |> fun t -> value t 0 in
  let marginal_scale = sqrt (1.0 +. obs_scale *. obs_scale) in
  let expected = Array.fold_left (fun total y ->
    total +. gaussian_log_density ~x:y ~mu:0.0 ~sigma:marginal_scale)
    0.0 y_values in
  check (float 1e-10) "observed optimal ELBO" expected actual

let rank_histogram draws truth =
  let repetitions = numel truth in
  let histogram = Array.make (List.length draws + 1) 0 in
  for i = 0 to repetitions - 1 do
    let rank = List.fold_left
      (fun rank draw -> if value draw i < value truth i then rank + 1 else rank)
      0 draws in
    histogram.(rank) <- histogram.(rank) + 1
  done;
  histogram

let check_uniform_ranks label histogram =
  let total = Array.fold_left ( + ) 0 histogram in
  let expected = float_of_int total /. float_of_int (Array.length histogram) in
  let chi_square = Array.fold_left (fun sum count ->
    let d = float_of_int count -. expected in sum +. d *. d /. expected)
    0.0 histogram in
  (* Fixed Threefry streams make this deterministic.  The cutoff is above the
     99.9% chi-square quantile for 15 degrees of freedom. *)
  check bool (Printf.sprintf "%s ranks (chi-square %.3f)" label chi_square)
    true (chi_square < 38.0)

let rank_chi_square histogram =
  let total = Array.fold_left ( + ) 0 histogram in
  let expected = float_of_int total /. float_of_int (Array.length histogram) in
  Array.fold_left (fun sum count ->
    let d = float_of_int count -. expected in sum +. d *. d /. expected)
    0.0 histogram

let posterior_parameters obs_scale y =
  let denominator = 1.0 +. obs_scale *. obs_scale in
  Array.map (fun x -> x /. denominator) y,
  obs_scale /. sqrt denominator

let extract_trace name trace = List.assoc name trace

let test_sbc_closed_form () =
  let repetitions = 2048 and posterior_draws = 15 and obs_scale = 0.7 in
  let frame = [|repetitions|] in
  let model, guide = observed_vi_program frame in
  let shapes = [
    ("prior_mu", [||]); ("prior_scale", [||]); ("obs_scale", [||]);
    ("q_mu", frame); ("q_rho", frame);
  ] in
  let fixed = [("prior_mu", scalar 0.0); ("prior_scale", scalar 1.0);
               ("obs_scale", scalar obs_scale)] in
  let model = Transform.Expand_rank.expand ~senv:shapes model in
  let _, generated, _ = Ast.Simulate.simulate ~run_key:150L fixed model in
  let z_true = extract_trace "z" generated and y = extract_trace "y" generated in
  let y_values = Array.init repetitions (value y) in
  let q_mu, q_scale = posterior_parameters obs_scale y_values in
  let guide_env = ("q_mu", tensor frame q_mu)
    :: ("q_rho", filled frame (log q_scale)) :: fixed in
  let guide = Transform.Expand_rank.expand ~senv:shapes guide in
  let draws = List.init posterior_draws (fun draw ->
    let _, trace, _ = Ast.Simulate.simulate
      ~run_key:(Int64.of_int (10_000 + draw)) guide_env guide in
    extract_trace "z" trace) in
  check_uniform_ranks "closed posterior z" (rank_histogram draws z_true);
  let joint z i =
    gaussian_log_density ~x:(value z i) ~mu:0.0 ~sigma:1.0
    +. gaussian_log_density ~x:(value y i) ~mu:(value z i) ~sigma:obs_scale in
  let true_joint = tensor frame (Array.init repetitions (joint z_true)) in
  let draw_joints = List.map (fun z ->
    tensor frame (Array.init repetitions (joint z))) draws in
  check_uniform_ranks "closed posterior log joint"
    (rank_histogram draw_joints true_joint)

type tensor_adam = { m : View.Tensor.t; v : View.Tensor.t }

let adam_tensor_ascent ~step ~learning_rate state parameter gradient =
  let beta1 = 0.9 and beta2 = 0.999 and epsilon = 1e-8 in
  let c1 = 1.0 -. beta1 ** float_of_int step in
  let c2 = 1.0 -. beta2 ** float_of_int step in
  for i = 0 to numel parameter - 1 do
    let g = value gradient i in
    let m = beta1 *. value state.m i +. (1.0 -. beta1) *. g in
    let v = beta2 *. value state.v i +. (1.0 -. beta2) *. g *. g in
    View.Buf.set state.m.buf i m;
    View.Buf.set state.v.buf i v;
    View.Buf.set parameter.buf i
      (value parameter i +. learning_rate *. (m /. c1)
       /. (sqrt (v /. c2) +. epsilon))
  done

let test_sbc_language_vi () =
  let repetitions = 512 and groups = 3 and posterior_draws = 15 in
  let obs_scale = 0.7 and frame = [|repetitions; groups|] in
  let model, guide = observed_vi_program frame in
  let shapes = [
    ("prior_mu", [||]); ("prior_scale", [||]); ("obs_scale", [||]);
    ("q_mu", frame); ("q_rho", frame); ("y_obs", frame);
  ] in
  let fixed = [("prior_mu", scalar 0.0); ("prior_scale", scalar 1.0);
               ("obs_scale", scalar obs_scale)] in
  let expanded_model = Transform.Expand_rank.expand ~senv:shapes model in
  let _, generated, _ = Ast.Simulate.simulate ~run_key:160L fixed expanded_model in
  let z_true = extract_trace "z" generated and y = extract_trace "y" generated in
  let program = Transform.build_elbo ~observed:[("y", var "y_obs")]
    ~model ~guide ~env_shapes:shapes in
  let gp = Transform.grad ~param_shapes:[("q_mu", frame); ("q_rho", frame)]
    ~data_shapes:(program.noise @ [("prior_mu", [||]); ("prior_scale", [||]);
      ("obs_scale", [||]); ("y_obs", frame)]) program.elbo in
  let q_mu = filled frame 0.0 and q_rho = filled frame 0.0 in
  let mu_adam = {m = filled frame 0.0; v = filled frame 0.0}
  and rho_adam = {m = filled frame 0.0; v = filled frame 0.0} in
  for step = 1 to 1500 do
    let env = Transform.noise_env program ~run_key:(Int64.of_int (20_000 + step))
      @ [("q_mu", q_mu); ("q_rho", q_rho); ("y_obs", y)] @ fixed in
    let _, gradients = Ast.Eval.eval_grad env
      ~primal_bindings:gp.primal_bindings ~loss_body:gp.loss_body
      ~grad_bindings:gp.grad_bindings ~grad_bodies:gp.grad_bodies in
    let learning_rate = if step <= 500 then 0.01 else 0.002 in
    adam_tensor_ascent ~step ~learning_rate mu_adam q_mu
      (List.assoc "q_mu" gradients);
    adam_tensor_ascent ~step ~learning_rate rho_adam q_rho
      (List.assoc "q_rho" gradients)
  done;
  let exact_mu, exact_scale = posterior_parameters obs_scale
    (Array.init (numel y) (value y)) in
  let mean_mu_error = ref 0.0 and mean_scale_error = ref 0.0 in
  for i = 0 to numel q_mu - 1 do
    mean_mu_error := !mean_mu_error +. Float.abs (value q_mu i -. exact_mu.(i));
    mean_scale_error := !mean_scale_error
      +. Float.abs (exp (value q_rho i) -. exact_scale)
  done;
  mean_mu_error := !mean_mu_error /. float_of_int (numel q_mu);
  mean_scale_error := !mean_scale_error /. float_of_int (numel q_rho);
  Printf.printf "\nSBC 1b VI mean errors: mu=%.6f scale=%.6f\n"
    !mean_mu_error !mean_scale_error;
  check bool "VI posterior mean converged" true (!mean_mu_error < 0.03);
  check bool "VI posterior scale converged" true (!mean_scale_error < 0.03);
  let expanded_guide = Transform.Expand_rank.expand ~senv:shapes guide in
  let posterior_env = [("q_mu", q_mu); ("q_rho", q_rho)] @ fixed in
  let draws = List.init posterior_draws (fun draw ->
    let _, trace, _ = Ast.Simulate.simulate
      ~run_key:(Int64.of_int (30_000 + draw)) posterior_env expanded_guide in
    extract_trace "z" trace) in
  for group = 0 to groups - 1 do
    let select t = tensor [|repetitions|]
      (Array.init repetitions (fun repetition ->
        value t (repetition * groups + group))) in
    check_uniform_ranks (Printf.sprintf "VI z[%d]" group)
      (rank_histogram (List.map select draws) (select z_true))
  done;
  let joint_per_repetition z = tensor [|repetitions|]
    (Array.init repetitions (fun repetition ->
      let total = ref 0.0 in
      for group = 0 to groups - 1 do
        let i = repetition * groups + group in
        total := !total
          +. gaussian_log_density ~x:(value z i) ~mu:0.0 ~sigma:1.0
          +. gaussian_log_density ~x:(value y i) ~mu:(value z i)
               ~sigma:obs_scale
      done;
      !total)) in
  check_uniform_ranks "VI log joint"
    (rank_histogram (List.map joint_per_repetition draws)
       (joint_per_repetition z_true))

let ragged_objective () =
  let half = const (scalar 0.5) in
  let log_two_pi = const (scalar (log (2.0 *. Float.pi))) in
  let sigma = prim Exp [var "log_sigma"] in
  (* Sanitize padding before density evaluation, then mask its contribution.
     The inner mask keeps invalid residuals out of reverse-mode residuals. *)
  let safe_y = rank 0 Mask [var "y"; var "mask"] in
  let standardized = rank 0 Div [rank 0 Sub [safe_y; var "mu"]; sigma] in
  let log_density = rank 0 Neg [rank 0 Add [
    rank 0 Add [rank 0 Mul [half; log_two_pi]; prim Log [sigma]];
    rank 0 Mul [half; rank 0 Mul [standardized; standardized]];
  ]] in
  let cells = rank 0 Mask [log_density; var "mask"] in
  prim (Sum_axis 0) [prim (Sum_axis 0) [cells]]

let test_ragged_mask_invariance () =
  let frame = [|2; 4|] in
  let objective = ragged_objective () in
  let gp = Transform.grad
    ~param_shapes:[("mu", [||]); ("log_sigma", [||])]
    ~data_shapes:[("y", frame); ("mask", frame)] objective in
  let mask = tensor frame [|1.0; 1.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0|] in
  let y_a = tensor frame [|0.2; -0.4; 7.0; 8.0; 1.1; 9.0; 10.0; 11.0|] in
  let y_b = tensor frame
    [|0.2; -0.4; Float.nan; infinity; 1.1; neg_infinity; -1e300; 1e300|] in
  let eval y =
    let env = [("mu", scalar 0.3); ("log_sigma", scalar (log 1.2));
               ("y", y); ("mask", mask)] in
    let loss, grads = Ast.Eval.eval_grad env
      ~primal_bindings:gp.primal_bindings ~loss_body:gp.loss_body
      ~grad_bindings:gp.grad_bindings ~grad_bodies:gp.grad_bodies in
    value loss 0, List.map (fun (name, gradient) -> name, value gradient 0) grads
  in
  let loss_a, grads_a = eval y_a and loss_b, grads_b = eval y_b in
  check bool "masked density bit invariant" true (loss_a = loss_b);
  List.iter (fun (name, expected) ->
    check bool (name ^ " gradient bit invariant") true
      (expected = List.assoc name grads_b)) grads_a

let hierarchical_observed_program frame_n frame_ng =
  let model =
    let_ "mu" (sample "mu" frame_n
      (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_mu_scale"))
      (let_ "tau" (sample "tau" frame_n
        (Ast.Half_normal.half_normal ~sigma:"prior_tau_scale"))
        (let_ "a" (sample "a" frame_ng
          (Ast.Normal.normal ~mu:"mu" ~sigma:"tau"))
          (sample "y" frame_ng
            (Ast.Normal.normal ~mu:"a" ~sigma:"obs_scale"))))
  in
  let guide =
    let_ "q_mu_scale" (prim Exp [var "q_mu_rho"])
      (let_ "mu" (sample "mu" frame_n
        (Ast.Normal.normal ~mu:"q_mu_loc" ~sigma:"q_mu_scale"))
        (let_ "q_tau_scale" (prim Exp [var "q_tau_rho"])
          (let_ "tau" (sample "tau" frame_n
            (Ast.Log_normal.log_normal ~mu:"q_tau_loc" ~sigma:"q_tau_scale"))
            (let_ "q_a_scale" (prim Exp [var "q_a_rho"])
              (sample "a" frame_ng
                (Ast.Normal.normal ~mu:"q_a_loc" ~sigma:"q_a_scale"))))))
  in
  model, guide

let test_leading_frame_hierarchical_coupling () =
  let frame_n = [|2|] and frame_ng = [|2; 3|] in
  let model, _ = hierarchical_observed_program frame_n frame_ng in
  let shapes = [("prior_mu", [||]); ("prior_mu_scale", [||]);
    ("prior_tau_scale", [||]); ("obs_scale", [||])] in
  let env = [("prior_mu", scalar 0.0); ("prior_mu_scale", scalar 2.0);
    ("prior_tau_scale", scalar 1.0); ("obs_scale", scalar 0.5)] in
  let sites = Ast.Sites.collect_sites model in
  let expanded = Transform.Expand_rank.expand ~senv:shapes model in
  let _, trace, _ = Ast.Simulate.simulate ~sites ~run_key:169L env expanded in
  let bindings, _ = Transform.Reparam.reparam ~sites model
    |> Transform.Reparam.elim_samples ~sites in
  let noise = Ast.Sites.draw_noise ~run_key:169L sites in
  List.iter (fun site ->
    let expression = Transform.Forward.wrap_bindings bindings
      (var (Ast.Sites.trace_name site))
      |> Transform.Expand_rank.expand ~senv:(shapes @ List.map
        (fun (name, sample) ->
          name, sample.View.Tensor.view.View.Ndview.shape) noise) in
    let actual = Ast.Eval.eval (noise @ env) expression
    and expected = List.assoc site.Ast.Sites.name trace in
    for i = 0 to numel expected - 1 do
      check bool (Printf.sprintf "%s[%d] leading-frame coupling" site.name i)
        true (value actual i = value expected i)
    done) sites

let test_batch_fwd_rejects_nonleading_reference () =
  let expression = sample "z" [|2; 3|]
    (Ast.Normal.normal ~mu:"bad_mu" ~sigma:"sigma")
    |> Transform.Expand_rank.expand
         ~senv:[("bad_mu", [|3|]); ("sigma", [||])] in
  check_raises "referenced env shape must lead-agree"
    (Failure "batch fwd: variable 'bad_mu' does not lead-agree with frame")
    (fun () -> ignore (Ast.Simulate.simulate ~run_key:168L
      [("bad_mu", filled [|3|] 0.0); ("sigma", scalar 1.0);
       ("unrelated", filled [|2; 3; 4|] 0.0)] expression));
  let good = sample "z" [|2; 3|]
    (Ast.Normal.normal ~mu:"mu" ~sigma:"sigma")
    |> Transform.Expand_rank.expand ~senv:[("mu", [||]); ("sigma", [||])] in
  ignore (Ast.Simulate.simulate ~run_key:168L
    [("mu", scalar 0.0); ("sigma", scalar 1.0);
     ("unrelated", filled [|3|] 0.0)] good)

let test_hierarchical_vi_sbc_diagnostic () =
  let repetitions = 256 and groups = 6 and posterior_draws = 15 in
  let frame_n = [|repetitions|] and frame_ng = [|repetitions; groups|] in
  let model, guide = hierarchical_observed_program frame_n frame_ng in
  let shapes = [
    ("prior_mu", [||]); ("prior_mu_scale", [||]);
    ("prior_tau_scale", [||]); ("obs_scale", [||]);
    ("q_mu_loc", frame_n); ("q_mu_rho", frame_n);
    ("q_tau_loc", frame_n); ("q_tau_rho", frame_n);
    ("q_a_loc", frame_ng); ("q_a_rho", frame_ng); ("y_obs", frame_ng);
  ] in
  let fixed = [("prior_mu", scalar 0.0); ("prior_mu_scale", scalar 2.0);
               ("prior_tau_scale", scalar 1.0); ("obs_scale", scalar 0.5)] in
  let expanded_model = Transform.Expand_rank.expand ~senv:shapes model in
  let _, generated, _ = Ast.Simulate.simulate ~run_key:170L fixed expanded_model in
  let mu_true = extract_trace "mu" generated
  and tau_true = extract_trace "tau" generated
  and a_true = extract_trace "a" generated
  and y = extract_trace "y" generated in
  let program = Transform.build_elbo ~observed:[("y", var "y_obs")]
    ~model ~guide ~env_shapes:shapes in
  let param_shapes = [("q_mu_loc", frame_n); ("q_mu_rho", frame_n);
    ("q_tau_loc", frame_n); ("q_tau_rho", frame_n);
    ("q_a_loc", frame_ng); ("q_a_rho", frame_ng)] in
  let gp = Transform.grad ~param_shapes
    ~data_shapes:(program.noise @ [("prior_mu", [||]);
      ("prior_mu_scale", [||]); ("prior_tau_scale", [||]);
      ("obs_scale", [||]); ("y_obs", frame_ng)]) program.elbo in
  let params = [
    ("q_mu_loc", filled frame_n 0.0); ("q_mu_rho", filled frame_n (log 0.8));
    ("q_tau_loc", filled frame_n (log 0.8));
    ("q_tau_rho", filled frame_n (log 0.35));
    ("q_a_loc", tensor frame_ng (Array.init (numel y) (value y)));
    ("q_a_rho", filled frame_ng (log 0.6));
  ] in
  let states = List.map (fun (name, parameter) ->
    name, {m = filled parameter.View.Tensor.view.View.Ndview.shape 0.0;
           v = filled parameter.View.Tensor.view.View.Ndview.shape 0.0}) params in
  let eval step =
    let env = Transform.noise_env program ~run_key:(Int64.of_int step)
      @ params @ [("y_obs", y)] @ fixed in
    Ast.Eval.eval_grad env ~primal_bindings:gp.primal_bindings
      ~loss_body:gp.loss_body ~grad_bindings:gp.grad_bindings
      ~grad_bodies:gp.grad_bodies
  in
  let initial, _ = eval 40_000 in
  for step = 1 to 1200 do
    let _, gradients = eval (40_000 + step) in
    let learning_rate = if step <= 500 then 0.01 else 0.003 in
    List.iter (fun (name, parameter) ->
      adam_tensor_ascent ~step ~learning_rate (List.assoc name states) parameter
        (List.assoc name gradients)) params
  done;
  let final, _ = eval 40_000 in
  check bool "hierarchical ELBO improves" true (value final 0 > value initial 0);
  let expanded_guide = Transform.Expand_rank.expand ~senv:shapes guide in
  let posterior_env = params @ fixed in
  let traces = List.init posterior_draws (fun draw ->
    let _, trace, _ = Ast.Simulate.simulate
      ~run_key:(Int64.of_int (50_000 + draw)) posterior_env expanded_guide in
    trace) in
  let draws name = List.map (extract_trace name) traces in
  let mu_hist = rank_histogram (draws "mu") mu_true
  and tau_hist = rank_histogram (draws "tau") tau_true
  and a_hist = rank_histogram (draws "a") a_true in
  let mu_chi = rank_chi_square mu_hist and tau_chi = rank_chi_square tau_hist
  and a_chi = rank_chi_square a_hist in
  let half_normal_ld x scale =
    0.5 *. log (2.0 /. Float.pi) -. log scale -. 0.5 *. (x /. scale) ** 2.0 in
  let joint_per_repetition trace =
    let mu = extract_trace "mu" trace and tau = extract_trace "tau" trace
    and a = extract_trace "a" trace in
    tensor frame_n (Array.init repetitions (fun repetition ->
      let total = ref (gaussian_log_density ~x:(value mu repetition)
        ~mu:0.0 ~sigma:2.0 +. half_normal_ld (value tau repetition) 1.0) in
      for group = 0 to groups - 1 do
        let i = repetition * groups + group in
        total := !total
          +. gaussian_log_density ~x:(value a i) ~mu:(value mu repetition)
               ~sigma:(value tau repetition)
          +. gaussian_log_density ~x:(value y i) ~mu:(value a i) ~sigma:0.5
      done;
      !total))
  in
  let true_trace = [("mu", mu_true); ("tau", tau_true); ("a", a_true)] in
  let joint_chi = rank_chi_square
    (rank_histogram (List.map joint_per_repetition traces)
       (joint_per_repetition true_trace)) in
  let endpoint_coverage histogram =
    let total = Array.fold_left ( + ) 0 histogram in
    1.0 -. float_of_int (histogram.(0) + histogram.(Array.length histogram - 1))
      /. float_of_int total in
  let tau_bottom = float_of_int tau_hist.(0) /. float_of_int repetitions
  and tau_top = float_of_int tau_hist.(posterior_draws)
    /. float_of_int repetitions in
  Printf.printf "\nSBC stage 2 diagnostic (not a completion criterion): \
mu=%.3f tau=%.3f a=%.3f log_joint=%.3f; \
tau rank=0/15 %.1f%%/%.1f%%, endpoint coverage mu/tau/a=%.1f%%/%.1f%%/%.1f%%\n"
    mu_chi tau_chi a_chi joint_chi (100.0 *. tau_bottom) (100.0 *. tau_top)
    (100.0 *. endpoint_coverage mu_hist)
    (100.0 *. endpoint_coverage tau_hist)
    (100.0 *. endpoint_coverage a_hist);
  List.iter (fun (label, statistic) ->
    check bool (label ^ " diagnostic finite") true (Float.is_finite statistic))
    [("mu", mu_chi); ("tau", tau_chi); ("a", a_chi);
     ("log joint", joint_chi)]

let logistic_glmm_program groups observations =
  let frame_g = [|groups|] and frame_go = [|groups; observations|] in
  let logits = rank 0 Add
    [rank 0 Mul [var "gamma"; var "x_obs"]; var "a"] in
  let one = const (scalar 1.0) in
  let log_likelihood = rank 0 Add [
    rank 0 Mul [var "y_obs"; prim Logsigmoid [logits]];
    rank 0 Mul [rank 0 Sub [one; var "y_obs"];
      prim Logsigmoid [rank 0 Neg [logits]]];
  ] in
  let model =
    let_ "tau" (sample "tau" [||]
      (Ast.Half_normal.half_normal ~sigma:"prior_tau_scale"))
      (let_ "a" (sample "a" frame_g
        (Ast.Normal.normal ~mu:"zero" ~sigma:"tau"))
        (score log_likelihood))
  in
  let guide =
    let_ "q_tau_scale" (prim Exp [var "q_tau_rho"])
      (let_ "tau" (sample "tau" [||]
        (Ast.Log_normal.log_normal ~mu:"q_tau_loc" ~sigma:"q_tau_scale"))
        (let_ "q_a_scale" (prim Exp [var "q_a_rho"])
          (sample "a" frame_g
            (Ast.Normal.normal ~mu:"q_a_loc" ~sigma:"q_a_scale"))))
  in
  model, guide, frame_g, frame_go

let test_logistic_glmm_learning () =
  let groups = 64 and observations = 20 in
  let true_gamma = 1.2 and true_tau = 0.7 in
  let model, guide, frame_g, frame_go =
    logistic_glmm_program groups observations in
  let latent_generator = sample "a" frame_g
    (Ast.Normal.normal ~mu:"zero" ~sigma:"true_tau")
    |> Transform.Expand_rank.expand
         ~senv:[("zero", [||]); ("true_tau", [||])] in
  let _, latent_trace, _ = Ast.Simulate.simulate ~run_key:180L
    [("zero", scalar 0.0); ("true_tau", scalar true_tau)] latent_generator in
  let true_a = extract_trace "a" latent_trace in
  let x = View.Tensor.make frame_go and y = View.Tensor.make frame_go in
  let key = Prng.Threefry.make_key ~run_key:181L
    ~namespace:Prng.Threefry.ns_data in
  for group = 0 to groups - 1 do
    for observation = 0 to observations - 1 do
      let i = group * observations + observation in
      let xv = (float_of_int observation -. 9.5) /. 5.0 in
      let eta = true_gamma *. xv +. value true_a group in
      let probability = if eta >= 0.0 then 1.0 /. (1.0 +. exp (-.eta))
        else let e = exp eta in e /. (1.0 +. e) in
      let ctr = Prng.Threefry.make_ctr ~site_id:2 ~component:1 ~frame_index:i in
      let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
      View.Buf.set x.buf i xv;
      View.Buf.set y.buf i
        (if Prng.Threefry.to_open_unit bits < probability then 1.0 else 0.0)
    done
  done;
  let shapes = [("prior_tau_scale", [||]); ("zero", [||]);
    ("gamma", [||]); ("x_obs", frame_go); ("y_obs", frame_go);
    ("q_tau_loc", [||]); ("q_tau_rho", [||]);
    ("q_a_loc", frame_g); ("q_a_rho", frame_g)] in
  let program = Transform.build_elbo ~observed:[] ~model ~guide
    ~env_shapes:shapes in
  let param_shapes = [("gamma", [||]); ("q_tau_loc", [||]);
    ("q_tau_rho", [||]); ("q_a_loc", frame_g); ("q_a_rho", frame_g)] in
  let gp = Transform.grad ~param_shapes
    ~data_shapes:(program.noise @ [("prior_tau_scale", [||]); ("zero", [||]);
      ("x_obs", frame_go); ("y_obs", frame_go)]) program.elbo in
  let params = [("gamma", scalar 0.0); ("q_tau_loc", scalar (log 0.8));
    ("q_tau_rho", scalar (log 0.35)); ("q_a_loc", filled frame_g 0.0);
    ("q_a_rho", filled frame_g (log 0.6))] in
  let states = List.map (fun (name, parameter) ->
    name, {m = filled parameter.View.Tensor.view.View.Ndview.shape 0.0;
           v = filled parameter.View.Tensor.view.View.Ndview.shape 0.0}) params in
  let fixed = [("prior_tau_scale", scalar 1.0); ("zero", scalar 0.0);
               ("x_obs", x); ("y_obs", y)] in
  let eval step =
    let env = Transform.noise_env program ~run_key:(Int64.of_int step)
      @ params @ fixed in
    Ast.Eval.eval_grad env ~primal_bindings:gp.primal_bindings
      ~loss_body:gp.loss_body ~grad_bindings:gp.grad_bindings
      ~grad_bodies:gp.grad_bodies
  in
  let initial, _ = eval 60_000 in
  for step = 1 to 1500 do
    let _, gradients = eval (60_000 + step) in
    let learning_rate = if step <= 700 then 0.01 else 0.003 in
    List.iter (fun (name, parameter) ->
      adam_tensor_ascent ~step ~learning_rate (List.assoc name states) parameter
        (List.assoc name gradients)) params
  done;
  let final, _ = eval 60_000 in
  let gamma = value (List.assoc "gamma" params) 0
  and tau_median = exp (value (List.assoc "q_tau_loc" params) 0) in
  Printf.printf "\nLogistic GLMM: gamma=%.4f (true %.4f), \
tau median=%.4f (true %.4f)\n" gamma true_gamma tau_median true_tau;
  check bool "GLMM ELBO improves" true (value final 0 > value initial 0);
  check bool "fixed effect recovered" true (Float.abs (gamma -. true_gamma) < 0.25);
  check bool "random-effect scale recovered" true
    (Float.is_finite tau_median && Float.abs (tau_median -. true_tau) < 0.3)

let test_observed_site () =
  let model, guide = observed_program 3 in
  let shapes =
    [
      ("prior_mu", [||]); ("prior_scale", [||]); ("obs_scale", [||]);
      ("q_mu", [|3|]); ("q_scale", [|3|]); ("y_obs", [|3|]);
    ] in
  let env =
    [
      ("prior_mu", scalar 0.0); ("prior_scale", scalar 1.0);
      ("obs_scale", scalar 0.4); ("q_mu", tensor [|3|] [|0.1; -0.2; 0.3|]);
      ("q_scale", tensor [|3|] [|0.8; 0.9; 0.7|]);
    ] in
  let expanded = Transform.Expand_rank.expand ~senv:shapes model in
  let _, generated_trace, _ = Ast.Simulate.simulate ~run_key:121L env expanded in
  let _, assessed = Ast.Assess.assess env expanded generated_trace in
  let slots = List.map (fun (name, sample) -> name, const sample) generated_trace in
  let symbolic = Transform.Assess_expr.assess_expr
    ~env_shapes:shapes expanded slots |> Ast.Eval.eval env in
  check (float 1e-12) "generated y assessed identically" (value assessed 0)
    (value symbolic 0);
  let y = List.assoc "y" generated_trace in
  let program = Transform.build_elbo ~observed:[("y", var "y_obs")]
    ~model ~guide ~env_shapes:shapes in
  check (list string) "only latent sites draw guide noise" ["z"]
    (List.map (fun site -> site.Ast.Sites.name) program.sites);
  let result = Ast.Eval.eval
    (Transform.noise_env program ~run_key:122L @ (("y_obs", y) :: env))
    program.elbo |> fun t -> value t 0 in
  check bool "observed ELBO is finite" true (Float.is_finite result)

let test_observed_partition_errors () =
  let model, guide = observed_program 3 in
  let shapes =
    [
      ("prior_mu", [||]); ("prior_scale", [||]); ("obs_scale", [||]);
      ("q_mu", [|3|]); ("q_scale", [|3|]); ("y_obs", [|3|]);
    ] in
  check_raises "model site must be latent or observed"
    (Transform.Reparam.Trace_mismatch "model site 'y' not found in guide")
    (fun () -> ignore (Transform.build_elbo ~observed:[]
      ~model ~guide ~env_shapes:shapes));
  check_raises "observed cannot also be latent"
    (Transform.Reparam.Trace_mismatch
       "observed site 'z' also appears in guide")
    (fun () -> ignore (Transform.build_elbo ~observed:[("z", var "y_obs");
      ("y", var "y_obs")] ~model ~guide ~env_shapes:shapes))

let test_discrete_observed_allowed () =
  let weights = tensor [|2|] [|1.0; 3.0|] in
  let model = sample "y" [||] (D_categorical (const weights)) in
  let guide = const (scalar 0.0) in
  let program = Transform.build_elbo ~observed:[("y", var "y_obs")]
    ~model ~guide ~env_shapes:[("y_obs", [||])] in
  let actual = Ast.Eval.eval [("y_obs", scalar 1.0)] program.elbo
    |> fun result -> value result 0 in
  check (float 1e-12) "discrete observed log density" (log 0.75) actual

let () =
  run "Phase 12"
    [
      ( "12-1 HalfNormal",
        [
          test_case "inv after fwd" `Quick test_half_normal_inverse;
          test_case "closed density" `Quick test_half_normal_density;
          test_case "normalizes" `Slow test_half_normal_normalizes;
          test_case "sigma FD" `Quick test_half_normal_sigma_fd;
          test_case "LogNormal density" `Quick test_log_normal_density;
          test_case "runtime support" `Quick test_support_runtime;
          test_case "symbolic declared support" `Quick
            test_symbolic_declared_support;
          test_case "support declarations" `Quick test_support_declarations;
          test_case "support partial order" `Quick test_support_partial_order;
        ] );
      ( "12-2 hierarchical sites",
        [
          test_case "assess equals assess_expr" `Quick
            test_hierarchical_assess_expr;
          test_case "coupling" `Quick test_hierarchical_coupling;
          test_case "all guide parameters FD" `Quick
            test_hierarchical_all_parameter_fd;
          test_case "support inclusion" `Quick test_support_check;
        ] );
      ( "12-3 observed sites",
        [
          test_case "simulate then assess" `Quick test_observed_site;
          test_case "latent observed partition" `Quick
            test_observed_partition_errors;
          test_case "discrete observed" `Quick test_discrete_observed_allowed;
        ] );
      ( "12-4 SBC implementation invariants",
        [
          test_case "observed optimal ELBO" `Quick test_observed_optimal_elbo;
          test_case "1a closed posterior ranks" `Slow test_sbc_closed_form;
          test_case "1b language VI ranks" `Slow test_sbc_language_vi;
        ] );
      ( "12-5 ragged masking",
        [test_case "padding is bit invisible" `Quick
           test_ragged_mask_invariance] );
      ( "12-6 hierarchical VI",
        [test_case "leading-frame coupling" `Quick
           test_leading_frame_hierarchical_coupling;
         test_case "reject nonleading fwd variable" `Quick
           test_batch_fwd_rejects_nonleading_reference;
         test_case "SBC stage 2 diagnostic" `Slow
           test_hierarchical_vi_sbc_diagnostic] );
      ( "12-7 logistic GLMM",
        [test_case "synthetic learning" `Slow test_logistic_glmm_learning] );
    ]
