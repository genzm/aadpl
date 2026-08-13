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
  check_raises "assess_expr rejects value outside base support"
    (Ast.Eval.Eval_error (loc, "value outside Uniform support (0,1)"))
    (fun () -> ignore (Ast.Eval.eval [("sigma", scalar 1.0)] symbolic))

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
    [S_real; S_positive; S_unit_interval; S_finite;
     S_product (S_positive, S_real)] in
  List.iter (fun a ->
    check bool "support reflexive" true (Ast.Sites.support_subset a a)) supports;
  List.iter (fun a -> List.iter (fun b -> List.iter (fun c ->
    if Ast.Sites.support_subset a b && Ast.Sites.support_subset b c then
      check bool "support transitive" true (Ast.Sites.support_subset a c))
    supports) supports) supports

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

let test_discrete_observed_rejected () =
  let weights = tensor [|2|] [|1.0; 1.0|] in
  let model = sample "y" [||] (D_categorical (const weights)) in
  let guide = const (scalar 0.0) in
  check_raises "discrete observation deferred to Phase 13"
    (Transform.Reparam.Trace_mismatch
       "observed site 'y' must be continuous")
    (fun () -> ignore (Transform.build_elbo
      ~observed:[("y", var "y_obs")] ~model ~guide
      ~env_shapes:[("y_obs", [||])]))

let () =
  run "Phase 12"
    [
      ( "12-1 HalfNormal",
        [
          test_case "inv after fwd" `Quick test_half_normal_inverse;
          test_case "closed density" `Quick test_half_normal_density;
          test_case "normalizes" `Slow test_half_normal_normalizes;
          test_case "sigma FD" `Quick test_half_normal_sigma_fd;
          test_case "runtime support" `Quick test_support_runtime;
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
          test_case "continuous only" `Quick test_discrete_observed_rejected;
        ] );
    ]
