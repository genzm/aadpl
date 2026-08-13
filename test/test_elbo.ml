(* Phase 11-4: ELBO assembly + FD gradient check.
   Stage 1: expression-level ELBO matches value-level (eval only).
   Stage 2: ELBO gradient via Transform.grad matches central finite differences.
   Erfinv JVP enters the seed path for the first time here. *)

open Alcotest
open Ast.Types

let mk_scalar f =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 f;
  t

let scalar_val (t : View.Tensor.t) = View.Buf.get t.buf 0

let tensor_of_array shape values =
  let t = View.Tensor.make shape in
  Array.iteri (fun i value -> View.Buf.set t.buf i value) values;
  t

let tensor_get (t : View.Tensor.t) i = View.Buf.get t.buf i

(* ── Stage 1: ELBO eval matches value-level ── *)

let test_elbo_eval () =
  (* Model: z ~ Normal(0, 1),  Guide: z ~ Normal(mu_g, sigma_g) *)
  let model = sample "z" [||] (Ast.Normal.normal ~mu:"mu_m" ~sigma:"sigma_m") in
  let guide = sample "z" [||] (Ast.Normal.normal ~mu:"mu_g" ~sigma:"sigma_g") in
  let mu_m = mk_scalar 0.0 in
  let sigma_m = mk_scalar 1.0 in
  let mu_g = mk_scalar 0.3 in
  let sigma_g = mk_scalar 0.8 in

  (* Value-level ELBO: simulate guide → z, assess model - assess guide *)
  let _, trace_g, _ =
    Ast.Simulate.simulate ~run_key:42L
      [ ("mu_g", mu_g); ("sigma_g", sigma_g) ]
      guide
  in
  let z = List.assoc "z" trace_g in
  let _, ld_model =
    Ast.Assess.assess
      [ ("mu_m", mu_m); ("sigma_m", sigma_m) ]
      model
      [ ("z", z) ]
  in
  let _, ld_guide =
    Ast.Assess.assess
      [ ("mu_g", mu_g); ("sigma_g", sigma_g) ]
      guide
      [ ("z", z) ]
  in
  let elbo_val = scalar_val ld_model -. scalar_val ld_guide in

  (* Expression-level ELBO *)
  let env_shapes =
    [ ("mu_m", [||]); ("sigma_m", [||]); ("mu_g", [||]); ("sigma_g", [||]) ]
  in
  let program = Transform.build_elbo ~observed:[] ~model ~guide ~env_shapes in
  check
    (list (pair string (array int)))
    "noise"
    [ ("%u.z", [||]) ]
    program.noise;
  let guide_noise = Transform.noise_env program ~run_key:42L in
  let model_noise = Ast.Sites.draw_noise ~run_key:42L program.sites in
  check bool "guide/model namespaces differ" true
    (scalar_val (List.assoc "%u.z" guide_noise)
    <> scalar_val (List.assoc "%u.z" model_noise));

  let noise = Ast.Sites.draw_noise ~run_key:42L program.sites in

  let env =
    [
      ("mu_m", mu_m); ("sigma_m", sigma_m); ("mu_g", mu_g); ("sigma_g", sigma_g);
    ]
  in
  let elbo_sym = scalar_val (Ast.Eval.eval (noise @ env) program.elbo) in
  check (float 1e-12) "ELBO eval" elbo_val elbo_sym

let test_noise_namespaces () =
  let guide = sample "z" [||] D_uniform in
  let sites = Ast.Sites.collect_sites guide in
  let program = Transform.build_elbo ~observed:[] ~model:guide ~guide ~env_shapes:[] in
  let _, trace, _ = Ast.Simulate.simulate ~sites ~run_key:42L [] guide in
  let model_noise = Ast.Sites.draw_noise ~run_key:42L sites in
  let guide_noise = Ast.Sites.draw_noise
    ~namespace:Prng.Threefry.ns_guide ~run_key:42L sites in
  let helper_noise = Transform.noise_env program ~run_key:42L in
  let traced = scalar_val (List.assoc "z" trace) in
  let model_u = scalar_val (List.assoc "%u.z" model_noise) in
  let guide_u = scalar_val (List.assoc "%u.z" guide_noise) in
  let helper_u = scalar_val (List.assoc "%u.z" helper_noise) in
  check bool "simulate couples to ns_model" true (traced = model_u);
  check bool "ns_guide differs from ns_model" true (guide_u <> model_u);
  check bool "noise_env uses ns_guide" true (helper_u = guide_u)

(* ── Stage 2: ELBO gradient vs central FD ── *)

let test_elbo_grad () =
  let model = sample "z" [||] (Ast.Normal.normal ~mu:"mu_m" ~sigma:"sigma_m") in
  let guide = sample "z" [||] (Ast.Normal.normal ~mu:"mu_g" ~sigma:"sigma_g") in
  let env_shapes =
    [ ("mu_m", [||]); ("sigma_m", [||]); ("mu_g", [||]); ("sigma_g", [||]) ]
  in
  let program = Transform.build_elbo ~observed:[] ~model ~guide ~env_shapes in

  (* grad wrt guide params *)
  let param_shapes = [ ("mu_g", [||]); ("sigma_g", [||]) ] in
  let data_shapes = program.noise @ [ ("mu_m", [||]); ("sigma_m", [||]) ] in
  let gp = Transform.grad ~param_shapes ~data_shapes program.elbo in

  (* Evaluation point *)
  let mu_g_val = 0.3 in
  let sigma_g_val = 0.8 in
  let u_val = 0.37 in
  let base_env =
    [
      ("mu_m", mk_scalar 0.0);
      ("sigma_m", mk_scalar 1.0);
      ("%u.z", mk_scalar u_val);
    ]
  in
  let make_env mg sg =
    ("mu_g", mk_scalar mg) :: ("sigma_g", mk_scalar sg) :: base_env
  in
  let env = make_env mu_g_val sigma_g_val in

  (* AD gradients *)
  let d_mu_g = scalar_val (Ast.Eval.eval env (List.assoc "mu_g" gp.grads)) in
  let d_sigma_g =
    scalar_val (Ast.Eval.eval env (List.assoc "sigma_g" gp.grads))
  in

  (* Central FD *)
  let eps = 1e-5 in
  let eval_loss e = scalar_val (Ast.Eval.eval e gp.loss) in
  let fd_mu_g =
    (eval_loss (make_env (mu_g_val +. eps) sigma_g_val)
    -. eval_loss (make_env (mu_g_val -. eps) sigma_g_val))
    /. (2.0 *. eps)
  in
  let fd_sigma_g =
    (eval_loss (make_env mu_g_val (sigma_g_val +. eps))
    -. eval_loss (make_env mu_g_val (sigma_g_val -. eps)))
    /. (2.0 *. eps)
  in

  (* Relative error check *)
  let rel_err ad fd =
    let denom = max (abs_float fd) 1e-10 in
    abs_float (ad -. fd) /. denom
  in
  let tol = 1e-6 in
  let r_mu = rel_err d_mu_g fd_mu_g in
  let r_sigma = rel_err d_sigma_g fd_sigma_g in
  check bool (Printf.sprintf "d_mu_g rel_err=%.2e" r_mu) true (r_mu < tol);
  check bool
    (Printf.sprintf "d_sigma_g rel_err=%.2e" r_sigma)
    true (r_sigma < tol)

(* ── Stage 2b: coupling — u fixed under psi perturbation ── *)

let test_elbo_coupling () =
  (* Verify: same u value produces consistent gradients across different (mu_g, sigma_g).
     The key coupling property: perturbing psi doesn't change u. *)
  let model = sample "z" [||] (Ast.Normal.normal ~mu:"mu_m" ~sigma:"sigma_m") in
  let guide = sample "z" [||] (Ast.Normal.normal ~mu:"mu_g" ~sigma:"sigma_g") in
  let env_shapes =
    [ ("mu_m", [||]); ("sigma_m", [||]); ("mu_g", [||]); ("sigma_g", [||]) ]
  in
  let program = Transform.build_elbo ~observed:[] ~model ~guide ~env_shapes in
  let param_shapes = [ ("mu_g", [||]); ("sigma_g", [||]) ] in
  let data_shapes = program.noise @ [ ("mu_m", [||]); ("sigma_m", [||]) ] in
  let gp = Transform.grad ~param_shapes ~data_shapes program.elbo in

  (* FD at TWO different operating points — both should pass *)
  let u_val = 0.62 in
  let base_env =
    [
      ("mu_m", mk_scalar 0.0);
      ("sigma_m", mk_scalar 1.0);
      ("%u.z", mk_scalar u_val);
    ]
  in
  let eps = 1e-5 in
  let tol = 1e-6 in
  List.iter
    (fun (mg, sg) ->
      let make_env mg' sg' =
        ("mu_g", mk_scalar mg') :: ("sigma_g", mk_scalar sg') :: base_env
      in
      let env = make_env mg sg in
      let d_mu = scalar_val (Ast.Eval.eval env (List.assoc "mu_g" gp.grads)) in
      let d_sigma =
        scalar_val (Ast.Eval.eval env (List.assoc "sigma_g" gp.grads))
      in
      let eval_loss e = scalar_val (Ast.Eval.eval e gp.loss) in
      let fd_mu =
        (eval_loss (make_env (mg +. eps) sg)
        -. eval_loss (make_env (mg -. eps) sg))
        /. (2.0 *. eps)
      in
      let fd_sigma =
        (eval_loss (make_env mg (sg +. eps))
        -. eval_loss (make_env mg (sg -. eps)))
        /. (2.0 *. eps)
      in
      let rel ad fd = abs_float (ad -. fd) /. max (abs_float fd) 1e-10 in
      let r1 = rel d_mu fd_mu in
      let r2 = rel d_sigma fd_sigma in
      check bool
        (Printf.sprintf "mu_g=%.1f sigma_g=%.1f d_mu rel=%.2e" mg sg r1)
        true (r1 < tol);
      check bool
        (Printf.sprintf "mu_g=%.1f sigma_g=%.1f d_sigma rel=%.2e" mg sg r2)
        true (r2 < tol))
    [ (0.3, 0.8); (-1.0, 1.5) ]

(* ── MLP guide: local encoder outputs participate in density + pathwise AD ── *)

let test_mlp_guide_fd () =
  let model =
    sample "z" [| 2 |] (Ast.Normal.normal ~mu:"mu_m" ~sigma:"sigma_m")
  in
  let guide =
    let_ "h2"
      (rank 2 Matmul [ var "x"; var "w" ])
      (let_ "mu_vec"
         (rank 0 Relu [ prim (Apply_view [ Vreshape [| 2 |] ]) [ var "h2" ] ])
         (sample "z" [| 2 |] (Ast.Normal.normal ~mu:"mu_vec" ~sigma:"sigma_g")))
  in
  let env_shapes =
    [
      ("x", [| 1; 2 |]);
      ("w", [| 2; 2 |]);
      ("mu_m", [||]);
      ("sigma_m", [||]);
      ("sigma_g", [||]);
    ]
  in
  let program = Transform.build_elbo ~observed:[] ~model ~guide ~env_shapes in
  check
    (list (pair string (array int)))
    "MLP noise"
    [ ("%u.z", [| 2 |]) ]
    program.noise;
  let gp =
    Transform.grad
      ~param_shapes:[ ("w", [| 2; 2 |]) ]
      ~data_shapes:
        (program.noise
        @ [
            ("x", [| 1; 2 |]);
            ("mu_m", [||]);
            ("sigma_m", [||]);
            ("sigma_g", [||]);
          ])
      program.elbo
  in
  let x = tensor_of_array [| 1; 2 |] [| 0.4; -0.7 |] in
  let w0 = [| 1.0; 0.5; -0.2; -0.7 |] in
  let base_env =
    [
      ("x", x);
      ("mu_m", mk_scalar 0.0);
      ("sigma_m", mk_scalar 1.0);
      ("sigma_g", mk_scalar 0.8);
      ("%u.z", tensor_of_array [| 2 |] [| 0.3; 0.7 |]);
    ]
  in
  let make_env values = ("w", tensor_of_array [| 2; 2 |] values) :: base_env in
  let grad = Ast.Eval.eval (make_env w0) (List.assoc "w" gp.grads) in
  let eps = 1e-5 in
  Array.iteri
    (fun i _ ->
      let wp = Array.copy w0 and wm = Array.copy w0 in
      wp.(i) <- wp.(i) +. eps;
      wm.(i) <- wm.(i) -. eps;
      let lp = scalar_val (Ast.Eval.eval (make_env wp) gp.loss) in
      let lm = scalar_val (Ast.Eval.eval (make_env wm) gp.loss) in
      let fd = (lp -. lm) /. (2.0 *. eps) in
      let ad = tensor_get grad i in
      let err = abs_float (ad -. fd) /. max (abs_float fd) 1e-10 in
      check bool (Printf.sprintf "MLP w[%d] FD rel=%.2e" i err) true (err < 1e-5))
    w0

(* ── Multiple sites: no return-value-as-latent convention ── *)

let test_two_site_value_match () =
  let program_expr ~mu_z ~sigma_z ~mu_w ~sigma_w =
    let_ "z"
      (sample "z" [||] (Ast.Normal.normal ~mu:mu_z ~sigma:sigma_z))
      (let_ "w"
         (sample "w" [||] (Ast.Normal.normal ~mu:mu_w ~sigma:sigma_w))
         (prim Add [ var "z"; var "w" ]))
  in
  let model = program_expr ~mu_z:"mz" ~sigma_z:"sz" ~mu_w:"mw" ~sigma_w:"sw" in
  let guide =
    program_expr ~mu_z:"qz" ~sigma_z:"qsz" ~mu_w:"qw" ~sigma_w:"qsw"
  in
  let env_shapes =
    List.map
      (fun name -> (name, [||]))
      [ "mz"; "sz"; "mw"; "sw"; "qz"; "qsz"; "qw"; "qsw" ]
  in
  let built = Transform.build_elbo ~observed:[] ~model ~guide ~env_shapes in
  check
    (list (pair string (array int)))
    "two noises"
    [ ("%u.z", [||]); ("%u.w", [||]) ]
    built.noise;
  let env =
    [
      ("mz", mk_scalar 0.0);
      ("sz", mk_scalar 1.0);
      ("mw", mk_scalar 0.2);
      ("sw", mk_scalar 1.3);
      ("qz", mk_scalar 0.4);
      ("qsz", mk_scalar 0.7);
      ("qw", mk_scalar (-0.3));
      ("qsw", mk_scalar 1.1);
    ]
  in
  let _, trace_z, _ = Ast.Simulate.simulate ~run_key:123L env guide in
  let _, model_ld = Ast.Assess.assess env model trace_z in
  let _, guide_ld = Ast.Assess.assess env guide trace_z in
  let expected = scalar_val model_ld -. scalar_val guide_ld in
  let noise_env = Ast.Sites.draw_noise ~run_key:123L built.sites in
  let actual = scalar_val (Ast.Eval.eval (noise_env @ env) built.elbo) in
  check (float 1e-12) "two-site ELBO" expected actual

(* ── Test suite ── *)

let () =
  run "ELBO"
    [
      ( "eval",
        [
          test_case "value match" `Quick test_elbo_eval;
          test_case "noise namespaces" `Quick test_noise_namespaces;
        ] );
      ( "grad",
        [
          test_case "FD check" `Quick test_elbo_grad;
          test_case "coupling" `Quick test_elbo_coupling;
          test_case "MLP guide FD" `Quick test_mlp_guide_fd;
        ] );
      ( "multi-site",
        [ test_case "value match" `Quick test_two_site_value_match ] );
    ]
