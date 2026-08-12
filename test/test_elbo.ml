(* Phase 11-4: ELBO assembly + FD gradient check.
   Stage 1: expression-level ELBO matches value-level (eval only).
   Stage 2: ELBO gradient via Transform.grad matches central finite differences.
   Erfinv JVP enters the seed path for the first time here. *)

open Alcotest
open Ast.Types

let mk_scalar f =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 f; t

let scalar_val (t : View.Tensor.t) = View.Buf.get t.buf 0

(* Build the ELBO expression for model/guide pair.
   Returns (elbo_expr, z_expr) where z_expr maps %u.z → z via guide's fwd.
   Free variables: mu_m, sigma_m, mu_g, sigma_g, %u.z *)
let build_elbo_expr ~model ~guide ~env_shapes =
  let guide_r = Transform.Reparam.reparam guide in
  let z_expr = match guide_r with
    | Let (_, _, _, fwd_body) -> fwd_body
    | _ -> failwith "unexpected reparam structure" in
  let senv_z = List.filter (fun (s, _) ->
    s = "mu_g" || s = "sigma_g" || s = "%u.z"
  ) env_shapes in
  let z_expr = Transform.Expand_rank.expand ~senv:senv_z z_expr in
  let slots = [("z", z_expr)] in
  let model_ld = Transform.Assess_expr.assess_expr ~ns:"m."
    ~env_shapes model slots in
  let guide_ld = Transform.Assess_expr.assess_expr ~ns:"g."
    ~env_shapes guide slots in
  (prim Sub [model_ld; guide_ld], z_expr)

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
  let (_, trace_g, _) = Ast.Simulate.simulate ~run_key:42L
    [("mu_g", mu_g); ("sigma_g", sigma_g)] guide in
  let z = List.assoc "z" trace_g in
  let (_, ld_model) = Ast.Assess.assess
    [("mu_m", mu_m); ("sigma_m", sigma_m)] model [("z", z)] in
  let (_, ld_guide) = Ast.Assess.assess
    [("mu_g", mu_g); ("sigma_g", sigma_g)] guide [("z", z)] in
  let elbo_val = scalar_val ld_model -. scalar_val ld_guide in

  (* Expression-level ELBO *)
  let env_shapes = [("mu_m", [||]); ("sigma_m", [||]);
                    ("mu_g", [||]); ("sigma_g", [||]); ("%u.z", [||])] in
  let (elbo_expr, _) = build_elbo_expr ~model ~guide ~env_shapes in

  (* Get u from reparam'd guide (trace records the base uniform sample) *)
  let guide_r = Transform.Reparam.reparam guide in
  let senv = [("mu_g", [||]); ("sigma_g", [||])] in
  let guide_re = Transform.Expand_rank.expand ~senv guide_r in
  let (_, trace_re, _) = Ast.Simulate.simulate ~run_key:42L
    [("mu_g", mu_g); ("sigma_g", sigma_g)] guide_re in
  let u = List.assoc "z" trace_re in

  let env = [("mu_m", mu_m); ("sigma_m", sigma_m);
             ("mu_g", mu_g); ("sigma_g", sigma_g); ("%u.z", u)] in
  let elbo_sym = scalar_val (Ast.Eval.eval env elbo_expr) in
  check (float 1e-12) "ELBO eval" elbo_val elbo_sym

(* ── Stage 2: ELBO gradient vs central FD ── *)

let test_elbo_grad () =
  let model = sample "z" [||] (Ast.Normal.normal ~mu:"mu_m" ~sigma:"sigma_m") in
  let guide = sample "z" [||] (Ast.Normal.normal ~mu:"mu_g" ~sigma:"sigma_g") in
  let env_shapes = [("mu_m", [||]); ("sigma_m", [||]);
                    ("mu_g", [||]); ("sigma_g", [||]); ("%u.z", [||])] in
  let (elbo_expr, _) = build_elbo_expr ~model ~guide ~env_shapes in

  (* grad wrt guide params *)
  let param_shapes = [("mu_g", [||]); ("sigma_g", [||])] in
  let data_shapes = [("%u.z", [||]); ("mu_m", [||]); ("sigma_m", [||])] in
  let gp = Transform.grad ~param_shapes ~data_shapes elbo_expr in

  (* Evaluation point *)
  let mu_g_val = 0.3 in
  let sigma_g_val = 0.8 in
  let u_val = 0.37 in
  let base_env = [("mu_m", mk_scalar 0.0); ("sigma_m", mk_scalar 1.0);
                  ("%u.z", mk_scalar u_val)] in
  let make_env mg sg = ("mu_g", mk_scalar mg) :: ("sigma_g", mk_scalar sg) :: base_env in
  let env = make_env mu_g_val sigma_g_val in

  (* AD gradients *)
  let d_mu_g = scalar_val (Ast.Eval.eval env (List.assoc "mu_g" gp.grads)) in
  let d_sigma_g = scalar_val (Ast.Eval.eval env (List.assoc "sigma_g" gp.grads)) in

  (* Central FD *)
  let eps = 1e-5 in
  let eval_loss e = scalar_val (Ast.Eval.eval e gp.loss) in
  let fd_mu_g =
    (eval_loss (make_env (mu_g_val +. eps) sigma_g_val) -.
     eval_loss (make_env (mu_g_val -. eps) sigma_g_val)) /. (2.0 *. eps) in
  let fd_sigma_g =
    (eval_loss (make_env mu_g_val (sigma_g_val +. eps)) -.
     eval_loss (make_env mu_g_val (sigma_g_val -. eps))) /. (2.0 *. eps) in

  (* Relative error check *)
  let rel_err ad fd =
    let denom = max (abs_float fd) 1e-10 in
    abs_float (ad -. fd) /. denom in
  let tol = 1e-6 in
  let r_mu = rel_err d_mu_g fd_mu_g in
  let r_sigma = rel_err d_sigma_g fd_sigma_g in
  check bool (Printf.sprintf "d_mu_g rel_err=%.2e" r_mu) true (r_mu < tol);
  check bool (Printf.sprintf "d_sigma_g rel_err=%.2e" r_sigma) true (r_sigma < tol)

(* ── Stage 2b: coupling — u fixed under psi perturbation ── *)

let test_elbo_coupling () =
  (* Verify: same u value produces consistent gradients across different (mu_g, sigma_g).
     The key coupling property: perturbing psi doesn't change u. *)
  let model = sample "z" [||] (Ast.Normal.normal ~mu:"mu_m" ~sigma:"sigma_m") in
  let guide = sample "z" [||] (Ast.Normal.normal ~mu:"mu_g" ~sigma:"sigma_g") in
  let env_shapes = [("mu_m", [||]); ("sigma_m", [||]);
                    ("mu_g", [||]); ("sigma_g", [||]); ("%u.z", [||])] in
  let (elbo_expr, _) = build_elbo_expr ~model ~guide ~env_shapes in
  let param_shapes = [("mu_g", [||]); ("sigma_g", [||])] in
  let data_shapes = [("%u.z", [||]); ("mu_m", [||]); ("sigma_m", [||])] in
  let gp = Transform.grad ~param_shapes ~data_shapes elbo_expr in

  (* FD at TWO different operating points — both should pass *)
  let u_val = 0.62 in
  let base_env = [("mu_m", mk_scalar 0.0); ("sigma_m", mk_scalar 1.0);
                  ("%u.z", mk_scalar u_val)] in
  let eps = 1e-5 in
  let tol = 1e-6 in
  List.iter (fun (mg, sg) ->
    let make_env mg' sg' =
      ("mu_g", mk_scalar mg') :: ("sigma_g", mk_scalar sg') :: base_env in
    let env = make_env mg sg in
    let d_mu = scalar_val (Ast.Eval.eval env (List.assoc "mu_g" gp.grads)) in
    let d_sigma = scalar_val (Ast.Eval.eval env (List.assoc "sigma_g" gp.grads)) in
    let eval_loss e = scalar_val (Ast.Eval.eval e gp.loss) in
    let fd_mu =
      (eval_loss (make_env (mg +. eps) sg) -.
       eval_loss (make_env (mg -. eps) sg)) /. (2.0 *. eps) in
    let fd_sigma =
      (eval_loss (make_env mg (sg +. eps)) -.
       eval_loss (make_env mg (sg -. eps))) /. (2.0 *. eps) in
    let rel ad fd = abs_float (ad -. fd) /. max (abs_float fd) 1e-10 in
    let r1 = rel d_mu fd_mu in
    let r2 = rel d_sigma fd_sigma in
    check bool (Printf.sprintf "mu_g=%.1f sigma_g=%.1f d_mu rel=%.2e" mg sg r1)
      true (r1 < tol);
    check bool (Printf.sprintf "mu_g=%.1f sigma_g=%.1f d_sigma rel=%.2e" mg sg r2)
      true (r2 < tol)
  ) [(0.3, 0.8); (-1.0, 1.5)]

(* ── Test suite ── *)

let () =
  run "ELBO" [
    "eval", [
      test_case "value match"      `Quick test_elbo_eval;
    ];
    "grad", [
      test_case "FD check"         `Quick test_elbo_grad;
      test_case "coupling"         `Quick test_elbo_coupling;
    ];
  ]
