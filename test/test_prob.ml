open Alcotest
open Ast.Types

let tensor_of a shape =
  let t = View.Tensor.make shape in
  Array.iteri (fun i v -> View.Buf.set t.buf i v) a;
  t

let scalar_val (t : View.Tensor.t) = View.Buf.get t.buf 0
let mk_scalar f = let t = View.Tensor.make [||] in View.Buf.set t.buf 0 f; t

(* ── 10-2: AST + pp + site check ── *)

let test_pp_sample () =
  let e = sample "z" [||] D_uniform in
  let s = Format.asprintf "%a" pp e in
  check bool "pp sample non-empty" true (String.length s > 0)

let test_pp_score () =
  let e = score (const (mk_scalar 1.0)) in
  let s = Format.asprintf "%a" pp e in
  check bool "pp score non-empty" true (String.length s > 0)

let test_check_sites_ok () =
  let e = let_ "a" (sample "x" [||] D_uniform)
            (sample "y" [||] D_uniform) in
  check_sites e;
  check pass "no exception" () ()

let test_check_sites_dup () =
  let e = let_ "a" (sample "x" [||] D_uniform)
            (sample "x" [||] D_uniform) in
  match check_sites e with
  | exception Duplicate_site (_, "x") -> check pass "caught dup" () ()
  | _ -> fail "expected Duplicate_site"

(* ── 10-3: simulate ── *)

let test_simulate_reproducibility () =
  let e = sample "z" [||] D_uniform in
  let (_, t1, _) = Ast.Simulate.simulate ~run_key:42L [] e in
  let (_, t2, _) = Ast.Simulate.simulate ~run_key:42L [] e in
  let v1 = scalar_val (List.assoc "z" t1) in
  let v2 = scalar_val (List.assoc "z" t2) in
  check (float 0.0) "same run_key → same trace" v1 v2

let test_simulate_different_keys () =
  let e = sample "z" [||] D_uniform in
  let (_, t1, _) = Ast.Simulate.simulate ~run_key:42L [] e in
  let (_, t2, _) = Ast.Simulate.simulate ~run_key:43L [] e in
  let v1 = scalar_val (List.assoc "z" t1) in
  let v2 = scalar_val (List.assoc "z" t2) in
  if v1 = v2 then fail "different run_key should differ"

let test_simulate_score () =
  (* Score contributions accumulate *)
  let e = let_ "_" (score (const (mk_scalar 1.5)))
            (let_ "_" (score (const (mk_scalar (-0.5))))
               (const (mk_scalar 0.0))) in
  let (_, _, lw) = Ast.Simulate.simulate ~run_key:0L [] e in
  check (float 1e-12) "log weight" 1.0 (scalar_val lw)

let test_simulate_coupling () =
  (* Coupling: same (runKey, site, index) → same raw uniform, regardless of θ.
     Use D_uniform directly to test at bit level — no erf/erfinv involved. *)
  let e = sample "u" [|4|] D_uniform in
  (* Two runs with different environments (θ doesn't affect Uniform draws) *)
  let env1 = [("theta", mk_scalar 1.0)] in
  let env2 = [("theta", mk_scalar 999.0)] in
  let (_, t1, _) = Ast.Simulate.simulate ~run_key:99L env1 e in
  let (_, t2, _) = Ast.Simulate.simulate ~run_key:99L env2 e in
  let u1 = List.assoc "u" t1 in
  let u2 = List.assoc "u" t2 in
  for i = 0 to 3 do
    let a = View.Buf.get u1.buf i in
    let b = View.Buf.get u2.buf i in
    (* Bit-exact: no tolerance, = comparison *)
    if a <> b then
      fail (Printf.sprintf "coupling[%d]: %g <> %g" i a b)
  done;
  (* Also verify via Threefry directly: same (key, ctr) → same output *)
  let key = Prng.Threefry.make_key ~run_key:99L
      ~namespace:Prng.Threefry.ns_model in
  for i = 0 to 3 do
    let ctr = Prng.Threefry.make_ctr ~site_id:0 ~component:1 ~frame_index:i in
    let (r0, _) = Prng.Threefry.threefry2x64 ~key ~ctr in
    let expected = Prng.Threefry.to_open_unit r0 in
    let actual = View.Buf.get u1.buf i in
    if actual <> expected then
      fail (Printf.sprintf "coupling vs Threefry[%d]: %g <> %g" i actual expected)
  done;
  check pass "coupling bit-exact" () ()

let test_simulate_categorical () =
  let w = tensor_of [| 0.0; 0.0; 1.0 |] [|3|] in
  let e = sample "k" [||] (D_categorical (const w)) in
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:7L [] e in
  let k = scalar_val (List.assoc "k" trace) in
  check (float 0.0) "categorical deterministic" 2.0 k

(* ── 10-4: Erf / Erfinv ── *)

let test_erf_known () =
  check (float 1e-6) "erf(0)" 0.0 (Ast.Eval.erf_impl 0.0);
  check (float 1e-6) "erf(1)" 0.84270079294971487 (Ast.Eval.erf_impl 1.0);
  check (float 1e-6) "erf(-1)" (-0.84270079294971487) (Ast.Eval.erf_impl (-1.0))

let test_erfinv_known () =
  check (float 1e-6) "erfinv(0)" 0.0 (Ast.Eval.erfinv_impl 0.0)

let test_erfinv_roundtrip () =
  (* erfinv(erf(x)) = x for various x *)
  let xs = [| -2.0; -1.0; -0.5; 0.0; 0.3; 0.7; 1.0; 1.5; 2.0 |] in
  Array.iter (fun x ->
    let y = Ast.Eval.erfinv_impl (Ast.Eval.erf_impl x) in
    check (float 1e-5) (Printf.sprintf "erfinv(erf(%.1f))" x) x y
  ) xs

let test_inv_fwd_identity () =
  (* For Normal: inv(fwd(u)) = u for u ∈ (0,1) *)
  let mu = 3.0 in
  let sigma = 2.0 in
  let fwd u =
    mu +. sigma *. (sqrt 2.0) *. Ast.Eval.erfinv_impl (2.0 *. u -. 1.0) in
  let inv x =
    0.5 *. (1.0 +. Ast.Eval.erf_impl ((x -. mu) /. (sigma *. sqrt 2.0))) in
  let us = [| 0.01; 0.1; 0.25; 0.5; 0.75; 0.9; 0.99 |] in
  Array.iter (fun u ->
    let recovered = inv (fwd u) in
    check (float 1e-5) (Printf.sprintf "inv(fwd(%.2f))" u) u recovered
  ) us

let test_erf_endpoint_stability () =
  (* Near ±1 erfinv should not produce NaN/inf for inputs in (-1,1) *)
  let xs = [| 0.999; 0.9999; 0.99999; -0.999; -0.9999 |] in
  Array.iter (fun x ->
    let y = Ast.Eval.erfinv_impl x in
    if Float.is_nan y || Float.is_infinite y then
      fail (Printf.sprintf "erfinv(%.5f) = %g" x y)
  ) xs;
  check pass "endpoints stable" () ()

let test_erf_jvp_fd () =
  (* Finite-difference check for erf JVP *)
  let x_val = mk_scalar 0.7 in
  let seed = mk_scalar 1.0 in
  let env = [("x", (x_val, seed))] in
  let e = prim Erf [var "x"] in
  let (_, tangent) = Ast.Jvp.jvp_eval env e in
  let jvp_val = scalar_val tangent in
  let h = 1e-6 in
  let f x = Ast.Eval.erf_impl x in
  let fd = (f (0.7 +. h) -. f (0.7 -. h)) /. (2.0 *. h) in
  check (float 1e-5) "erf JVP vs FD" fd jvp_val

let test_erfinv_jvp_fd () =
  let x_val = mk_scalar 0.5 in
  let seed = mk_scalar 1.0 in
  let env = [("x", (x_val, seed))] in
  let e = prim Erfinv [var "x"] in
  let (_, tangent) = Ast.Jvp.jvp_eval env e in
  let jvp_val = scalar_val tangent in
  let h = 1e-6 in
  let f x = Ast.Eval.erfinv_impl x in
  let fd = (f (0.5 +. h) -. f (0.5 -. h)) /. (2.0 *. h) in
  check (float 1e-5) "erfinv JVP vs FD" fd jvp_val

(* ── 10-5: assess ── *)

let test_assess_uniform_normalization () =
  (* ∫₀¹ exp(log_density(Uniform, x)) dx = 1 *)
  let n = 10000 in
  let h = 1.0 /. float_of_int n in
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    let x = (float_of_int i +. 0.5) *. h in
    let ld = Ast.Assess.log_density D_uniform (mk_scalar x) [] in
    sum := !sum +. exp ld *. h
  done;
  check (float 1e-6) "uniform normalization" 1.0 !sum

let test_assess_normal_normalization () =
  (* ∫ exp(log_density(Normal(0,1), x)) dx ≈ 1 over [-10,10] *)
  let env = [("mu", mk_scalar 0.0); ("sigma", mk_scalar 1.0)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let n = 100000 in
  let lo = -10.0 and hi = 10.0 in
  let h = (hi -. lo) /. float_of_int n in
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    let x = lo +. (float_of_int i +. 0.5) *. h in
    let ld = Ast.Assess.log_density dist (mk_scalar x) env in
    sum := !sum +. exp ld *. h
  done;
  check (float 1e-3) "normal normalization" 1.0 !sum

let test_assess_categorical_normalization () =
  let w = tensor_of [| 2.0; 3.0; 5.0 |] [|3|] in
  let dist = D_categorical (const w) in
  let sum = ref 0.0 in
  for k = 0 to 2 do
    let ld = Ast.Assess.log_density dist (mk_scalar (float_of_int k)) [] in
    sum := !sum +. exp ld
  done;
  check (float 1e-10) "categorical normalization" 1.0 !sum

let test_assess_normal_closed_form () =
  (* log p(x; μ, σ) = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)² *)
  let mu = 2.0 and sigma = 0.5 in
  let env = [("mu", mk_scalar mu); ("sigma", mk_scalar sigma)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let xs = [| -1.0; 0.0; 1.0; 2.0; 3.0; 4.0 |] in
  Array.iter (fun x ->
    let ld = Ast.Assess.log_density dist (mk_scalar x) env in
    let expected =
      -0.5 *. log (2.0 *. Float.pi) -. log sigma
      -. 0.5 *. ((x -. mu) /. sigma) ** 2.0 in
    check (float 1e-12) (Printf.sprintf "normal ld at x=%.1f" x) expected ld
  ) xs

let test_assess_simulate_roundtrip () =
  (* simulate then assess with same trace: return values should match *)
  let env = [("mu", mk_scalar 1.0); ("sigma", mk_scalar 2.0)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [||] dist in
  let (v_sim, trace, _) = Ast.Simulate.simulate ~run_key:77L env e in
  let (v_assess, _) = Ast.Assess.assess env e trace in
  check (float 1e-12) "roundtrip value"
    (scalar_val v_sim) (scalar_val v_assess)

let test_elbo_smoke () =
  (* ELBO = assess(model, t) - assess(guide, t)
     Simple model: z ~ Normal(0,1), score(z)
     Guide: z ~ Normal(1, 0.5) *)
  let model =
    let_ "z" (sample "z" [||]
      (Ast.Normal.normal ~mu:"m_mu" ~sigma:"m_sigma"))
      (let_ "_" (score (var "z"))
         (var "z")) in
  let guide =
    sample "z" [||]
      (Ast.Normal.normal ~mu:"g_mu" ~sigma:"g_sigma") in
  let model_env = [("m_mu", mk_scalar 0.0); ("m_sigma", mk_scalar 1.0)] in
  let guide_env = [("g_mu", mk_scalar 1.0); ("g_sigma", mk_scalar 0.5)] in
  (* Simulate from guide *)
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:123L guide_env guide in
  (* Assess both *)
  let (_, log_p) = Ast.Assess.assess model_env model trace in
  let (_, log_q) = Ast.Assess.assess guide_env guide trace in
  let elbo = scalar_val log_p -. scalar_val log_q in
  (* Just check it's a finite number — not testing convergence *)
  if Float.is_nan elbo || Float.is_infinite elbo then
    fail (Printf.sprintf "ELBO should be finite, got %g" elbo);
  check pass "ELBO smoke" () ()

(* ── 10-6: frame-based Sample ── *)

let test_simulate_frame () =
  let e = sample "z" [|5|] D_uniform in
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:42L [] e in
  let z = List.assoc "z" trace in
  check (array int) "frame shape" [|5|] z.view.View.Ndview.shape;
  (* All values in (0,1) *)
  for i = 0 to 4 do
    let v = View.Buf.get z.buf i in
    if v <= 0.0 || v >= 1.0 then
      fail (Printf.sprintf "z[%d] = %g not in (0,1)" i v)
  done

let test_frame_vs_scalar_loop () =
  (* frame version should match scalar loop *)
  let n = 8 in
  let e_frame = sample "z" [|n|] D_uniform in
  let (_, trace_f, _) = Ast.Simulate.simulate ~run_key:42L [] e_frame in
  let z_frame = List.assoc "z" trace_f in
  (* Scalar loop: simulate n times with same key, each with site_id=0 *)
  let key = Prng.Threefry.make_key ~run_key:42L
      ~namespace:Prng.Threefry.ns_model in
  for i = 0 to n - 1 do
    let ctr = Prng.Threefry.make_ctr ~site_id:0 ~component:1 ~frame_index:i in
    let (r0, _) = Prng.Threefry.threefry2x64 ~key ~ctr in
    let expected = Prng.Threefry.to_open_unit r0 in
    let actual = View.Buf.get z_frame.buf i in
    check (float 1e-15) (Printf.sprintf "frame[%d]" i) expected actual
  done

let test_frame_normal () =
  let env = [("mu", mk_scalar 0.0); ("sigma", mk_scalar 1.0)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [|100|] dist in
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:42L env e in
  let z = List.assoc "z" trace in
  check (array int) "frame normal shape" [|100|] z.view.View.Ndview.shape;
  (* Basic sanity: mean should be near 0 *)
  let sum = ref 0.0 in
  for i = 0 to 99 do
    sum := !sum +. View.Buf.get z.buf i
  done;
  let mean = !sum /. 100.0 in
  if Float.abs mean > 1.0 then
    fail (Printf.sprintf "mean = %g, expected near 0" mean)

let test_density_frame_sum () =
  (* log_density of frame sample = sum of individual log_densities *)
  let env = [("mu", mk_scalar 0.0); ("sigma", mk_scalar 1.0)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [|5|] dist in
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:42L env e in
  let z = List.assoc "z" trace in
  (* Assess the full frame *)
  let (_, ld_full) = Ast.Assess.assess env e trace in
  (* Sum of individual densities *)
  let sum_ld = ref 0.0 in
  for i = 0 to 4 do
    let xi = View.Buf.get z.buf i in
    let mu = 0.0 and sigma = 1.0 in
    let expected =
      -0.5 *. log (2.0 *. Float.pi) -. log sigma
      -. 0.5 *. ((xi -. mu) /. sigma) ** 2.0 in
    sum_ld := !sum_ld +. expected
  done;
  check (float 1e-12) "density frame sum" !sum_ld (scalar_val ld_full)

let test_frame_batch_instrumented () =
  (* Verify that frame=[|G|] D_pushforward evaluates fwd once, not G times.
     With G=100 Normal samples, the fwd expression has 5 prims
     (add, mul, mul, erfinv, sub, mul). Each should be called once. *)
  let env = [("mu", mk_scalar 0.0); ("sigma", mk_scalar 1.0)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let g = 100 in
  let e = sample "z" [|g|] dist in
  Ast.Eval.enable_stats ();
  Ast.Eval.reset_stats ();
  let _ = Ast.Simulate.simulate ~run_key:42L env e in
  (* Check erfinv is called exactly once (not G times) *)
  let erfinv_stats =
    Hashtbl.find_opt Ast.Eval.stats.kernel_stats "erfinv" in
  Ast.Eval.disable_stats ();
  (match erfinv_stats with
   | Some ps ->
     check int "erfinv calls" 1 ps.Ast.Eval.calls;
     check int "erfinv elems" g ps.Ast.Eval.total_elems
   | None -> fail "erfinv kernel not recorded")

(* ── Test suite ── *)

let () =
  run "Prob" [
    "AST", [
      test_case "pp sample"       `Quick test_pp_sample;
      test_case "pp score"        `Quick test_pp_score;
      test_case "check sites ok"  `Quick test_check_sites_ok;
      test_case "check sites dup" `Quick test_check_sites_dup;
    ];
    "simulate", [
      test_case "reproducibility"   `Quick test_simulate_reproducibility;
      test_case "different keys"    `Quick test_simulate_different_keys;
      test_case "score"             `Quick test_simulate_score;
      test_case "coupling"          `Quick test_simulate_coupling;
      test_case "categorical"       `Quick test_simulate_categorical;
    ];
    "erf/erfinv", [
      test_case "erf known"           `Quick test_erf_known;
      test_case "erfinv known"        `Quick test_erfinv_known;
      test_case "erfinv roundtrip"    `Quick test_erfinv_roundtrip;
      test_case "inv∘fwd identity"    `Quick test_inv_fwd_identity;
      test_case "endpoint stability"  `Quick test_erf_endpoint_stability;
      test_case "erf JVP FD"         `Quick test_erf_jvp_fd;
      test_case "erfinv JVP FD"      `Quick test_erfinv_jvp_fd;
    ];
    "assess", [
      test_case "uniform norm"      `Quick test_assess_uniform_normalization;
      test_case "normal norm"       `Quick test_assess_normal_normalization;
      test_case "categorical norm"  `Quick test_assess_categorical_normalization;
      test_case "normal closed"     `Quick test_assess_normal_closed_form;
      test_case "sim→assess"        `Quick test_assess_simulate_roundtrip;
      test_case "ELBO smoke"        `Quick test_elbo_smoke;
    ];
    "frame", [
      test_case "uniform frame"      `Quick test_simulate_frame;
      test_case "frame vs scalar"    `Quick test_frame_vs_scalar_loop;
      test_case "frame normal"       `Quick test_frame_normal;
      test_case "density frame sum"  `Quick test_density_frame_sum;
      test_case "batch instrumented" `Quick test_frame_batch_instrumented;
    ];
  ]
