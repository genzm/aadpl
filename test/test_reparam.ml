open Alcotest
open Ast.Types

let mk_scalar f =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 f; t

let scalar_val (t : View.Tensor.t) = View.Buf.get t.buf 0

(* ── is_reparammed ── *)

let test_is_reparammed_uniform () =
  let e = sample "z" [||] D_uniform in
  check bool "uniform" true (Transform.Reparam.is_reparammed e)

let test_is_reparammed_normal () =
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [||] dist in
  check bool "before reparam" false (Transform.Reparam.is_reparammed e);
  let e' = Transform.Reparam.reparam e in
  check bool "after reparam" true (Transform.Reparam.is_reparammed e')

(* ── reparam output structure ── *)

let test_reparam_uniform_unchanged () =
  let e = sample "z" [||] D_uniform in
  let e' = Transform.Reparam.reparam e in
  (* Should remain a Sample with D_uniform *)
  (match e' with
   | Sample (_, "z", _, D_uniform) -> check pass "unchanged" () ()
   | _ -> fail "expected Sample with D_uniform")

let test_reparam_normal_structure () =
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [||] dist in
  let e' = Transform.Reparam.reparam e in
  (* Should be Let(%u.z, Sample(z, [], D_uniform), fwd_body) *)
  (match e' with
   | Let (_, "%u.z", Sample (_, "z", _, D_uniform), _body) ->
     check pass "structure" () ()
   | _ ->
     let s = Format.asprintf "%a" pp e' in
     fail (Printf.sprintf "unexpected structure: %s" s))

(* ── Bit-exact coupling: simulate(e) vs eval(reparam(e)) ── *)

let test_coupling_scalar () =
  (* Normal(μ,σ): simulate should produce the same z as eval(reparam(e)) *)
  let mu = mk_scalar 2.0 in
  let sigma = mk_scalar 0.5 in
  let env = [("mu", mu); ("sigma", sigma)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [||] dist in
  (* simulate path *)
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:42L env e in
  let z_sim = scalar_val (List.assoc "z" trace) in
  (* reparam + expand + eval path *)
  let e_r = Transform.Reparam.reparam e in
  let senv = [("mu", [||]); ("sigma", [||])] in
  let e_re = Transform.Expand_rank.expand ~senv e_r in
  (* eval needs the Sample to be handled — use simulate on reparammed expr *)
  let (v_reparam, _, _) = Ast.Simulate.simulate ~run_key:42L env e_re in
  let z_reparam = scalar_val v_reparam in
  (* Bit-exact: = comparison, no tolerance *)
  if z_sim <> z_reparam then
    fail (Printf.sprintf "coupling: sim=%g reparam=%g" z_sim z_reparam);
  check pass "scalar coupling" () ()

let test_coupling_frame () =
  let mu = mk_scalar 0.0 in
  let sigma = mk_scalar 1.0 in
  let env = [("mu", mu); ("sigma", sigma)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let g = 10 in
  let e = sample "z" [|g|] dist in
  (* simulate path: return value = fwd(u) = Normal samples *)
  let (v_sim, _, _) = Ast.Simulate.simulate ~run_key:77L env e in
  (* reparam path: return value = fwd expression applied to u *)
  let e_r = Transform.Reparam.reparam e in
  let senv = [("mu", [||]); ("sigma", [||])] in
  let e_re = Transform.Expand_rank.expand ~senv e_r in
  let (v_reparam, _, _) = Ast.Simulate.simulate ~run_key:77L env e_re in
  for i = 0 to g - 1 do
    let a = View.Buf.get v_sim.buf i in
    let b = View.Buf.get v_reparam.buf i in
    if a <> b then
      fail (Printf.sprintf "coupling[%d]: sim=%g reparam=%g" i a b)
  done;
  check pass "frame coupling" () ()

(* ── Density consistency: assess with reparam'd trace ── *)

let test_density_consistency () =
  let mu = mk_scalar 1.0 in
  let sigma = mk_scalar 2.0 in
  let env = [("mu", mu); ("sigma", sigma)] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [||] dist in
  (* Get z from reparam path *)
  let e_r = Transform.Reparam.reparam e in
  let senv = [("mu", [||]); ("sigma", [||])] in
  let e_re = Transform.Expand_rank.expand ~senv e_r in
  let (_, trace_r, _) = Ast.Simulate.simulate ~run_key:55L env e_re in
  let z = List.assoc "z" trace_r in
  (* assess with original (pre-reparam) expression *)
  let trace_z = [("z", z)] in
  let (_, ld) = Ast.Assess.assess env e trace_z in
  (* Compare with closed form *)
  let zv = scalar_val z in
  let expected =
    -0.5 *. log (2.0 *. Float.pi) -. log 2.0
    -. 0.5 *. ((zv -. 1.0) /. 2.0) ** 2.0 in
  check (float 1e-12) "density consistency" expected (scalar_val ld)

(* ── reparam preserves value: the return value matches ── *)

let test_reparam_return_value () =
  (* Build: let z = Sample(Normal) in z*z + 1 *)
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = let_ "z" (sample "z" [||] dist)
            (prim Add [prim Mul [var "z"; var "z"]; const (mk_scalar 1.0)]) in
  let env = [("mu", mk_scalar 0.0); ("sigma", mk_scalar 1.0)] in
  (* simulate original *)
  let (v_orig, _, _) = Ast.Simulate.simulate ~run_key:33L env e in
  (* reparam then simulate *)
  let e_r = Transform.Reparam.reparam e in
  let senv = [("mu", [||]); ("sigma", [||])] in
  let e_re = Transform.Expand_rank.expand ~senv e_r in
  let (v_reparam, _, _) = Ast.Simulate.simulate ~run_key:33L env e_re in
  if scalar_val v_orig <> scalar_val v_reparam then
    fail (Printf.sprintf "return: orig=%g reparam=%g"
            (scalar_val v_orig) (scalar_val v_reparam));
  check pass "return value" () ()

(* ── Categorical unchanged ── *)

let test_reparam_categorical_unchanged () =
  let w = View.Tensor.make [|3|] in
  View.Buf.set w.buf 0 1.0; View.Buf.set w.buf 1 2.0; View.Buf.set w.buf 2 3.0;
  let e = sample "k" [||] (D_categorical (const w)) in
  let e' = Transform.Reparam.reparam e in
  check bool "categorical reparammed" true (Transform.Reparam.is_reparammed e');
  (match e' with
   | Sample (_, "k", _, D_categorical _) -> check pass "unchanged" () ()
   | _ -> fail "expected Sample with D_categorical")

(* ── assess_expr: symbolic density matches value-level assess ── *)

let test_assess_expr_scalar () =
  let mu = mk_scalar 1.0 in
  let sigma = mk_scalar 2.0 in
  let env = [("mu", mu); ("sigma", sigma)] in
  let env_shapes = [("mu", [||]); ("sigma", [||])] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = sample "z" [||] dist in
  (* Get z value from simulate *)
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:42L env e in
  let z = List.assoc "z" trace in
  (* Value-level assess *)
  let (_, ld_val) = Ast.Assess.assess env e [("z", z)] in
  (* Expression-level assess *)
  let slots = [("z", const z)] in
  let ld_expr = Transform.Assess_expr.assess_expr ~env_shapes e slots in
  let ld_sym = Ast.Eval.eval env ld_expr in
  check (float 1e-12) "scalar assess_expr" (scalar_val ld_val) (scalar_val ld_sym)

let test_assess_expr_frame () =
  let mu = mk_scalar 0.0 in
  let sigma = mk_scalar 1.0 in
  let env = [("mu", mu); ("sigma", sigma)] in
  let env_shapes = [("mu", [||]); ("sigma", [||])] in
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let g = 10 in
  let e = sample "z" [|g|] dist in
  (* Get z value from simulate *)
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:77L env e in
  let z = List.assoc "z" trace in
  (* Value-level assess *)
  let (_, ld_val) = Ast.Assess.assess env e [("z", z)] in
  (* Expression-level assess *)
  let slots = [("z", const z)] in
  let ld_expr = Transform.Assess_expr.assess_expr ~env_shapes e slots in
  let ld_sym = Ast.Eval.eval env ld_expr in
  check (float 1e-12) "frame assess_expr" (scalar_val ld_val) (scalar_val ld_sym)

let test_assess_expr_with_score () =
  (* model: let z = Sample(Normal) in Score(z); z *)
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = let_ "z" (sample "z" [||] dist) (let_ "_s" (score (var "z")) (var "z")) in
  let mu = mk_scalar 0.0 in
  let sigma = mk_scalar 1.0 in
  let env = [("mu", mu); ("sigma", sigma)] in
  let env_shapes = [("mu", [||]); ("sigma", [||])] in
  let (_, trace, _) = Ast.Simulate.simulate ~run_key:99L env e in
  let z = List.assoc "z" trace in
  (* Value-level assess *)
  let (_, ld_val) = Ast.Assess.assess env e [("z", z)] in
  (* Expression-level assess *)
  let slots = [("z", const z)] in
  let ld_expr = Transform.Assess_expr.assess_expr ~env_shapes e slots in
  let ld_sym = Ast.Eval.eval env ld_expr in
  check (float 1e-12) "score assess_expr" (scalar_val ld_val) (scalar_val ld_sym)

(* ── discrete seed prohibition ── *)

(* Simple substring check without Str dependency *)
let contains s sub =
  let len_s = String.length s and len_sub = String.length sub in
  if len_sub > len_s then false
  else
    let rec check i =
      if i > len_s - len_sub then false
      else if String.sub s i len_sub = sub then true
      else check (i + 1)
    in check 0

let test_discrete_in_grad () =
  (* D_categorical Sample passed directly to grad → Grad_error with loc *)
  let w = View.Tensor.make [|3|] in
  View.Buf.set w.buf 0 1.0; View.Buf.set w.buf 1 2.0; View.Buf.set w.buf 2 3.0;
  let e = let_ "k" (sample "k" [||] (D_categorical (const w))) (var "k") in
  let raised = ref false in
  (try
     let _ = Transform.grad ~param_shapes:[] e in ()
   with Transform.Grad_error (_, msg) ->
     raised := true;
     check bool "mentions discrete" true (contains msg "discrete");
     check bool "mentions site name" true (contains msg "'k'"));
  check bool "Grad_error raised" true !raised

let test_continuous_sample_in_grad () =
  (* Unreparammed continuous Sample → Grad_error mentioning reparam *)
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let e = let_ "z" (sample "z" [||] dist) (var "z") in
  let raised = ref false in
  (try
     let _ = Transform.grad ~param_shapes:[("mu", [||]); ("sigma", [||])] e in ()
   with Transform.Grad_error (_, msg) ->
     raised := true;
     check bool "mentions reparam" true (contains msg "reparam"));
  check bool "Grad_error raised" true !raised

let test_score_in_grad () =
  (* Score passed to grad → Grad_error mentioning assess_expr *)
  let e = let_ "_s" (score (var "x")) (var "x") in
  let raised = ref false in
  (try
     let _ = Transform.grad ~param_shapes:[("x", [||])] e in ()
   with Transform.Grad_error (_, msg) ->
     raised := true;
     check bool "mentions assess_expr" true (contains msg "assess_expr"));
  check bool "Grad_error raised" true !raised

(* ── trace compatibility ── *)

let test_trace_compat_ok () =
  (* Matching sites → no exception *)
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let model = let_ "z" (sample "z" [|10|] dist) (var "z") in
  let guide = let_ "z" (sample "z" [|10|] dist) (var "z") in
  check pass "no exception"
    () (Transform.Reparam.check_trace_compat ~model ~guide)

let test_trace_compat_missing_site () =
  (* Guide has site not in model → Trace_mismatch *)
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let model = let_ "z" (sample "z" [||] dist) (var "z") in
  let guide = let_ "w" (sample "w" [||] dist) (var "w") in
  let raised = ref false in
  (try
     Transform.Reparam.check_trace_compat ~model ~guide
   with Transform.Reparam.Trace_mismatch msg ->
     raised := true;
     check bool "mentions site" true (contains msg "'w'"));
  check bool "Trace_mismatch raised" true !raised

let test_trace_compat_frame_mismatch () =
  (* Same site name, different frames → Trace_mismatch *)
  let dist = Ast.Normal.normal ~mu:"mu" ~sigma:"sigma" in
  let model = let_ "z" (sample "z" [|10|] dist) (var "z") in
  let guide = let_ "z" (sample "z" [|5|] dist) (var "z") in
  let raised = ref false in
  (try
     Transform.Reparam.check_trace_compat ~model ~guide
   with Transform.Reparam.Trace_mismatch msg ->
     raised := true;
     check bool "mentions frame mismatch" true (contains msg "frame mismatch"));
  check bool "Trace_mismatch raised" true !raised

(* ── Test suite ── *)

let () =
  run "Reparam" [
    "postcondition", [
      test_case "uniform"         `Quick test_is_reparammed_uniform;
      test_case "normal"          `Quick test_is_reparammed_normal;
    ];
    "structure", [
      test_case "uniform noop"    `Quick test_reparam_uniform_unchanged;
      test_case "normal"          `Quick test_reparam_normal_structure;
      test_case "categorical"     `Quick test_reparam_categorical_unchanged;
    ];
    "coupling", [
      test_case "scalar"          `Quick test_coupling_scalar;
      test_case "frame"           `Quick test_coupling_frame;
    ];
    "consistency", [
      test_case "density"         `Quick test_density_consistency;
      test_case "return value"    `Quick test_reparam_return_value;
    ];
    "assess_expr", [
      test_case "scalar"          `Quick test_assess_expr_scalar;
      test_case "frame"           `Quick test_assess_expr_frame;
      test_case "with score"      `Quick test_assess_expr_with_score;
    ];
    "discrete_guard", [
      test_case "categorical"     `Quick test_discrete_in_grad;
      test_case "continuous"      `Quick test_continuous_sample_in_grad;
      test_case "score"           `Quick test_score_in_grad;
    ];
    "trace_compat", [
      test_case "matching"        `Quick test_trace_compat_ok;
      test_case "missing site"    `Quick test_trace_compat_missing_site;
      test_case "frame mismatch"  `Quick test_trace_compat_frame_mismatch;
    ];
  ]
