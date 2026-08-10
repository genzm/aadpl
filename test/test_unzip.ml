open View
open Ast.Types

(* --- helpers --- *)

let tensor_of_list shape vals =
  let t = Tensor.make shape in
  List.iteri (fun i v -> Buf.set t.buf i v) vals;
  t

let tensor_get (t : Tensor.t) i =
  let s = t.view.Ndview.shape in
  let r = Array.length s in
  let idx = Array.make r 0 in
  let tmp = ref i in
  for k = r - 1 downto 0 do
    idx.(k) <- !tmp mod s.(k);
    tmp := !tmp / s.(k)
  done;
  Buf.get t.buf (Ndview.index_of t.view idx)

let numel (t : Tensor.t) = Ndview.numel t.view

let check_equal msg (a : Tensor.t) (b : Tensor.t) =
  let n = numel a in
  Alcotest.(check int) (msg ^ " numel") n (numel b);
  for i = 0 to n - 1 do
    let va = tensor_get a i in
    let vb = tensor_get b i in
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "%s[%d]" msg i) va vb
  done

(* Run forward + unzip, check:
   1. primal_bindings + primal_out eval = original eval
   2. all bindings + tangent_out eval = jvp tangent
   3. linearity check passes (implicit — unzip would fail otherwise)
   env: (name, primal, tangent) list *)
let check_unzip msg env expr =
  Transform.Forward.reset_gensym ();
  let fwd = Transform.Forward.forward expr in
  let seeds = List.map (fun (s, _, _) ->
    Transform.Forward.tangent_name s) env in
  let uz = Transform.Unzip.unzip fwd ~seeds in
  (* --- check 1: primal matches original eval --- *)
  let primal_env = List.map (fun (s, v, _) -> (s, v)) env in
  let p_full = Transform.Forward.wrap_bindings
    uz.primal_bindings uz.primal_out in
  let primal_unzip = Ast.Eval.eval primal_env p_full in
  let primal_ref = Ast.Eval.eval primal_env expr in
  check_equal (msg ^ " primal") primal_ref primal_unzip;
  (* --- check 2: tangent matches jvp_eval tangent --- *)
  let tang_env = List.map (fun (s, _, t) ->
    (Transform.Forward.tangent_name s, t)) env in
  let full_env = primal_env @ tang_env in
  let all_bs = uz.primal_bindings @ uz.tangent_bindings in
  let t_full = Transform.Forward.wrap_bindings all_bs uz.tangent_out in
  let tangent_unzip = Ast.Eval.eval full_env t_full in
  let dual_env = List.map (fun (s, v, t) -> (s, (v, t))) env in
  let (_, tangent_ref) = Ast.Jvp.jvp_eval dual_env expr in
  check_equal (msg ^ " tangent") tangent_ref tangent_unzip

(* === map1 === *)

let test_neg () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.5;(-1.);2.] in
  check_unzip "neg" [("x", x, dx)] (prim Neg [var "x"])

let test_exp () =
  let x = tensor_of_list [|3|] [0.1;0.5;1.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);0.3] in
  check_unzip "exp" [("x", x, dx)] (prim Exp [var "x"])

let test_log () =
  let x = tensor_of_list [|3|] [0.5;1.0;2.0] in
  let dx = tensor_of_list [|3|] [1.;0.5;(-0.3)] in
  check_unzip "log" [("x", x, dx)] (prim Log [var "x"])

let test_sqrt () =
  let x = tensor_of_list [|3|] [1.0;4.0;9.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);2.] in
  check_unzip "sqrt" [("x", x, dx)] (prim Sqrt [var "x"])

let test_relu () =
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  check_unzip "relu" [("x", x, dx)] (prim Relu [var "x"])

let test_step () =
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  check_unzip "step" [("x", x, dx)] (prim Step [var "x"])

(* === map2 === *)

let test_mul () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_unzip "mul" [("x", x, dx); ("y", y, dy)]
    (prim Mul [var "x"; var "y"])

let test_div () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_unzip "div" [("x", x, dx); ("y", y, dy)]
    (prim Div [var "x"; var "y"])

let test_max2 () =
  let x = tensor_of_list [|3|] [1.;5.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [2.;4.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_unzip "max2" [("x", x, dx); ("y", y, dy)]
    (prim Max2 [var "x"; var "y"])

(* === reduce === *)

let test_sum_axis () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_unzip "sum_axis" [("x", x, dx)] (prim (Sum_axis 1) [var "x"])

let test_max_axis () =
  let x = tensor_of_list [|2;3|] [1.;3.;2.;6.;4.;5.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_unzip "max_axis" [("x", x, dx)] (prim (Max_axis 1) [var "x"])

(* === structural === *)

let test_transpose () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_unzip "transpose" [("x", x, dx)]
    (prim (Apply_view [Vtranspose [|1;0|]]) [var "x"])

let test_broadcast () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  check_unzip "broadcast" [("x", x, dx)]
    (prim (Apply_view [Vbroadcast (0, 2)]) [var "x"])

(* === matmul === *)

let test_matmul () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let da = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  let b = tensor_of_list [|3;2|] [7.;8.;9.;10.;11.;12.] in
  let db = tensor_of_list [|3;2|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  check_unzip "matmul" [("a", a, da); ("b", b, db)]
    (prim Matmul [var "a"; var "b"])

(* === composed === *)

let test_exp_sum () =
  let x = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  let dx = tensor_of_list [|4|] [1.;0.;0.;0.] in
  check_unzip "exp_sum" [("x", x, dx)]
    (prim (Sum_axis 0) [prim Exp [var "x"]])

let test_matmul_relu () =
  let a = tensor_of_list [|2;3|] [0.1;(-0.2);0.3;0.4;0.5;(-0.6)] in
  let da = tensor_of_list [|2;3|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  let b = tensor_of_list [|3;2|] [1.;(-1.);2.;1.;(-1.);2.] in
  let db = tensor_of_list [|3;2|] [0.1;0.1;0.1;0.1;0.1;0.1] in
  check_unzip "matmul_relu" [("a", a, da); ("b", b, db)]
    (prim Relu [prim Matmul [var "a"; var "b"]])

let test_let_x_sq () =
  let a = tensor_of_list [|3|] [2.;3.;4.] in
  let da = tensor_of_list [|3|] [1.;1.;1.] in
  check_unzip "let_x_x*x" [("a", a, da)]
    (let_ "x" (var "a") (prim Mul [var "x"; var "x"]))

(* === linearity check: verify classification is correct === *)

let test_exp_classification () =
  (* For exp(x): forward produces:
     bindings: [(%p0, x); (%r1, exp(%p0))]
     tangent:  mul(%r1, %x.t)
     %r1 depends on %p0 which depends on x (not a seed).
     So %r1 should be in primal_bindings, not tangent. *)
  Transform.Forward.reset_gensym ();
  let x = tensor_of_list [|2|] [1.;2.] in
  let dx = tensor_of_list [|2|] [1.;1.] in
  let expr = prim Exp [var "x"] in
  let fwd = Transform.Forward.forward expr in
  let seeds = [Transform.Forward.tangent_name "x"] in
  let uz = Transform.Unzip.unzip fwd ~seeds in
  (* exp(%p0) should be in primal bindings (no tangent dependency) *)
  let has_exp_in_primal = List.exists (fun (_, rhs) ->
    match rhs with Prim (_, Exp, _) -> true | _ -> false
  ) uz.primal_bindings in
  Alcotest.(check bool) "exp in primal" true has_exp_in_primal;
  (* tangent bindings should contain only linear ops *)
  let has_exp_in_tangent = List.exists (fun (_, rhs) ->
    match rhs with Prim (_, Exp, _) -> true | _ -> false
  ) uz.tangent_bindings in
  Alcotest.(check bool) "no exp in tangent" false has_exp_in_tangent;
  (* and eval should still match *)
  check_unzip "exp_class" [("x", x, dx)] expr

let test_mul_classification () =
  (* For mul(x,y): forward produces tangent = add(mul(%x.t, y), mul(x, %y.t))
     The Mul nodes in tangent have one tangent-dep and one non-dep arg — linear *)
  Transform.Forward.reset_gensym ();
  let x = tensor_of_list [|2|] [1.;2.] in
  let dx = tensor_of_list [|2|] [0.1;0.2] in
  let y = tensor_of_list [|2|] [3.;4.] in
  let dy = tensor_of_list [|2|] [0.3;0.4] in
  check_unzip "mul_class" [("x", x, dx); ("y", y, dy)]
    (prim Mul [var "x"; var "y"])

(* === Dense layer through expand + forward + unzip === *)

let test_dense () =
  let xs = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let w = tensor_of_list [|3;2|] [1.;0.;0.;1.;1.;1.] in
  let bias = tensor_of_list [|2|] [0.1;0.2] in
  let dxs = Tensor.make [|2;3|] in
  let dw = tensor_of_list [|3;2|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  let dbias = Tensor.make [|2|] in
  let expr =
    let_ "xs" (const xs)
      (let_ "w" (const w)
        (let_ "bias" (const bias)
          (rank 1 Add [var "bias"; rank 2 Matmul [var "xs"; var "w"]]))) in
  let expanded = Transform.Expand_rank.expand expr in
  check_unzip "dense" [("xs", xs, dxs); ("w", w, dw); ("bias", bias, dbias)]
    expanded

let () =
  let open Alcotest in
  run "unzip" [
    "map1", [
      test_case "neg" `Quick test_neg;
      test_case "exp" `Quick test_exp;
      test_case "log" `Quick test_log;
      test_case "sqrt" `Quick test_sqrt;
      test_case "relu" `Quick test_relu;
      test_case "step" `Quick test_step;
    ];
    "map2", [
      test_case "mul" `Quick test_mul;
      test_case "div" `Quick test_div;
      test_case "max2" `Quick test_max2;
    ];
    "reduce", [
      test_case "sum_axis" `Quick test_sum_axis;
      test_case "max_axis" `Quick test_max_axis;
    ];
    "structural", [
      test_case "transpose" `Quick test_transpose;
      test_case "broadcast" `Quick test_broadcast;
    ];
    "linalg", [
      test_case "matmul" `Quick test_matmul;
    ];
    "composed", [
      test_case "exp then sum" `Quick test_exp_sum;
      test_case "matmul then relu" `Quick test_matmul_relu;
      test_case "let x = a in x*x" `Quick test_let_x_sq;
    ];
    "classification", [
      test_case "exp residual in primal" `Quick test_exp_classification;
      test_case "mul linearity" `Quick test_mul_classification;
    ];
    "integration", [
      test_case "dense (expand+forward+unzip)" `Quick test_dense;
    ];
  ]
