open View
open Ast.Types

(* --- helpers --- *)

let tensor_of_list shape vals =
  let t = Tensor.make shape in
  List.iteri (fun i v -> Buf.set t.buf i v) vals;
  t

let numel (t : Tensor.t) = Ndview.numel t.view

(* Finite-difference verification:
   tangent ≈ (f(x + ε*v) - f(x)) / ε
   where v is the tangent direction.

   For multi-input functions, we perturb all inputs simultaneously
   along their respective tangent directions. *)

let eps = 1e-7

(* Read element i from a tensor, respecting its view *)
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

(* Perturb tensor: x + ε * dx *)
let perturb (x : Tensor.t) (dx : Tensor.t) : Tensor.t =
  let s = x.view.Ndview.shape in
  let n = Ndview.numel x.view in
  let r = Tensor.make s in
  for i = 0 to n - 1 do
    Buf.set r.buf i (tensor_get x i +. eps *. tensor_get dx i)
  done;
  r

(* Compare tangent with finite difference, element-wise *)
let check_fd msg (tangent : Tensor.t) (f_x : Tensor.t) (f_xpev : Tensor.t) =
  let n = numel tangent in
  for i = 0 to n - 1 do
    let t = tensor_get tangent i in
    let fd = (tensor_get f_xpev i -. tensor_get f_x i) /. eps in
    let scale = max 1.0 (max (abs_float t) (abs_float fd)) in
    let rel = abs_float (t -. fd) /. scale in
    Alcotest.(check bool)
      (Printf.sprintf "%s[%d]: tangent=%.6g fd=%.6g rel=%.2e" msg i t fd rel)
      true (rel < 1e-5)
  done

(* --- single-input FD check --- *)

let check_jvp_1 msg expr_fn x dx =
  let (_, tangent) =
    Ast.Jvp.jvp_eval [("x", (x, dx))] (expr_fn (var "x")) in
  let f_x = Ast.Eval.eval [("x", x)] (expr_fn (var "x")) in
  let x_pert = perturb x dx in
  let f_xpev = Ast.Eval.eval [("x", x_pert)] (expr_fn (var "x")) in
  check_fd msg tangent f_x f_xpev

(* --- two-input FD check --- *)

let check_jvp_2 msg expr_fn x dx y dy =
  let (_, tangent) =
    Ast.Jvp.jvp_eval [("x", (x, dx)); ("y", (y, dy))]
      (expr_fn (var "x") (var "y")) in
  let f_xy = Ast.Eval.eval [("x", x); ("y", y)]
    (expr_fn (var "x") (var "y")) in
  let x_pert = perturb x dx in
  let y_pert = perturb y dy in
  let f_pert = Ast.Eval.eval [("x", x_pert); ("y", y_pert)]
    (expr_fn (var "x") (var "y")) in
  check_fd msg tangent f_xy f_pert

(* === map1 tests === *)

let test_neg () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.5;(-1.);2.] in
  check_jvp_1 "neg" (fun v -> prim Neg [v]) x dx

let test_exp () =
  let x = tensor_of_list [|3|] [0.1;0.5;1.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);0.3] in
  check_jvp_1 "exp" (fun v -> prim Exp [v]) x dx

let test_log () =
  let x = tensor_of_list [|3|] [0.5;1.0;2.0] in
  let dx = tensor_of_list [|3|] [1.;0.5;(-0.3)] in
  check_jvp_1 "log" (fun v -> prim Log [v]) x dx

let test_sqrt () =
  let x = tensor_of_list [|3|] [1.0;4.0;9.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);2.] in
  check_jvp_1 "sqrt" (fun v -> prim Sqrt [v]) x dx

let test_relu () =
  (* avoid x=0 where relu is non-differentiable *)
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  check_jvp_1 "relu" (fun v -> prim Relu [v]) x dx

let test_step () =
  (* step is piecewise constant — tangent must be zero everywhere *)
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  let (primal, tangent) =
    Ast.Jvp.jvp_eval [("x", (x, dx))] (prim Step [var "x"]) in
  (* primal: [0, 1, 1, 0] *)
  Alcotest.(check (float 1e-10)) "p[0]" 0.0 (tensor_get primal 0);
  Alcotest.(check (float 1e-10)) "p[1]" 1.0 (tensor_get primal 1);
  Alcotest.(check (float 1e-10)) "p[2]" 1.0 (tensor_get primal 2);
  Alcotest.(check (float 1e-10)) "p[3]" 0.0 (tensor_get primal 3);
  (* tangent: all zero *)
  for i = 0 to 3 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "t[%d]" i) 0.0 (tensor_get tangent i)
  done

(* === map2 tests === *)

let test_add () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_jvp_2 "add" (fun a b -> prim Add [a; b]) x dx y dy

let test_sub () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_jvp_2 "sub" (fun a b -> prim Sub [a; b]) x dx y dy

let test_mul () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_jvp_2 "mul" (fun a b -> prim Mul [a; b]) x dx y dy

let test_div () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_jvp_2 "div" (fun a b -> prim Div [a; b]) x dx y dy

let test_max2 () =
  (* avoid ties where max2 is non-differentiable *)
  let x = tensor_of_list [|3|] [1.;5.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [2.;4.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_jvp_2 "max2" (fun a b -> prim Max2 [a; b]) x dx y dy

(* === reduce tests === *)

let test_sum_axis () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_jvp_1 "sum_axis" (fun v -> prim (Sum_axis 1) [v]) x dx

let test_max_axis () =
  (* avoid ties *)
  let x = tensor_of_list [|2;3|] [1.;3.;2.;6.;4.;5.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_jvp_1 "max_axis" (fun v -> prim (Max_axis 1) [v]) x dx

(* === structural tests (linear — tangent = same op) === *)

let test_transpose () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_jvp_1 "transpose" (fun v -> prim (Transpose [|1;0|]) [v]) x dx

let test_broadcast () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  check_jvp_1 "broadcast" (fun v -> prim (Broadcast (0, 2)) [v]) x dx

let test_slice () =
  let x = tensor_of_list [|4|] [1.;2.;3.;4.] in
  let dx = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  check_jvp_1 "slice" (fun v -> prim (Slice [|(1,3,1)|]) [v]) x dx

let test_reshape () =
  let x = tensor_of_list [|6|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|6|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_jvp_1 "reshape" (fun v -> prim (Reshape [|2;3|]) [v]) x dx

let test_gather () =
  let x = tensor_of_list [|5|] [10.;20.;30.;40.;50.] in
  let dx = tensor_of_list [|5|] [0.1;0.2;0.3;0.4;0.5] in
  check_jvp_1 "gather" (fun v -> prim (Gather (0, [|1;3;0|])) [v]) x dx

(* === matmul === *)

let test_matmul () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let da = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  let b = tensor_of_list [|3;2|] [7.;8.;9.;10.;11.;12.] in
  let db = tensor_of_list [|3;2|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  check_jvp_2 "matmul" (fun a b -> prim Matmul [a; b]) a da b db

(* === composed expressions === *)

let test_composed_exp_sum () =
  (* sum(exp(x)) — chain rule *)
  let x = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  let dx = tensor_of_list [|4|] [1.;0.;0.;0.] in
  check_jvp_1 "exp_then_sum"
    (fun v -> prim (Sum_axis 0) [prim Exp [v]]) x dx

let test_composed_matmul_relu () =
  (* relu(A @ B) *)
  let a = tensor_of_list [|2;3|] [0.1;(-0.2);0.3;0.4;0.5;(-0.6)] in
  let da = tensor_of_list [|2;3|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  let b = tensor_of_list [|3;2|] [1.;(-1.);2.;1.;(-1.);2.] in
  let db = tensor_of_list [|3;2|] [0.1;0.1;0.1;0.1;0.1;0.1] in
  check_jvp_2 "matmul_relu"
    (fun a b -> prim Relu [prim Matmul [a; b]]) a da b db

let test_let_binding () =
  (* let x = a in x * x  (square, derivative = 2*a*da) *)
  let a = tensor_of_list [|3|] [2.;3.;4.] in
  let da = tensor_of_list [|3|] [1.;1.;1.] in
  let (_, tangent) =
    Ast.Jvp.jvp_eval [("a", (a, da))]
      (let_ "x" (var "a") (prim Mul [var "x"; var "x"])) in
  (* d(x^2) = 2*x*dx = 2*[2,3,4]*[1,1,1] = [4,6,8] *)
  Alcotest.(check (float 1e-10)) "t[0]" 4.0 (tensor_get tangent 0);
  Alcotest.(check (float 1e-10)) "t[1]" 6.0 (tensor_get tangent 1);
  Alcotest.(check (float 1e-10)) "t[2]" 8.0 (tensor_get tangent 2)

(* === Dense layer (matmul + bias) through expand + jvp === *)

let test_dense_jvp () =
  let xs = tensor_of_list [|2;2;3|]
    [1.;2.;3.; 4.;5.;6.; 7.;8.;9.; 10.;11.;12.] in
  let w = tensor_of_list [|3;2|] [1.;0.; 0.;1.; 1.;1.] in
  let bias = tensor_of_list [|2|] [0.1; 0.2] in
  let dxs = Tensor.make [|2;2;3|] in  (* zero tangent for xs *)
  let dw = tensor_of_list [|3;2|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  let dbias = Tensor.make [|2|] in    (* zero tangent for bias *)
  (* Dense = add⎉1 [bias; matmul⎉2 [xs; w]]
     Use let bindings so expand can infer shapes from Const *)
  let expr =
    let_ "xs" (const xs)
      (let_ "w" (const w)
        (let_ "bias" (const bias)
          (rank 1 Add [var "bias"; rank 2 Matmul [var "xs"; var "w"]]))) in
  let expanded = Transform.Expand_rank.expand expr in
  let (primal, tangent) =
    Ast.Jvp.jvp_eval
      [("xs", (xs, dxs)); ("w", (w, dw)); ("bias", (bias, dbias))]
      expanded in
  (* Verify primal matches eval *)
  let primal_ref = Ast.Eval.eval
    [("xs", xs); ("w", w); ("bias", bias)] expanded in
  let n = numel primal in
  for i = 0 to n - 1 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "primal[%d]" i)
      (tensor_get primal_ref i) (tensor_get primal i)
  done;
  (* FD check: perturb w only *)
  let w_pert = perturb w dw in
  let result_pert = Ast.Eval.eval
    [("xs", xs); ("w", w_pert); ("bias", bias)] expanded in
  check_fd "dense_jvp" tangent primal_ref result_pert

let () =
  let open Alcotest in
  run "jvp" [
    "map1", [
      test_case "neg" `Quick test_neg;
      test_case "exp" `Quick test_exp;
      test_case "log" `Quick test_log;
      test_case "sqrt" `Quick test_sqrt;
      test_case "relu" `Quick test_relu;
      test_case "step (zero tangent)" `Quick test_step;
    ];
    "map2", [
      test_case "add" `Quick test_add;
      test_case "sub" `Quick test_sub;
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
      test_case "slice" `Quick test_slice;
      test_case "reshape" `Quick test_reshape;
      test_case "gather" `Quick test_gather;
    ];
    "linalg", [
      test_case "matmul" `Quick test_matmul;
    ];
    "composed", [
      test_case "exp then sum" `Quick test_composed_exp_sum;
      test_case "matmul then relu" `Quick test_composed_matmul_relu;
      test_case "let x = a in x*x" `Quick test_let_binding;
      test_case "Dense layer (expand + jvp)" `Quick test_dense_jvp;
    ];
  ]
