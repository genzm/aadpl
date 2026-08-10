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

(* Compare two tensors element-wise *)
let check_equal msg (a : Tensor.t) (b : Tensor.t) =
  let n = numel a in
  Alcotest.(check int) (msg ^ " numel") n (numel b);
  for i = 0 to n - 1 do
    let va = tensor_get a i in
    let vb = tensor_get b i in
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "%s[%d]" msg i) va vb
  done

(* Run forward transform and eval, compare against jvp_eval.
   env: (name, primal, tangent) list *)
let check_forward msg env expr =
  Transform.Forward.reset_gensym ();
  let (bs, p_expr, t_expr) = Transform.Forward.forward expr in
  (* Build eval env: primal bindings + tangent bindings *)
  let eval_env = List.map (fun (s, v, _) -> (s, v)) env in
  let tang_env = List.map (fun (s, _, t) ->
    (Transform.Forward.tangent_name s, t)) env in
  let full_env = eval_env @ tang_env in
  (* Wrap bindings, then eval *)
  let p_full = Transform.Forward.wrap_bindings bs p_expr in
  let t_full = Transform.Forward.wrap_bindings bs t_expr in
  let primal_fwd = Ast.Eval.eval full_env p_full in
  let tangent_fwd = Ast.Eval.eval full_env t_full in
  (* Reference: jvp_eval *)
  let dual_env = List.map (fun (s, v, t) -> (s, (v, t))) env in
  let (primal_ref, tangent_ref) = Ast.Jvp.jvp_eval dual_env expr in
  check_equal (msg ^ " primal") primal_ref primal_fwd;
  check_equal (msg ^ " tangent") tangent_ref tangent_fwd

(* === map1 tests === *)

let test_neg () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.5;(-1.);2.] in
  check_forward "neg" [("x", x, dx)] (prim Neg [var "x"])

let test_exp () =
  let x = tensor_of_list [|3|] [0.1;0.5;1.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);0.3] in
  check_forward "exp" [("x", x, dx)] (prim Exp [var "x"])

let test_log () =
  let x = tensor_of_list [|3|] [0.5;1.0;2.0] in
  let dx = tensor_of_list [|3|] [1.;0.5;(-0.3)] in
  check_forward "log" [("x", x, dx)] (prim Log [var "x"])

let test_sqrt () =
  let x = tensor_of_list [|3|] [1.0;4.0;9.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);2.] in
  check_forward "sqrt" [("x", x, dx)] (prim Sqrt [var "x"])

let test_relu () =
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  check_forward "relu" [("x", x, dx)] (prim Relu [var "x"])

let test_step () =
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  check_forward "step" [("x", x, dx)] (prim Step [var "x"])

(* === map2 tests === *)

let test_add () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_forward "add" [("x", x, dx); ("y", y, dy)]
    (prim Add [var "x"; var "y"])

let test_sub () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_forward "sub" [("x", x, dx); ("y", y, dy)]
    (prim Sub [var "x"; var "y"])

let test_mul () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_forward "mul" [("x", x, dx); ("y", y, dy)]
    (prim Mul [var "x"; var "y"])

let test_div () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_forward "div" [("x", x, dx); ("y", y, dy)]
    (prim Div [var "x"; var "y"])

let test_max2 () =
  let x = tensor_of_list [|3|] [1.;5.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [2.;4.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_forward "max2" [("x", x, dx); ("y", y, dy)]
    (prim Max2 [var "x"; var "y"])

(* === reduce === *)

let test_sum_axis () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_forward "sum_axis" [("x", x, dx)] (prim (Sum_axis 1) [var "x"])

let test_max_axis () =
  (* avoid ties *)
  let x = tensor_of_list [|2;3|] [1.;3.;2.;6.;4.;5.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_forward "max_axis" [("x", x, dx)] (prim (Max_axis 1) [var "x"])

(* === select_axis eval test === *)

let test_select_axis_eval () =
  (* select_axis(x, idx) along axis 0: x[2x3], idx[3] *)
  let x = tensor_of_list [|2;3|] [10.;20.;30.;40.;50.;60.] in
  let idx = tensor_of_list [|3|] [1.;0.;1.] in
  let result = Ast.Eval.eval [("x", x); ("i", idx)]
    (prim (Select_axis 0) [var "x"; var "i"]) in
  (* idx=[1,0,1] selects row 1,0,1 for columns 0,1,2 → [40, 20, 60] *)
  Alcotest.(check (float 1e-10)) "sel[0]" 40.0 (tensor_get result 0);
  Alcotest.(check (float 1e-10)) "sel[1]" 20.0 (tensor_get result 1);
  Alcotest.(check (float 1e-10)) "sel[2]" 60.0 (tensor_get result 2)

(* === argmax_axis eval test === *)

let test_argmax_axis_eval () =
  let x = tensor_of_list [|2;3|] [1.;3.;2.;6.;4.;5.] in
  let result = Ast.Eval.eval [("x", x)]
    (prim (Argmax_axis 1) [var "x"]) in
  (* row 0: max at col 1; row 1: max at col 0 *)
  Alcotest.(check (float 1e-10)) "am[0]" 1.0 (tensor_get result 0);
  Alcotest.(check (float 1e-10)) "am[1]" 0.0 (tensor_get result 1)

(* === structural === *)

let test_transpose () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_forward "transpose" [("x", x, dx)]
    (prim (Apply_view [Vtranspose [|1;0|]]) [var "x"])

let test_broadcast () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  check_forward "broadcast" [("x", x, dx)]
    (prim (Apply_view [Vbroadcast (0, 2)]) [var "x"])

let test_slice () =
  let x = tensor_of_list [|4|] [1.;2.;3.;4.] in
  let dx = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  check_forward "slice" [("x", x, dx)] (prim (Apply_view [Vslice [|(1,3,1)|]]) [var "x"])

let test_reshape () =
  let x = tensor_of_list [|6|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|6|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_forward "reshape" [("x", x, dx)] (prim (Apply_view [Vreshape [|2;3|]]) [var "x"])

let test_gather () =
  let x = tensor_of_list [|5|] [10.;20.;30.;40.;50.] in
  let dx = tensor_of_list [|5|] [0.1;0.2;0.3;0.4;0.5] in
  check_forward "gather" [("x", x, dx)]
    (prim (Gather (0, [|1;3;0|])) [var "x"])

(* === matmul === *)

let test_matmul () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let da = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  let b = tensor_of_list [|3;2|] [7.;8.;9.;10.;11.;12.] in
  let db = tensor_of_list [|3;2|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  check_forward "matmul" [("a", a, da); ("b", b, db)]
    (prim Matmul [var "a"; var "b"])

(* === composed === *)

let test_exp_sum () =
  let x = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  let dx = tensor_of_list [|4|] [1.;0.;0.;0.] in
  check_forward "exp_sum" [("x", x, dx)]
    (prim (Sum_axis 0) [prim Exp [var "x"]])

let test_matmul_relu () =
  let a = tensor_of_list [|2;3|] [0.1;(-0.2);0.3;0.4;0.5;(-0.6)] in
  let da = tensor_of_list [|2;3|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  let b = tensor_of_list [|3;2|] [1.;(-1.);2.;1.;(-1.);2.] in
  let db = tensor_of_list [|3;2|] [0.1;0.1;0.1;0.1;0.1;0.1] in
  check_forward "matmul_relu" [("a", a, da); ("b", b, db)]
    (prim Relu [prim Matmul [var "a"; var "b"]])

let test_let_binding () =
  let a = tensor_of_list [|3|] [2.;3.;4.] in
  let da = tensor_of_list [|3|] [1.;1.;1.] in
  check_forward "let_x_x*x" [("a", a, da)]
    (let_ "x" (var "a") (prim Mul [var "x"; var "x"]))

let () =
  let open Alcotest in
  run "forward" [
    "map1", [
      test_case "neg" `Quick test_neg;
      test_case "exp" `Quick test_exp;
      test_case "log" `Quick test_log;
      test_case "sqrt" `Quick test_sqrt;
      test_case "relu" `Quick test_relu;
      test_case "step" `Quick test_step;
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
    "new_prims", [
      test_case "select_axis eval" `Quick test_select_axis_eval;
      test_case "argmax_axis eval" `Quick test_argmax_axis_eval;
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
      test_case "exp then sum" `Quick test_exp_sum;
      test_case "matmul then relu" `Quick test_matmul_relu;
      test_case "let x = a in x*x" `Quick test_let_binding;
    ];
  ]
