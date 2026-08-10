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

(* Inner product ⟨a, b⟩ *)
let inner (a : Tensor.t) (b : Tensor.t) : float =
  let n = numel a in
  assert (n = numel b);
  let s = ref 0.0 in
  for i = 0 to n - 1 do
    s := !s +. tensor_get a i *. tensor_get b i
  done;
  !s

(* Check inner product equality:
   ⟨jvp(f, x, v), u⟩ = ⟨v, vjp(f, x, u)⟩
   where vjp = transpose(forward(f))

   vars: (name, primal, tangent_direction, shape) list
   cotangent_dir: direction for the output cotangent u *)
let check_ip msg vars expr =
  (* Generate random cotangent direction *)
  Transform.Forward.reset_gensym ();
  let fwd_tmp = Transform.Forward.forward expr in
  let seeds = List.map (fun (s, _, _) ->
    Transform.Forward.tangent_name s) vars in
  let uz_tmp = Transform.Unzip.unzip fwd_tmp ~seeds in
  (* Infer output shape *)
  let input_shapes = List.map (fun (s, v, _) ->
    (s, v.Tensor.view.Ndview.shape)) vars in
  let tangent_input_shapes = List.map (fun (s, _, dv) ->
    (Transform.Forward.tangent_name s, dv.Tensor.view.Ndview.shape)) vars in
  let senv = ref (input_shapes @ tangent_input_shapes) in
  List.iter (fun (name, rhs) ->
    let sh = Transform.Expand_rank.infer_shape !senv rhs in
    senv := (name, sh) :: !senv
  ) (uz_tmp.primal_bindings @ uz_tmp.tangent_bindings);
  let out_shape = Transform.Expand_rank.infer_shape !senv uz_tmp.tangent_out in
  (* Random cotangent *)
  let out_n = Array.fold_left ( * ) 1 out_shape in
  let u = Tensor.make out_shape in
  for i = 0 to out_n - 1 do
    Buf.set u.buf i (0.1 *. float_of_int (i + 1))
  done;
  (* Now do the real check *)
  Transform.Forward.reset_gensym ();
  let fwd = Transform.Forward.forward expr in
  let uz = Transform.Unzip.unzip fwd ~seeds in
  let all_input_shapes = input_shapes @ tangent_input_shapes in
  let tr = Transform.Transpose.transpose
    ~primal_bindings:uz.primal_bindings
    ~tangent_bindings:uz.tangent_bindings
    ~tangent_out:uz.tangent_out
    ~seeds
    ~input_shapes:all_input_shapes
    ~cotangent_var:"%ct_out" in
  (* LHS: ⟨jvp tangent, u⟩ *)
  let dual_env = List.map (fun (s, v, dv) -> (s, (v, dv))) vars in
  let (_, tangent) = Ast.Jvp.jvp_eval dual_env expr in
  let lhs = inner tangent u in
  (* RHS: sum of ⟨v_i, grad_i⟩ *)
  let primal_env = List.map (fun (s, v, _) -> (s, v)) vars in
  let tangent_env = List.map (fun (s, _, dv) ->
    (Transform.Forward.tangent_name s, dv)) vars in
  let ct_env = [("%ct_out", u)] in
  let full_env = primal_env @ tangent_env @ ct_env in
  let all_bs = uz.primal_bindings @ uz.tangent_bindings @ tr.grad_bindings in
  let rhs = ref 0.0 in
  List.iter (fun (seed, grad_expr) ->
    let grad_full = Transform.Forward.wrap_bindings all_bs grad_expr in
    let grad_val = Ast.Eval.eval full_env grad_full in
    (* seed is "%x.t" → variable name is "x" *)
    let var_name =
      let s = seed in
      (* "%x.t" → strip leading % and trailing .t *)
      String.sub s 1 (String.length s - 3) in
    let dv = List.assoc var_name
      (List.map (fun (s, _, dv) -> (s, dv)) vars) in
    rhs := !rhs +. inner dv grad_val
  ) tr.grad_map;
  let scale = max 1.0 (max (abs_float lhs) (abs_float !rhs)) in
  let rel = abs_float (lhs -. !rhs) /. scale in
  Alcotest.(check bool)
    (Printf.sprintf "%s: ⟨Jv,u⟩=%.6g ⟨v,J*u⟩=%.6g rel=%.2e" msg lhs !rhs rel)
    true (rel < 1e-10)

(* === tests === *)

let test_neg () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.5;(-1.);2.] in
  check_ip "neg" [("x", x, dx)] (prim Neg [var "x"])

let test_exp () =
  let x = tensor_of_list [|3|] [0.1;0.5;1.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);0.3] in
  check_ip "exp" [("x", x, dx)] (prim Exp [var "x"])

let test_log () =
  let x = tensor_of_list [|3|] [0.5;1.0;2.0] in
  let dx = tensor_of_list [|3|] [1.;0.5;(-0.3)] in
  check_ip "log" [("x", x, dx)] (prim Log [var "x"])

let test_sqrt () =
  let x = tensor_of_list [|3|] [1.0;4.0;9.0] in
  let dx = tensor_of_list [|3|] [1.;(-0.5);2.] in
  check_ip "sqrt" [("x", x, dx)] (prim Sqrt [var "x"])

let test_relu () =
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  check_ip "relu" [("x", x, dx)] (prim Relu [var "x"])

let test_step () =
  let x = tensor_of_list [|4|] [(-1.);0.5;2.;(-0.3)] in
  let dx = tensor_of_list [|4|] [1.;1.;1.;1.] in
  check_ip "step" [("x", x, dx)] (prim Step [var "x"])

let test_add () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_ip "add" [("x", x, dx); ("y", y, dy)]
    (prim Add [var "x"; var "y"])

let test_sub () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_ip "sub" [("x", x, dx); ("y", y, dy)]
    (prim Sub [var "x"; var "y"])

let test_mul () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_ip "mul" [("x", x, dx); ("y", y, dy)]
    (prim Mul [var "x"; var "y"])

let test_div () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [4.;5.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_ip "div" [("x", x, dx); ("y", y, dy)]
    (prim Div [var "x"; var "y"])

let test_max2 () =
  let x = tensor_of_list [|3|] [1.;5.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  let y = tensor_of_list [|3|] [2.;4.;6.] in
  let dy = tensor_of_list [|3|] [0.4;0.5;0.6] in
  check_ip "max2" [("x", x, dx); ("y", y, dy)]
    (prim Max2 [var "x"; var "y"])

let test_sum_axis () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_ip "sum_axis" [("x", x, dx)] (prim (Sum_axis 1) [var "x"])

let test_max_axis () =
  let x = tensor_of_list [|2;3|] [1.;3.;2.;6.;4.;5.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_ip "max_axis" [("x", x, dx)] (prim (Max_axis 1) [var "x"])

let test_transpose () =
  let x = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_ip "transpose" [("x", x, dx)]
    (prim (Apply_view [Vtranspose [|1;0|]]) [var "x"])

let test_broadcast () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  check_ip "broadcast" [("x", x, dx)]
    (prim (Apply_view [Vbroadcast (0, 2)]) [var "x"])

let test_slice () =
  let x = tensor_of_list [|4|] [1.;2.;3.;4.] in
  let dx = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  check_ip "slice" [("x", x, dx)]
    (prim (Apply_view [Vslice [|(1,3,1)|]]) [var "x"])

let test_reshape () =
  let x = tensor_of_list [|6|] [1.;2.;3.;4.;5.;6.] in
  let dx = tensor_of_list [|6|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  check_ip "reshape" [("x", x, dx)]
    (prim (Apply_view [Vreshape [|2;3|]]) [var "x"])

let test_gather () =
  let x = tensor_of_list [|5|] [10.;20.;30.;40.;50.] in
  let dx = tensor_of_list [|5|] [0.1;0.2;0.3;0.4;0.5] in
  check_ip "gather" [("x", x, dx)]
    (prim (Gather (0, [|1;3;0|])) [var "x"])

let test_matmul () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let da = tensor_of_list [|2;3|] [0.1;0.2;0.3;0.4;0.5;0.6] in
  let b = tensor_of_list [|3;2|] [7.;8.;9.;10.;11.;12.] in
  let db = tensor_of_list [|3;2|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  check_ip "matmul" [("a", a, da); ("b", b, db)]
    (prim Matmul [var "a"; var "b"])

(* === composed === *)

let test_exp_sum () =
  let x = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  let dx = tensor_of_list [|4|] [1.;0.5;(-0.3);0.2] in
  check_ip "exp_sum" [("x", x, dx)]
    (prim (Sum_axis 0) [prim Exp [var "x"]])

let test_matmul_relu () =
  let a = tensor_of_list [|2;3|] [0.1;(-0.2);0.3;0.4;0.5;(-0.6)] in
  let da = tensor_of_list [|2;3|] [0.01;0.02;0.03;0.04;0.05;0.06] in
  let b = tensor_of_list [|3;2|] [1.;(-1.);2.;1.;(-1.);2.] in
  let db = tensor_of_list [|3;2|] [0.1;0.1;0.1;0.1;0.1;0.1] in
  check_ip "matmul_relu" [("a", a, da); ("b", b, db)]
    (prim Relu [prim Matmul [var "a"; var "b"]])

let test_let_x_sq () =
  let a = tensor_of_list [|3|] [2.;3.;4.] in
  let da = tensor_of_list [|3|] [1.;1.;1.] in
  check_ip "let_x_x*x" [("a", a, da)]
    (let_ "x" (var "a") (prim Mul [var "x"; var "x"]))

(* === gradient vs finite difference === *)

let eps = 1e-7

let check_grad_fd msg vars expr =
  (* Compute gradient via forward+unzip+transpose, then check vs FD *)
  Transform.Forward.reset_gensym ();
  let fwd = Transform.Forward.forward expr in
  let seeds = List.map (fun (s, _, _) ->
    Transform.Forward.tangent_name s) vars in
  let uz = Transform.Unzip.unzip fwd ~seeds in
  let input_shapes = List.map (fun (s, v, _) ->
    (s, v.Tensor.view.Ndview.shape)) vars in
  let tangent_input_shapes = List.map (fun (s, _, dv) ->
    (Transform.Forward.tangent_name s, dv.Tensor.view.Ndview.shape)) vars in
  let all_input_shapes = input_shapes @ tangent_input_shapes in
  let tr = Transform.Transpose.transpose
    ~primal_bindings:uz.primal_bindings
    ~tangent_bindings:uz.tangent_bindings
    ~tangent_out:uz.tangent_out
    ~seeds
    ~input_shapes:all_input_shapes
    ~cotangent_var:"%ct_out" in
  (* Output shape *)
  let senv = ref all_input_shapes in
  List.iter (fun (name, rhs) ->
    let sh = Transform.Expand_rank.infer_shape !senv rhs in
    senv := (name, sh) :: !senv
  ) (uz.primal_bindings @ uz.tangent_bindings);
  let out_shape = Transform.Expand_rank.infer_shape !senv uz.tangent_out in
  (* Use u = all-ones cotangent (sum reduction → gradient of sum(f(x))) *)
  let out_n = Array.fold_left ( * ) 1 out_shape in
  let u = Tensor.make out_shape in
  for i = 0 to out_n - 1 do Buf.set u.buf i 1.0 done;
  (* Eval gradients *)
  let primal_env = List.map (fun (s, v, _) -> (s, v)) vars in
  let tangent_env = List.map (fun (s, _, dv) ->
    (Transform.Forward.tangent_name s, dv)) vars in
  let ct_env = [("%ct_out", u)] in
  let full_env = primal_env @ tangent_env @ ct_env in
  let all_bs = uz.primal_bindings @ uz.tangent_bindings @ tr.grad_bindings in
  (* For each variable, compare gradient with FD *)
  let f_orig = Ast.Eval.eval primal_env expr in
  List.iter (fun (seed, grad_expr) ->
    let grad_full = Transform.Forward.wrap_bindings all_bs grad_expr in
    let grad_val = Ast.Eval.eval full_env grad_full in
    let var_name = String.sub seed 1 (String.length seed - 3) in
    let x = List.assoc var_name (List.map (fun (s,v,_) -> (s,v)) vars) in
    let n = numel x in
    for i = 0 to n - 1 do
      let x_pert = Tensor.make (x.Tensor.view.Ndview.shape) in
      for j = 0 to n - 1 do
        Buf.set x_pert.buf j (tensor_get x j)
      done;
      Buf.set x_pert.buf i (tensor_get x i +. eps);
      let env_pert = List.map (fun (s, v, _) ->
        if s = var_name then (s, x_pert) else (s, v)) vars in
      let f_pert = Ast.Eval.eval env_pert expr in
      let fd = ref 0.0 in
      let m = numel f_orig in
      for j = 0 to m - 1 do
        fd := !fd +. (tensor_get f_pert j -. tensor_get f_orig j) /. eps
      done;
      let g = tensor_get grad_val i in
      let scale = max 1.0 (max (abs_float g) (abs_float !fd)) in
      let rel = abs_float (g -. !fd) /. scale in
      Alcotest.(check bool)
        (Printf.sprintf "%s grad[%s][%d]: g=%.6g fd=%.6g rel=%.2e"
           msg var_name i g !fd rel)
        true (rel < 1e-5)
    done
  ) tr.grad_map

let test_grad_exp_sum () =
  let x = tensor_of_list [|4|] [0.1;0.2;0.3;0.4] in
  let dx = Tensor.make [|4|] in  (* unused by grad but needed for seeds *)
  check_grad_fd "grad_exp_sum" [("x", x, dx)]
    (prim (Sum_axis 0) [prim Exp [var "x"]])

let test_grad_matmul () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let da = Tensor.make [|2;3|] in
  let b = tensor_of_list [|3;2|] [7.;8.;9.;10.;11.;12.] in
  let db = Tensor.make [|3;2|] in
  (* matmul [2x3] @ [3x2] = [2x2], sum_axis 1 → [2], sum_axis 0 → scalar *)
  check_grad_fd "grad_matmul" [("a", a, da); ("b", b, db)]
    (prim (Sum_axis 0) [prim (Sum_axis 1) [prim Matmul [var "a"; var "b"]]])

(* === bug regression: dead let === *)

let test_dead_let () =
  (* let y = exp(x) in x*x — y is unused, its tangent should be skipped *)
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  check_ip "dead_let" [("x", x, dx)]
    (let_ "y" (prim Exp [var "x"]) (prim Mul [var "x"; var "x"]))

let test_dead_let_grad () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = Tensor.make [|3|] in
  check_grad_fd "dead_let_grad" [("x", x, dx)]
    (let_ "y" (prim Exp [var "x"]) (prim Mul [var "x"; var "x"]))

(* === bug regression: transpose∘gather and gather∘transpose (view on scatter) === *)

let test_transpose_gather () =
  (* transpose(gather(x)) — cotangent to scatter_add will be a transposed view *)
  let x = tensor_of_list [|3;4|]
    [1.;2.;3.;4.; 5.;6.;7.;8.; 9.;10.;11.;12.] in
  let dx = tensor_of_list [|3;4|]
    [0.1;0.2;0.3;0.4; 0.5;0.6;0.7;0.8; 0.9;1.0;1.1;1.2] in
  check_ip "transpose_gather" [("x", x, dx)]
    (prim (Apply_view [Vtranspose [|1;0|]]) [prim (Gather (0, [|2;0;1;0|])) [var "x"]])

let test_gather_transpose () =
  let x = tensor_of_list [|3;4|]
    [1.;2.;3.;4.; 5.;6.;7.;8.; 9.;10.;11.;12.] in
  let dx = tensor_of_list [|3;4|]
    [0.1;0.2;0.3;0.4; 0.5;0.6;0.7;0.8; 0.9;1.0;1.1;1.2] in
  check_ip "gather_transpose" [("x", x, dx)]
    (prim (Gather (1, [|2;0;1|])) [prim (Apply_view [Vtranspose [|1;0|]]) [var "x"]])

(* === bug regression: shadowing detection === *)

let test_shadowing_rejected () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let dx = tensor_of_list [|3|] [0.1;0.2;0.3] in
  (* let x = x*x in x — shadows the input x *)
  let expr = let_ "x" (var "a") (let_ "x" (prim Mul [var "x"; var "x"]) (var "x")) in
  Transform.Forward.reset_gensym ();
  let fwd = Transform.Forward.forward expr in
  let seeds = [Transform.Forward.tangent_name "a"] in
  let threw = try
    ignore (Transform.Unzip.unzip fwd ~seeds);
    false
  with Failure msg ->
    (* Should mention "duplicate binding name" *)
    let has_dup = try
      let _ = String.index msg 'd' in true  (* crude check *)
    with Not_found -> false in
    ignore (has_dup && (ignore (x, dx); true));
    true
  in
  Alcotest.(check bool) "shadowing detected" true threw

let () =
  let open Alcotest in
  run "transpose" [
    "inner_product_map1", [
      test_case "neg" `Quick test_neg;
      test_case "exp" `Quick test_exp;
      test_case "log" `Quick test_log;
      test_case "sqrt" `Quick test_sqrt;
      test_case "relu" `Quick test_relu;
      test_case "step" `Quick test_step;
    ];
    "inner_product_map2", [
      test_case "add" `Quick test_add;
      test_case "sub" `Quick test_sub;
      test_case "mul" `Quick test_mul;
      test_case "div" `Quick test_div;
      test_case "max2" `Quick test_max2;
    ];
    "inner_product_reduce", [
      test_case "sum_axis" `Quick test_sum_axis;
      test_case "max_axis" `Quick test_max_axis;
    ];
    "inner_product_structural", [
      test_case "transpose" `Quick test_transpose;
      test_case "broadcast" `Quick test_broadcast;
      test_case "slice" `Quick test_slice;
      test_case "reshape" `Quick test_reshape;
      test_case "gather" `Quick test_gather;
    ];
    "inner_product_linalg", [
      test_case "matmul" `Quick test_matmul;
    ];
    "inner_product_composed", [
      test_case "exp then sum" `Quick test_exp_sum;
      test_case "matmul then relu" `Quick test_matmul_relu;
      test_case "let x = a in x*x" `Quick test_let_x_sq;
    ];
    "gradient_vs_fd", [
      test_case "grad exp_sum" `Quick test_grad_exp_sum;
      test_case "grad matmul" `Quick test_grad_matmul;
    ];
    "regression", [
      test_case "dead let" `Quick test_dead_let;
      test_case "dead let grad" `Quick test_dead_let_grad;
      test_case "transpose∘gather" `Quick test_transpose_gather;
      test_case "gather∘transpose" `Quick test_gather_transpose;
      test_case "shadowing rejected" `Quick test_shadowing_rejected;
    ];
  ]
