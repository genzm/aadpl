open View
open Ast.Types

(* --- helpers --- *)

let tensor_of_list shape vals =
  let t = Tensor.make shape in
  List.iteri (fun i v -> Buf.set t.buf i v) vals;
  t

let get t indices =
  Buf.get t.Tensor.buf (Ndview.index_of t.Tensor.view indices)

let eval_expr e = Ast.Eval.eval [] e

(* --- arithmetic identities --- *)

let test_add_comm () =
  let a = tensor_of_list [|2;2|] [1.;2.;3.;4.] in
  let b = tensor_of_list [|2;2|] [5.;6.;7.;8.] in
  let ab = eval_expr (prim Add [const a; const b]) in
  let ba = eval_expr (prim Add [const b; const a]) in
  for i = 0 to 3 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "comm[%d]" i)
      (Buf.get ab.buf i) (Buf.get ba.buf i)
  done

let test_mul_assoc () =
  let a = tensor_of_list [|3|] [2.;3.;4.] in
  let b = tensor_of_list [|3|] [5.;6.;7.] in
  let c = tensor_of_list [|3|] [8.;9.;10.] in
  let ab_c = eval_expr (prim Mul [prim Mul [const a; const b]; const c]) in
  let a_bc = eval_expr (prim Mul [const a; prim Mul [const b; const c]]) in
  for i = 0 to 2 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "assoc[%d]" i)
      (Buf.get ab_c.buf i) (Buf.get a_bc.buf i)
  done

(* exp(log(x)) = x *)
let test_exp_log_inverse () =
  let x = tensor_of_list [|4|] [1.;2.;3.;0.5] in
  let r = eval_expr (prim Exp [prim Log [const x]]) in
  List.iteri (fun i v ->
    Alcotest.(check (float 1e-10)) (Printf.sprintf "exp_log[%d]" i) v (Buf.get r.buf i))
    [1.;2.;3.;0.5]

(* neg(neg(x)) = x *)
let test_neg_involution () =
  let x = tensor_of_list [|3|] [1.;(-2.);3.] in
  let r = eval_expr (prim Neg [prim Neg [const x]]) in
  List.iteri (fun i v ->
    Alcotest.(check (float 1e-10)) (Printf.sprintf "neg_neg[%d]" i) v (Buf.get r.buf i))
    [1.;(-2.);3.]

(* --- matmul identity --- *)

let test_matmul_identity () =
  (* A @ I = A *)
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let eye = tensor_of_list [|3;3|] [1.;0.;0.; 0.;1.;0.; 0.;0.;1.] in
  let r = eval_expr (prim Matmul [const a; const eye]) in
  Alcotest.(check (float 1e-10)) "r[0,0]" 1.0 (get r [|0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,2]" 3.0 (get r [|0;2|]);
  Alcotest.(check (float 1e-10)) "r[1,1]" 5.0 (get r [|1;1|]);
  Alcotest.(check (float 1e-10)) "r[1,2]" 6.0 (get r [|1;2|])

(* (AB)^T = B^T A^T *)
let test_matmul_transpose () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let b = tensor_of_list [|3;2|] [7.;8.;9.;10.;11.;12.] in
  let ab_t = eval_expr
    (prim (Transpose [|1;0|]) [prim Matmul [const a; const b]]) in
  let bt_at = eval_expr
    (prim Matmul [prim (Transpose [|1;0|]) [const b];
                  prim (Transpose [|1;0|]) [const a]]) in
  for i = 0 to 1 do
    for j = 0 to 1 do
      Alcotest.(check (float 1e-10))
        (Printf.sprintf "t[%d,%d]" i j)
        (get ab_t [|i;j|]) (get bt_at [|i;j|])
    done
  done

(* --- let binding and sharing --- *)

let test_let_sharing () =
  (* let x = [1,2,3] in x + x = [2,4,6] *)
  let v = tensor_of_list [|3|] [1.;2.;3.] in
  let r = eval_expr (let_ "x" (const v) (prim Add [var "x"; var "x"])) in
  Alcotest.(check (float 1e-10)) "r[0]" 2.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "r[1]" 4.0 (Buf.get r.buf 1);
  Alcotest.(check (float 1e-10)) "r[2]" 6.0 (Buf.get r.buf 2)

let test_let_nested () =
  (* let a = [1,2] in let b = a * a in b + a = [2,6] *)
  let v = tensor_of_list [|2|] [1.;2.] in
  let r = eval_expr
    (let_ "a" (const v)
      (let_ "b" (prim Mul [var "a"; var "a"])
        (prim Add [var "b"; var "a"]))) in
  Alcotest.(check (float 1e-10)) "r[0]" 2.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "r[1]" 6.0 (Buf.get r.buf 1)

(* --- structural ops --- *)

let test_sum_axis () =
  (* sum_axis(0) [[1,2],[3,4]] = [4,6] *)
  let m = tensor_of_list [|2;2|] [1.;2.;3.;4.] in
  let r = eval_expr (prim (Sum_axis 0) [const m]) in
  Alcotest.(check (float 1e-10)) "s[0]" 4.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "s[1]" 6.0 (Buf.get r.buf 1)

let test_sum_all () =
  (* sum both axes: total = 10 *)
  let m = tensor_of_list [|2;2|] [1.;2.;3.;4.] in
  let r = eval_expr (prim (Sum_axis 0) [prim (Sum_axis 1) [const m]]) in
  Alcotest.(check (float 1e-10)) "total" 10.0 (Buf.get r.buf 0)

let test_reshape_then_sum () =
  (* reshape [1,2,3,4,5,6] to [2,3] then sum axis 1 = [6, 15] *)
  let v = tensor_of_list [|6|] [1.;2.;3.;4.;5.;6.] in
  let r = eval_expr
    (prim (Sum_axis 1) [prim (Reshape [|2;3|]) [const v]]) in
  Alcotest.(check (float 1e-10)) "s[0]" 6.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "s[1]" 15.0 (Buf.get r.buf 1)

(* --- view ops set shared, then compute still works --- *)

let test_transpose_is_view () =
  (* transpose returns a view (shared=true), not a copy *)
  let m = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let r = eval_expr (prim (Transpose [|1;0|]) [const m]) in
  Alcotest.(check bool) "shared" true r.View.Tensor.buf.View.Buf.shared;
  (* the value is correct *)
  Alcotest.(check (float 1e-10)) "t[0,0]" 1.0 (get r [|0;0|]);
  Alcotest.(check (float 1e-10)) "t[1,0]" 2.0 (get r [|1;0|]);
  Alcotest.(check (float 1e-10)) "t[2,1]" 6.0 (get r [|2;1|])

let test_broadcast_is_view () =
  let v = tensor_of_list [|3|] [10.;20.;30.] in
  let r = eval_expr (prim (Broadcast (0, 2)) [const v]) in
  Alcotest.(check bool) "shared" true r.View.Tensor.buf.View.Buf.shared;
  Alcotest.(check (float 1e-10)) "b[0,0]" 10.0 (get r [|0;0|]);
  Alcotest.(check (float 1e-10)) "b[1,2]" 30.0 (get r [|1;2|])

let test_view_then_compute () =
  (* transpose then add: transpose returns view, add reads through it *)
  let m = tensor_of_list [|2;2|] [1.;2.;3.;4.] in
  let r = eval_expr
    (prim Add [prim (Transpose [|1;0|]) [const m];
               prim (Transpose [|1;0|]) [const m]]) in
  (* [[1,3],[2,4]] + [[1,3],[2,4]] = [[2,6],[4,8]] *)
  Alcotest.(check (float 1e-10)) "r[0,0]" 2.0 (get r [|0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,1]" 6.0 (get r [|0;1|]);
  Alcotest.(check (float 1e-10)) "r[1,0]" 4.0 (get r [|1;0|]);
  Alcotest.(check (float 1e-10)) "r[1,1]" 8.0 (get r [|1;1|])

let test_broadcast_then_sum () =
  (* broadcast [1,2,3] to [2,3] then sum axis 0 = [2,4,6] *)
  let v = tensor_of_list [|3|] [1.;2.;3.] in
  let r = eval_expr
    (prim (Sum_axis 0) [prim (Broadcast (0, 2)) [const v]]) in
  Alcotest.(check (float 1e-10)) "s[0]" 2.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "s[1]" 4.0 (Buf.get r.buf 1);
  Alcotest.(check (float 1e-10)) "s[2]" 6.0 (Buf.get r.buf 2)

(* --- gather with indices in AST --- *)

let test_gather () =
  let v = tensor_of_list [|5|] [10.;20.;30.;40.;50.] in
  let r = eval_expr (prim (Gather (0, [|1;3;1|])) [const v]) in
  Alcotest.(check (float 1e-10)) "g[0]" 20.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "g[1]" 40.0 (Buf.get r.buf 1);
  Alcotest.(check (float 1e-10)) "g[2]" 20.0 (Buf.get r.buf 2)

(* --- shape mismatch error --- *)

let test_shape_mismatch () =
  let a = tensor_of_list [|2|] [1.;2.] in
  let b = tensor_of_list [|3|] [1.;2.;3.] in
  Alcotest.check_raises "shape mismatch"
    (Ast.Eval.Shape_mismatch (dummy_loc, "map2: shape mismatch"))
    (fun () -> ignore (eval_expr (prim Add [const a; const b])))

(* --- relu --- *)

let test_relu () =
  let x = tensor_of_list [|4|] [(-1.);0.;1.;(-0.5)] in
  let r = eval_expr (prim Relu [const x]) in
  Alcotest.(check (float 1e-10)) "r[0]" 0.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "r[1]" 0.0 (Buf.get r.buf 1);
  Alcotest.(check (float 1e-10)) "r[2]" 1.0 (Buf.get r.buf 2);
  Alcotest.(check (float 1e-10)) "r[3]" 0.0 (Buf.get r.buf 3)

(* --- pretty-printer round trip (just check it doesn't crash) --- *)

let test_pp () =
  let a = tensor_of_list [|2;2|] [1.;2.;3.;4.] in
  let e = let_ "x" (const a)
    (prim Matmul [var "x"; prim (Transpose [|1;0|]) [var "x"]]) in
  let s = Format.asprintf "%a" pp e in
  Alcotest.(check bool) "pp non-empty" true (String.length s > 0)

(* --- linear algebra: A(Bx) = (AB)x --- *)

let test_matmul_assoc () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let b = tensor_of_list [|3;2|] [7.;8.;9.;10.;11.;12.] in
  let x = tensor_of_list [|2;1|] [1.;2.] in
  let a_bx = eval_expr
    (prim Matmul [const a; prim Matmul [const b; const x]]) in
  let ab_x = eval_expr
    (prim Matmul [prim Matmul [const a; const b]; const x]) in
  for i = 0 to 1 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "assoc[%d]" i)
      (get a_bx [|i;0|]) (get ab_x [|i;0|])
  done

let () =
  let open Alcotest in
  run "eval" [
    "arithmetic", [
      test_case "add commutative" `Quick test_add_comm;
      test_case "mul associative" `Quick test_mul_assoc;
      test_case "exp(log(x)) = x" `Quick test_exp_log_inverse;
      test_case "neg involution" `Quick test_neg_involution;
      test_case "relu" `Quick test_relu;
    ];
    "linalg", [
      test_case "matmul identity" `Quick test_matmul_identity;
      test_case "(AB)^T = B^T A^T" `Quick test_matmul_transpose;
      test_case "A(Bx) = (AB)x" `Quick test_matmul_assoc;
    ];
    "let", [
      test_case "let sharing" `Quick test_let_sharing;
      test_case "let nested" `Quick test_let_nested;
    ];
    "structural", [
      test_case "sum_axis" `Quick test_sum_axis;
      test_case "sum all axes" `Quick test_sum_all;
      test_case "reshape then sum" `Quick test_reshape_then_sum;
      test_case "gather" `Quick test_gather;
    ];
    "view-ops", [
      test_case "transpose is view" `Quick test_transpose_is_view;
      test_case "broadcast is view" `Quick test_broadcast_is_view;
      test_case "view then compute" `Quick test_view_then_compute;
      test_case "broadcast then sum" `Quick test_broadcast_then_sum;
    ];
    "error", [
      test_case "shape mismatch" `Quick test_shape_mismatch;
    ];
    "pp", [
      test_case "pretty-printer" `Quick test_pp;
    ];
  ]
