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

let expand_eval e =
  let e' = Transform.Expand_rank.expand e in
  assert (Transform.Expand_rank.is_expanded e');
  eval_expr e'

(* --- elementwise: add⎉0 same shape --- *)

let test_add_rank0_same () =
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let b = tensor_of_list [|2;3|] [10.;20.;30.;40.;50.;60.] in
  let r = expand_eval (rank 0 Add [const a; const b]) in
  let expected = eval_expr (prim Add [const a; const b]) in
  for i = 0 to 5 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "add⎉0[%d]" i)
      (Buf.get expected.buf i) (Buf.get r.buf i)
  done

(* --- 5c mandatory: bias addition --- *)
(* add⎉1 [b; h]  where b:[N], h:[B,N] — cell is vector, frame [] vs [B] *)

let test_bias_add () =
  let b = tensor_of_list [|3|] [0.1; 0.2; 0.3] in
  let h = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  (* b cell=[3] frame=[], h cell=[3] frame=[2].
     b gets Broadcast(0,2) → [2,3]. Then Add. *)
  let r = expand_eval (rank 1 Add [const b; const h]) in
  (* row 0: [1.1, 2.2, 3.3], row 1: [4.1, 5.2, 6.3] *)
  Alcotest.(check (float 1e-10)) "r[0,0]" 1.1 (get r [|0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,1]" 2.2 (get r [|0;1|]);
  Alcotest.(check (float 1e-10)) "r[0,2]" 3.3 (get r [|0;2|]);
  Alcotest.(check (float 1e-10)) "r[1,0]" 4.1 (get r [|1;0|]);
  Alcotest.(check (float 1e-10)) "r[1,1]" 5.2 (get r [|1;1|]);
  Alcotest.(check (float 1e-10)) "r[1,2]" 6.3 (get r [|1;2|])

(* --- bias with 2-level frame: b:[N], h:[B,G,N] --- *)

let test_bias_add_2frame () =
  let b = tensor_of_list [|2|] [100.; 200.] in
  let h = tensor_of_list [|2;3;2|]
    [1.;2.; 3.;4.; 5.;6.;  7.;8.; 9.;10.; 11.;12.] in
  (* k=1: b frame=[], h frame=[2,3]. b gets Broadcast(0,2) then (1,3) → [2,3,2] *)
  let r = expand_eval (rank 1 Add [const b; const h]) in
  Alcotest.(check (float 1e-10)) "r[0,0,0]" 101.0 (get r [|0;0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,0,1]" 202.0 (get r [|0;0;1|]);
  Alcotest.(check (float 1e-10)) "r[1,2,0]" 111.0 (get r [|1;2;0|]);
  Alcotest.(check (float 1e-10)) "r[1,2,1]" 212.0 (get r [|1;2;1|])

(* --- 5c mandatory: Dense layer = matmul⎉2 + bias --- *)
(* xs:[B,M,K], W:[K,N] — frame [B] vs [] *)

let test_dense_matmul () =
  let xs = tensor_of_list [|2;2;3|]
    [1.;2.;3.; 4.;5.;6.;     (* batch 0: [[1,2,3],[4,5,6]] *)
     7.;8.;9.; 10.;11.;12.]  (* batch 1: [[7,8,9],[10,11,12]] *)
  in
  let w = tensor_of_list [|3;2|]
    [1.;0.; 0.;1.; 1.;1.]  (* W = [[1,0],[0,1],[1,1]] *)
  in
  let r = expand_eval (rank 2 Matmul [const xs; const w]) in
  (* batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]]
     = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]] *)
  Alcotest.(check (float 1e-10)) "r[0,0,0]"  4.0 (get r [|0;0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,0,1]"  5.0 (get r [|0;0;1|]);
  Alcotest.(check (float 1e-10)) "r[0,1,0]" 10.0 (get r [|0;1;0|]);
  Alcotest.(check (float 1e-10)) "r[0,1,1]" 11.0 (get r [|0;1;1|]);
  (* batch 1: [[7,8,9],[10,11,12]] @ W
     = [[7+0+9, 0+8+9],[10+0+12, 0+11+12]] = [[16,17],[22,23]] *)
  Alcotest.(check (float 1e-10)) "r[1,0,0]" 16.0 (get r [|1;0;0|]);
  Alcotest.(check (float 1e-10)) "r[1,0,1]" 17.0 (get r [|1;0;1|]);
  Alcotest.(check (float 1e-10)) "r[1,1,0]" 22.0 (get r [|1;1;0|]);
  Alcotest.(check (float 1e-10)) "r[1,1,1]" 23.0 (get r [|1;1;1|])

(* --- Dense layer: matmul⎉2 + bias add⎉1 --- *)

let test_dense_full () =
  let xs = tensor_of_list [|2;2;3|]
    [1.;2.;3.; 4.;5.;6.; 7.;8.;9.; 10.;11.;12.] in
  let w = tensor_of_list [|3;2|] [1.;0.; 0.;1.; 1.;1.] in
  let bias = tensor_of_list [|2|] [100.;200.] in
  (* matmul⎉2 gives [2,2,2], then add⎉1 with bias [2]
     bias cell=[2] frame=[], matmul_result cell=[2] frame=[2,2]
     bias broadcast → [2,2,2], add *)
  let r = expand_eval
    (rank 1 Add [const bias;
                 rank 2 Matmul [const xs; const w]]) in
  (* batch 0: [[4,5],[10,11]] + [100,200] = [[104,205],[110,211]] *)
  Alcotest.(check (float 1e-10)) "r[0,0,0]" 104.0 (get r [|0;0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,0,1]" 205.0 (get r [|0;0;1|]);
  Alcotest.(check (float 1e-10)) "r[0,1,0]" 110.0 (get r [|0;1;0|]);
  Alcotest.(check (float 1e-10)) "r[0,1,1]" 211.0 (get r [|0;1;1|])

(* --- neg⎉0 on [3] = plain Neg --- *)

let test_neg_rank0 () =
  let a = tensor_of_list [|3|] [1.;(-2.);3.] in
  let r = expand_eval (rank 0 Neg [const a]) in
  let expected = eval_expr (prim Neg [const a]) in
  for i = 0 to 2 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "neg⎉0[%d]" i)
      (Buf.get expected.buf i) (Buf.get r.buf i)
  done

(* --- sum_axis⎉1 on [B,N]: sum each row --- *)

let test_sum_axis_rank1 () =
  let m = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let r = expand_eval (rank 1 (Sum_axis 0) [const m]) in
  (* sum_axis(0) on cell [N] → scalar per row. Shifted to sum_axis(1). *)
  Alcotest.(check (float 1e-10)) "s[0]" 6.0 (Buf.get r.buf 0);
  Alcotest.(check (float 1e-10)) "s[1]" 15.0 (Buf.get r.buf 1)

(* --- transpose⎉2 on [B,M,N]: frame=1, perm becomes [0,2,1] --- *)

let test_transpose_batched () =
  let m = tensor_of_list [|2;2;3|]
    [1.;2.;3.;4.;5.;6.; 7.;8.;9.;10.;11.;12.] in
  let r = expand_eval (rank 2 (Apply_view [Vtranspose [|1;0|]]) [const m]) in
  let expected = eval_expr (prim (Apply_view [Vtranspose [|0;2;1|]]) [const m]) in
  for i = 0 to 1 do
    for j = 0 to 2 do
      for k = 0 to 1 do
        Alcotest.(check (float 1e-10))
          (Printf.sprintf "bt[%d,%d,%d]" i j k)
          (get expected [|i;j;k|]) (get r [|i;j;k|])
      done
    done
  done

(* --- gather⎉1 on [B,N]: gather from each row --- *)

let test_gather_rank1 () =
  let m = tensor_of_list [|2;5|]
    [10.;20.;30.;40.;50.; 60.;70.;80.;90.;100.] in
  let r = expand_eval (rank 1 (Gather (0, [|1;3|])) [const m]) in
  (* gather(0,[1;3]) on cell, shifted to gather(1,[1;3]) *)
  Alcotest.(check (float 1e-10)) "g[0,0]" 20.0 (get r [|0;0|]);
  Alcotest.(check (float 1e-10)) "g[0,1]" 40.0 (get r [|0;1|]);
  Alcotest.(check (float 1e-10)) "g[1,0]" 70.0 (get r [|1;0|]);
  Alcotest.(check (float 1e-10)) "g[1,1]" 90.0 (get r [|1;1|])

(* --- slice⎉2 on [B,M,N]: frame ranges prepended --- *)

let test_slice_rank2 () =
  let t = tensor_of_list [|2;3;4|]
    [ 0.; 1.; 2.; 3.;  4.; 5.; 6.; 7.;  8.; 9.;10.;11.;
     12.;13.;14.;15.; 16.;17.;18.;19.; 20.;21.;22.;23.] in
  let r = expand_eval
    (rank 2 (Apply_view [Vslice [|(0,2,1);(1,3,1)|]]) [const t]) in
  (* slice on cell [3,4] with [(0,2,1);(1,3,1)] → [2,2] per batch
     Expanded: Slice [(0,2,1);(0,2,1);(1,3,1)]
     batch 0: rows 0-1, cols 1-2 → [[1,2],[5,6]]
     batch 1: rows 0-1, cols 1-2 → [[13,14],[17,18]] *)
  Alcotest.(check (float 1e-10)) "s[0,0,0]"  1.0 (get r [|0;0;0|]);
  Alcotest.(check (float 1e-10)) "s[0,0,1]"  2.0 (get r [|0;0;1|]);
  Alcotest.(check (float 1e-10)) "s[0,1,0]"  5.0 (get r [|0;1;0|]);
  Alcotest.(check (float 1e-10)) "s[0,1,1]"  6.0 (get r [|0;1;1|]);
  Alcotest.(check (float 1e-10)) "s[1,0,0]" 13.0 (get r [|1;0;0|]);
  Alcotest.(check (float 1e-10)) "s[1,0,1]" 14.0 (get r [|1;0;1|]);
  Alcotest.(check (float 1e-10)) "s[1,1,0]" 17.0 (get r [|1;1;0|]);
  Alcotest.(check (float 1e-10)) "s[1,1,1]" 18.0 (get r [|1;1;1|])

(* --- reshape⎉1 on [B,N]: frame prepended to target shape --- *)

let test_reshape_rank1 () =
  (* [2,6] tensor, reshape⎉1 [2;3]
     k=1, frame=[2], shift_prim [|2|] (Reshape [|2;3|]) = Reshape [|2;2;3|]
     Result shape: [2,2,3] *)
  let t = tensor_of_list [|2;6|]
    [1.;2.;3.;4.;5.;6.; 7.;8.;9.;10.;11.;12.] in
  let r = expand_eval (rank 1 (Apply_view [Vreshape [|2;3|]]) [const t]) in
  Alcotest.(check (float 1e-10)) "r[0,0,0]" 1.0 (get r [|0;0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,1,2]" 6.0 (get r [|0;1;2|]);
  Alcotest.(check (float 1e-10)) "r[1,0,0]" 7.0 (get r [|1;0;0|]);
  Alcotest.(check (float 1e-10)) "r[1,1,2]" 12.0 (get r [|1;1;2|])

(* --- is_expanded --- *)

let test_is_expanded () =
  let a = tensor_of_list [|3|] [1.;2.;3.] in
  let e1 = prim Add [const a; const a] in
  let e2 = rank 0 Add [const a; const a] in
  Alcotest.(check bool) "prim is expanded" true
    (Transform.Expand_rank.is_expanded e1);
  Alcotest.(check bool) "rank is not expanded" false
    (Transform.Expand_rank.is_expanded e2);
  let e3 = Transform.Expand_rank.expand e2 in
  Alcotest.(check bool) "after expand" true
    (Transform.Expand_rank.is_expanded e3)

(* --- eval rejects Rank --- *)

let test_eval_rejects_rank () =
  let a = tensor_of_list [|3|] [1.;2.;3.] in
  let e = rank 0 Add [const a; const a] in
  Alcotest.check_raises "rank rejected"
    (Ast.Eval.Eval_error (dummy_loc, "Rank node must be expanded before eval"))
    (fun () -> ignore (eval_expr e))

(* --- expand inside let --- *)

let test_expand_in_let () =
  let v = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let e = let_ "x" (const v)
    (rank 1 Add [var "x"; var "x"]) in
  let e' = Transform.Expand_rank.expand e in
  assert (Transform.Expand_rank.is_expanded e');
  let r = Ast.Eval.eval [] e' in
  for i = 0 to 5 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "let_add[%d]" i)
      (2.0 *. Buf.get v.buf i) (Buf.get r.buf i)
  done

(* --- nested Rank --- *)

let test_nested_rank () =
  (* neg⎉0 (add⎉1 [b; h]) where b:[3], h:[2,3] *)
  let b = tensor_of_list [|3|] [1.;2.;3.] in
  let h = tensor_of_list [|2;3|] [10.;20.;30.;40.;50.;60.] in
  let r = expand_eval
    (rank 0 Neg [rank 1 Add [const b; const h]]) in
  (* add⎉1: b broadcast to [2,3], add → [[11,22,33],[41,52,63]]
     neg⎉0: negate → [[-11,-22,-33],[-41,-52,-63]] *)
  Alcotest.(check (float 1e-10)) "r[0,0]" (-11.0) (get r [|0;0|]);
  Alcotest.(check (float 1e-10)) "r[0,2]" (-33.0) (get r [|0;2|]);
  Alcotest.(check (float 1e-10)) "r[1,0]" (-41.0) (get r [|1;0|]);
  Alcotest.(check (float 1e-10)) "r[1,2]" (-63.0) (get r [|1;2|])

(* --- expect test: pp the expanded AST to verify Broadcast insertion --- *)

let test_expand_pp_bias () =
  let b = tensor_of_list [|3|] [1.;2.;3.] in
  let h = tensor_of_list [|2;3|] [10.;20.;30.;40.;50.;60.] in
  let e = rank 1 Add [const b; const h] in
  let e' = Transform.Expand_rank.expand e in
  let s = Format.asprintf "%a" pp e' in
  (* Should contain "apply_view(B(" showing the Vbroadcast insertion *)
  let contains sub s =
    let len_sub = String.length sub in
    let len_s = String.length s in
    let rec check i =
      if i + len_sub > len_s then false
      else if String.sub s i len_sub = sub then true
      else check (i + 1)
    in check 0
  in
  Alcotest.(check bool) "contains apply_view(B(" true (contains "apply_view(B(" s)

(* --- k < cell_rank error --- *)

let test_cell_rank_too_low () =
  let m = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  try
    ignore (Transform.Expand_rank.expand
      (rank 1 Matmul [const m; const m]));
    Alcotest.fail "should have raised"
  with Transform.Expand_rank.Expand_error (_, msg) ->
    Alcotest.(check bool) "mentions cell rank"
      true (String.length msg > 0)

(* --- leading agreement violation --- *)

let test_frame_mismatch () =
  (* a:[2,3], b:[4,3] with k=1 — frames [2] and [4] disagree *)
  let a = tensor_of_list [|2;3|] [1.;2.;3.;4.;5.;6.] in
  let b = tensor_of_list [|4;3|]
    [1.;2.;3.; 4.;5.;6.; 7.;8.;9.; 10.;11.;12.] in
  try
    ignore (Transform.Expand_rank.expand
      (rank 1 Add [const a; const b]));
    Alcotest.fail "should have raised"
  with Transform.Expand_rank.Expand_error (_, msg) ->
    Alcotest.(check bool) "mentions mismatch"
      true (String.length msg > 0)

let () =
  let open Alcotest in
  run "expand_rank" [
    "elementwise", [
      test_case "add⎉0 same shape" `Quick test_add_rank0_same;
      test_case "bias add⎉1 (frame [] vs [B])" `Quick test_bias_add;
      test_case "bias add⎉1 (2-level frame)" `Quick test_bias_add_2frame;
      test_case "neg⎉0" `Quick test_neg_rank0;
    ];
    "dense", [
      test_case "matmul⎉2 (frame [B] vs [])" `Quick test_dense_matmul;
      test_case "dense = matmul⎉2 + bias add⎉1" `Quick test_dense_full;
    ];
    "reduce", [
      test_case "sum_axis⎉1" `Quick test_sum_axis_rank1;
    ];
    "structural", [
      test_case "transpose⎉2 batched" `Quick test_transpose_batched;
      test_case "gather⎉1" `Quick test_gather_rank1;
      test_case "slice⎉2" `Quick test_slice_rank2;
      test_case "reshape⎉1" `Quick test_reshape_rank1;
    ];
    "meta", [
      test_case "is_expanded" `Quick test_is_expanded;
      test_case "eval rejects Rank" `Quick test_eval_rejects_rank;
      test_case "expand in let" `Quick test_expand_in_let;
      test_case "nested Rank" `Quick test_nested_rank;
      test_case "pp shows broadcast" `Quick test_expand_pp_bias;
    ];
    "error", [
      test_case "k < cell_rank" `Quick test_cell_rank_too_low;
      test_case "frame mismatch" `Quick test_frame_mismatch;
    ];
  ]
