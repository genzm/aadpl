(* Tests for fuse_views (view fusion pass).
   After old structural prims were deleted, this pass fuses adjacent
   Apply_view nodes: Apply_view s2 [Apply_view s1 [x]] → Apply_view (s1@s2) [x].
   Does not fuse across Let boundaries. *)

open View
open Ast.Types

(* --- helpers --- *)

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

let eval_with bindings expr =
  let env = List.map (fun (s, t) -> (s, (t : Tensor.t :> value))) bindings in
  Ast.Eval.eval env expr

let check_vals label r1 r2 =
  let n = numel r1 in
  Alcotest.(check int) (label ^ " numel") n (numel r2);
  for i = 0 to n - 1 do
    Alcotest.(check (float 1e-15)) (label ^ " [" ^ string_of_int i ^ "]")
      (tensor_get r1 i) (tensor_get r2 i)
  done

(* --- fusion --- *)

let test_fusion () =
  (* Apply_view[Vslice](Apply_view[Vtranspose](x)) → Apply_view [Vtranspose; Vslice] x *)
  let e = prim (Apply_view [Vslice [|(0,2,1);(1,3,1)|]])
    [prim (Apply_view [Vtranspose [|1;0|]]) [var "x"]] in
  let d = Transform.Desugar.fuse_views e in
  let is_single = match d with
    | Prim (_, Apply_view spec, [Var _]) -> List.length spec = 2
    | _ -> false in
  Alcotest.(check bool) "fused to single Apply_view" true is_single;
  let x = Tensor.make_random [|3;4|] in
  let r1 = eval_with [("x",x)] e in
  let r2 = eval_with [("x",x)] d in
  check_vals "fusion eval" r1 r2

let test_triple_fusion () =
  (* Apply_view[Vreshape](Apply_view[Vslice](Apply_view[Vtranspose](x)))
     → Apply_view [Vtranspose; Vslice; Vreshape] x *)
  let e = prim (Apply_view [Vreshape [|4|]])
    [prim (Apply_view [Vslice [|(0,2,1);(1,3,1)|]])
       [prim (Apply_view [Vtranspose [|1;0|]]) [var "x"]]] in
  let d = Transform.Desugar.fuse_views e in
  let spec_len = match d with
    | Prim (_, Apply_view spec, [Var _]) -> List.length spec
    | _ -> 0 in
  Alcotest.(check int) "3-way fusion" 3 spec_len;
  let x = Tensor.make_random [|3;4|] in
  check_vals "triple fusion" (eval_with [("x",x)] e) (eval_with [("x",x)] d)

let test_no_fusion_across_let () =
  (* let y = Apply_view[Vtranspose] x in Apply_view[Vslice] y — two separate Apply_views *)
  let e = let_ "y" (prim (Apply_view [Vtranspose [|1;0|]]) [var "x"])
            (prim (Apply_view [Vslice [|(0,2,1);(1,3,1)|]]) [var "y"]) in
  let d = Transform.Desugar.fuse_views e in
  let has_two = match d with
    | Let (_, "y", Prim (_, Apply_view _, _), Prim (_, Apply_view _, _)) -> true
    | _ -> false in
  Alcotest.(check bool) "no fusion across let" true has_two;
  let x = Tensor.make_random [|3;4|] in
  check_vals "let boundary" (eval_with [("x",x)] e) (eval_with [("x",x)] d)

(* --- mixed: non-structural ops are preserved --- *)

let test_mixed () =
  let x = Tensor.make_random [|3;4|] in
  let e = prim Exp [prim (Apply_view [Vtranspose [|1;0|]]) [var "x"]] in
  let d = Transform.Desugar.fuse_views e in
  check_vals "exp(transpose(x))"
    (eval_with [("x",x)] e) (eval_with [("x",x)] d)

let test_mixed_mul () =
  let x = Tensor.make_random [|3;4|] in
  let y = Tensor.make_random [|4;3|] in
  let e = prim Mul [prim (Apply_view [Vtranspose [|1;0|]]) [var "x"]; var "y"] in
  let d = Transform.Desugar.fuse_views e in
  check_vals "transpose(x) * y"
    (eval_with [("x",x); ("y",y)] e) (eval_with [("x",x); ("y",y)] d)

(* --- end-to-end pipeline: fuse_views → forward → unzip → transpose --- *)

let test_pipeline () =
  let in_shape = [|3;4|] in
  let out_shape = [|3|] in
  (* f(x) = sum_axis 0 (transpose(1,0) x)  — output shape [3] *)
  let expr = prim (Sum_axis 0) [prim (Apply_view [Vtranspose [|1;0|]]) [var "x"]] in
  Alcotest.(check bool) "is_desugared" true (Transform.Desugar.is_desugared expr);
  let run e =
    Transform.Forward.reset_gensym ();
    let fwd = Transform.Forward.forward e in
    let seed = Transform.Forward.tangent_name "x" in
    let uz = Transform.Unzip.unzip fwd ~seeds:[seed] in
    Transform.Transpose.transpose
      ~primal_bindings:uz.primal_bindings
      ~tangent_bindings:uz.tangent_bindings
      ~tangent_out:uz.tangent_out
      ~seeds:[seed]
      ~input_shapes:[("x", in_shape); (seed, in_shape)]
      ~cotangent_var:"ct"
  in
  let tr = run expr in
  let x = Tensor.make_random in_shape in
  let ct = Tensor.make_random out_shape in
  let grad = eval_with [("x",x); ("ct",ct)]
    (Transform.Forward.wrap_bindings tr.Transform.Transpose.grad_bindings
       (snd (List.hd tr.Transform.Transpose.grad_map))) in
  (* Gradient of sum_axis0(transpose(x)) w.r.t. x:
     transpose maps [3,4] → [4,3], sum_axis0 sums axis 0 of [4,3] → [3].
     z[k] = Σ_j x[k,j], so grad[i,j] = ct[i]. *)
  for i = 0 to 2 do
    for j = 0 to 3 do
      Alcotest.(check (float 1e-15))
        (Printf.sprintf "grad[%d,%d]" i j)
        (tensor_get ct i) (tensor_get grad (i * 4 + j))
    done
  done

let () =
  let open Alcotest in
  run "desugar" [
    "fusion", [
      test_case "adjacent" `Quick test_fusion;
      test_case "triple" `Quick test_triple_fusion;
      test_case "no fusion across let" `Quick test_no_fusion_across_let;
    ];
    "mixed", [
      test_case "exp(transpose)" `Quick test_mixed;
      test_case "transpose * y" `Quick test_mixed_mul;
    ];
    "pipeline", [
      test_case "fuse→forward→unzip→transpose" `Quick test_pipeline;
    ];
  ]
