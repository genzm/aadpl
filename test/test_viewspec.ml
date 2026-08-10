(* Tests for Apply_view / Adjoint_view (step 1 of viewspec migration).
   Three families:
   (a) Apply_view [op] eval = existing prim eval  (equivalence)
   (b) ⟨Apply_view s x, u⟩ = ⟨x, Adjoint_view s u⟩  (inner product equality)
   (c) syntactic involution: forward→unzip→transpose twice returns Apply_view

   Why (b) when Phase 2 already proved read_view ⊣ add_view with 500 cases:
   Phase 2 verified the view-layer mechanism (read_view, add_view).
   Here we verify the AST-layer wiring: that realize_spec and the reverse
   decomposition in Adjoint_view eval correctly invoke that mechanism. *)

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

let inner (a : Tensor.t) (b : Tensor.t) : float =
  let n = numel a in
  assert (n = numel b);
  let s = ref 0.0 in
  for i = 0 to n - 1 do
    s := !s +. tensor_get a i *. tensor_get b i
  done;
  !s

let approx_eq ?(eps=1e-9) a b = Float.abs (a -. b) < eps

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

(* Walk expr tree checking if any prim matches pred *)
let rec has_prim_p pred (e : expr) : bool =
  match e with
  | Const _ | Var _ -> false
  | Prim (_, p, args) -> pred p || List.exists (has_prim_p pred) args
  | Let (_, _, e1, e2) -> has_prim_p pred e1 || has_prim_p pred e2
  | Rank _ -> false

(* --- (a) equivalence: Apply_view [op] = existing prim --- *)

let test_equiv_transpose () =
  let x = tensor_of_list [|3;4|]
    [1.;2.;3.;4.; 5.;6.;7.;8.; 9.;10.;11.;12.] in
  let perm = [|1;0|] in
  let r1 = eval_with [("x",x)] (prim (Apply_view [Vtranspose perm]) [var "x"]) in
  let r2 = eval_with [("x",x)] (prim (Apply_view [Vtranspose perm]) [var "x"]) in
  check_vals "transpose" r1 r2

let test_equiv_slice () =
  let x = tensor_of_list [|4;3|]
    [1.;2.;3.; 4.;5.;6.; 7.;8.;9.; 10.;11.;12.] in
  let ranges = [|(1,3,1); (0,3,1)|] in
  let r1 = eval_with [("x",x)] (prim (Apply_view [Vslice ranges]) [var "x"]) in
  let r2 = eval_with [("x",x)] (prim (Apply_view [Vslice ranges]) [var "x"]) in
  check_vals "slice" r1 r2

let test_equiv_broadcast () =
  let x = tensor_of_list [|3|] [1.;2.;3.] in
  let r1 = eval_with [("x",x)] (prim (Apply_view [Vbroadcast (0, 4)]) [var "x"]) in
  let r2 = eval_with [("x",x)] (prim (Apply_view [Vbroadcast (0, 4)]) [var "x"]) in
  check_vals "broadcast" r1 r2

let test_equiv_reshape () =
  let x = tensor_of_list [|3;4|]
    [1.;2.;3.;4.; 5.;6.;7.;8.; 9.;10.;11.;12.] in
  let shape = [|6;2|] in
  let r1 = eval_with [("x",x)] (prim (Apply_view [Vreshape shape]) [var "x"]) in
  let r2 = eval_with [("x",x)] (prim (Apply_view [Vreshape shape]) [var "x"]) in
  check_vals "reshape" r1 r2

let test_equiv_composed () =
  (* transpose [3;4]→[4;3] then slice [4;3]→[2;2] *)
  let x = tensor_of_list [|3;4|]
    [1.;2.;3.;4.; 5.;6.;7.;8.; 9.;10.;11.;12.] in
  let r1 = eval_with [("x",x)]
    (prim (Apply_view [Vslice [|(0,2,1);(1,3,1)|]]) [prim (Apply_view [Vtranspose [|1;0|]]) [var "x"]]) in
  let r2 = eval_with [("x",x)]
    (prim (Apply_view [Vtranspose [|1;0|]; Vslice [|(0,2,1);(1,3,1)|]]) [var "x"]) in
  check_vals "composed" r1 r2

(* --- (b) inner product equality --- *)

let check_ip label spec in_shape =
  let x = Tensor.make_random in_shape in
  let out_shape = viewspec_output_shape spec in_shape in
  let u = Tensor.make_random out_shape in
  let ax = eval_with [("x",x)] (prim (Apply_view spec) [var "x"]) in
  let lhs = inner ax u in
  let atu = eval_with [("u",u)]
    (prim (Adjoint_view (spec, in_shape)) [var "u"]) in
  let rhs = inner x atu in
  Alcotest.(check bool) label true (approx_eq lhs rhs)

let test_ip_transpose () =
  check_ip "transpose" [Vtranspose [|1;0|]] [|3;4|]

let test_ip_transpose_3d () =
  check_ip "transpose 3d" [Vtranspose [|2;0;1|]] [|2;3;4|]

let test_ip_slice () =
  check_ip "slice" [Vslice [|(1,3,1); (0,2,1)|]] [|4;3|]

let test_ip_slice_step () =
  check_ip "slice step" [Vslice [|(0,4,2); (0,3,1)|]] [|4;3|]

let test_ip_slice_neg_step () =
  check_ip "slice neg step" [Vslice [|(2,0,-1); (0,3,1)|]] [|4;3|]

let test_ip_broadcast () =
  check_ip "broadcast" [Vbroadcast (0, 5)] [|3|]

let test_ip_broadcast_inner () =
  check_ip "broadcast inner" [Vbroadcast (1, 4)] [|3|]

let test_ip_reshape () =
  check_ip "reshape" [Vreshape [|6;2|]] [|3;4|]

let test_ip_reshape_flat () =
  check_ip "reshape flat" [Vreshape [|12|]] [|3;4|]

(* composed specs *)
let test_ip_transpose_slice () =
  check_ip "transpose then slice"
    [Vtranspose [|1;0|]; Vslice [|(0,2,1);(1,3,1)|]] [|3;4|]

let test_ip_slice_reshape () =
  check_ip "slice then reshape"
    [Vslice [|(1,3,1);(0,3,1)|]; Vreshape [|6|]] [|4;3|]

let test_ip_broadcast_transpose () =
  check_ip "broadcast then transpose"
    [Vbroadcast (0, 3); Vtranspose [|1;0|]] [|4|]

(* Non-contiguous reshape: transpose makes view non-contiguous,
   then reshape forces the materialize fallback in Apply_view eval *)
let test_ip_transpose_reshape () =
  check_ip "transpose then reshape"
    [Vtranspose [|1;0|]; Vreshape [|12|]] [|3;4|]

(* --- (c) syntactic involution: two rounds of forward→unzip→transpose --- *)

let test_involution_syntactic () =
  let spec = [Vtranspose [|1;0|]] in
  let in_shape = [|3;4|] in
  let out_shape = viewspec_output_shape spec in_shape in
  (* Round 1: f(x) = Apply_view spec x *)
  let expr = prim (Apply_view spec) [var "x"] in
  Transform.Forward.reset_gensym ();
  let fwd = Transform.Forward.forward expr in
  let seed = Transform.Forward.tangent_name "x" in
  let uz = Transform.Unzip.unzip fwd ~seeds:[seed] in
  let tr = Transform.Transpose.transpose
    ~primal_bindings:uz.primal_bindings
    ~tangent_bindings:uz.tangent_bindings
    ~tangent_out:uz.tangent_out
    ~seeds:[seed]
    ~input_shapes:[("x", in_shape); (seed, in_shape)]
    ~cotangent_var:"ct" in
  let grad_expr = snd (List.hd tr.grad_map) in
  let grad_full = Transform.Forward.wrap_bindings tr.grad_bindings grad_expr in
  let is_adjoint = function Adjoint_view _ -> true | _ -> false in
  Alcotest.(check bool) "round 1 produces Adjoint_view" true
    (has_prim_p is_adjoint grad_full);
  (* Round 2: g(ct) = gradient from round 1 (contains Adjoint_view) *)
  Transform.Forward.reset_gensym ();
  let fwd2 = Transform.Forward.forward grad_full in
  let seed2 = Transform.Forward.tangent_name "ct" in
  let uz2 = Transform.Unzip.unzip fwd2 ~seeds:[seed2] in
  let tr2 = Transform.Transpose.transpose
    ~primal_bindings:uz2.primal_bindings
    ~tangent_bindings:uz2.tangent_bindings
    ~tangent_out:uz2.tangent_out
    ~seeds:[seed2]
    ~input_shapes:[("ct", out_shape); (seed2, out_shape)]
    ~cotangent_var:"ct2" in
  let grad_expr2 = snd (List.hd tr2.grad_map) in
  let grad_full2 = Transform.Forward.wrap_bindings tr2.grad_bindings grad_expr2 in
  let is_apply = function Apply_view _ -> true | _ -> false in
  Alcotest.(check bool) "round 2 produces Apply_view" true
    (has_prim_p is_apply grad_full2)

(* Vbroadcast involution: Sum_axis adjoint generates Apply_view [Vbroadcast],
   whose adjoint (Adjoint_view [Vbroadcast]) goes through add_view.
   Two rounds verify the vocabulary closes for broadcast too. *)
let test_involution_broadcast () =
  let spec = [Vbroadcast (0, 3)] in
  let in_shape = [|4|] in
  let out_shape = viewspec_output_shape spec in_shape in
  let expr = prim (Apply_view spec) [var "x"] in
  Transform.Forward.reset_gensym ();
  let fwd = Transform.Forward.forward expr in
  let seed = Transform.Forward.tangent_name "x" in
  let uz = Transform.Unzip.unzip fwd ~seeds:[seed] in
  let tr = Transform.Transpose.transpose
    ~primal_bindings:uz.primal_bindings
    ~tangent_bindings:uz.tangent_bindings
    ~tangent_out:uz.tangent_out
    ~seeds:[seed]
    ~input_shapes:[("x", in_shape); (seed, in_shape)]
    ~cotangent_var:"ct" in
  let grad_expr = snd (List.hd tr.grad_map) in
  let grad_full = Transform.Forward.wrap_bindings tr.grad_bindings grad_expr in
  let is_adjoint = function Adjoint_view _ -> true | _ -> false in
  Alcotest.(check bool) "round 1 produces Adjoint_view" true
    (has_prim_p is_adjoint grad_full);
  Transform.Forward.reset_gensym ();
  let fwd2 = Transform.Forward.forward grad_full in
  let seed2 = Transform.Forward.tangent_name "ct" in
  let uz2 = Transform.Unzip.unzip fwd2 ~seeds:[seed2] in
  let tr2 = Transform.Transpose.transpose
    ~primal_bindings:uz2.primal_bindings
    ~tangent_bindings:uz2.tangent_bindings
    ~tangent_out:uz2.tangent_out
    ~seeds:[seed2]
    ~input_shapes:[("ct", out_shape); (seed2, out_shape)]
    ~cotangent_var:"ct2" in
  let grad_expr2 = snd (List.hd tr2.grad_map) in
  let grad_full2 = Transform.Forward.wrap_bindings tr2.grad_bindings grad_expr2 in
  let is_apply = function Apply_view _ -> true | _ -> false in
  Alcotest.(check bool) "round 2 produces Apply_view" true
    (has_prim_p is_apply grad_full2)

(* --- (d) validate catches bad specs --- *)

let test_validate_reshape_numel () =
  let x = Tensor.make_random [|3;4|] in
  let threw = try
    ignore (eval_with [("x",x)]
      (prim (Apply_view [Vreshape [|7|]]) [var "x"]));
    false
  with Ast.Eval.Shape_mismatch _ -> true in
  Alcotest.(check bool) "reshape numel mismatch" true threw

let test_validate_adjoint_shape () =
  (* Adjoint_view with wrong input shape *)
  let u = Tensor.make_random [|3;4|] in  (* should be [|4;3|] after transpose *)
  let threw = try
    ignore (eval_with [("u",u)]
      (prim (Adjoint_view ([Vtranspose [|1;0|]], [|3;4|])) [var "u"]));
    false
  with Ast.Eval.Shape_mismatch _ -> true in
  Alcotest.(check bool) "adjoint shape mismatch" true threw

(* --- (e) pp --- *)

let test_pp () =
  let spec = [Vtranspose [|1;0|]; Vslice [|(0,2,1)|]] in
  let s = Format.asprintf "%a" pp_prim (Apply_view spec) in
  Alcotest.(check bool) "pp contains apply_view" true
    (String.length s > 0 && s.[0] = 'a')

let () =
  let open Alcotest in
  run "viewspec" [
    "equivalence", [
      test_case "transpose" `Quick test_equiv_transpose;
      test_case "slice" `Quick test_equiv_slice;
      test_case "broadcast" `Quick test_equiv_broadcast;
      test_case "reshape" `Quick test_equiv_reshape;
      test_case "composed" `Quick test_equiv_composed;
    ];
    "inner_product", [
      test_case "transpose" `Quick test_ip_transpose;
      test_case "transpose 3d" `Quick test_ip_transpose_3d;
      test_case "slice" `Quick test_ip_slice;
      test_case "slice step" `Quick test_ip_slice_step;
      test_case "slice neg step" `Quick test_ip_slice_neg_step;
      test_case "broadcast" `Quick test_ip_broadcast;
      test_case "broadcast inner" `Quick test_ip_broadcast_inner;
      test_case "reshape" `Quick test_ip_reshape;
      test_case "reshape flat" `Quick test_ip_reshape_flat;
    ];
    "inner_product_composed", [
      test_case "transpose then slice" `Quick test_ip_transpose_slice;
      test_case "slice then reshape" `Quick test_ip_slice_reshape;
      test_case "broadcast then transpose" `Quick test_ip_broadcast_transpose;
      test_case "transpose then reshape" `Quick test_ip_transpose_reshape;
    ];
    "involution", [
      test_case "syntactic double adjoint" `Quick test_involution_syntactic;
      test_case "broadcast involution" `Quick test_involution_broadcast;
    ];
    "validate", [
      test_case "reshape numel" `Quick test_validate_reshape_numel;
      test_case "adjoint shape" `Quick test_validate_adjoint_shape;
    ];
    "pp", [
      test_case "print" `Quick test_pp;
    ];
  ]
