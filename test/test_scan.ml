open View
open Ast.Types

let tensor shape values =
  let result = Tensor.make shape in
  List.iteri (fun index value -> Buf.set result.buf index value) values;
  result

let scalar value = tensor [||] [value]

let flat_values value =
  let shape = value.Tensor.view.Ndview.shape in
  let count = Ndview.numel value.view in
  Array.init count (fun linear ->
    let index = Array.make (Array.length shape) 0 in
    let rest = ref linear in
    for axis = Array.length shape - 1 downto 0 do
      index.(axis) <- !rest mod shape.(axis);
      rest := !rest / shape.(axis)
    done;
    Buf.get value.buf (Ndview.index_of value.view index))

let check_values label expected actual =
  Alcotest.(check (array (float 0.0))) label expected (flat_values actual)

let additive_scan ?(collect = false) ?(reverse = false) input =
  scan ~steps:4
    ~carries:[("z", const (scalar 1.0), prim Add [var "z"; var "x"])]
    ~inputs:[("x", input)] ~collect ~reverse (var "z")

let test_unrolled_bit_agreement () =
  let input = tensor [|4|] [2.0; 3.0; 4.0; 5.0] in
  let scanned = Ast.Eval.eval [] (additive_scan (const input)) in
  let unrolled =
    let_ "z0" (const (scalar 1.0))
      (let_ "z1" (prim Add [var "z0"; const (scalar 2.0)])
        (let_ "z2" (prim Add [var "z1"; const (scalar 3.0)])
          (let_ "z3" (prim Add [var "z2"; const (scalar 4.0)])
            (let_ "z4" (prim Add [var "z3"; const (scalar 5.0)])
              (var "z4"))))) in
  let expected = Ast.Eval.eval [] unrolled in
  check_values "finite expansion" (flat_values expected) scanned

let test_collect_execution_order () =
  let input = tensor [|4|] [2.0; 3.0; 4.0; 5.0] in
  let result = Ast.Eval.eval [] (additive_scan ~collect:true (const input)) in
  check_values "post-update trajectory" [|3.0; 6.0; 10.0; 15.0|] result

let test_reverse_matches_reversed_forward () =
  let input = tensor [|4|] [2.0; 3.0; 4.0; 5.0] in
  let reverse_result =
    Ast.Eval.eval [] (additive_scan ~collect:true ~reverse:true (const input)) in
  let reversed_input =
    prim (Apply_view [Vslice [|(3, -1, -1)|]]) [const input] in
  let forward_result =
    Ast.Eval.eval [] (additive_scan ~collect:true reversed_input) in
  check_values "reverse input equivalence" (flat_values forward_result)
    reverse_result

let test_multiple_carries_are_simultaneous () =
  let expression =
    scan ~steps:3
      ~carries:[
        ("a", const (scalar 1.0), var "b");
        ("b", const (scalar 10.0), var "a");
      ] ~inputs:[] ~collect:true ~reverse:false
      (prim Sub [var "a"; var "b"]) in
  let result = Ast.Eval.eval [] expression in
  check_values "simultaneous updates" [|9.0; -9.0; 9.0|] result

let test_expand_rank_inside_body () =
  let initial = tensor [|2|] [1.0; 10.0] in
  let input = tensor [|3; 2|] [2.0; 3.0; 4.0; 5.0; 6.0; 7.0] in
  let expression =
    scan ~steps:3
      ~carries:[("z", const initial, rank 0 Add [var "z"; var "x"])]
      ~inputs:[("x", const input)] ~collect:true ~reverse:false (var "z") in
  let expanded = Transform.Expand_rank.expand expression in
  Alcotest.(check bool) "expanded body" true
    (Transform.Expand_rank.is_expanded expanded);
  check_values "rank body trajectory"
    [|3.0; 13.0; 7.0; 18.0; 13.0; 25.0|]
    (Ast.Eval.eval [] expanded)

let test_body_effects_are_rejected () =
  let sample_loc = { file = "scan-test"; line = 17; col = 3 } in
  let expression =
    scan ~steps:1
      ~carries:[("z", const (scalar 0.0),
        Sample (sample_loc, "inside", [||], D_uniform))]
      ~inputs:[] ~collect:false ~reverse:false (var "z") in
  match Ast.Eval.eval [] expression with
  | _ -> Alcotest.fail "Sample in Scan body was accepted"
  | exception Ast.Eval.Eval_error (loc, message) ->
      Alcotest.(check string) "effect loc file" "scan-test" loc.file;
      Alcotest.(check int) "effect loc line" 17 loc.line;
      Alcotest.(check string) "effect message"
        "Scan body must not contain Sample" message

let test_body_score_is_rejected () =
  let score_loc = { file = "scan-test"; line = 23; col = 5 } in
  let expression =
    scan ~steps:1
      ~carries:[("z", const (scalar 0.0),
        Score (score_loc, const (scalar 1.0)))]
      ~inputs:[] ~collect:false ~reverse:false (var "z") in
  match Ast.Eval.eval [] expression with
  | _ -> Alcotest.fail "Score in Scan body was accepted"
  | exception Ast.Eval.Eval_error (loc, message) ->
      Alcotest.(check string) "score loc file" "scan-test" loc.file;
      Alcotest.(check int) "score loc line" 23 loc.line;
      Alcotest.(check string) "score message"
        "Scan body must not contain Score" message

let test_grad_rejects_scan1 () =
  let scan_loc = { file = "scan-test"; line = 29; col = 1 } in
  let spec = {
    steps = 1;
    carries = [("z", const (scalar 0.0), var "z")];
    inputs = [];
    collect = false;
    reverse = false;
  } in
  let expression = Scan (scan_loc, spec, var "z") in
  match Transform.grad ~param_shapes:[] expression with
  | _ -> Alcotest.fail "Scan₁ reached grad"
  | exception Transform.Grad_error (loc, message) ->
      Alcotest.(check string) "grad loc file" "scan-test" loc.file;
      Alcotest.(check int) "grad loc line" 29 loc.line;
      Alcotest.(check string) "grad message"
        "Scan is not differentiable yet" message

let () =
  Alcotest.run "scan" [
    "eval", [
      Alcotest.test_case "finite expansion bit agreement" `Quick
        test_unrolled_bit_agreement;
      Alcotest.test_case "collect execution order" `Quick
        test_collect_execution_order;
      Alcotest.test_case "reverse equals reversed forward" `Quick
        test_reverse_matches_reversed_forward;
      Alcotest.test_case "multiple carries update simultaneously" `Quick
        test_multiple_carries_are_simultaneous;
      Alcotest.test_case "expand rank inside body" `Quick
        test_expand_rank_inside_body;
    ];
    "guards", [
      Alcotest.test_case "body effects rejected with loc" `Quick
        test_body_effects_are_rejected;
      Alcotest.test_case "body Score rejected with loc" `Quick
        test_body_score_is_rejected;
      Alcotest.test_case "grad rejects Scan1 with loc" `Quick
        test_grad_rejects_scan1;
    ];
  ]
