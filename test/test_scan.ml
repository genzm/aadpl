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

let test_cumulative_sum_and_product () =
  let values = [|0.9; 0.8; 0.7; 0.6|] in
  let input = tensor [|4|] (Array.to_list values) in
  let cumulative primitive =
    scan ~steps:4
      ~carries:[("acc", const (scalar 1.0),
        prim primitive [var "acc"; var "x"])]
      ~inputs:[("x", const input)] ~collect:true ~reverse:false (var "acc") in
  let expected_product = Array.copy values in
  let expected_sum = Array.copy values in
  for index = 1 to Array.length values - 1 do
    expected_product.(index) <- expected_product.(index - 1) *. values.(index);
    expected_sum.(index) <- expected_sum.(index - 1) +. values.(index)
  done;
  check_values "cumulative product / diffusion alpha-bar" expected_product
    (Ast.Eval.eval [] (cumulative Mul));
  let sum =
    scan ~steps:4
      ~carries:[("acc", const (scalar 0.0),
        prim Add [var "acc"; var "x"])]
      ~inputs:[("x", const input)] ~collect:true ~reverse:false (var "acc") in
  check_values "cumulative sum" expected_sum (Ast.Eval.eval [] sum)

let test_attention_head_split_view () =
  let input = tensor [|2; 3; 8|]
    (List.init 48 float_of_int) in
  let expression =
    prim (Apply_view [
      Vreshape [|2; 3; 2; 4|];
      Vtranspose [|0; 2; 1; 3|];
    ]) [const input] in
  Ast.Eval.reset_stats ();
  let result = Ast.Eval.eval [] expression in
  Alcotest.(check (array int)) "head shape" [|2; 2; 3; 4|]
    result.Tensor.view.Ndview.shape;
  Alcotest.(check int) "split and transpose stay a view" 0
    Ast.Eval.stats.materializations;
  check_values "head order"
    [|0.; 1.; 2.; 3.; 8.; 9.; 10.; 11.; 16.; 17.; 18.; 19.;
      4.; 5.; 6.; 7.; 12.; 13.; 14.; 15.; 20.; 21.; 22.; 23.;
      24.; 25.; 26.; 27.; 32.; 33.; 34.; 35.; 40.; 41.; 42.; 43.;
      28.; 29.; 30.; 31.; 36.; 37.; 38.; 39.; 44.; 45.; 46.; 47.|]
    result

let test_fuse_views_stops_at_scan_boundary () =
  let nested =
    prim (Apply_view [Vslice [|(0, 2, 1)|]])
      [prim (Apply_view [Vreshape [|2|]]) [var "z"]] in
  let expression =
    scan ~steps:1 ~carries:[("z", const (tensor [|2|] [1.0; 2.0]), nested)]
      ~inputs:[] ~collect:false ~reverse:false (var "z") in
  match Transform.Desugar.fuse_views expression with
  | Scan (_, { carries = [(_, _, Prim (_, Apply_view _,
      [Prim (_, Apply_view _, _)]))]; _ }, _) -> ()
  | _ -> Alcotest.fail "fuse_views crossed the Scan boundary"

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

let test_grad_time_varying_product () =
  let expression =
    scan ~steps:4
      ~carries:[("z", var "z0", prim Mul [var "a"; var "z"])]
      ~inputs:[("a", var "coefficients")] ~collect:false ~reverse:false
      (var "z") in
  let program = Transform.grad
    ~param_shapes:[("z0", [||]); ("coefficients", [|4|])] expression in
  let loss, gradients = Ast.Eval.eval_grad
    [("z0", scalar 1.0);
     ("coefficients", tensor [|4|] [2.0; 3.0; 5.0; 7.0])]
    ~primal_bindings:program.primal_bindings ~loss_body:program.loss_body
    ~grad_bindings:program.grad_bindings ~grad_bodies:program.grad_bodies in
  check_values "Scan grad loss" [|210.0|] loss;
  check_values "Scan grad z0" [|210.0|] (List.assoc "z0" gradients);
  check_values "Scan grad coefficients" [|105.0; 70.0; 42.0; 30.0|]
    (List.assoc "coefficients" gradients)

let test_time_varying_product_jvp () =
  let expression =
    scan ~steps:4
      ~carries:[("z", var "z0", prim Mul [var "a"; var "z"])]
      ~inputs:[("a", var "coefficients")] ~collect:false ~reverse:false
      (var "z") in
  Transform.Forward.reset_gensym ();
  let forward = Transform.Forward.forward expression in
  let seeds = [Transform.Forward.tangent_name "z0";
    Transform.Forward.tangent_name "coefficients"] in
  let unzipped = Transform.Unzip.unzip forward ~seeds in
  let evaluate coefficient_tangent =
    let environment = [
      ("z0", scalar 1.0);
      ("coefficients", tensor [|4|] [2.0; 3.0; 5.0; 7.0]);
      (Transform.Forward.tangent_name "z0", scalar 0.0);
      (Transform.Forward.tangent_name "coefficients", coefficient_tangent);
    ] in
    Transform.Forward.wrap_bindings
      (unzipped.primal_bindings @ unzipped.tangent_bindings)
      unzipped.tangent_out
    |> Ast.Eval.eval environment
    |> flat_values
    |> fun values -> values.(0) in
  let expected = [|105.0; 70.0; 42.0; 30.0|] in
  Array.iteri (fun index derivative ->
    let direction = Tensor.make [|4|] in
    Buf.set direction.buf index 1.0;
    Alcotest.(check (float 0.0))
      (Printf.sprintf "pre-update carry at step %d" index)
      derivative (evaluate direction)) expected

let test_nonlinear_residual_jvp_fd () =
  let expression =
    scan ~steps:3
      ~carries:[("z", var "z0",
        prim Add [prim Exp [var "z"]; var "x"])]
      ~inputs:[("x", var "inputs")] ~collect:false ~reverse:false
      (var "z") in
  let inputs = tensor [|3|] [0.1; -0.2; 0.05] in
  let evaluate z0 =
    Ast.Eval.eval [("z0", scalar z0); ("inputs", inputs)] expression
    |> flat_values |> fun values -> values.(0) in
  Transform.Forward.reset_gensym ();
  let unzipped = Transform.Forward.forward expression
    |> fun result -> Transform.Unzip.unzip result
      ~seeds:[Transform.Forward.tangent_name "z0"] in
  let tangent =
    Transform.Forward.wrap_bindings
      (unzipped.primal_bindings @ unzipped.tangent_bindings)
      unzipped.tangent_out
    |> Ast.Eval.eval [
      ("z0", scalar 0.1); ("inputs", inputs);
      (Transform.Forward.tangent_name "z0", scalar 1.0);
      (Transform.Forward.tangent_name "inputs", tensor [|3|] [0.0; 0.0; 0.0]);
    ]
    |> flat_values |> fun values -> values.(0) in
  let epsilon = 1e-6 in
  let finite_difference =
    (evaluate (0.1 +. epsilon) -. evaluate (0.1 -. epsilon)) /. (2.0 *. epsilon) in
  Alcotest.(check (float 1e-5)) "collected nonlinear residual"
    finite_difference tangent

let dot left right =
  let left = flat_values left and right = flat_values right in
  let total = ref 0.0 in
  Array.iteri (fun index value -> total := !total +. value *. right.(index)) left;
  !total

let test_scan_inner_product () =
  let expression =
    scan ~steps:4
      ~carries:[("z", var "z0", prim Mul [var "a"; var "z"])]
      ~inputs:[("a", var "coefficients")] ~collect:false ~reverse:false
      (var "z") in
  let coefficients = tensor [|4|] [2.0; 3.0; 5.0; 7.0]
  and coefficient_tangent = tensor [|4|] [0.2; -0.1; 0.3; 0.4] in
  let z0 = scalar 1.0 and z0_tangent = scalar 0.6 and cotangent = scalar 0.7 in
  let environment = [
    ("z0", z0); ("coefficients", coefficients);
    (Transform.Forward.tangent_name "z0", z0_tangent);
    (Transform.Forward.tangent_name "coefficients", coefficient_tangent);
    ("%ct", cotangent);
  ] in
  Transform.Forward.reset_gensym ();
  let unzipped = Transform.Forward.forward expression
    |> fun result -> Transform.Unzip.unzip result ~seeds:[
      Transform.Forward.tangent_name "z0";
      Transform.Forward.tangent_name "coefficients";
    ] in
  let tangent_value = Transform.Forward.wrap_bindings
    (unzipped.primal_bindings @ unzipped.tangent_bindings)
    unzipped.tangent_out |> Ast.Eval.eval environment in
  let seeds = [Transform.Forward.tangent_name "z0";
    Transform.Forward.tangent_name "coefficients"] in
  let shapes = [
    ("z0", [||]); ("coefficients", [|4|]);
    (Transform.Forward.tangent_name "z0", [||]);
    (Transform.Forward.tangent_name "coefficients", [|4|]);
  ] in
  let transposed = Transform.Transpose.transpose
    ~primal_bindings:unzipped.primal_bindings
    ~tangent_bindings:unzipped.tangent_bindings
    ~tangent_out:unzipped.tangent_out ~seeds ~input_shapes:shapes
    ~cotangent_var:"%ct" in
  let gradient seed =
    Transform.Forward.wrap_bindings
      (unzipped.primal_bindings @ transposed.grad_bindings)
      (List.assoc seed transposed.grad_map)
    |> Ast.Eval.eval environment in
  let right = dot (gradient (Transform.Forward.tangent_name "z0")) z0_tangent
    +. dot (gradient (Transform.Forward.tangent_name "coefficients"))
      coefficient_tangent in
  Alcotest.(check (float 1e-10)) "<Jv,u> = <v,J^T u>"
    (dot tangent_value cotangent) right

let test_rnn_all_parameter_fd () =
  let two = const (scalar 2.0) and one = const (scalar 1.0) in
  let affine = rank 0 Add [
    prim Matmul [var "w"; var "h"];
    prim Matmul [var "u"; var "x"];
  ] in
  let tanh = rank 0 Sub [
    rank 0 Mul [two; rank 0 Exp [rank 0 Logsigmoid [rank 0 Mul [two; affine]]]];
    one;
  ] in
  let expression =
    scan ~steps:3 ~carries:[("h", const (tensor [|1; 1|] [0.15]), tanh)]
      ~inputs:[("x", var "xs")] ~collect:false ~reverse:false
      (prim (Sum_axis 0) [prim (Sum_axis 0) [var "h"]]) in
  let shapes = [("w", [|1; 1|]); ("u", [|1; 1|])]
  and data_shapes = [("xs", [|3; 1; 1|])] in
  let xs = tensor [|3; 1; 1|] [0.2; -0.4; 0.7]
  and w = tensor [|1; 1|] [0.35]
  and u = tensor [|1; 1|] [-0.25] in
  let program = Transform.grad ~param_shapes:shapes ~data_shapes expression in
  let _, gradients = Ast.Eval.eval_grad [("w", w); ("u", u); ("xs", xs)]
    ~primal_bindings:program.primal_bindings ~loss_body:program.loss_body
    ~grad_bindings:program.grad_bindings ~grad_bodies:program.grad_bodies in
  let loss w_value u_value =
    let expanded = Transform.Expand_rank.expand ~senv:(shapes @ data_shapes)
      expression in
    Ast.Eval.eval [("w", tensor [|1; 1|] [w_value]);
      ("u", tensor [|1; 1|] [u_value]); ("xs", xs)] expanded
    |> flat_values |> fun values -> values.(0) in
  let epsilon = 1e-6 in
  List.iter (fun (name, center, other, vary_w) ->
    let positive = if vary_w then loss (center +. epsilon) other
      else loss other (center +. epsilon) in
    let negative = if vary_w then loss (center -. epsilon) other
      else loss other (center -. epsilon) in
    let finite_difference = (positive -. negative) /. (2.0 *. epsilon) in
    let actual = List.assoc name gradients |> flat_values |> fun values -> values.(0) in
    Alcotest.(check (float 1e-5)) (name ^ " RNN FD") finite_difference actual)
    [("w", 0.35, -0.25, true); ("u", -0.25, 0.35, false)]

let test_collect_and_reverse_grad_fd () =
  let coefficients = tensor [|4|] [0.8; 1.1; 0.7; 1.2] in
  let collected =
    scan ~steps:4
      ~carries:[("z", const (scalar 1.0), prim Mul [var "a"; var "z"])]
      ~inputs:[("a", var "coefficients")] ~collect:true ~reverse:false
      (prim (Sum_axis 0) [var "z"]) in
  let program = Transform.grad ~param_shapes:[("coefficients", [|4|])]
    collected in
  let _, gradients = Ast.Eval.eval_grad [("coefficients", coefficients)]
    ~primal_bindings:program.primal_bindings ~loss_body:program.loss_body
    ~grad_bindings:program.grad_bindings ~grad_bodies:program.grad_bodies in
  let actual = List.assoc "coefficients" gradients |> flat_values in
  let epsilon = 1e-6 in
  Array.iteri (fun index center ->
    let evaluate delta =
      let values = [0.8; 1.1; 0.7; 1.2] in
      let values = List.mapi (fun i value ->
        if i = index then center +. delta else value) values in
      Ast.Eval.eval [("coefficients", tensor [|4|] values)] collected
      |> flat_values |> fun result -> result.(0) in
    let expected = (evaluate epsilon -. evaluate (-.epsilon)) /. (2.0 *. epsilon) in
    Alcotest.(check (float 1e-6))
      (Printf.sprintf "collect coefficient %d FD" index) expected actual.(index))
    (flat_values coefficients);
  let reverse_expression =
    scan ~steps:3 ~carries:[("z", const (scalar 0.1),
      prim Add [prim Exp [prim Mul [var "w"; var "z"]]; var "x"])]
      ~inputs:[("x", const (tensor [|3|] [0.2; -0.1; 0.3]))]
      ~collect:false ~reverse:true (var "z") in
  let reverse_program = Transform.grad ~param_shapes:[("w", [||])]
    reverse_expression in
  let _, reverse_gradients = Ast.Eval.eval_grad [("w", scalar 0.2)]
    ~primal_bindings:reverse_program.primal_bindings
    ~loss_body:reverse_program.loss_body
    ~grad_bindings:reverse_program.grad_bindings
    ~grad_bodies:reverse_program.grad_bodies in
  let reverse_loss w = Ast.Eval.eval [("w", scalar w)] reverse_expression
    |> flat_values |> fun values -> values.(0) in
  let expected = (reverse_loss (0.2 +. epsilon) -. reverse_loss (0.2 -. epsilon))
    /. (2.0 *. epsilon) in
  let actual = List.assoc "w" reverse_gradients |> flat_values
    |> fun values -> values.(0) in
  Alcotest.(check (float 1e-5)) "reverse Scan FD" expected actual

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
      Alcotest.test_case "cumulative sum and product" `Quick
        test_cumulative_sum_and_product;
      Alcotest.test_case "attention head split remains a view" `Quick
        test_attention_head_split_view;
      Alcotest.test_case "fuse views stops at boundary" `Quick
        test_fuse_views_stops_at_scan_boundary;
    ];
    "guards", [
      Alcotest.test_case "body effects rejected with loc" `Quick
        test_body_effects_are_rejected;
      Alcotest.test_case "body Score rejected with loc" `Quick
        test_body_score_is_rejected;
      Alcotest.test_case "grad crosses Scan boundary" `Quick
        test_grad_time_varying_product;
    ];
    "forward", [
      Alcotest.test_case "time-varying product JVP uses pre-update carry" `Quick
        test_time_varying_product_jvp;
      Alcotest.test_case "nonlinear residual JVP agrees with FD" `Quick
        test_nonlinear_residual_jvp_fd;
      Alcotest.test_case "Scan inner-product identity" `Quick
        test_scan_inner_product;
      Alcotest.test_case "one-layer RNN all-parameter FD" `Quick
        test_rnn_all_parameter_fd;
      Alcotest.test_case "collect and reverse gradients agree with FD" `Quick
        test_collect_and_reverse_grad_fd;
    ];
  ]
