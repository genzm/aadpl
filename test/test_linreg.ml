(* Phase 8: linear regression — first end-to-end learning.
   Verifies: grad pipeline on real loss, gradient correctness,
   monotone loss decrease, convergence to OLS closed-form solution. *)

open View
open Ast.Types

(* --- tensor helpers --- *)

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

let tensor_of_array shape (vals : float array) =
  let t = Tensor.make shape in
  Array.iteri (fun i v -> Buf.set t.buf i v) vals;
  t

let scalar v = tensor_of_array [||] [|v|]
let scalar_val (t : Tensor.t) = Buf.get t.buf 0

let eval_expr env e =
  let env' = List.map (fun (s, t) -> (s, (t : Tensor.t :> value))) env in
  Ast.Eval.eval env' e

(* SGD update: w ← w − lr * g *)
let sgd_update (env : (string * Tensor.t) list) (grads : (string * Tensor.t) list)
    ~(lr : float) : (string * Tensor.t) list =
  List.map (fun (name, w) ->
    match List.assoc_opt name grads with
    | None -> (name, w)
    | Some g ->
      let n = numel w in
      let w' = Tensor.make w.view.Ndview.shape in
      for i = 0 to n - 1 do
        Buf.set w'.buf i (tensor_get w i -. lr *. tensor_get g i)
      done;
      (name, w')
  ) env

(* --- 8-1: L = Σ w², single variable, no Matmul --- *)

let test_8_1_grad () =
  (* loss(w) = sum(w * w) where w : [3] *)
  let loss_expr =
    prim (Sum_axis 0) [prim Mul [var "w"; var "w"]] in
  let gp = Transform.grad ~param_shapes:[("w", [|3|])] loss_expr in
  let w = tensor_of_array [|3|] [|1.0; 2.0; 3.0|] in
  let env = [("w", w)] in
  (* Expected gradient: d/dw sum(w²) = 2w *)
  let g = eval_expr env (List.assoc "w" gp.grads) in
  for i = 0 to 2 do
    Alcotest.(check (float 1e-12))
      (Printf.sprintf "8-1 grad[%d]" i)
      (2.0 *. tensor_get w i) (tensor_get g i)
  done

let test_8_1_loss_decreases () =
  let loss_expr =
    prim (Sum_axis 0) [prim Mul [var "w"; var "w"]] in
  let gp = Transform.grad ~param_shapes:[("w", [|3|])] loss_expr in
  let env = ref [("w", tensor_of_array [|3|] [|1.0; 2.0; 3.0|])] in
  let prev_loss = ref infinity in
  for _step = 1 to 100 do
    let l = scalar_val (eval_expr !env gp.loss) in
    Alcotest.(check bool) "loss decreases" true (l < !prev_loss);
    prev_loss := l;
    let g = eval_expr !env (List.assoc "w" gp.grads) in
    env := sgd_update !env [("w", g)] ~lr:0.1
  done;
  Alcotest.(check bool) "loss near zero" true (!prev_loss < 1e-10)

(* --- 8-2: L = (1/N) Σ (Xw − y)², with Matmul --- *)

let gen_linreg_data ~n ~d ~noise_std =
  Random.init 42;
  let w_true = Tensor.make_random [|d|] in
  let x = Tensor.make_random [|n; d|] in
  (* y = X @ w_true + noise *)
  let y_clean = eval_expr [("x", x); ("w", w_true)]
    (prim Matmul [var "x"; prim (Apply_view [Vreshape [|d; 1|]]) [var "w"]]) in
  let y = Tensor.make [|n|] in
  for i = 0 to n - 1 do
    let noise = noise_std *. (Random.float 2.0 -. 1.0) in
    Buf.set y.buf i (tensor_get y_clean i +. noise)
  done;
  (x, y, w_true)

(* OLS closed-form: w* = (XᵀX)⁻¹ Xᵀy *)
let ols_solve (x : Tensor.t) (y : Tensor.t) : Tensor.t =
  let n = x.view.Ndview.shape.(0) in
  let d = x.view.Ndview.shape.(1) in
  (* XᵀX : [d, d] *)
  let xtx = Array.make_matrix d d 0.0 in
  for i = 0 to d - 1 do
    for j = 0 to d - 1 do
      let s = ref 0.0 in
      for k = 0 to n - 1 do
        s := !s +. tensor_get x (k * d + i) *. tensor_get x (k * d + j)
      done;
      xtx.(i).(j) <- !s
    done
  done;
  (* Xᵀy : [d] *)
  let xty = Array.make d 0.0 in
  for i = 0 to d - 1 do
    let s = ref 0.0 in
    for k = 0 to n - 1 do
      s := !s +. tensor_get x (k * d + i) *. tensor_get y k
    done;
    xty.(i) <- !s
  done;
  (* Gaussian elimination with partial pivoting *)
  let a = Array.init d (fun i -> Array.init (d + 1) (fun j ->
    if j < d then xtx.(i).(j) else xty.(i))) in
  for col = 0 to d - 1 do
    (* pivot *)
    let max_row = ref col in
    for row = col + 1 to d - 1 do
      if abs_float a.(row).(col) > abs_float a.(!max_row).(col) then
        max_row := row
    done;
    let tmp = a.(col) in a.(col) <- a.(!max_row); a.(!max_row) <- tmp;
    (* eliminate *)
    let pivot = a.(col).(col) in
    for j = col to d do a.(col).(j) <- a.(col).(j) /. pivot done;
    for row = 0 to d - 1 do
      if row <> col then begin
        let factor = a.(row).(col) in
        for j = col to d do
          a.(row).(j) <- a.(row).(j) -. factor *. a.(col).(j)
        done
      end
    done
  done;
  let w_star = Tensor.make [|d|] in
  for i = 0 to d - 1 do Buf.set w_star.buf i a.(i).(d) done;
  w_star

let make_linreg_loss_expr ~n ~d (x : Tensor.t) (y : Tensor.t) =
  (* loss(w) = (1/N) sum((X @ w_col - y)²)
     where w : [d], w_col = reshape(w, [d,1]),
     X @ w_col : [n,1], squeeze to [n] via reshape *)
  let inv_n = scalar (1.0 /. float_of_int n) in
  let_ "pred"
    (prim (Apply_view [Vreshape [|n|]])
       [prim Matmul [const x;
                     prim (Apply_view [Vreshape [|d; 1|]]) [var "w"]]])
    (let_ "diff" (prim Sub [var "pred"; const y])
       (prim Mul [const inv_n;
                  prim (Sum_axis 0) [prim Mul [var "diff"; var "diff"]]]))

let test_8_2_grad_exact () =
  let n = 50 and d = 3 in
  let (x, y, _w_true) = gen_linreg_data ~n ~d ~noise_std:0.1 in
  let loss_expr = make_linreg_loss_expr ~n ~d x y in
  let gp = Transform.grad ~param_shapes:[("w", [|d|])] loss_expr in
  let w0 = Tensor.make_random [|d|] in
  let env = [("w", w0)] in
  (* Closed-form gradient: (2/N) Xᵀ(Xw − y) *)
  let xw = eval_expr env
    (prim (Apply_view [Vreshape [|n|]])
       [prim Matmul [const x;
                     prim (Apply_view [Vreshape [|d; 1|]]) [var "w"]]]) in
  let residual = Tensor.make [|n|] in
  for i = 0 to n - 1 do
    Buf.set residual.buf i (tensor_get xw i -. tensor_get y i)
  done;
  let expected_grad = Tensor.make [|d|] in
  for j = 0 to d - 1 do
    let s = ref 0.0 in
    for i = 0 to n - 1 do
      s := !s +. tensor_get x (i * d + j) *. tensor_get residual i
    done;
    Buf.set expected_grad.buf j (2.0 *. !s /. float_of_int n)
  done;
  let g = eval_expr env (List.assoc "w" gp.grads) in
  for j = 0 to d - 1 do
    Alcotest.(check (float 1e-10))
      (Printf.sprintf "8-2 grad[%d]" j)
      (tensor_get expected_grad j) (tensor_get g j)
  done

let test_8_2_converge () =
  let n = 100 and d = 3 in
  let (x, y, _w_true) = gen_linreg_data ~n ~d ~noise_std:0.05 in
  let loss_expr = make_linreg_loss_expr ~n ~d x y in
  let gp = Transform.grad ~param_shapes:[("w", [|d|])] loss_expr in
  let w_star = ols_solve x y in
  let env = ref [("w", Tensor.make [|d|])] in  (* w0 = zeros *)
  let prev_loss = ref infinity in
  for _step = 1 to 2000 do
    let l = scalar_val (eval_expr !env gp.loss) in
    Alcotest.(check bool) "loss decreases" true (l < !prev_loss +. 1e-15);
    prev_loss := l;
    let g = eval_expr !env (List.assoc "w" gp.grads) in
    env := sgd_update !env [("w", g)] ~lr:0.01
  done;
  (* Check convergence to OLS *)
  let w_final = List.assoc "w" !env in
  let max_err = ref 0.0 in
  let max_abs = ref 0.0 in
  for j = 0 to d - 1 do
    let err = abs_float (tensor_get w_final j -. tensor_get w_star j) in
    let abs_val = abs_float (tensor_get w_star j) in
    max_err := max !max_err err;
    max_abs := max !max_abs abs_val
  done;
  let rel_err = !max_err /. (max !max_abs 1e-10) in
  Alcotest.(check bool)
    (Printf.sprintf "converged to OLS (rel_err=%.2e)" rel_err)
    true (rel_err < 1e-3)

(* --- 8-3: bias via rank (Apply_view[Vbroadcast] adjoint in training) --- *)

let make_linreg_bias_loss_expr ~n ~d (x : Tensor.t) (y : Tensor.t) =
  (* loss(w, b) = (1/N) sum((X@w_col + b - y)²)
     b : scalar [||], broadcast to [n] via rank 0 Add *)
  let inv_n = scalar (1.0 /. float_of_int n) in
  let_ "xw"
    (prim (Apply_view [Vreshape [|n|]])
       [prim Matmul [const x;
                     prim (Apply_view [Vreshape [|d; 1|]]) [var "w"]]])
    (let_ "pred" (rank 0 Add [var "b"; var "xw"])
       (let_ "diff" (prim Sub [var "pred"; const y])
          (prim Mul [const inv_n;
                     prim (Sum_axis 0) [prim Mul [var "diff"; var "diff"]]])))

let test_8_3_bias_grad () =
  let n = 50 and d = 3 in
  let (x, y, _w_true) = gen_linreg_data ~n ~d ~noise_std:0.1 in
  let loss_expr = make_linreg_bias_loss_expr ~n ~d x y in
  let gp = Transform.grad
    ~param_shapes:[("w", [|d|]); ("b", [||])] loss_expr in
  let w0 = Tensor.make_random [|d|] in
  let b0 = scalar 0.5 in
  let env = [("w", w0); ("b", b0)] in
  (* Evaluate gradients — just check they have correct shapes *)
  let gw = eval_expr env (List.assoc "w" gp.grads) in
  let gb = eval_expr env (List.assoc "b" gp.grads) in
  Alcotest.(check int) "gw rank" 1 (Array.length gw.view.Ndview.shape);
  Alcotest.(check int) "gw dim" d gw.view.Ndview.shape.(0);
  Alcotest.(check int) "gb rank" 0 (Array.length gb.view.Ndview.shape);
  (* Numerical gradient check for bias *)
  let eps = 1e-5 in
  let b_plus = scalar (0.5 +. eps) in
  let b_minus = scalar (0.5 -. eps) in
  let l_plus = scalar_val (eval_expr [("w",w0);("b",b_plus)] gp.loss) in
  let l_minus = scalar_val (eval_expr [("w",w0);("b",b_minus)] gp.loss) in
  let fd_grad = (l_plus -. l_minus) /. (2.0 *. eps) in
  let ad_grad = scalar_val gb in
  Alcotest.(check (float 1e-4)) "bias grad vs finite diff" fd_grad ad_grad

let test_8_3_bias_converge () =
  let n = 100 and d = 3 in
  let (x, y, _w_true) = gen_linreg_data ~n ~d ~noise_std:0.05 in
  let loss_expr = make_linreg_bias_loss_expr ~n ~d x y in
  let gp = Transform.grad
    ~param_shapes:[("w", [|d|]); ("b", [||])] loss_expr in
  let env = ref [("w", Tensor.make [|d|]); ("b", scalar 0.0)] in
  let prev_loss = ref infinity in
  for _step = 1 to 2000 do
    let l = scalar_val (eval_expr !env gp.loss) in
    Alcotest.(check bool) "loss decreases" true (l < !prev_loss +. 1e-15);
    prev_loss := l;
    let gw = eval_expr !env (List.assoc "w" gp.grads) in
    let gb = eval_expr !env (List.assoc "b" gp.grads) in
    env := sgd_update !env [("w", gw); ("b", gb)] ~lr:0.01
  done;
  (* OLS via augmented X̃ = [X, 1]: solve [w*; b*] = (X̃ᵀX̃)⁻¹ X̃ᵀy *)
  let x_aug = Tensor.make [|n; d + 1|] in
  for i = 0 to n - 1 do
    for j = 0 to d - 1 do
      Buf.set x_aug.buf (i * (d + 1) + j) (tensor_get x (i * d + j))
    done;
    Buf.set x_aug.buf (i * (d + 1) + d) 1.0
  done;
  let wb_star = ols_solve x_aug y in
  let w_final = List.assoc "w" !env in
  let b_final = List.assoc "b" !env in
  let max_err = ref 0.0 in
  let max_abs = ref 0.0 in
  for j = 0 to d - 1 do
    let err = abs_float (tensor_get w_final j -. tensor_get wb_star j) in
    let abs_val = abs_float (tensor_get wb_star j) in
    max_err := max !max_err err;
    max_abs := max !max_abs abs_val
  done;
  let b_err = abs_float (scalar_val b_final -. tensor_get wb_star d) in
  max_err := max !max_err b_err;
  max_abs := max !max_abs (abs_float (tensor_get wb_star d));
  let rel_err = !max_err /. (max !max_abs 1e-10) in
  Alcotest.(check bool)
    (Printf.sprintf "converged to OLS (rel_err=%.2e)" rel_err)
    true (rel_err < 1e-2)

(* --- 9-0: stats instrumentation smoke test --- *)

let test_9_0_stats () =
  let n = 50 and d = 3 in
  let (x, y, _) = gen_linreg_data ~n ~d ~noise_std:0.1 in
  let loss_expr = make_linreg_loss_expr ~n ~d x y in
  let gp = Transform.grad ~param_shapes:[("w", [|d|])] loss_expr in
  let env = ref [("w", Tensor.make [|d|])] in
  Ast.Eval.reset_stats ();
  Ast.Eval.enable_stats ();
  for _step = 1 to 10 do
    let _l = scalar_val (eval_expr !env gp.loss) in
    let g = eval_expr !env (List.assoc "w" gp.grads) in
    env := sgd_update !env [("w", g)] ~lr:0.01
  done;
  Printf.printf "\n--- 9-0 stats report (10 steps, N=%d, D=%d) ---\n" n d;
  Ast.Eval.report ();
  Ast.Eval.disable_stats ();
  (* Verify stats were actually collected *)
  let s = Ast.Eval.stats in
  Alcotest.(check bool) "buffers allocated > 0"
    true (s.buffers_allocated > 0);
  Alcotest.(check bool) "kernel stats non-empty"
    true (Hashtbl.length s.kernel_stats > 0)

let () =
  let open Alcotest in
  run "linreg" [
    "8-1 w²", [
      test_case "gradient = 2w" `Quick test_8_1_grad;
      test_case "loss decreases to zero" `Quick test_8_1_loss_decreases;
    ];
    "8-2 linreg", [
      test_case "gradient = (2/N)Xᵀ(Xw−y)" `Quick test_8_2_grad_exact;
      test_case "converge to OLS" `Quick test_8_2_converge;
    ];
    "8-3 bias", [
      test_case "bias grad shape + fd check" `Quick test_8_3_bias_grad;
      test_case "bias converge" `Quick test_8_3_bias_converge;
    ];
    "9-0 stats", [
      test_case "instrumentation smoke" `Quick test_9_0_stats;
    ];
  ]
