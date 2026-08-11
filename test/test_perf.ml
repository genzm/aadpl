(* Phase 9-1: size distribution measurement.
   Three workloads — linreg, MLP, N-parallel linreg — evaluated
   with stats enabled.  The printed tables are the deliverable;
   the asserts just sanity-check the instrumentation. *)

open View
open Ast.Types

(* --- tensor helpers --- *)

let tensor_of_array shape (vals : float array) =
  let t = Tensor.make shape in
  Array.iteri (fun i v -> Buf.set t.buf i v) vals;
  t

let scalar v = tensor_of_array [||] [|v|]
let scalar_val (t : Tensor.t) = Buf.get t.buf 0

let eval_expr env e =
  let env' = List.map (fun (s, t) -> (s, (t : Tensor.t :> value))) env in
  Ast.Eval.eval env' e

(* --- workload 1: linreg (n=50, d=3, 10 steps) --- *)

let bench_linreg () =
  Random.init 42;
  let n = 50 and d = 3 in
  let x = Tensor.make_random [|n; d|] in
  let w_true = Tensor.make_random [|d|] in
  let y_clean = eval_expr [("x", x); ("w", w_true)]
    (prim Matmul [var "x"; prim (Apply_view [Vreshape [|d;1|]]) [var "w"]]) in
  let y = Tensor.make [|n|] in
  for i = 0 to n - 1 do
    Buf.set y.buf i (Buf.get y_clean.buf (Ndview.index_of y_clean.view
      (let idx = Array.make (Array.length y_clean.view.Ndview.shape) 0 in
       let tmp = ref i in
       for k = Array.length y_clean.view.Ndview.shape - 1 downto 0 do
         idx.(k) <- !tmp mod y_clean.view.Ndview.shape.(k);
         tmp := !tmp / y_clean.view.Ndview.shape.(k)
       done; idx)))
  done;
  let inv_n = scalar (1.0 /. float_of_int n) in
  let loss_expr =
    let_ "pred"
      (prim (Apply_view [Vreshape [|n|]])
         [prim Matmul [const x;
                       prim (Apply_view [Vreshape [|d;1|]]) [var "w"]]])
      (let_ "diff" (prim Sub [var "pred"; const y])
         (prim Mul [const inv_n;
                    prim (Sum_axis 0) [prim Mul [var "diff"; var "diff"]]])) in
  let gp = Transform.grad ~param_shapes:[("w", [|d|])] loss_expr in
  let env = ref [("w", Tensor.make [|d|])] in
  Ast.Eval.reset_stats ();
  Ast.Eval.enable_stats ();
  for _step = 1 to 10 do
    let _l = scalar_val (eval_expr !env gp.loss) in
    let g = eval_expr !env (List.assoc "w" gp.grads) in
    (* sgd update *)
    let w = List.assoc "w" !env in
    let w' = Tensor.make [|d|] in
    for i = 0 to d - 1 do
      Buf.set w'.buf i (Buf.get w.buf i -. 0.01 *. Buf.get g.buf i)
    done;
    env := [("w", w')]
  done;
  Ast.Eval.disable_stats ();
  Printf.printf "\n=== Workload 1: linreg (n=%d, d=%d, 10 steps) ===\n" n d;
  Ast.Eval.report ();
  let s = Ast.Eval.stats in
  Alcotest.(check bool) "linreg: kernels recorded"
    true (Hashtbl.length s.kernel_stats > 0)

(* --- workload 2: MLP forward+grad (784→400→10, batch 64) --- *)

let bench_mlp () =
  Random.init 42;
  let batch = 64 and d_in = 784 and d_hid = 400 and d_out = 10 in
  (* dummy data *)
  let x  = Tensor.make_random [|batch; d_in|] in
  let w1 = Tensor.make_random [|d_in; d_hid|] in
  let b1 = Tensor.make_random [|d_hid|] in
  let w2 = Tensor.make_random [|d_hid; d_out|] in
  let b2 = Tensor.make_random [|d_out|] in
  (* labels: random class indices as float tensor *)
  let y  = Tensor.make [|batch|] in
  for i = 0 to batch - 1 do
    Buf.set y.buf i (float_of_int (Random.int d_out))
  done;
  (* MLP expression:
     h = relu(x @ W1 + b1)          — rank 0 Add for bias broadcast
     logits = h @ W2 + b2
     lse = log(sum_axis(1, exp(logits)))  : [batch]
     picked = select_axis(1, logits, y)   : [batch]
     loss = (1/B) * sum_axis(0, lse - picked)  *)
  let inv_b = scalar (1.0 /. float_of_int batch) in
  let loss_expr =
    let_ "h"
      (prim Relu [rank 1 Add [prim Matmul [const x; var "w1"]; var "b1"]])
      (let_ "logits"
         (rank 1 Add [prim Matmul [var "h"; var "w2"]; var "b2"])
         (let_ "m" (prim (Max_axis 1) [var "logits"])
            (let_ "shifted" (rank 0 Sub [var "logits"; var "m"])
               (let_ "lse"
                  (rank 0 Add [var "m";
                     prim Log [prim (Sum_axis 1) [prim Exp [var "shifted"]]]])
                  (let_ "picked"
                     (prim (Select_axis 1) [var "logits"; const y])
                     (prim Mul [const inv_b;
                                prim (Sum_axis 0)
                                  [prim Sub [var "lse"; var "picked"]]]))))))
  in
  let param_shapes = [
    ("w1", [|d_in; d_hid|]);
    ("b1", [|d_hid|]);
    ("w2", [|d_hid; d_out|]);
    ("b2", [|d_out|]);
  ] in
  let gp = Transform.grad ~param_shapes loss_expr in
  let env = [("w1", w1); ("b1", b1); ("w2", w2); ("b2", b2)] in
  Ast.Eval.reset_stats ();
  Ast.Eval.enable_stats ();
  (* forward + grad, 1 step (grad already runs loss internally) *)
  let _loss = scalar_val (eval_expr env gp.loss) in
  List.iter (fun (name, gexpr) ->
    let _g = eval_expr env gexpr in
    ignore name
  ) gp.grads;
  Ast.Eval.disable_stats ();
  Printf.printf "\n=== Workload 2: MLP (%d→%d→%d, batch %d, 1 eval) ===\n"
    d_in d_hid d_out batch;
  Ast.Eval.report ();
  let s = Ast.Eval.stats in
  (* gemm should be in >=10k bin *)
  let matmul_stats = Hashtbl.find_opt s.kernel_stats "matmul" in
  (match matmul_stats with
   | Some ps ->
     Alcotest.(check bool) "MLP: matmul hits >=10k bin"
       true (ps.bin_large > 0)
   | None -> Alcotest.fail "MLP: no matmul recorded")

(* --- workload 3: N-parallel linreg via rank --- *)

let bench_parallel () =
  Random.init 42;
  let n_problems = 1000 and n = 50 and d = 3 in
  (* X : [N, n, d],  y : [N, n],  w : [N, d] *)
  let x_big = Tensor.make_random [|n_problems; n; d|] in
  let y_big = Tensor.make_random [|n_problems; n|] in
  let w_big = Tensor.make_random [|n_problems; d|] in
  (* Per-problem loss at cell rank:
       pred = reshape(Matmul(X_i, reshape(w_i, [d,1])), [n])
       diff = pred - y_i
       loss_i = (1/n) * sum(diff * diff)
     Lift everything with rank.

     But Matmul has cell_rank=2. X_i:[n,d] is rank 2 = cell_rank,
     so the frame for X comes from the batch dim N.
     w_i:[d] is rank 1, but Matmul needs rank 2, so we still need
     the reshape trick: reshape(w,[N,d]) → [N,d,1] then matmul⎉2.
     Result [N,n,1] → reshape to [N,n].

     Actually, let's be explicit with the reshapes and use rank 0 for
     the elementwise, rank 1 for the sum: *)
  let inv_n = scalar (1.0 /. float_of_int n) in
  let loss_expr =
    let_ "w_col"
      (prim (Apply_view [Vreshape [|n_problems; d; 1|]]) [var "w"])
      (let_ "xw"
         (* matmul⎉2: X:[N,n,d], w_col:[N,d,1] → [N,n,1] *)
         (rank 2 Matmul [var "x"; var "w_col"])
         (let_ "pred"
            (prim (Apply_view [Vreshape [|n_problems; n|]]) [var "xw"])
            (let_ "diff"
               (rank 0 Sub [var "pred"; var "y"])
               (let_ "sq"
                  (rank 0 Mul [var "diff"; var "diff"])
                  (rank 0 Mul [const inv_n;
                               rank 1 (Sum_axis 0) [var "sq"]])))))
  in
  let param_shapes = [
    ("w", [|n_problems; d|]);
    ("x", [|n_problems; n; d|]);
    ("y", [|n_problems; n|]);
  ] in
  (* Just expand + eval, no grad (too slow for 1000 problems, not needed for Q1) *)
  let expanded = Transform.Expand_rank.expand ~senv:param_shapes loss_expr in
  let expanded = Transform.Desugar.fuse_views expanded in
  let env = [("x", x_big); ("y", y_big); ("w", w_big)] in
  Ast.Eval.reset_stats ();
  Ast.Eval.enable_stats ();
  let _result = eval_expr env expanded in
  Ast.Eval.disable_stats ();
  Printf.printf "\n=== Workload 3: N-parallel linreg (N=%d, n=%d, d=%d, 1 forward) ===\n"
    n_problems n d;
  Ast.Eval.report ();
  let s = Ast.Eval.stats in
  Alcotest.(check bool) "parallel: kernels recorded"
    true (Hashtbl.length s.kernel_stats > 0);
  (* Q1 check: are elementwise ops batched (few calls, large elems)
     or per-problem (many calls, small elems)?
     Just report — the user will judge from the table. *)
  let total_calls = Hashtbl.fold (fun _ (ps : Ast.Eval.prim_stats) acc ->
    acc + ps.calls) s.kernel_stats 0 in
  Printf.printf "  → total kernel calls: %d\n" total_calls

(* --- workload 4: N-parallel linreg with grad (9-2) --- *)

let bench_parallel_grad () =
  Random.init 42;
  let n_problems = 1000 and n = 50 and d = 3 in
  let x_big = Tensor.make_random [|n_problems; n; d|] in
  let y_big = Tensor.make_random [|n_problems; n|] in
  (* loss(w) where w:[N,d].
     Per-problem MSE, then sum across N to get scalar.
     x, y are const (not differentiated). *)
  let inv_n = scalar (1.0 /. float_of_int n) in
  let inv_np = scalar (1.0 /. float_of_int n_problems) in
  let loss_expr =
    let_ "w_col"
      (prim (Apply_view [Vreshape [|n_problems; d; 1|]]) [var "w"])
      (let_ "xw"
         (rank 2 Matmul [const x_big; var "w_col"])
         (let_ "pred"
            (prim (Apply_view [Vreshape [|n_problems; n|]]) [var "xw"])
            (let_ "diff"
               (rank 0 Sub [var "pred"; const y_big])
               (let_ "sq"
                  (rank 0 Mul [var "diff"; var "diff"])
                  (let_ "per_problem"
                     (rank 0 Mul [const inv_n;
                                  rank 1 (Sum_axis 0) [var "sq"]])
                     (prim Mul [const inv_np;
                                prim (Sum_axis 0) [var "per_problem"]]))))))
  in
  let param_shapes = [("w", [|n_problems; d|])] in
  let gp = Transform.grad ~param_shapes loss_expr in
  let w0 = Tensor.make_random [|n_problems; d|] in
  let env = ref [("w", w0)] in
  Ast.Eval.reset_stats ();
  Ast.Eval.enable_stats ();
  (* 1 step: eval loss + grad + sgd update *)
  let loss_val = scalar_val (eval_expr !env gp.loss) in
  let gw = eval_expr !env (List.assoc "w" gp.grads) in
  (* sgd update *)
  let w_cur = List.assoc "w" !env in
  let w_new = Tensor.make [|n_problems; d|] in
  let total = n_problems * d in
  for i = 0 to total - 1 do
    Buf.set w_new.buf i (Buf.get w_cur.buf i -. 0.01 *. Buf.get gw.buf i)
  done;
  env := [("w", w_new)];
  let loss_val2 = scalar_val (eval_expr !env gp.loss) in
  Ast.Eval.disable_stats ();
  Printf.printf "\n=== Workload 4: N-parallel linreg grad (N=%d, n=%d, d=%d, 1 step) ===\n"
    n_problems n d;
  Ast.Eval.report ();
  Printf.printf "  loss: %.6f → %.6f\n" loss_val loss_val2;
  Alcotest.(check bool) "loss decreased" true (loss_val2 < loss_val);
  let s = Ast.Eval.stats in
  let total_calls = Hashtbl.fold (fun _ (ps : Ast.Eval.prim_stats) acc ->
    acc + ps.calls) s.kernel_stats 0 in
  Printf.printf "  → total kernel calls: %d\n" total_calls

(* --- 9-3: microbench — c_node / c_elem measurement --- *)

let bench_micro () =
  Random.init 42;
  let iters = 1000 in
  (* --- elementwise (add): vary numel --- *)
  let sizes = [| 10; 100; 1000; 10000; 100000 |] in
  Printf.printf "\n=== 9-3 microbench: add (elementwise) ===\n";
  Printf.printf "%10s %10s %12s %12s\n" "numel" "iters" "total(ms)" "per-call(μs)";
  Printf.printf "%s\n" (String.make 50 '-');
  Array.iter (fun sz ->
    let a = Tensor.make_random [|sz|] in
    let b = Tensor.make_random [|sz|] in
    let expr = prim Add [const a; const b] in
    let env = [] in
    (* warmup *)
    for _ = 1 to 10 do ignore (eval_expr env expr) done;
    let t0 = Unix.gettimeofday () in
    for _ = 1 to iters do ignore (eval_expr env expr) done;
    let dt = Unix.gettimeofday () -. t0 in
    let per_call_us = dt /. float_of_int iters *. 1e6 in
    Printf.printf "%10d %10d %12.2f %12.2f\n" sz iters (dt *. 1000.) per_call_us
  ) sizes;
  (* --- matmul: vary M (M×M @ M×M) --- *)
  let ms = [| 4; 16; 64; 128; 256 |] in
  Printf.printf "\n=== 9-3 microbench: matmul (M×M @ M×M) ===\n";
  Printf.printf "%10s %10s %12s %12s %12s\n"
    "M" "iters" "total(ms)" "per-call(μs)" "elems(M³)";
  Printf.printf "%s\n" (String.make 62 '-');
  Array.iter (fun m ->
    let a = Tensor.make_random [|m; m|] in
    let b = Tensor.make_random [|m; m|] in
    let expr = prim Matmul [const a; const b] in
    let env = [] in
    let n_iter = max 1 (min iters (100000000 / (m * m * m + 1))) in
    (* warmup *)
    for _ = 1 to (min 10 n_iter) do ignore (eval_expr env expr) done;
    let t0 = Unix.gettimeofday () in
    for _ = 1 to n_iter do ignore (eval_expr env expr) done;
    let dt = Unix.gettimeofday () -. t0 in
    let per_call_us = dt /. float_of_int n_iter *. 1e6 in
    Printf.printf "%10d %10d %12.2f %12.2f %12d\n"
      m n_iter (dt *. 1000.) per_call_us (m * m * m)
  ) ms;
  (* --- sum_axis: vary numel --- *)
  Printf.printf "\n=== 9-3 microbench: sum_axis(0) ===\n";
  Printf.printf "%10s %10s %12s %12s\n" "numel" "iters" "total(ms)" "per-call(μs)";
  Printf.printf "%s\n" (String.make 50 '-');
  Array.iter (fun sz ->
    let a = Tensor.make_random [|sz|] in
    let expr = prim (Sum_axis 0) [const a] in
    let env = [] in
    for _ = 1 to 10 do ignore (eval_expr env expr) done;
    let t0 = Unix.gettimeofday () in
    for _ = 1 to iters do ignore (eval_expr env expr) done;
    let dt = Unix.gettimeofday () -. t0 in
    let per_call_us = dt /. float_of_int iters *. 1e6 in
    Printf.printf "%10d %10d %12.2f %12.2f\n" sz iters (dt *. 1000.) per_call_us
  ) sizes;
  (* just pass *)
  Alcotest.(check pass) "microbench ran" () ()

(* --- 9-4: gemm comparison — naive vs BLAS --- *)

let bench_gemm () =
  Random.init 42;
  let ms = [| 16; 64; 128; 256; 512 |] in
  Printf.printf "\n=== 9-4 gemm comparison: naive vs BLAS (Accelerate) ===\n";
  Printf.printf "%6s %10s %10s %10s %10s\n"
    "M" "naive(μs)" "BLAS(μs)" "speedup" "BLAS GFLOPS";
  Printf.printf "%s\n" (String.make 56 '-');
  Array.iter (fun m ->
    let a = Tensor.make_random [|m; m|] in
    let b = Tensor.make_random [|m; m|] in
    let flops = 2.0 *. (float_of_int m ** 3.0) in
    (* naive — call Kernel.Naive directly, bypassing eval's BLAS path *)
    let n_iter = max 1 (min 200 (100000000 / (m * m * m + 1))) in
    let naive_dst = Buf.create (m * m) in
    for _ = 1 to (min 3 n_iter) do
      Kernel.Naive.matmul ~a:a.buf ~view_a:a.view ~b:b.buf ~view_b:b.view
        ~dst:naive_dst ~nframe:0
    done;
    let t0 = Unix.gettimeofday () in
    for _ = 1 to n_iter do
      Kernel.Naive.matmul ~a:a.buf ~view_a:a.view ~b:b.buf ~view_b:b.view
        ~dst:naive_dst ~nframe:0
    done;
    let dt_naive = (Unix.gettimeofday () -. t0) /. float_of_int n_iter in
    (* BLAS *)
    let c_buf = Buf.create (m * m) in
    let n_blas = max 1 (min 2000 (1000000000 / (m * m * m + 1))) in
    for _ = 1 to (min 3 n_blas) do
      Blas_stub.dgemm ~m ~n:m ~k:m a.buf b.buf c_buf
    done;
    let t0 = Unix.gettimeofday () in
    for _ = 1 to n_blas do
      Blas_stub.dgemm ~m ~n:m ~k:m a.buf b.buf c_buf
    done;
    let dt_blas = (Unix.gettimeofday () -. t0) /. float_of_int n_blas in
    let speedup = dt_naive /. dt_blas in
    let gflops = flops /. dt_blas /. 1e9 in
    Printf.printf "%6d %10.1f %10.1f %10.1fx %10.2f\n"
      m (dt_naive *. 1e6) (dt_blas *. 1e6) speedup gflops;
    (* correctness: full-element comparison naive vs BLAS
       (naive_dst already computed by the benchmark loop above) *)
    Blas_stub.dgemm ~m ~n:m ~k:m a.buf b.buf c_buf;
    for i = 0 to m * m - 1 do
      let nv = Buf.get naive_dst i and bv = Buf.get c_buf i in
      let err = abs_float (bv -. nv) in
      if err > 1e-8 *. (abs_float nv +. 1e-15) then
        Alcotest.fail (Printf.sprintf "M=%d elem %d: BLAS=%.10g naive=%.10g"
          m i bv nv)
    done
  ) ms;
  Alcotest.(check pass) "gemm comparison ran" () ()

(* --- 9-6: MLP training (dummy MNIST-scale data) --- *)

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

(* --- idx file loader (MNIST binary format) --- *)

let read_int32_be ic =
  let b3 = input_byte ic in
  let b2 = input_byte ic in
  let b1 = input_byte ic in
  let b0 = input_byte ic in
  (b3 lsl 24) lor (b2 lsl 16) lor (b1 lsl 8) lor b0

let load_idx_images path =
  let ic = open_in_bin path in
  let _magic = read_int32_be ic in
  let n = read_int32_be ic in
  let rows = read_int32_be ic in
  let cols = read_int32_be ic in
  let dim = rows * cols in
  let t = Tensor.make [|n; dim|] in
  for i = 0 to n * dim - 1 do
    Buf.set t.buf i (float_of_int (input_byte ic) /. 255.0)
  done;
  close_in ic; t

let load_idx_labels path =
  let ic = open_in_bin path in
  let _magic = read_int32_be ic in
  let n = read_int32_be ic in
  let t = Tensor.make [|n|] in
  for i = 0 to n - 1 do
    Buf.set t.buf i (float_of_int (input_byte ic))
  done;
  close_in ic; t

let find_data_dir () =
  let candidates = ["data"; "../data"; "../../data"] in
  List.find_opt (fun p ->
    Sys.file_exists (p ^ "/train-images-idx3-ubyte")) candidates

let bench_mlp_train () =
  Random.init 42;
  let batch = 64 and d_in = 784 and d_hid = 400 and d_out = 10 in
  let x = Tensor.make_random [|batch; d_in|] in
  (* Xavier-ish initialization *)
  let w1 = Tensor.make_random [|d_in; d_hid|] in
  let scale1 = 1.0 /. sqrt (float_of_int d_in) in
  for i = 0 to d_in * d_hid - 1 do
    Buf.set w1.buf i (Buf.get w1.buf i *. scale1)
  done;
  let b1 = Tensor.make [|d_hid|] in
  let w2 = Tensor.make_random [|d_hid; d_out|] in
  let scale2 = 1.0 /. sqrt (float_of_int d_hid) in
  for i = 0 to d_hid * d_out - 1 do
    Buf.set w2.buf i (Buf.get w2.buf i *. scale2)
  done;
  let b2 = Tensor.make [|d_out|] in
  let y = Tensor.make [|batch|] in
  for i = 0 to batch - 1 do
    Buf.set y.buf i (float_of_int (Random.int d_out))
  done;
  let inv_b = scalar (1.0 /. float_of_int batch) in
  let loss_expr =
    let_ "h"
      (prim Relu [rank 1 Add [prim Matmul [const x; var "w1"]; var "b1"]])
      (let_ "logits"
         (rank 1 Add [prim Matmul [var "h"; var "w2"]; var "b2"])
         (let_ "m" (prim (Max_axis 1) [var "logits"])
            (let_ "shifted" (rank 0 Sub [var "logits"; var "m"])
               (let_ "lse"
                  (rank 0 Add [var "m";
                     prim Log [prim (Sum_axis 1) [prim Exp [var "shifted"]]]])
                  (let_ "picked"
                     (prim (Select_axis 1) [var "logits"; const y])
                     (prim Mul [const inv_b;
                                prim (Sum_axis 0)
                                  [prim Sub [var "lse"; var "picked"]]]))))))
  in
  let param_shapes = [
    ("w1", [|d_in; d_hid|]);
    ("b1", [|d_hid|]);
    ("w2", [|d_hid; d_out|]);
    ("b2", [|d_out|]);
  ] in
  let gp = Transform.grad ~param_shapes loss_expr in
  let params = ref [("w1", w1); ("b1", b1); ("w2", w2); ("b2", b2)] in
  let lr = 0.01 in
  let steps = 50 in
  Ast.Eval.reset_stats ();
  Ast.Eval.enable_stats ();
  let losses = Array.make steps 0.0 in
  let t0 = Unix.gettimeofday () in
  for step = 0 to steps - 1 do
    let env = List.map (fun (s, t) -> (s, (t : Tensor.t :> value))) !params in
    let (loss_v, grad_list) = Ast.Eval.eval_grad env
      ~primal_bindings:gp.primal_bindings
      ~loss_body:gp.loss_body
      ~grad_bodies:gp.grad_bodies in
    losses.(step) <- scalar_val loss_v;
    (* SGD update *)
    params := List.map (fun (name, w) ->
      match List.assoc_opt name grad_list with
      | None -> (name, w)
      | Some g ->
        let n = numel w in
        let w' = Tensor.make w.view.Ndview.shape in
        for i = 0 to n - 1 do
          Buf.set w'.buf i (tensor_get w i -. lr *. tensor_get g i)
        done;
        (name, w')
    ) !params
  done;
  let dt = Unix.gettimeofday () -. t0 in
  Ast.Eval.disable_stats ();
  Printf.printf "\n=== 9-6 MLP training (%d→%d→%d, batch %d, %d steps) ===\n"
    d_in d_hid d_out batch steps;
  Printf.printf "  loss: %.4f → %.4f (%.1f ms total, %.1f ms/step)\n"
    losses.(0) losses.(steps - 1) (dt *. 1000.) (dt *. 1000. /. float_of_int steps);
  Ast.Eval.report ();
  (* Check loss decreased *)
  Alcotest.(check bool) "loss decreased"
    true (losses.(steps - 1) < losses.(0));
  (* Check monotone decrease (with tolerance for float noise) *)
  let monotone = ref true in
  for i = 1 to steps - 1 do
    if losses.(i) > losses.(i - 1) +. 1e-10 then monotone := false
  done;
  Printf.printf "  monotone: %b\n" !monotone;
  (* Check matmul count: with eval_grad, primal should not be doubled *)
  let s = Ast.Eval.stats in
  let mm = Hashtbl.find s.kernel_stats "matmul" in
  Printf.printf "  matmul calls: %d (%d per step)\n" mm.calls (mm.calls / steps)

(* --- MNIST: real data training to 95%+ --- *)

let bench_mnist () =
  match find_data_dir () with
  | None ->
    Printf.printf "MNIST data not found — run via dune exec test/test_perf.exe\n";
    Alcotest.(check pass) "skipped (no data)" () ()
  | Some dir ->
  Random.init 42;
  let train_x = load_idx_images (dir ^ "/train-images-idx3-ubyte") in
  let train_y = load_idx_labels (dir ^ "/train-labels-idx1-ubyte") in
  let test_x  = load_idx_images (dir ^ "/t10k-images-idx3-ubyte") in
  let test_y  = load_idx_labels (dir ^ "/t10k-labels-idx1-ubyte") in
  let batch = 64 and d_in = 784 and d_hid = 400 and d_out = 10 in
  let n_train = train_x.view.Ndview.shape.(0) in
  let n_test  = test_x.view.Ndview.shape.(0) in
  Printf.printf "\n=== MNIST MLP (%d→%d→%d, batch %d) ===\n"
    d_in d_hid d_out batch;
  Printf.printf "  train: %d, test: %d\n" n_train n_test;
  (* He init — make_random returns [-1,1], already centered *)
  let w1 = Tensor.make_random [|d_in; d_hid|] in
  let s1 = sqrt (2.0 /. float_of_int d_in) in
  for i = 0 to d_in * d_hid - 1 do
    Buf.set w1.buf i (Buf.get w1.buf i *. s1)
  done;
  let b1 = Tensor.make [|d_hid|] in
  let w2 = Tensor.make_random [|d_hid; d_out|] in
  let s2 = sqrt (2.0 /. float_of_int d_hid) in
  for i = 0 to d_hid * d_out - 1 do
    Buf.set w2.buf i (Buf.get w2.buf i *. s2)
  done;
  let b2 = Tensor.make [|d_out|] in
  (* Loss: x, y are data vars; w1,b1,w2,b2 are params *)
  let inv_b = scalar (1.0 /. float_of_int batch) in
  let loss_expr =
    let_ "h"
      (prim Relu [rank 1 Add [prim Matmul [var "x"; var "w1"]; var "b1"]])
      (let_ "logits"
         (rank 1 Add [prim Matmul [var "h"; var "w2"]; var "b2"])
         (let_ "m" (prim (Max_axis 1) [var "logits"])
            (let_ "shifted" (rank 0 Sub [var "logits"; var "m"])
               (let_ "lse"
                  (rank 0 Add [var "m";
                     prim Log [prim (Sum_axis 1) [prim Exp [var "shifted"]]]])
                  (let_ "picked"
                     (prim (Select_axis 1) [var "logits"; var "y"])
                     (prim Mul [const inv_b;
                                prim (Sum_axis 0)
                                  [prim Sub [var "lse"; var "picked"]]]))))))
  in
  let param_shapes = [
    ("w1", [|d_in; d_hid|]); ("b1", [|d_hid|]);
    ("w2", [|d_hid; d_out|]); ("b2", [|d_out|]);
  ] in
  let data_shapes = [("x", [|batch; d_in|]); ("y", [|batch|])] in
  let gp = Transform.grad ~param_shapes ~data_shapes loss_expr in
  (* Forward-only expression for accuracy *)
  let logits_expr =
    let_ "h" (prim Relu [rank 1 Add [prim Matmul [var "x"; var "w1"]; var "b1"]])
      (rank 1 Add [prim Matmul [var "h"; var "w2"]; var "b2"]) in
  let all_shapes = param_shapes @ data_shapes in
  let logits_expanded =
    Transform.Desugar.fuse_views
      (Transform.Expand_rank.expand ~senv:all_shapes logits_expr) in
  (* Test accuracy *)
  let compute_accuracy params =
    let correct = ref 0 in
    let n_batches = n_test / batch in
    for bi = 0 to n_batches - 1 do
      let x_b = Tensor.make [|batch; d_in|] in
      let y_b = Tensor.make [|batch|] in
      let off = bi * batch in
      for s = 0 to batch - 1 do
        let src = off + s in
        for c = 0 to d_in - 1 do
          Buf.set x_b.buf (s * d_in + c)
            (Buf.get test_x.buf (src * d_in + c))
        done;
        Buf.set y_b.buf s (Buf.get test_y.buf src)
      done;
      let logits = eval_expr (("x", x_b) :: ("y", y_b) :: params) logits_expanded in
      for s = 0 to batch - 1 do
        let best = ref 0 in
        let best_v = ref neg_infinity in
        for c = 0 to d_out - 1 do
          let v = Buf.get logits.buf (s * d_out + c) in
          if v > !best_v then (best := c; best_v := v)
        done;
        if !best = int_of_float (Buf.get y_b.buf s) then incr correct
      done
    done;
    float_of_int !correct /. float_of_int (n_batches * batch)
  in
  let params = ref [("w1", w1); ("b1", b1); ("w2", w2); ("b2", b2)] in
  let n_epochs = 5 in
  let lr = 0.1 in
  let t_total = ref 0.0 in
  for epoch = 0 to n_epochs - 1 do
    let idx = Array.init n_train Fun.id in
    for i = n_train - 1 downto 1 do
      let j = Random.int (i + 1) in
      let tmp = idx.(i) in idx.(i) <- idx.(j); idx.(j) <- tmp
    done;
    let n_batches = n_train / batch in
    let epoch_loss = ref 0.0 in
    let t0 = Unix.gettimeofday () in
    for bi = 0 to n_batches - 1 do
      let x_b = Tensor.make [|batch; d_in|] in
      let y_b = Tensor.make [|batch|] in
      for s = 0 to batch - 1 do
        let src = idx.(bi * batch + s) in
        for c = 0 to d_in - 1 do
          Buf.set x_b.buf (s * d_in + c)
            (Buf.get train_x.buf (src * d_in + c))
        done;
        Buf.set y_b.buf s (Buf.get train_y.buf src)
      done;
      let env = List.map (fun (s, t) -> (s, (t : Tensor.t :> value)))
        (("x", x_b) :: ("y", y_b) :: !params) in
      let (loss_v, grad_list) = Ast.Eval.eval_grad env
        ~primal_bindings:gp.primal_bindings
        ~loss_body:gp.loss_body
        ~grad_bodies:gp.grad_bodies in
      epoch_loss := !epoch_loss +. scalar_val loss_v;
      params := List.map (fun (name, w) ->
        match List.assoc_opt name grad_list with
        | None -> (name, w)
        | Some g ->
          let n = numel w in
          let w' = Tensor.make w.view.Ndview.shape in
          for i = 0 to n - 1 do
            Buf.set w'.buf i (Buf.get w.buf i -. lr *. Buf.get g.buf i)
          done;
          (name, w')
      ) !params
    done;
    let dt = Unix.gettimeofday () -. t0 in
    t_total := !t_total +. dt;
    let avg_loss = !epoch_loss /. float_of_int n_batches in
    let acc = compute_accuracy !params in
    Printf.printf "  epoch %d: loss=%.4f  acc=%.4f  (%.1fs, %.1fms/step)\n"
      (epoch + 1) avg_loss acc dt (dt *. 1000. /. float_of_int n_batches)
  done;
  let final_acc = compute_accuracy !params in
  Printf.printf "  total: %.1fs  final accuracy: %.2f%%\n"
    !t_total (final_acc *. 100.0);
  Alcotest.(check bool) "MNIST accuracy >= 95%"
    true (final_acc >= 0.95)

let () =
  let open Alcotest in
  run "perf" [
    "9-1 linreg", [
      test_case "size distribution" `Quick bench_linreg;
    ];
    "9-1 MLP", [
      test_case "size distribution" `Quick bench_mlp;
    ];
    "9-1 parallel", [
      test_case "size distribution" `Quick bench_parallel;
    ];
    "9-2 parallel-grad", [
      test_case "N-parallel linreg with grad" `Quick bench_parallel_grad;
    ];
    "9-3 microbench", [
      test_case "c_node and c_elem" `Slow bench_micro;
    ];
    "9-4 gemm", [
      test_case "naive vs BLAS" `Slow bench_gemm;
    ];
    "9-6 MLP-train", [
      test_case "MLP training" `Slow bench_mlp_train;
    ];
    "MNIST", [
      test_case "95%+ accuracy" `Slow bench_mnist;
    ];
  ]
