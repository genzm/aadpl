open View

(* === Dual pair: sum_axis* = broadcast, broadcast* = sum_axis ===

   Principle: both sides of the inner product must close in base space.
   Forward goes through σ* (read from base via composed view),
   adjoint goes back through σ_* (add_view to base).
   This is the same pattern as test_gather_scatter_adjoint.

   Non-circular: sum_axis has its own axis loop,
   broadcast inserts stride 0. Independent implementations. *)

(* Direction 1: ⟨sum_axis(σ*(x)), g⟩ = ⟨x, σ_*(broadcast(g))⟩
   forward  = sum_axis through composed view v (axis loop + index_of)
   adjoint  = broadcast g to logical shape, then add_view through v to base *)
let test_sum_star_eq_broadcast =
  QCheck.Test.make ~count:500 ~name:"sum_axis* = broadcast (composed views)"
    (QCheck.make (
      let open QCheck.Gen in
      let* (v, _) = Gen_view.gen_composed in
      let r = Ndview.rank v in
      let* axis = int_range 0 (r - 1) in
      return (v, axis)))
    (fun (v, axis) ->
      let bs = Ndview.base_size v in
      let n = Ndview.numel v in
      let x = Buf.create bs in
      Buf.fill_random x;
      let reduced_shape = Array.init (Ndview.rank v - 1) (fun i ->
        if i < axis then v.Ndview.shape.(i) else v.Ndview.shape.(i + 1)) in
      let reduced_n = Array.fold_left ( * ) 1 reduced_shape in
      (* forward: sum_axis reads x through composed view v *)
      let y = Buf.create reduced_n in
      Kernel.Naive.sum_axis ~src:x ~view:v ~axis ~dst:y;
      let g = Buf.create reduced_n in
      Buf.fill_random g;
      let lhs = Buf.dot y g in
      (* adjoint: broadcast g to logical shape, then add_view through v to base *)
      let g_bcast_view = Ndview.broadcast (Ndview.contiguous reduced_shape)
                           ~axis ~size:v.Ndview.shape.(axis) in
      let g_expanded = Buf.create n in
      Tensor.read_view ~src:g ~view:g_bcast_view ~dst:g_expanded;
      let acc = Buf.create bs in
      Tensor.add_view ~src:g_expanded ~view:v ~acc;
      let rhs = Buf.dot x acc in
      let scale = max 1.0 (max (abs_float lhs) (abs_float rhs)) in
      let tol = 1e-14 *. float_of_int n +. 1e-12 in
      abs_float (lhs -. rhs) /. scale < tol)

(* Direction 2: ⟨read_view(x, broadcast(v)), g⟩ = ⟨x, add_view(sum_axis(g), v)⟩
   forward  = read_view through broadcast(v) (stride-0 on top of composed view)
   adjoint  = sum_axis g along broadcast axis, then add_view through v to base *)
let test_broadcast_star_eq_sum =
  QCheck.Test.make ~count:500 ~name:"broadcast* = sum_axis (composed views)"
    (QCheck.make (
      let open QCheck.Gen in
      let* (v, _) = Gen_view.gen_composed in
      let r = Ndview.rank v in
      let* axis = int_range 0 r in
      let* size = int_range 1 5 in
      return (v, axis, size)))
    (fun (v, axis, size) ->
      let bs = Ndview.base_size v in
      let n = Ndview.numel v in
      let x = Buf.create bs in
      Buf.fill_random x;
      (* forward: broadcast composed view, read x through it *)
      let bv = Ndview.broadcast v ~axis ~size in
      let bn = Ndview.numel bv in
      let bx = Buf.create bn in
      Tensor.read_view ~src:x ~view:bv ~dst:bx;
      let g = Buf.create bn in
      Buf.fill_random g;
      let lhs = Buf.dot bx g in
      (* adjoint: sum g along broadcast axis, then add_view through v to base *)
      let g_summed = Buf.create n in
      Kernel.Naive.sum_axis ~src:g ~view:(Ndview.contiguous bv.Ndview.shape)
        ~axis ~dst:g_summed;
      let acc = Buf.create bs in
      Tensor.add_view ~src:g_summed ~view:v ~acc;
      let rhs = Buf.dot x acc in
      let scale = max 1.0 (max (abs_float lhs) (abs_float rhs)) in
      let tol = 1e-14 *. float_of_int bn +. 1e-12 in
      abs_float (lhs -. rhs) /. scale < tol)

(* === Dual pair: gather* = scatter_add (composed views) ===
   forward  = gather reads x through composed view v
   adjoint  = scatter_add writes g back through v to base
   Inner product in base space. *)

let test_gather_scatter_adjoint =
  QCheck.Test.make ~count:500 ~name:"gather* = scatter_add (composed views)"
    (QCheck.make (
      let open QCheck.Gen in
      let* (v, _) = Gen_view.gen_composed in
      let r = Ndview.rank v in
      let* axis = int_range 0 (r - 1) in
      let n_axis = v.Ndview.shape.(axis) in
      let* m = int_range 1 6 in
      let* indices = list_size (return m) (int_range 0 (n_axis - 1)) in
      return (v, axis, Array.of_list indices)))
    (fun (v, axis, indices) ->
      let bs = Ndview.base_size v in
      let n = Ndview.numel v in
      let x = Buf.create bs in
      Buf.fill_random x;
      let m = Array.length indices in
      let out_shape = Array.init (Ndview.rank v) (fun k ->
        if k = axis then m else v.Ndview.shape.(k)) in
      let out_n = Array.fold_left ( * ) 1 out_shape in
      (* forward: gather through composed view *)
      let y = Buf.create out_n in
      Kernel.Naive.gather ~src:x ~view:v ~axis ~indices ~dst:y;
      let g = Buf.create out_n in
      Buf.fill_random g;
      let lhs = Buf.dot y g in
      (* adjoint: scatter_add back into base space *)
      let acc = Buf.create bs in
      Kernel.Naive.scatter_add ~src:g ~view:v ~axis ~indices ~acc;
      let rhs = Buf.dot x acc in
      let scale = max 1.0 (max (abs_float lhs) (abs_float rhs)) in
      let tol = 1e-14 *. float_of_int (max n out_n) +. 1e-12 in
      abs_float (lhs -. rhs) /. scale < tol)

(* === matmul transpose rules: ⟨AB, G⟩ = ⟨A, GB^T⟩ = ⟨B, A^TG⟩ === *)

let test_matmul_adjoint_a =
  QCheck.Test.make ~count:500 ~name:"matmul: dA = G @ B^T"
    (QCheck.make (
      let open QCheck.Gen in
      let* m = int_range 1 5 in
      let* k = int_range 1 5 in
      let* n = int_range 1 5 in
      return (m, k, n)))
    (fun (m, k, n) ->
      let a = Buf.create (m * k) in Buf.fill_random a;
      let b = Buf.create (k * n) in Buf.fill_random b;
      let g = Buf.create (m * n) in Buf.fill_random g;
      let va = Ndview.contiguous [|m; k|] in
      let vb = Ndview.contiguous [|k; n|] in
      let vg = Ndview.contiguous [|m; n|] in
      let ab = Buf.create (m * n) in
      Kernel.Naive.matmul ~nframe:0 ~a ~view_a:va ~b ~view_b:vb ~dst:ab;
      let lhs = Buf.dot ab g in
      let vb_t = Ndview.transpose vb ~perm:[|1; 0|] in
      let gb_t = Buf.create (m * k) in
      Kernel.Naive.matmul ~nframe:0 ~a:g ~view_a:vg ~b ~view_b:vb_t ~dst:gb_t;
      let rhs = Buf.dot a gb_t in
      let scale = max 1.0 (max (abs_float lhs) (abs_float rhs)) in
      let tol = 1e-14 *. float_of_int (m * k * n) +. 1e-12 in
      abs_float (lhs -. rhs) /. scale < tol)

let test_matmul_adjoint_b =
  QCheck.Test.make ~count:500 ~name:"matmul: dB = A^T @ G"
    (QCheck.make (
      let open QCheck.Gen in
      let* m = int_range 1 5 in
      let* k = int_range 1 5 in
      let* n = int_range 1 5 in
      return (m, k, n)))
    (fun (m, k, n) ->
      let a = Buf.create (m * k) in Buf.fill_random a;
      let b = Buf.create (k * n) in Buf.fill_random b;
      let g = Buf.create (m * n) in Buf.fill_random g;
      let va = Ndview.contiguous [|m; k|] in
      let vb = Ndview.contiguous [|k; n|] in
      let vg = Ndview.contiguous [|m; n|] in
      let ab = Buf.create (m * n) in
      Kernel.Naive.matmul ~nframe:0 ~a ~view_a:va ~b ~view_b:vb ~dst:ab;
      let lhs = Buf.dot ab g in
      let va_t = Ndview.transpose va ~perm:[|1; 0|] in
      let a_t_g = Buf.create (k * n) in
      Kernel.Naive.matmul ~nframe:0 ~a ~view_a:va_t ~b:g ~view_b:vg ~dst:a_t_g;
      let rhs = Buf.dot b a_t_g in
      let scale = max 1.0 (max (abs_float lhs) (abs_float rhs)) in
      let tol = 1e-14 *. float_of_int (m * k * n) +. 1e-12 in
      abs_float (lhs -. rhs) /. scale < tol)

(* === map1/map2 through composed views ===
   map1 with f=id must agree with read_view (independent implementations).
   map2 with f=(+.) must agree with read_view of each input, summed. *)

let test_map1_agrees_read_view =
  QCheck.Test.make ~count:500 ~name:"map1 id = read_view (composed views)"
    (QCheck.make Gen_view.gen_view)
    (fun v ->
      let bs = Ndview.base_size v in
      let n = Ndview.numel v in
      let x = Buf.create bs in
      Buf.fill_random x;
      let y1 = Buf.create n in
      Kernel.Naive.map1 ~f:Fun.id ~src:x ~view:v ~dst:y1;
      let y2 = Buf.create n in
      Tensor.read_view ~src:x ~view:v ~dst:y2;
      let ok = ref true in
      for i = 0 to n - 1 do
        if Buf.get y1 i <> Buf.get y2 i then ok := false
      done;
      !ok)

let test_map2_agrees_read_views =
  QCheck.Test.make ~count:500 ~name:"map2 (+) = read + read (composed views)"
    (QCheck.make Gen_view.gen_view)
    (fun v ->
      let bs = Ndview.base_size v in
      let n = Ndview.numel v in
      let x1 = Buf.create bs in Buf.fill_random x1;
      let x2 = Buf.create bs in Buf.fill_random x2;
      let y = Buf.create n in
      Kernel.Naive.map2 ~f:( +. ) ~src1:x1 ~view1:v ~src2:x2 ~view2:v ~dst:y;
      let r1 = Buf.create n in
      Tensor.read_view ~src:x1 ~view:v ~dst:r1;
      let r2 = Buf.create n in
      Tensor.read_view ~src:x2 ~view:v ~dst:r2;
      let ok = ref true in
      for i = 0 to n - 1 do
        if Buf.get y i <> Buf.get r1 i +. Buf.get r2 i then ok := false
      done;
      !ok)

(* === Hand-calculated tests === *)

let test_sum_axis_hand () =
  (* [[1,2,3],[4,5,6]] sum axis=0 → [5,7,9] *)
  let x = Buf.create 6 in
  List.iteri (fun i v -> Buf.set x i v) [1.;2.;3.;4.;5.;6.];
  let y = Buf.create 3 in
  Kernel.Naive.sum_axis ~src:x ~view:(Ndview.contiguous [|2;3|]) ~axis:0 ~dst:y;
  Alcotest.(check (float 1e-10)) "s[0]" 5.0 (Buf.get y 0);
  Alcotest.(check (float 1e-10)) "s[1]" 7.0 (Buf.get y 1);
  Alcotest.(check (float 1e-10)) "s[2]" 9.0 (Buf.get y 2);
  (* sum axis=1 → [6,15] *)
  let z = Buf.create 2 in
  Kernel.Naive.sum_axis ~src:x ~view:(Ndview.contiguous [|2;3|]) ~axis:1 ~dst:z;
  Alcotest.(check (float 1e-10)) "s[0]" 6.0 (Buf.get z 0);
  Alcotest.(check (float 1e-10)) "s[1]" 15.0 (Buf.get z 1)

let test_max_axis_hand () =
  (* [[1,5,3],[4,2,6]] max axis=1 → [5,6], argmax=[1,2] *)
  let x = Buf.create 6 in
  List.iteri (fun i v -> Buf.set x i v) [1.;5.;3.;4.;2.;6.];
  let y = Buf.create 2 in
  let argmax = Array.make 2 0 in
  Kernel.Naive.max_axis ~src:x ~view:(Ndview.contiguous [|2;3|])
    ~axis:1 ~dst:y ~dst_argmax:argmax;
  Alcotest.(check (float 1e-10)) "max[0]" 5.0 (Buf.get y 0);
  Alcotest.(check (float 1e-10)) "max[1]" 6.0 (Buf.get y 1);
  Alcotest.(check int) "argmax[0]" 1 argmax.(0);
  Alcotest.(check int) "argmax[1]" 2 argmax.(1)

let test_matmul_hand () =
  (* [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]] *)
  let a = Buf.create 4 in
  List.iteri (fun i v -> Buf.set a i v) [1.;2.;3.;4.];
  let b = Buf.create 4 in
  List.iteri (fun i v -> Buf.set b i v) [5.;6.;7.;8.];
  let c = Buf.create 4 in
  Kernel.Naive.matmul ~nframe:0 ~a ~view_a:(Ndview.contiguous [|2;2|])
                      ~b ~view_b:(Ndview.contiguous [|2;2|]) ~dst:c;
  Alcotest.(check (float 1e-10)) "c[0,0]" 19.0 (Buf.get c 0);
  Alcotest.(check (float 1e-10)) "c[0,1]" 22.0 (Buf.get c 1);
  Alcotest.(check (float 1e-10)) "c[1,0]" 43.0 (Buf.get c 2);
  Alcotest.(check (float 1e-10)) "c[1,1]" 50.0 (Buf.get c 3)

let test_matmul_nonsquare () =
  (* [1,2,3] @ [[4],[5],[6]] = [[32]] *)
  let a = Buf.create 3 in
  List.iteri (fun i v -> Buf.set a i v) [1.;2.;3.];
  let b = Buf.create 3 in
  List.iteri (fun i v -> Buf.set b i v) [4.;5.;6.];
  let c = Buf.create 1 in
  Kernel.Naive.matmul ~nframe:0 ~a ~view_a:(Ndview.contiguous [|1;3|])
                      ~b ~view_b:(Ndview.contiguous [|3;1|]) ~dst:c;
  Alcotest.(check (float 1e-10)) "c[0,0]" 32.0 (Buf.get c 0)

let test_matmul_transposed () =
  (* A = [[1,2],[3,4]], B stored [5,6,7,8] = [[5,6],[7,8]].
     Transposed view: B^T = [[5,7],[6,8]].
     A @ B^T = [[1*5+2*6, 1*7+2*8],[3*5+4*6, 3*7+4*8]] = [[17,23],[39,53]] *)
  let a = Buf.create 4 in
  List.iteri (fun i v -> Buf.set a i v) [1.;2.;3.;4.];
  let b = Buf.create 4 in
  List.iteri (fun i v -> Buf.set b i v) [5.;6.;7.;8.];
  let c = Buf.create 4 in
  let vb = Ndview.transpose (Ndview.contiguous [|2;2|]) ~perm:[|1;0|] in
  Kernel.Naive.matmul ~nframe:0 ~a ~view_a:(Ndview.contiguous [|2;2|])
                      ~b ~view_b:vb ~dst:c;
  Alcotest.(check (float 1e-10)) "c[0,0]" 17.0 (Buf.get c 0);
  Alcotest.(check (float 1e-10)) "c[0,1]" 23.0 (Buf.get c 1);
  Alcotest.(check (float 1e-10)) "c[1,0]" 39.0 (Buf.get c 2);
  Alcotest.(check (float 1e-10)) "c[1,1]" 53.0 (Buf.get c 3)

let test_map1 () =
  let x = Buf.create 4 in
  List.iteri (fun i v -> Buf.set x i v) [1.;(-2.);3.;0.];
  let y = Buf.create 4 in
  Kernel.Naive.map1 ~f:(fun x -> x *. x) ~src:x
    ~view:(Ndview.contiguous [|4|]) ~dst:y;
  Alcotest.(check (float 1e-10)) "y[0]" 1.0 (Buf.get y 0);
  Alcotest.(check (float 1e-10)) "y[1]" 4.0 (Buf.get y 1);
  Alcotest.(check (float 1e-10)) "y[2]" 9.0 (Buf.get y 2);
  Alcotest.(check (float 1e-10)) "y[3]" 0.0 (Buf.get y 3)

let test_map2 () =
  let a = Buf.create 3 in
  let b = Buf.create 3 in
  List.iteri (fun i v -> Buf.set a i v) [1.;2.;3.];
  List.iteri (fun i v -> Buf.set b i v) [10.;20.;30.];
  let c = Buf.create 3 in
  let v = Ndview.contiguous [|3|] in
  Kernel.Naive.map2 ~f:( +. ) ~src1:a ~view1:v ~src2:b ~view2:v ~dst:c;
  Alcotest.(check (float 1e-10)) "c[0]" 11.0 (Buf.get c 0);
  Alcotest.(check (float 1e-10)) "c[1]" 22.0 (Buf.get c 1);
  Alcotest.(check (float 1e-10)) "c[2]" 33.0 (Buf.get c 2)

let test_gather_hand () =
  (* [10,20,30,40,50] gather indices=[1,3,1] → [20,40,20] *)
  let x = Buf.create 5 in
  List.iteri (fun i v -> Buf.set x i v) [10.;20.;30.;40.;50.];
  let y = Buf.create 3 in
  Kernel.Naive.gather ~src:x ~view:(Ndview.contiguous [|5|])
    ~axis:0 ~indices:[|1;3;1|] ~dst:y;
  Alcotest.(check (float 1e-10)) "y[0]" 20.0 (Buf.get y 0);
  Alcotest.(check (float 1e-10)) "y[1]" 40.0 (Buf.get y 1);
  Alcotest.(check (float 1e-10)) "y[2]" 20.0 (Buf.get y 2)

let test_scatter_add_hand () =
  (* scatter_add [1,2,3] with indices=[1,3,1] into size-5 acc *)
  (* acc[1] += 1+3 = 4, acc[3] += 2 *)
  let g = Buf.create 3 in
  List.iteri (fun i v -> Buf.set g i v) [1.;2.;3.];
  let acc = Buf.create 5 in
  Kernel.Naive.scatter_add ~src:g ~view:(Ndview.contiguous [|5|])
    ~axis:0 ~indices:[|1;3;1|] ~acc;
  Alcotest.(check (float 1e-10)) "acc[0]" 0.0 (Buf.get acc 0);
  Alcotest.(check (float 1e-10)) "acc[1]" 4.0 (Buf.get acc 1);
  Alcotest.(check (float 1e-10)) "acc[2]" 0.0 (Buf.get acc 2);
  Alcotest.(check (float 1e-10)) "acc[3]" 2.0 (Buf.get acc 3);
  Alcotest.(check (float 1e-10)) "acc[4]" 0.0 (Buf.get acc 4)

let () =
  let open Alcotest in
  run "kernel" [
    "dual-pair", List.map (fun t -> QCheck_alcotest.to_alcotest t) [
      test_sum_star_eq_broadcast;
      test_broadcast_star_eq_sum;
      test_gather_scatter_adjoint;
      test_matmul_adjoint_a;
      test_matmul_adjoint_b;
    ];
    "view-agree", List.map (fun t -> QCheck_alcotest.to_alcotest t) [
      test_map1_agrees_read_view;
      test_map2_agrees_read_views;
    ];
    "hand", [
      test_case "sum_axis" `Quick test_sum_axis_hand;
      test_case "max_axis" `Quick test_max_axis_hand;
      test_case "matmul 2x2" `Quick test_matmul_hand;
      test_case "matmul non-square" `Quick test_matmul_nonsquare;
      test_case "matmul transposed" `Quick test_matmul_transposed;
      test_case "map1" `Quick test_map1;
      test_case "map2" `Quick test_map2;
      test_case "gather" `Quick test_gather_hand;
      test_case "scatter_add" `Quick test_scatter_add_hand;
    ];
  ]
