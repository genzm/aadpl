open View.Ndview

(* === Phase 1 properties === *)

let test_shadow_agrees =
  QCheck.Test.make ~count:500 ~name:"shadow agrees with index_of"
    (QCheck.make Gen_view.gen_composed)
    (fun (v, sh) ->
      let ok = ref true in
      iter_indices v.shape (fun _ idx ->
        if index_of v idx <> sh idx then ok := false);
      !ok)

let test_contiguous_index_of =
  QCheck.Test.make ~count:500 ~name:"contiguous index_of = row-major"
    (QCheck.make (QCheck.Gen.map fst Gen_view.gen_contiguous))
    (fun v ->
      let ok = ref true in
      let i = ref 0 in
      iter_indices v.shape (fun _ idx ->
        if index_of v idx <> !i then ok := false;
        incr i);
      !ok)

let test_broadcast_index_of =
  QCheck.Test.make ~count:500 ~name:"broadcast index_of"
    (QCheck.make (
      let open QCheck.Gen in
      let* (v, _) = Gen_view.gen_contiguous in
      let r = rank v in
      let* axis = int_range 0 r in
      let* size = int_range 1 4 in
      let bv = broadcast v ~axis ~size in
      return (v, bv, axis)))
    (fun (v, bv, axis) ->
      let ok = ref true in
      iter_indices bv.shape (fun _ idx ->
        let orig_idx = Array.init (rank v) (fun i ->
          if i < axis then idx.(i) else idx.(i + 1)) in
        if index_of bv idx <> index_of v orig_idx then ok := false);
      !ok)

let gen_contiguous_rank2plus =
  let open QCheck.Gen in
  let* r = int_range 2 4 in
  let* dims = list_size (return r) (int_range 1 5) in
  return (contiguous (Array.of_list dims))

let test_transpose_involution =
  QCheck.Test.make ~count:500 ~name:"transpose inverse"
    (QCheck.make (
      let open QCheck.Gen in
      let* v = gen_contiguous_rank2plus in
      let r = rank v in
      let* perm = Gen_view.gen_perm r in
      let inv_perm = Array.make r 0 in
      Array.iteri (fun i p -> inv_perm.(p) <- i) perm;
      let tv = transpose (transpose v ~perm) ~perm:inv_perm in
      return (v, tv)))
    (fun (v, tv) ->
      v.shape = tv.shape && v.strides = tv.strides && v.offset = tv.offset)

let test_composed_view_valid =
  QCheck.Test.make ~count:500 ~name:"composed view indices in range"
    (QCheck.make Gen_view.gen_view)
    (fun v ->
      let bs = base_size v in
      let ok = ref true in
      iter_indices v.shape (fun _ idx ->
        let pos = index_of v idx in
        if pos < 0 || pos >= bs then ok := false);
      !ok)

(* Both sides: is_injective must agree with exhaustive check *)
let test_is_injective_exact =
  QCheck.Test.make ~count:500 ~name:"is_injective exact"
    (QCheck.make Gen_view.gen_view)
    (fun v ->
      let positions = Array.map (index_of v) (all_indices v.shape) in
      Array.sort compare positions;
      let actually = ref true in
      for i = 0 to Array.length positions - 2 do
        if positions.(i) = positions.(i + 1) then actually := false
      done;
      is_injective v = !actually)

let test_is_injective_stride0 =
  QCheck.Test.make ~count:500 ~name:"is_injective false on stride 0 (size>1)"
    (QCheck.make (
      let open QCheck.Gen in
      let* (v, _) = Gen_view.gen_contiguous in
      let r = rank v in
      let* axis = int_range 0 r in
      let* size = int_range 2 4 in
      return (broadcast v ~axis ~size)))
    (fun v ->
      not (is_injective v))

let test_numel =
  QCheck.Test.make ~count:500 ~name:"numel = number of indices"
    (QCheck.make Gen_view.gen_view)
    (fun v ->
      numel v = Array.length (all_indices v.shape))

(* === Phase 2: adjointness === *)

let test_adjointness =
  QCheck.Test.make ~count:500 ~name:"adjointness: read_view / add_view"
    (QCheck.make Gen_view.gen_view)
    (fun v ->
      let bs = base_size v in
      let n = numel v in
      let x = View.Buf.create bs in
      View.Buf.fill_random x;
      let g = View.Buf.create n in
      View.Buf.fill_random g;
      let read_out = View.Buf.create n in
      View.Tensor.read_view ~src:x ~view:v ~dst:read_out;
      let lhs = View.Buf.dot read_out g in
      let acc = View.Buf.create bs in
      View.Tensor.add_view ~src:g ~view:v ~acc;
      let rhs = View.Buf.dot x acc in
      let scale = max 1.0 (max (abs_float lhs) (abs_float rhs)) in
      let tol = 1e-14 *. float_of_int (numel v) +. 1e-12 in
      abs_float (lhs -. rhs) /. scale < tol)

let test_adjoint_broadcast_sums () =
  let base = View.Buf.create 3 in
  View.Buf.set base 0 1.0; View.Buf.set base 1 2.0; View.Buf.set base 2 3.0;
  let v = broadcast (contiguous [|3|]) ~axis:0 ~size:2 in
  let out = View.Buf.create 6 in
  View.Tensor.read_view ~src:base ~view:v ~dst:out;
  Alcotest.(check (float 1e-10)) "read[0]" 1.0 (View.Buf.get out 0);
  Alcotest.(check (float 1e-10)) "read[3]" 1.0 (View.Buf.get out 3);
  Alcotest.(check (float 1e-10)) "read[5]" 3.0 (View.Buf.get out 5);
  let g = View.Buf.create 6 in
  View.Buf.set g 0 1.0; View.Buf.set g 4 1.0;
  let acc = View.Buf.create 3 in
  View.Tensor.add_view ~src:g ~view:v ~acc;
  Alcotest.(check (float 1e-10)) "add[0]" 1.0 (View.Buf.get acc 0);
  Alcotest.(check (float 1e-10)) "add[1]" 1.0 (View.Buf.get acc 1);
  Alcotest.(check (float 1e-10)) "add[2]" 0.0 (View.Buf.get acc 2)

let () =
  let open Alcotest in
  run "view" [
    "phase1", List.map (fun t -> QCheck_alcotest.to_alcotest t) [
      test_shadow_agrees;
      test_contiguous_index_of;
      test_broadcast_index_of;
      test_transpose_involution;
      test_composed_view_valid;
      test_is_injective_exact;
      test_is_injective_stride0;
      test_numel;
    ];
    "phase2", List.map (fun t -> QCheck_alcotest.to_alcotest t) [
      test_adjointness;
    ] @ [
      test_case "broadcast adjoint sums" `Quick test_adjoint_broadcast_sums;
    ];
  ]
