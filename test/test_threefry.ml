open Alcotest
open Prng

(* ── Reference vector tests ── *)

let hex s = Int64.of_string ("0x" ^ s)

let check_vector name ~key ~ctr ~expect =
  let (r0, r1) = Threefry.threefry2x64 ~key ~ctr in
  let (e0, e1) = expect in
  check string (name ^ " [0]") (Printf.sprintf "%Lx" e0) (Printf.sprintf "%Lx" r0);
  check string (name ^ " [1]") (Printf.sprintf "%Lx" e1) (Printf.sprintf "%Lx" r1)

let test_vector_zeros () =
  check_vector "zeros"
    ~key:(0L, 0L) ~ctr:(0L, 0L)
    ~expect:(hex "c2b6e3a8c2c69865", hex "6f81ed42f350084d")

let test_vector_ones () =
  check_vector "ones"
    ~key:(Int64.minus_one, Int64.minus_one)
    ~ctr:(Int64.minus_one, Int64.minus_one)
    ~expect:(hex "e02cb7c4d95d277a", hex "d06633d0893b8b68")

let test_vector_pi_e () =
  check_vector "pi_e"
    ~key:(hex "a4093822299f31d0", hex "082efa98ec4e6c89")
    ~ctr:(hex "243f6a8885a308d3", hex "13198a2e03707344")
    ~expect:(hex "263c7d30bb0f0af1", hex "56be8361d3311526")

(* ── Reproducibility ── *)

let test_reproducibility () =
  let key = Threefry.make_key ~run_key:42L ~namespace:Threefry.ns_model in
  let ctr = Threefry.make_ctr ~site_id:7 ~component:1 ~frame_index:3 in
  let (a0, a1) = Threefry.threefry2x64 ~key ~ctr in
  let (b0, b1) = Threefry.threefry2x64 ~key ~ctr in
  check int64 "r0" a0 b0;
  check int64 "r1" a1 b1

(* ── Sensitivity: changing ctr by 1 produces different output ── *)

let test_sensitivity () =
  let key = Threefry.make_key ~run_key:0L ~namespace:0L in
  let (a0, _) = Threefry.threefry2x64 ~key ~ctr:(0L, 0L) in
  let (b0, _) = Threefry.threefry2x64 ~key ~ctr:(0L, 1L) in
  let (c0, _) = Threefry.threefry2x64 ~key ~ctr:(1L, 0L) in
  if a0 = b0 then fail "ctr[1]+1 should differ";
  if a0 = c0 then fail "ctr[0]+1 should differ";
  if b0 = c0 then fail "both perturbations should differ";
  check pass "sensitivity" () ()

let test_counter_boundaries () =
  let cases =
    [
      (0, 0, 0);
      (0, 1, 47_039_999);
      (1, 0, 47_040_000);
      (0xffff_ffff, 0xffff_ffff, max_int);
    ]
  in
  let counters = List.map (fun (site_id, component, frame_index) ->
    Threefry.make_ctr ~site_id ~component ~frame_index) cases in
  check int "counter encoding is injective" (List.length counters)
    (List.length (List.sort_uniq compare counters));
  check_raises "negative frame_index"
    (Invalid_argument "Threefry.make_ctr: negative frame_index")
    (fun () -> ignore (Threefry.make_ctr ~site_id:0 ~component:1 ~frame_index:(-1)));
  check_raises "site_id above 32 bits"
    (Invalid_argument "Threefry.make_ctr: site_id outside unsigned 32-bit range")
    (fun () -> ignore (Threefry.make_ctr
      ~site_id:0x1_0000_0000 ~component:1 ~frame_index:0));
  check_raises "component above 32 bits"
    (Invalid_argument "Threefry.make_ctr: component outside unsigned 32-bit range")
    (fun () -> ignore (Threefry.make_ctr
      ~site_id:0 ~component:0x1_0000_0000 ~frame_index:0))

(* ── Namespace sensitivity ── *)

let test_namespace_sensitivity () =
  let k1 = Threefry.make_key ~run_key:42L ~namespace:Threefry.ns_model in
  let k2 = Threefry.make_key ~run_key:42L ~namespace:Threefry.ns_data in
  let ctr = Threefry.make_ctr ~site_id:0 ~component:1 ~frame_index:0 in
  let (a0, _) = Threefry.threefry2x64 ~key:k1 ~ctr in
  let (b0, _) = Threefry.threefry2x64 ~key:k2 ~ctr in
  if a0 = b0 then fail "different namespace should differ";
  check pass "namespace" () ()

(* ── to_open_unit range ── *)

let test_open_unit_range () =
  (* Check boundary values *)
  let f0 = Threefry.to_open_unit 0L in
  let fmax = Threefry.to_open_unit Int64.minus_one in
  if f0 <= 0.0 then fail (Printf.sprintf "to_open_unit(0) = %g <= 0" f0);
  if fmax >= 1.0 then fail (Printf.sprintf "to_open_unit(max) = %g >= 1" fmax);
  check pass "open unit range" () ()

(* ── Moment test: 10^6 samples ── *)

let test_moments () =
  let n = 1_000_000 in
  let key = Threefry.make_key ~run_key:123L ~namespace:0L in
  let sum = ref 0.0 in
  let sum2 = ref 0.0 in
  for i = 0 to n - 1 do
    let ctr = Threefry.make_ctr ~site_id:0 ~component:1 ~frame_index:i in
    let (r0, r1) = Threefry.threefry2x64 ~key ~ctr in
    let u0 = Threefry.to_open_unit r0 in
    let u1 = Threefry.to_open_unit r1 in
    sum := !sum +. u0 +. u1;
    sum2 := !sum2 +. u0 *. u0 +. u1 *. u1
  done;
  let count = float_of_int (2 * n) in
  let mean = !sum /. count in
  let mean_sq = !sum2 /. count in
  (* E[U] = 0.5, Var[U] = 1/12, E[U^2] = 1/3 *)
  (* 3σ for mean of 2M samples: 3 * sqrt(1/12 / 2e6) ≈ 0.00061 *)
  let tol_mean = 0.001 in
  let tol_sq = 0.001 in
  if Float.abs (mean -. 0.5) > tol_mean then
    fail (Printf.sprintf "E[X] = %.6f, expected 0.5 (tol %.4f)" mean tol_mean);
  if Float.abs (mean_sq -. (1.0 /. 3.0)) > tol_sq then
    fail (Printf.sprintf "E[X^2] = %.6f, expected 0.333... (tol %.4f)" mean_sq tol_sq);
  check pass "moments" () ()

(* ── Test suite ── *)

let () =
  run "Threefry" [
    "reference vectors", [
      test_case "zeros"  `Quick test_vector_zeros;
      test_case "ones"   `Quick test_vector_ones;
      test_case "pi_e"   `Quick test_vector_pi_e;
    ];
    "properties", [
      test_case "reproducibility"        `Quick test_reproducibility;
      test_case "sensitivity"            `Quick test_sensitivity;
      test_case "counter boundaries"     `Quick test_counter_boundaries;
      test_case "namespace sensitivity"  `Quick test_namespace_sensitivity;
      test_case "open unit range"        `Quick test_open_unit_range;
      test_case "moments"                `Quick test_moments;
    ];
  ]
