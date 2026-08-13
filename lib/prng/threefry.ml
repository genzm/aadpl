(* Threefry-2x64: counter-based PRNG from Salmon et al. 2011 (Random123).
   20 rounds, no external dependencies. *)

(* Skein parity constant *)
let skein_parity = 0x1BD11BDAA9FC1A22L

(* Rotation constants for Threefry-2x64 (8 values, cycled) *)
let rotations = [| 16; 42; 12; 31; 16; 32; 24; 21 |]

let rotate_left (x : int64) (n : int) : int64 =
  Int64.logor (Int64.shift_left x n) (Int64.shift_right_logical x (64 - n))

let threefry2x64 ~key:(k0, k1) ~ctr:(c0, c1) : int64 * int64 =
  let k2 = Int64.logxor (Int64.logxor k0 k1) skein_parity in
  let ks = [| k0; k1; k2 |] in
  let x0 = ref (Int64.add c0 ks.(0)) in
  let x1 = ref (Int64.add c1 ks.(1)) in
  for r = 0 to 19 do
    x0 := Int64.add !x0 !x1;
    x1 := Int64.logxor (rotate_left !x1 rotations.(r mod 8)) !x0;
    if (r + 1) mod 4 = 0 then begin
      let inj = (r + 1) / 4 in
      x0 := Int64.add !x0 ks.(inj mod 3);
      x1 := Int64.add (Int64.add !x1 ks.((inj + 1) mod 3)) (Int64.of_int inj)
    end
  done;
  (!x0, !x1)

(* Convert Int64 to float in open interval (0, 1).
   Uses upper 52 bits: ((x >>> 12) + 0.5) * 2^{-52}.
   Range: [2^{-53}, 1 - 2^{-53}].
   (53-bit version rounds to 1.0 at max due to IEEE-754 tie-breaking.) *)
let to_open_unit (x : int64) : float =
  let u52 = Int64.shift_right_logical x 12 in
  (Int64.to_float u52 +. 0.5) *. (1.0 /. 4503599627370496.0)

(* Key construction: (run_key, namespace) *)
let make_key ~run_key ~namespace : int64 * int64 = (run_key, namespace)

(* Counter construction: structurally injective.
   ctr[0] = site_id (upper 32 bits) | component (lower 32 bits)
   ctr[1] = frame_index
   component encodes D_product tree path: root=1, left=2k, right=2k+1.
   Injective for depth ≤ 31. *)
let make_ctr ~site_id ~component ~frame_index : int64 * int64 =
  ( Int64.logor
      (Int64.shift_left (Int64.of_int site_id) 32)
      (Int64.of_int component),
    Int64.of_int frame_index )

(* Namespace constants *)
let ns_init = 0L
let ns_model = 1L
let ns_data = 2L
let ns_guide = 3L
