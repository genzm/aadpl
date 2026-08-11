(* Minimal BLAS binding: cblas_dgemm via macOS Accelerate.

   Contract:
   - All arrays are row-major (CblasRowMajor), NoTrans.
   - Caller must ensure a has m*k, b has k*n, c has m*n contiguous elements.
   - c is overwritten: C := A * B.
   - Transposed inputs must be materialized by the caller before passing. *)

open Bigarray

external dgemm_raw : int -> int -> int ->
  (float, float64_elt, c_layout) Array1.t ->
  (float, float64_elt, c_layout) Array1.t ->
  (float, float64_elt, c_layout) Array1.t -> unit
  = "cblas_dgemm_stub_bytecode" "cblas_dgemm_stub"

let dgemm ~m ~n ~k (a : View.Buf.t) (b : View.Buf.t) (c : View.Buf.t) =
  dgemm_raw m n k a.data b.data c.data
