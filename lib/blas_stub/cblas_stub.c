#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/bigarray.h>
#include <Accelerate/Accelerate.h>

/* cblas_dgemm_stub(m, n, k, a, b, c)
   C = A * B  where A:[m,k], B:[k,n], C:[m,n]
   a, b, c are float64 bigarray1 (c_layout).
   All row-major (CblasRowMajor). */
CAMLprim value cblas_dgemm_stub(value vm, value vn, value vk,
                                value va, value vb, value vc) {
  int m = Int_val(vm);
  int n = Int_val(vn);
  int k = Int_val(vk);
  double *a = Caml_ba_data_val(va);
  double *b = Caml_ba_data_val(vb);
  double *c = Caml_ba_data_val(vc);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k, 1.0, a, k, b, n, 0.0, c, n);
  return Val_unit;
}

CAMLprim value cblas_dgemm_stub_bytecode(value *argv, int argc) {
  (void)argc;
  return cblas_dgemm_stub(argv[0], argv[1], argv[2],
                          argv[3], argv[4], argv[5]);
}
