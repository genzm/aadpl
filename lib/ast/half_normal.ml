(* HalfNormal(σ) as a direct pushforward of D_uniform.
   fwd: u -> σ √2 erfinv(u)
   inv: x -> erf(x / (σ √2))
   where σ is a positive variable bound in the enclosing environment. *)

open Types

let mk_scalar f =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 f;
  const t

let half_normal ~sigma : dist =
  let sqrt2 = mk_scalar (sqrt 2.0) in
  D_pushforward
    {
      fwd_var = "_u";
      fwd =
        prim Mul
          [var sigma; prim Mul [sqrt2; prim Erfinv [var "_u"]]];
      inv_var = "_x";
      inv =
        prim Erf
          [prim Div [var "_x"; prim Mul [var sigma; sqrt2]]];
      support = S_positive;
      base = D_uniform;
    }
