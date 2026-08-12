(* Normal(μ, σ) as D_pushforward of D_uniform.
   fwd: u → μ + σ * √2 * erfinv(2u - 1)
   inv: x → 0.5 * (1 + erf((x - μ) / (σ√2)))
   where μ and σ are variable names bound in the enclosing environment. *)

open Types

let mk_scalar f =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 f; const t

let normal ~mu ~sigma : dist =
  let sqrt2 = mk_scalar (sqrt 2.0) in
  let half = mk_scalar 0.5 in
  let one = mk_scalar 1.0 in
  let two = mk_scalar 2.0 in
  D_pushforward {
    fwd_var = "_u";
    fwd =
      (* μ + σ * √2 * erfinv(2u - 1) *)
      prim Add [
        var mu;
        prim Mul [
          var sigma;
          prim Mul [
            sqrt2;
            prim Erfinv [prim Sub [prim Mul [two; var "_u"]; one]]
          ]
        ]
      ];
    inv_var = "_x";
    inv =
      (* 0.5 * (1 + erf((x - μ) / (σ√2))) *)
      prim Mul [
        half;
        prim Add [
          one;
          prim Erf [prim Div [prim Sub [var "_x"; var mu];
                              prim Mul [var sigma; sqrt2]]]
        ]
      ];
    base = D_uniform;
  }
