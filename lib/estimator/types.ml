(* Estimator IR.

     Model AST --[elbo, importance, ...]--> Estimator IR --[lowering]--> Tensor AST

   This layer records WHICH statistical quantity is wanted.  It does not record
   how that quantity is estimated: choosing an estimator is the lowering's job,
   so one objective can be lowered pathwise, by exact enumeration, or by a score
   function without the objective itself changing.

   Keeping the two apart is what makes the estimator checkable.  An objective is
   a statement about p and q that can be compared against a closed form; a
   lowering is a rewrite whose output can be compared against another lowering
   of the same objective. *)

type expectation = {
  sites : Ast.Sites.site list;
      (* Static sites of [proposal], in Sites.collect_sites order.  These ids are
         the ones the Threefry counter uses, so a lowering that draws noise must
         use this list unchanged or the draws move. *)
  proposal : Ast.Types.expr;
      (* The program whose Sample nodes draw z.  Deliberately NOT rank-expanded:
         Reparam.reparam rewrites distribution [fwd] bodies through wrap_rank0
         and expects them unexpanded.  Lowerings expand after reparameterizing. *)
  body : Ast.Types.expr;
      (* f(z).  Already rank-expanded.  Reads z through the trace variables
         %tr.<site>, which the lowering is responsible for binding. *)
  env_shapes : (string * int array) list;
}

type t =
  | Deterministic of Ast.Types.expr
      (* No expectation to take; the expression is already a tensor program. *)
  | Expect of expectation
      (* E_{z ~ proposal} [ body ] *)
