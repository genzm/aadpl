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
  log_density : Ast.Types.expr;
      (* log q(z), the proposal's own density, read through the same trace
         variables as [body].  An estimator that differentiates the MEASURE
         needs it; one that differentiates the PATH does not, which is why it
         sits here rather than being folded into the body. *)
  supports : (string * Ast.Types.support) list;
      (* The resolved support of each proposal site, by name.  Resolving it needs
         shape inference over the proposal, which only the objective has in hand,
         so it is settled once here rather than re-derived by each lowering. *)
  env_shapes : (string * int array) list;
}

type t =
  | Deterministic of Ast.Types.expr
      (* No expectation to take; the expression is already a tensor program. *)
  | Expect of expectation
      (* E_{z ~ proposal} [ body ] *)

(* What every lowering produces: an ordinary tensor program, plus the stochastic
   inputs it expects to be supplied from outside -- uniforms %u.<site> for a
   pathwise lowering, draws %tr.<site> for a sampling one.  The list is empty
   exactly when the lowering is exact rather than sampled, which is how a caller
   tells the two apart. *)
type program = {
  loss : Ast.Types.expr;
  sites : Ast.Sites.site list;
  noise : (string * int array) list;
}
