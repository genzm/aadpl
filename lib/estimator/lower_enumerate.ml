(* Exact enumeration.

     E_{z ~ q} [ f(z) ]  =  sum over the joint support of  q(z) * f(z)

   The result holds no Sample and no noise variable: it is the expectation, not
   an estimate of it.  Transform.grad then differentiates it with the ordinary
   AD, so `grad (lower_enumerate objective)` is the EXACT gradient of an
   expectation -- the golden reference a sampling estimator gets checked
   against, and the reason enumeration is worth having before any
   score-function machinery.

   The weight of an assignment is exp of the proposal's own log-density, which
   the IR already carries.  That costs nothing in generality: a proposal whose
   later sites are conditioned on earlier draws is weighted correctly without
   this code ever looking at a distribution, only at the support sizes.

   Each assignment gets its own copy of the body, so the program grows with the
   product of the supports.  Beyond [max_assignments] the right answer is to
   give the enumeration its own leading array axis the way Quadrature does, and
   reduce along it, rather than to replicate. *)

open Ast.Types

exception Enumerate_error of string

let max_assignments = 4096

let categories ~supports (site : Ast.Sites.site) =
  match List.assoc_opt site.name supports with
  | Some (S_finite count) -> count
  | _ ->
      raise
        (Enumerate_error
           (Printf.sprintf "site '%s' does not have a finite support" site.name))

(* Every assignment of a category to each site, in row-major order. *)
let assignments counts =
  List.fold_left
    (fun acc count ->
      List.concat_map
        (fun prefix -> List.init count (fun category -> prefix @ [ category ]))
        acc)
    [ [] ] counts

let check_enumerable (site : Ast.Sites.site) =
  (match Strategy.of_site site with
  | Strategy.Enumerate -> ()
  | other ->
      raise
        (Enumerate_error
           (Printf.sprintf "site '%s' is %s, not enumerable" site.name
              (Strategy.to_string other))));
  if site.frame <> [||] then
    raise
      (Enumerate_error
         (Printf.sprintf "enumeration is over scalar sites; '%s' has a frame"
            site.name))

let lower (objective : Types.t) : Types.program =
  match objective with
  | Types.Deterministic loss -> { Types.loss; sites = []; noise = [] }
  | Types.Expect { sites; body; log_density; supports; env_shapes; _ } ->
      if sites = [] then
        raise
          (Enumerate_error "an expectation over no sites has nothing to enumerate");
      List.iter check_enumerable sites;
      let counts = List.map (categories ~supports) sites in
      let total = List.fold_left ( * ) 1 counts in
      if total > max_assignments then
        raise
          (Enumerate_error
             (Printf.sprintf
                "%d assignments exceeds the replication limit of %d; this needs \
                 an enumeration axis rather than replicated bodies"
                total max_assignments));
      let traces = List.map Ast.Sites.trace_name sites in
      let term index assignment =
        let substitute e =
          List.fold_left2
            (fun e trace category ->
              Transform.Assess_expr.subst ~from:trace
                ~to_:(Lit.scalar (float_of_int category))
                e)
            e traces assignment
        in
        let tag = Printf.sprintf "e%d" index in
        let weight =
          prim Exp [ substitute log_density |> Rename.binders ~tag:(tag ^ "q") ]
        in
        rank 0 Mul [ weight; substitute body |> Rename.binders ~tag ]
      in
      let loss =
        Lit.sum (List.mapi term (assignments counts))
        |> Transform.Expand_rank.expand ~senv:env_shapes
        |> Transform.Desugar.fuse_views
      in
      (* No noise: the expectation is computed, not sampled. *)
      { Types.loss; sites; noise = [] }
