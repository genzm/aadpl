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

   Two forms produce the same sum.  [replicate] gives each assignment its own
   copy of the body, so the program grows with the product of the supports;
   [batched:true] instead binds each trace to the column of its coordinates and
   lifts the body once to that axis, the way Quadrature does.  The replicating
   form is the default and is kept as the reference: a test holds the two to
   1e-12 on both value and gradient, which is what makes the batched form
   trustworthy. *)

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

(* One [K] tensor per site, holding that site's coordinate in each assignment. *)
let coordinate_tensors ~count assignments position =
  let tensor = View.Tensor.make [| count |] in
  List.iteri
    (fun index assignment ->
      View.Buf.set tensor.buf index
        (float_of_int (List.nth assignment position)))
    assignments;
  const tensor

(* One program for every assignment at once: bind each trace to the column of
   its coordinates, lift the body and the density to that axis, and reduce.
   Same sum as [replicate], but the program no longer grows with the support. *)
let batch ~traces ~body ~log_density assignments =
  let count = List.length assignments in
  let bindings =
    List.mapi
      (fun position trace ->
        (trace, coordinate_tensors ~count assignments position))
      traces
  in
  let lift e = Batch.lift ~size:count ~batched:traces e in
  (* body already contains a copy of log q; one rename keeps the binders of the
     second copy distinct.  One, not one per assignment. *)
  let weight = prim Exp [ lift (Rename.binders ~tag:"b" log_density) ] in
  Transform.Forward.wrap_let_bindings bindings
    (prim (Sum_axis 0) [ prim Mul [ weight; lift body ] ])

(* One copy of the body per assignment, weighted and summed. *)
let replicate ~traces ~body ~log_density assignments =
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
  Lit.sum (List.mapi term assignments)

let lower ?(batched = false) (objective : Types.t) : Types.program =
  match objective with
  | Types.Deterministic loss -> { Types.loss; sites = []; noise = [] }
  | Types.Expect { sites; body; log_density; supports; env_shapes; _ } ->
      if sites = [] then
        raise
          (Enumerate_error "an expectation over no sites has nothing to enumerate");
      List.iter check_enumerable sites;
      let counts = List.map (categories ~supports) sites in
      let total = List.fold_left ( * ) 1 counts in
      if (not batched) && total > max_assignments then
        raise
          (Enumerate_error
             (Printf.sprintf
                "%d assignments exceeds the replication limit of %d; pass \
                 ~batched:true to enumerate along an axis instead"
                total max_assignments));
      let traces = List.map Ast.Sites.trace_name sites in
      let build = if batched then batch else replicate in
      let loss =
        build ~traces ~body ~log_density (assignments counts)
        |> Transform.Expand_rank.expand ~senv:env_shapes
        |> Transform.Desugar.fuse_views
      in
      (* No noise: the expectation is computed, not sampled. *)
      { Types.loss; sites; noise = [] }
