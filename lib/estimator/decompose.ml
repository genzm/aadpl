(* Additive decomposition of a log-density expression.

   assess_expr emits gensymmed Lets wrapped around Add trees, one term per
   density or score contribution.  Splitting that back apart is what lets an
   estimator attribute cost to sites: a score term for site i may drop every
   term that cannot depend on z_i, which changes the variance without changing
   the expectation.

   The binder names are unique (they are gensymmed), so hoisting them to the top
   cannot capture; [split] checks that rather than assuming it. *)

open Ast.Types

exception Decompose_error of string

type t = {
  bindings : (string * expr) list;  (* hoisted, in dependency order *)
  terms : expr list;                (* the summands, in program order *)
}

let split (e : expr) : t =
  let rec go e =
    match e with
    | Let (_, name, rhs, body) ->
        let inner = go body in
        { inner with bindings = (name, rhs) :: inner.bindings }
    | Prim (_, Add, [ left; right ]) ->
        let left = go left and right = go right in
        {
          bindings = left.bindings @ right.bindings;
          terms = left.terms @ right.terms;
        }
    | Prim (loc, Sub, [ left; right ]) ->
        let left = go left and right = go right in
        {
          bindings = left.bindings @ right.bindings;
          terms =
            left.terms @ List.map (fun t -> Prim (loc, Neg, [ t ])) right.terms;
        }
    | _ -> { bindings = []; terms = [ e ] }
  in
  let result = go e in
  let names = List.map fst result.bindings in
  if List.length names <> List.length (List.sort_uniq String.compare names) then
    raise (Decompose_error "hoisting would shadow: binder names are not unique");
  result

(* Rebuild the expression the split came from.  Used to check the split against
   the original rather than trusting it. *)
let recombine (t : t) : expr =
  List.fold_right
    (fun (name, rhs) body -> let_ name rhs body)
    t.bindings (Lit.sum t.terms)

(* Which of [traces] an expression can reach, following the hoisted bindings.
   The bindings are in dependency order, so one forward pass is enough. *)
let reach (t : t) ~(traces : string list) : expr -> string list =
  let step table e =
    Transform.Assess_expr.free_vars e
    |> List.concat_map (fun name ->
           if List.mem name traces then [ name ]
           else match List.assoc_opt name table with Some r -> r | None -> [])
    |> List.sort_uniq String.compare
  in
  let table =
    List.fold_left
      (fun table (name, rhs) -> (name, step table rhs) :: table)
      [] t.bindings
  in
  step table
