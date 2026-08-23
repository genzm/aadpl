(* Alpha-renaming of Let binders.

   A lowering that uses a sub-expression more than once -- one copy per category,
   or an integrand alongside the proposal density it already contains -- would
   otherwise repeat the binder names that assess_expr gensymmed, and Unzip
   requires every binding in a program to be unique.  Tagging keeps the copies
   apart while preserving the original name, so the printed IR stays readable. *)

open Ast.Types

exception Rename_error of string

let binders ~(tag : string) (e : expr) : expr =
  let fresh name =
    let bare =
      if String.length name > 0 && name.[0] = '%' then
        String.sub name 1 (String.length name - 1)
      else name
    in
    "%" ^ tag ^ "." ^ bare
  in
  let rec go e =
    match e with
    | Const _ | Var _ -> e
    | Prim (loc, primitive, args) -> Prim (loc, primitive, List.map go args)
    | Rank (loc, cell, primitive, args) ->
        Rank (loc, cell, primitive, List.map go args)
    | Let (loc, name, rhs, body) ->
        let renamed = fresh name in
        Let
          ( loc,
            renamed,
            go rhs,
            go (Transform.Assess_expr.subst ~from:name ~to_:(var renamed) body)
          )
    | Scan _ -> raise (Rename_error "Scan in a lowered body")
    | Sample _ | Score _ -> raise (Rename_error "Sample or Score in a lowered body")
  in
  go e
