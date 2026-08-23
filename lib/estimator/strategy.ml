(* Which estimator a site admits.

   The rule is given to the CONSTRUCTORS of the distribution algebra, not to
   individual distributions.  Normal, LogNormal and HalfNormal are all built out
   of D_pushforward, so one line covers them and any distribution a user later
   builds the same way.  Adding a per-distribution table here would undo the
   reason the algebra exists. *)

open Ast.Types

type t =
  | Pathwise
      (* z = g(u) with u from a fixed base: differentiate through the map. *)
  | Enumerate  (* finite support: sum over it exactly. *)
  | Score_function
      (* differentiate the measure instead of the path.  Not yet lowered. *)
  | Unsupported

let rec of_dist = function
  | D_uniform -> Pathwise
  | D_pushforward { base; _ } -> of_dist base
  | D_categorical _ -> Enumerate
  | D_product _ -> Unsupported
(* a product of independent draws needs a composite lowering *)

let of_site (site : Ast.Sites.site) = of_dist site.dist

(* The strategy a whole objective admits.  Sites that disagree would need a
   lowering that mixes estimators per site, which does not exist yet, so they
   are reported as unsupported rather than silently lowered by whichever
   strategy happens to come first. *)
let of_sites sites =
  match List.sort_uniq compare (List.map of_site sites) with
  | [] -> Pathwise
  | [ single ] -> single
  | _ -> Unsupported

let to_string = function
  | Pathwise -> "pathwise"
  | Enumerate -> "enumerate"
  | Score_function -> "score_function"
  | Unsupported -> "unsupported"
