(* Fuse adjacent Apply_view nodes.
   Apply_view s2 [Apply_view s1 [x]] → Apply_view (s1 @ s2) [x].
   Does not fuse across Let boundaries.
   Precondition: expand has been run (no Rank nodes). *)

open Ast.Types

let rec fuse_views (e : expr) : expr =
  match e with
  | Const _ | Var _ -> e
  | Let (loc, s, e1, e2) ->
    Let (loc, s, fuse_views e1, fuse_views e2)
  | Scan _ -> e
  | Rank _ -> failwith "fuse_views: Rank must be expanded first"
  | Sample _ -> failwith "fuse_views: Sample not supported"
  | Score _ -> failwith "fuse_views: Score not supported"
  | Prim (loc, p, args) ->
    let args' = List.map fuse_views args in
    match p, args' with
    | Apply_view spec, [x] -> fuse loc spec x
    (* Note: Adjoint_view composition (Adjoint(s1,ts1)[Adjoint(s2,mid)[x]] =
       Adjoint(s1@s2,ts1)[x]) is valid but only arises in second-order AD.
       Left unfused — each stage goes through add_view, so correctness is unaffected. *)
    | _ -> Prim (loc, p, args')

(* If inner expression is already Apply_view, merge specs (left-to-right) *)
and fuse loc spec x =
  match x with
  | Prim (_, Apply_view inner_spec, [inner_x]) ->
    Prim (loc, Apply_view (inner_spec @ spec), [inner_x])
  | _ ->
    Prim (loc, Apply_view spec, [x])

(* Backward-compatible alias *)
let desugar = fuse_views

(* is_desugared: always true now that old structural prims are deleted.
   Retained for test compatibility; checks only for Rank nodes. *)
let rec is_desugared (e : expr) : bool =
  match e with
  | Const _ | Var _ -> true
  | Let (_, _, e1, e2) -> is_desugared e1 && is_desugared e2
  | Scan (_, scan, continuation) ->
    List.for_all (fun (_, init, next) ->
      is_desugared init && is_desugared next) scan.carries
    && List.for_all (fun (_, input) -> is_desugared input) scan.inputs
    && is_desugared continuation
  | Prim (_, _, args) -> List.for_all is_desugared args
  | Rank _ -> false
  | Sample _ -> true
  | Score _ -> true
