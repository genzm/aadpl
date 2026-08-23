(* Lift an expanded tensor program to a leading batch axis.

   Every free variable named in [batched] gains a leading axis of [size]; every
   expression that depends on one gains it too, and anything that does not is
   broadcast to match.  That is precisely what Expand_rank does to a Rank node's
   frame, so the axis arithmetic here is Expand_rank.shift_prim rather than a
   second implementation of it -- the batch axis IS a frame axis.

   Lower_enumerate uses this to evaluate every assignment at once rather than
   replicating the body per assignment.  The replicated form is kept as the
   reference the batched form is checked against, on the same principle that
   keeps Kernel.Naive alive next to BLAS: an implementation with nothing to
   compare against has lost its standard. *)

open Ast.Types

exception Batch_error of string

let lift ~(size : int) ~(batched : string list) (e : expr) : expr =
  if size <= 0 then raise (Batch_error "batch size must be positive");
  let broadcast e = prim (Apply_view [ Vbroadcast (0, size) ]) [ e ] in
  let rec go batched e =
    match e with
    | Const _ -> (e, false)
    | Var (_, name) -> (e, List.mem name batched)
    | Let (loc, name, rhs, body) ->
        let rhs, rhs_is_batched = go batched rhs in
        (* a rebinding shadows whatever the name meant outside *)
        let batched =
          if rhs_is_batched then name :: batched
          else List.filter (fun bound -> bound <> name) batched
        in
        let body, body_is_batched = go batched body in
        (Let (loc, name, rhs, body), body_is_batched)
    | Prim (loc, primitive, args) ->
        let args = List.map (go batched) args in
        if not (List.exists snd args) then
          (Prim (loc, primitive, List.map fst args), false)
        else
          let args =
            List.map
              (fun (arg, is_batched) -> if is_batched then arg else broadcast arg)
              args
          in
          ( Prim
              (loc, Transform.Expand_rank.shift_prim [| size |] primitive, args),
            true )
    | Rank _ ->
        raise (Batch_error "Rank must be expanded before batching")
    | Scan _ -> raise (Batch_error "Scan cannot be batched")
    | Sample _ | Score _ ->
        raise (Batch_error "Sample or Score in a batched program")
  in
  let result, is_batched = go batched e in
  if is_batched then result else broadcast result
