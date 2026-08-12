(* assess_expr: build a log-density expression from a probabilistic program.
   Given a model expression and slot mappings (site name → value expression),
   produces a single expression that evaluates to log p(trace).

   For D_pushforward, the Jacobian is computed symbolically:
   Forward.forward is applied to the raw (scalar) inv expression.
   The result is then wrap_rank0'd + expand_rank'd for frame broadcasting.

   All gensym'd variables use the "a." namespace prefix to avoid collision
   with Transform.grad's forward pass. *)

open Ast.Types

let mk_scalar f =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 f; const t

(* Collect free variables from an expression *)
let free_vars (e : expr) : string list =
  let tbl = Hashtbl.create 8 in
  let rec go bound e =
    match e with
    | Const _ -> ()
    | Var (_, s) ->
      if not (List.mem s bound) then Hashtbl.replace tbl s ()
    | Prim (_, _, args) -> List.iter (go bound) args
    | Let (_, s, e1, e2) -> go bound e1; go (s :: bound) e2
    | Rank (_, _, _, args) -> List.iter (go bound) args
    | Sample _ | Score _ -> ()
  in
  go [] e;
  Hashtbl.fold (fun k () acc -> k :: acc) tbl []

(* Assert no Sample/Score nodes appear inside Prim arguments.
   Samples must be let-bound so their density is captured by go_bind. *)
let assert_no_sample_in_args loc args =
  let rec has_sample = function
    | Sample _ | Score _ -> true
    | Const _ | Var _ -> false
    | Prim (_, _, args) -> List.exists has_sample args
    | Let (_, _, e1, e2) -> has_sample e1 || has_sample e2
    | Rank (_, _, _, args) -> List.exists has_sample args
  in
  if List.exists has_sample args then
    failwith (Printf.sprintf "assess_expr: Sample/Score must be let-bound, not nested in Prim (at %s:%d)" loc.file loc.line)

(* Sum over all frame axes *)
let sum_frame frame e =
  let n = Array.length frame in
  let rec go i acc = if i = 0 then acc else go (i - 1) (prim (Sum_axis 0) [acc]) in
  go n e

(* Build raw (scalar) log-density expression for a distribution.
   Forward AD is applied to the raw inv (no wrap_rank0/expand).
   Returns None for zero contribution (D_uniform). *)
let rec log_density_raw dist x =
  match dist with
  | D_uniform -> None
  | D_categorical _ ->
    failwith "assess_expr: D_categorical log-density not yet implemented (Phase 12+)"
  | D_pushforward { inv_var; inv; base; _ } ->
    let fvs = free_vars inv in
    let other_vars = List.filter (fun v -> v <> inv_var) fvs in
    (* Forward AD on raw (scalar) inv expression *)
    let (bs, primal, tangent) = Forward.forward inv in
    (* Seed bindings: inv_var → x, tangent seeds *)
    let seeds =
      (inv_var, x) ::
      (Forward.tangent_name inv_var, mk_scalar 1.0) ::
      List.map (fun v -> (Forward.tangent_name v, mk_scalar 0.0)) other_vars in
    (* u = inv(x), jac = d(inv)/dx *)
    let u_var = Forward.gensym "u" in
    let jac_var = Forward.gensym "jac" in
    let base_ld = log_density_raw base (var u_var) in
    let log_jac = prim Log [var jac_var] in
    let ld = match base_ld with
      | None -> log_jac
      | Some bld -> prim Add [bld; log_jac] in
    let body = let_ u_var primal (let_ jac_var tangent ld) in
    Some (Forward.wrap_bindings seeds (Forward.wrap_bindings bs body))
  | D_product _ -> failwith "assess_expr: D_product not supported"

(* Build log-density expression, handling frame broadcasting.
   For frame samples, wraps the raw expression in Rank(0,...) + expand
   so scalar constants and env variables broadcast against frame-shaped inputs. *)
let log_density_expr ~env_shapes frame dist x =
  match log_density_raw dist x with
  | None -> None
  | Some raw ->
    if frame = [||] then Some raw
    else
      let wrapped = Reparam.wrap_rank0 raw in
      (* senv: free variables in the raw expression that need shapes *)
      let senv = List.filter_map (fun v ->
        match List.assoc_opt v env_shapes with
        | Some sh -> Some (v, sh)
        | None -> None
      ) (free_vars raw) in
      Some (Expand_rank.expand ~senv wrapped)

let assess_expr ~(env_shapes : (string * int array) list)
    (e : expr) (slots : (string * expr) list) : expr =
  Forward.reset_gensym ();
  Forward.with_ns "a." (fun () ->
    let rec go e =
      match e with
      | Const _ | Var _ -> mk_scalar 0.0
      | Let (l, s, e1, e2) ->
        let (v1, ld1) = go_bind e1 in
        let ld2 = go e2 in
        Let (l, s, v1, prim Add [ld1; ld2])
      | Prim (l, _, args) ->
        assert_no_sample_in_args l args;
        mk_scalar 0.0
      | Score (_, e) -> rewrite e
      | Sample (_, name, frame, dist) ->
        density_of name frame dist
      | Rank _ -> failwith "assess_expr: Rank must be expanded first"
    and go_bind e =
      match e with
      | Sample (_, name, frame, dist) ->
        (List.assoc name slots, density_of name frame dist)
      | _ -> (rewrite e, go e)
    and density_of name frame dist =
      let slot = List.assoc name slots in
      match log_density_expr ~env_shapes frame dist slot with
      | None -> mk_scalar 0.0
      | Some ld ->
        if frame = [||] then ld
        else sum_frame frame ld
    and rewrite e =
      match e with
      | Const _ | Var _ -> e
      | Let (l, s, e1, e2) ->
        let (v1, _) = go_bind e1 in
        Let (l, s, v1, rewrite e2)
      | Prim (l, p, args) ->
        assert_no_sample_in_args l args;
        Prim (l, p, List.map rewrite args)
      | Sample (_, name, _, _) -> List.assoc name slots
      | Score _ -> mk_scalar 0.0
      | Rank _ -> failwith "assess_expr: Rank"
    in
    go e
  )
