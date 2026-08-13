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
  View.Buf.set t.buf 0 f;
  const t

(* Collect free variables from an expression *)
let free_vars (e : expr) : string list =
  let tbl = Hashtbl.create 8 in
  let rec go bound e =
    match e with
    | Const _ -> ()
    | Var (_, s) -> if not (List.mem s bound) then Hashtbl.replace tbl s ()
    | Prim (_, _, args) -> List.iter (go bound) args
    | Let (_, s, e1, e2) ->
        go bound e1;
        go (s :: bound) e2
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
    failwith
      (Printf.sprintf
         "assess_expr: Sample/Score must be let-bound, not nested in Prim (at \
          %s:%d)"
         loc.file loc.line)

(* Sum over all frame axes *)
let sum_frame frame e =
  let n = Array.length frame in
  let rec go i acc =
    if i = 0 then acc else go (i - 1) (prim (Sum_axis 0) [ acc ])
  in
  go n e

(* Substitute Var(from) → to_ in an expression, respecting Let shadowing *)
let rec subst ~from ~(to_ : expr) (e : expr) : expr =
  match e with
  | Var (_, s) when s = from -> to_
  | Var _ | Const _ -> e
  | Prim (l, p, args) -> Prim (l, p, List.map (subst ~from ~to_) args)
  | Let (l, s, e1, e2) ->
      let e1' = subst ~from ~to_ e1 in
      if s = from then Let (l, s, e1', e2)
      else Let (l, s, e1', subst ~from ~to_ e2)
  | Rank (l, k, p, args) -> Rank (l, k, p, List.map (subst ~from ~to_) args)
  | Sample (l, name, frame, dist) ->
      Sample (l, name, frame, subst_dist ~from ~to_ dist)
  | Score (l, e) -> Score (l, subst ~from ~to_ e)

and subst_dist ~from ~to_ = function
  | D_uniform -> D_uniform
  | D_categorical weights -> D_categorical (subst ~from ~to_ weights)
  | D_pushforward { fwd_var; fwd; inv_var; inv; support; base } ->
      let fwd = if fwd_var = from then fwd else subst ~from ~to_ fwd in
      let inv = if inv_var = from then inv else subst ~from ~to_ inv in
      D_pushforward
        { fwd_var; fwd; inv_var; inv; support;
          base = subst_dist ~from ~to_ base }
  | D_product (a, b) ->
      D_product (subst_dist ~from ~to_ a, subst_dist ~from ~to_ b)

(* Build raw (scalar) log-density expression for a distribution.
   Forward AD is applied to the raw inv (no wrap_rank0/expand).
   Returns None for zero contribution (D_uniform).

   Tangent seeds are INLINED via substitution (not Let-bound) so the output
   is a pure primal expression. This is critical: if tangent-named variables
   appeared as Let bindings, a subsequent Forward.forward pass (Transform.grad)
   would auto-generate bindings with the same names, causing Unzip collisions.

   inv_var is renamed to a gensym'd name for the same reason. *)
let rec log_density_raw loc dist x =
  match dist with
  | D_uniform ->
      Some (Prim (loc, Log_unit_density, [x]))
  | D_categorical _ ->
      failwith
        "assess_expr: D_categorical log-density not yet implemented (Phase 12+)"
  | D_pushforward { inv_var; inv; base; _ } ->
      (* Rename inv_var to a unique gensym'd name *)
      let local_var = Forward.gensym "x" in
      let inv = subst ~from:inv_var ~to_:(var local_var) inv in
      let fvs = free_vars inv in
      let other_vars = List.filter (fun v -> v <> local_var) fvs in
      (* Forward AD on raw (scalar) inv expression *)
      let bs, primal, tangent = Forward.forward inv in
      (* Inline tangent seeds: substitute tangent-named free vars with constants.
       inv_var tangent → 1.0 (derivative wrt itself), others → 0.0. *)
      let inline e =
        let e =
          subst ~from:(Forward.tangent_name local_var) ~to_:(mk_scalar 1.0) e
        in
        List.fold_left
          (fun acc v ->
            subst ~from:(Forward.tangent_name v) ~to_:(mk_scalar 0.0) acc)
          e other_vars
      in
      let bs' = List.map (fun (name, e) -> (name, inline e)) bs in
      let tangent' = inline tangent in
      (* Primal seed: bind local_var to x (the slot value) *)
      let primal_seed = [ (local_var, x) ] in
      (* u = inv(x), jac = d(inv)/dx *)
      let u_var = Forward.gensym "u" in
      let jac_var = Forward.gensym "jac" in
      let base_ld = log_density_raw loc base (var u_var) in
      let log_jac = prim Log [ var jac_var ] in
      let ld =
        match base_ld with
        | None -> log_jac
        | Some bld -> prim Add [ bld; log_jac ]
      in
      let body = let_ u_var primal (let_ jac_var tangent' ld) in
      Some (Forward.wrap_bindings primal_seed (Forward.wrap_bindings bs' body))
  | D_product _ -> failwith "assess_expr: D_product not supported"

(* Build log-density expression, handling frame broadcasting.
   For frame samples, wraps the raw expression in Rank(0,...) + expand
   so scalar constants and env variables broadcast against frame-shaped inputs. *)
let log_density_expr ~env_shapes ~loc _frame dist x =
  match log_density_raw loc dist x with
  | None -> None
  | Some raw ->
      let raw = prim Add
        [Prim (loc, Log_support_density (Ast.Sites.dist_support dist), [x]); raw]
      in
      let wrapped = Reparam.wrap_rank0 raw in
      (* senv: free variables in the raw expression that need shapes *)
      let senv =
        List.filter_map
          (fun v ->
            match List.assoc_opt v env_shapes with
            | Some sh -> Some (v, sh)
            | None -> None)
          (free_vars raw)
      in
      Some (Expand_rank.expand ~senv wrapped)

let assess_expr ?(ns = "a.") ~(env_shapes : (string * int array) list)
    (e : expr) (slots : (string * expr) list) : expr =
  Forward.reset_gensym ();
  Forward.with_ns ns (fun () ->
      let rec go senv e =
        match e with
        | Const _ | Var _ -> mk_scalar 0.0
        | Let (l, s, e1, e2) ->
            let v1, ld1 = go_bind senv e1 in
            let sh1 = Expand_rank.infer_shape senv v1 in
            let local = Forward.gensym "let" in
            let e2 = subst ~from:s ~to_:(var local) e2 in
            let ld2 = go ((local, sh1) :: senv) e2 in
            Let (l, local, v1, prim Add [ ld1; ld2 ])
        | Prim (l, _, args) ->
            assert_no_sample_in_args l args;
            mk_scalar 0.0
        | Score (_, e) ->
            let value = rewrite senv e in
            let shape = Expand_rank.infer_shape senv value in
            if shape = [||] then value else sum_frame shape value
        | Sample (loc, name, frame, dist) ->
            density_of senv loc name frame dist
        | Rank _ -> failwith "assess_expr: Rank must be expanded first"
      and go_bind senv e =
        match e with
        | Sample (loc, name, frame, dist) ->
            (List.assoc name slots, density_of senv loc name frame dist)
        | _ -> (rewrite senv e, go senv e)
      and density_of senv loc name frame dist =
        let slot = List.assoc name slots in
        match log_density_expr ~env_shapes:senv ~loc frame dist slot with
        | None -> mk_scalar 0.0
        | Some ld -> if frame = [||] then ld else sum_frame frame ld
      and rewrite senv e =
        match e with
        | Const _ | Var _ -> e
        | Let (l, s, e1, e2) ->
            let v1, _ = go_bind senv e1 in
            let sh1 = Expand_rank.infer_shape senv v1 in
            let local = Forward.gensym "let" in
            let e2 = subst ~from:s ~to_:(var local) e2 in
            Let (l, local, v1, rewrite ((local, sh1) :: senv) e2)
        | Prim (l, p, args) ->
            assert_no_sample_in_args l args;
            Prim (l, p, List.map (rewrite senv) args)
        | Sample (_, name, _, _) -> List.assoc name slots
        | Score _ -> mk_scalar 0.0
        | Rank _ -> failwith "assess_expr: Rank"
      in
      go env_shapes e)
