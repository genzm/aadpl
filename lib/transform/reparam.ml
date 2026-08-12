(* reparam: AST-to-AST transform that eliminates D_pushforward from Sample nodes.
   Sample(name, frame, D_pushforward{fwd_var; fwd; base}) becomes:
     Let(%u.name, Sample(name, frame, base), subst fwd_var→%u.name in wrap_rank0(fwd))

   Site name is preserved on the base Sample so that Threefry counter
   (site_id, component, frame_index) produces the same random draws as
   simulate's D_pushforward branch — this is the coupling/key-inheritance
   rule (§6.7).

   inv is discarded: density computation uses assess on the original
   (pre-reparam) expression.

   fwd is wrapped in Rank(0,...) so that expand_rank's leading agreement
   handles frame broadcasting correctly when fwd contains scalar constants
   alongside frame-shaped variables. *)

open Ast.Types

let u_name site = "%u." ^ site

(* Wrap each Prim in fwd as Rank(0, p, args) so expand_rank can broadcast
   scalar constants against frame-shaped variables. Var/Const/Let pass through. *)
let rec wrap_rank0 (e : expr) : expr =
  match e with
  | Const _ | Var _ -> e
  | Prim (l, p, args) -> Rank (l, 0, p, List.map wrap_rank0 args)
  | Let (l, s, e1, e2) -> Let (l, s, wrap_rank0 e1, wrap_rank0 e2)
  | Rank _ | Sample _ | Score _ ->
    failwith "wrap_rank0: non-elementwise construct in fwd"

(* Substitute fwd_var → replacement in an expression *)
let rec subst_var ~from ~to_ (e : expr) : expr =
  match e with
  | Var (_, s) when s = from -> to_
  | Var _ | Const _ -> e
  | Prim (l, p, args) -> Prim (l, p, List.map (subst_var ~from ~to_) args)
  | Let (l, s, e1, e2) ->
    let e1' = subst_var ~from ~to_ e1 in
    if s = from then Let (l, s, e1', e2)  (* shadowed *)
    else Let (l, s, e1', subst_var ~from ~to_ e2)
  | Rank (l, k, p, args) -> Rank (l, k, p, List.map (subst_var ~from ~to_) args)
  | Sample _ | Score _ -> failwith "subst_var: unexpected Sample/Score in fwd"

(* Reparameterize a distribution, returning the base dist and the fwd expression
   with fwd_var substituted to the given variable name. Recurses through nested
   D_pushforward (e.g. Normal = affine ∘ Φ⁻¹ ∘ Uniform unfolds in 1 step since
   base=D_uniform, but a LogNormal = exp ∘ Normal would unfold in 2). *)
let rec reparam_dist (loc : loc) (name : string) (dist : dist) : dist * expr option =
  match dist with
  | D_uniform | D_categorical _ -> (dist, None)
  | D_pushforward { fwd_var; fwd; base; _ } ->
    let un = u_name name in
    let (base', inner_fwd) = reparam_dist loc name base in
    let fwd_body = wrap_rank0 (subst_var ~from:fwd_var ~to_:(var un) fwd) in
    let full_fwd = match inner_fwd with
      | None -> fwd_body
      | Some inner ->
        (* Nested: inner maps u→intermediate, fwd maps intermediate→final.
           Compose: substitute un in fwd with (inner applied to un). *)
        subst_var ~from:un ~to_:inner fwd_body in
    (base', Some full_fwd)
  | D_product _ ->
    failwith "reparam: D_product in base not supported (Phase 12)"

let rec reparam (e : expr) : expr =
  match e with
  | Const _ | Var _ -> e
  | Prim (l, p, args) -> Prim (l, p, List.map reparam args)
  | Let (l, s, e1, e2) -> Let (l, s, reparam e1, reparam e2)
  | Rank (l, k, p, args) -> Rank (l, k, p, List.map reparam args)
  | Score (l, e) -> Score (l, reparam e)
  | Sample (l, name, frame, dist) ->
    let (base, fwd_opt) = reparam_dist l name dist in
    (match fwd_opt with
     | None -> Sample (l, name, frame, base)  (* D_uniform/D_categorical: unchanged *)
     | Some fwd_body ->
       let un = u_name name in
       Let (l, un, Sample (l, name, frame, base), fwd_body))

(* is_reparammed: true if no Sample has D_pushforward as its dist *)
let rec is_reparammed (e : expr) : bool =
  match e with
  | Const _ | Var _ -> true
  | Prim (_, _, args) -> List.for_all is_reparammed args
  | Let (_, _, e1, e2) -> is_reparammed e1 && is_reparammed e2
  | Rank (_, _, _, args) -> List.for_all is_reparammed args
  | Score (_, e) -> is_reparammed e
  | Sample (_, _, _, dist) -> dist_is_primitive dist

and dist_is_primitive = function
  | D_uniform | D_categorical _ -> true
  | D_pushforward _ -> false
  | D_product (a, b) -> dist_is_primitive a && dist_is_primitive b
