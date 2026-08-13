(* reparam: AST-to-AST transform that eliminates D_pushforward from Sample nodes.
   Sample(name, frame, D_pushforward{fwd_var; fwd; base}) becomes:
     Let(%u.name, Sample(name, frame, base),
       Let(%tr.name, subst fwd_var→%u.name in wrap_rank0(fwd), %tr.name))

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
let trace_name site = "%tr." ^ site

(* Wrap each Prim in fwd as Rank(0, p, args) so expand_rank can broadcast
   scalar constants against frame-shaped variables. Var/Const/Let pass through. *)
let rec wrap_rank0 (e : expr) : expr =
  match e with
  | Const _ | Var _ -> e
  | Prim (l, p, args) -> Rank (l, 0, p, List.map wrap_rank0 args)
  | Let (l, s, e1, e2) -> Let (l, s, wrap_rank0 e1, wrap_rank0 e2)
  | Rank (l, _, p, args) -> Rank (l, 0, p, List.map wrap_rank0 args)
  | Sample _ | Score _ ->
      failwith "wrap_rank0: non-elementwise construct in fwd"

(* Substitute fwd_var → replacement in an expression *)
let rec subst_var ~from ~to_ (e : expr) : expr =
  match e with
  | Var (_, s) when s = from -> to_
  | Var _ | Const _ -> e
  | Prim (l, p, args) -> Prim (l, p, List.map (subst_var ~from ~to_) args)
  | Let (l, s, e1, e2) ->
      let e1' = subst_var ~from ~to_ e1 in
      if s = from then Let (l, s, e1', e2) (* shadowed *)
      else Let (l, s, e1', subst_var ~from ~to_ e2)
  | Rank (l, k, p, args) -> Rank (l, k, p, List.map (subst_var ~from ~to_) args)
  | Sample _ | Score _ -> failwith "subst_var: unexpected Sample/Score in fwd"

(* Reparameterize a distribution, returning the base dist and the fwd expression
   with fwd_var substituted to the given variable name. Recurses through nested
   D_pushforward (e.g. Normal = affine ∘ Φ⁻¹ ∘ Uniform unfolds in 1 step since
   base=D_uniform, but a LogNormal = exp ∘ Normal would unfold in 2). *)
let rec reparam_dist (loc : loc) (name : string) (dist : dist) :
    dist * expr option =
  match dist with
  | D_uniform | D_categorical _ -> (dist, None)
  | D_pushforward { fwd_var; fwd; base; _ } ->
      let un = u_name name in
      let base', inner_fwd = reparam_dist loc name base in
      let fwd_body = wrap_rank0 (subst_var ~from:fwd_var ~to_:(var un) fwd) in
      let full_fwd =
        match inner_fwd with
        | None -> fwd_body
        | Some inner ->
            (* Nested: inner maps u→intermediate, fwd maps intermediate→final.
           Compose: substitute un in fwd with (inner applied to un). *)
            subst_var ~from:un ~to_:inner fwd_body
      in
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
  | Sample (l, name, frame, dist) -> (
      let base, fwd_opt = reparam_dist l name dist in
      match fwd_opt with
      | None ->
          let tr = trace_name name in
          Let (l, tr, Sample (l, name, frame, base), var tr)
      | Some fwd_body ->
          let un = u_name name in
          let tr = trace_name name in
          Let
            (l, un, Sample (l, name, frame, base), Let (l, tr, fwd_body, var tr))
      )

(* Eliminate primitive samples from a reparameterized guide and flatten all
   Let nodes into a shared binding list.  Uniform samples become free noise
   variables; their generated self-bindings are omitted.  The result expression
   is returned for completeness, although build_elbo only needs the bindings. *)
exception Elim_error of loc * string

let elim_samples (e : expr) : Forward.bindings * expr =
  let rec go e =
    match e with
    | Const _ | Var _ -> ([], e)
    | Sample (_, name, _, D_uniform) -> ([], var (u_name name))
    | Sample (l, name, _, D_categorical _) ->
        raise
          (Elim_error
             ( l,
               Printf.sprintf "discrete site '%s' cannot be reparameterized"
                 name ))
    | Sample (l, name, _, D_pushforward _) ->
        raise
          (Elim_error
             (l, Printf.sprintf "site '%s' was not reparameterized" name))
    | Sample (l, name, _, D_product _) ->
        raise
          (Elim_error
             ( l,
               Printf.sprintf "product site '%s' is not supported (Phase 12)"
                 name ))
    | Score (_, _) -> ([], mk_zero ())
    | Let (_, s, e1, e2) ->
        let bs1, r1 = go e1 in
        let bs2, r2 = go e2 in
        let binding =
          match r1 with Var (_, s') when s = s' -> [] | _ -> [ (s, r1) ]
        in
        (bs1 @ binding @ bs2, r2)
    | Prim (l, p, args) ->
        let bindings, args' = go_list args in
        (bindings, Prim (l, p, args'))
    | Rank (l, k, p, args) ->
        let bindings, args' = go_list args in
        (bindings, Rank (l, k, p, args'))
  and go_list args =
    List.fold_left
      (fun (bindings, rev_args) arg ->
        let bs, arg' = go arg in
        (bindings @ bs, arg' :: rev_args))
      ([], []) args
    |> fun (bindings, rev_args) -> (bindings, List.rev rev_args)
  and mk_zero () =
    let t = View.Tensor.make [||] in
    const t
  in
  go e

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

(* --- collect_sites: gather (name, frame) pairs from Sample nodes --- *)

let collect_sites (e : expr) : (string * int array) list =
  let acc = ref [] in
  let rec walk = function
    | Sample (_, name, frame, dist) ->
        acc := (name, frame) :: !acc;
        walk_dist dist
    | Score (_, e) -> walk e
    | Const _ | Var _ -> ()
    | Prim (_, _, args) -> List.iter walk args
    | Let (_, _, e1, e2) ->
        walk e1;
        walk e2
    | Rank (_, _, _, args) -> List.iter walk args
  and walk_dist = function
    | D_uniform -> ()
    | D_categorical e -> walk e
    | D_pushforward { fwd; inv; base; _ } ->
        walk fwd;
        walk inv;
        walk_dist base
    | D_product (a, b) ->
        walk_dist a;
        walk_dist b
  in
  walk e;
  List.rev !acc

(* --- check_trace_compat: verify model and guide have matching sites --- *)

exception Trace_mismatch of string

let check_trace_compat ~model ~guide =
  let model_sites = collect_sites model in
  let guide_sites = collect_sites guide in
  let pp_frame frame =
    String.concat "," (List.map string_of_int (Array.to_list frame))
  in
  List.iter
    (fun (name, guide_frame) ->
      match List.assoc_opt name model_sites with
      | None ->
          raise
            (Trace_mismatch
               (Printf.sprintf "guide site '%s' not found in model" name))
      | Some model_frame ->
          if model_frame <> guide_frame then
            raise
              (Trace_mismatch
                 (Printf.sprintf
                    "site '%s' frame mismatch: model=[%s] guide=[%s]" name
                    (pp_frame model_frame) (pp_frame guide_frame))))
    guide_sites;
  List.iter
    (fun (name, _) ->
      match List.assoc_opt name guide_sites with
        | None ->
            raise
              (Trace_mismatch
                 (Printf.sprintf "model site '%s' not found in guide" name))
        | Some _ -> ())
    model_sites
