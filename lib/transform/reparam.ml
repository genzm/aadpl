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

let u_name = Ast.Sites.noise_name_of_name
let trace_name = Ast.Sites.trace_name_of_name

(* Wrap each Prim in fwd as Rank(0, p, args) so expand_rank can broadcast
   scalar constants against frame-shaped variables. Var/Const/Let pass through. *)
let rec wrap_rank0 (e : expr) : expr =
  match e with
  | Const _ | Var _ -> e
  | Prim (l, p, args) -> Rank (l, 0, p, List.map wrap_rank0 args)
  | Let (l, s, e1, e2) -> Let (l, s, wrap_rank0 e1, wrap_rank0 e2)
  | Rank (l, 0, p, args) -> Rank (l, 0, p, List.map wrap_rank0 args)
  | Rank _ -> failwith "wrap_rank0: non-elementwise Rank in fwd"
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
let rec reparam_dist (loc : loc) (site : Ast.Sites.site) (dist : dist) :
    dist * expr option =
  match dist with
  | D_uniform | D_categorical _ -> (dist, None)
  | D_pushforward { fwd_var; fwd; base; _ } ->
      let un = Ast.Sites.noise_name site in
      let base', inner_fwd = reparam_dist loc site base in
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

let reparam ?sites (e : expr) : expr =
  let sites = Option.value sites ~default:(Ast.Sites.collect_sites e) in
  let rec go (e : expr) : expr =
    match e with
    | Const _ | Var _ -> e
    | Prim (l, p, args) -> Prim (l, p, List.map go args)
    | Let (l, s, e1, e2) -> Let (l, s, go e1, go e2)
    | Rank (l, k, p, args) -> Rank (l, k, p, List.map go args)
    | Score (l, e) -> Score (l, go e)
    | Sample (l, name, frame, dist) -> (
        let site = Ast.Sites.find name sites in
        let base, fwd_opt = reparam_dist l site dist in
        match fwd_opt with
        | None ->
            let tr = Ast.Sites.trace_name site in
            Let (l, tr, Sample (l, name, frame, base), var tr)
        | Some fwd_body ->
            let un = Ast.Sites.noise_name site in
            let tr = Ast.Sites.trace_name site in
            Let
              ( l,
                un,
                Sample (l, name, frame, base),
                Let (l, tr, fwd_body, var tr) ))
  in
  go e

(* Eliminate primitive samples from a reparameterized guide and flatten all
   Let nodes into a shared binding list.  Uniform samples become free noise
   variables; their generated self-bindings are omitted.  The result expression
   is returned for completeness, although build_elbo only needs the bindings. *)
exception Elim_error of loc * string

let elim_samples ~(sites : Ast.Sites.site list) (e : expr) :
    Forward.bindings * expr =
  let rec go e =
    match e with
    | Const _ | Var _ -> ([], e)
    | Sample (_, name, _, D_uniform) ->
        ([], var (Ast.Sites.noise_name (Ast.Sites.find name sites)))
    | Sample (loc, name, _, _) ->
        raise
          (Elim_error
             ( loc,
               Printf.sprintf "site '%s' violates elim_samples precondition"
                 name ))
    | Score (loc, _) ->
        raise (Elim_error (loc, "Score violates elim_samples precondition"))
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
  in
  let bindings, result = go e in
  let names = List.map fst bindings in
  if List.length names <> List.length (List.sort_uniq String.compare names) then
    raise
      (Elim_error
         (loc_of e, "binder names must be unique across the flattened guide"));
  (bindings, result)

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

let collect_sites = Ast.Sites.collect_sites

exception Guide_error of loc * string

let check_guide (e : expr) : unit =
  let rec check_dist loc name = function
    | D_uniform -> ()
    | D_pushforward { base; _ } -> check_dist loc name base
    | D_categorical _ ->
        raise
          (Guide_error (loc, Printf.sprintf "guide site '%s' is discrete" name))
    | D_product _ ->
        raise
          (Guide_error
             ( loc,
               Printf.sprintf
                 "guide site '%s' uses D_product, which is not \
                  reparameterizable"
                 name ))
  in
  let rec walk = function
    | Const _ | Var _ -> ()
    | Prim (_, _, args) | Rank (_, _, _, args) -> List.iter walk args
    | Let (_, _, e1, e2) ->
        walk e1;
        walk e2
    | Score (loc, _) ->
        raise (Guide_error (loc, "guide must not contain Score"))
    | Sample (loc, name, _, dist) -> check_dist loc name dist
  in
  check_sites e;
  walk e

(* --- check_trace_compat: verify model and guide have matching sites --- *)

exception Trace_mismatch of string

exception Support_mismatch of loc * string

let check_support_compat ~model_sites ~guide_sites =
  List.iter
    (fun (guide_site : Ast.Sites.site) ->
      match List.find_opt (fun site ->
        site.Ast.Sites.name = guide_site.name) model_sites with
      | None -> ()
      | Some model_site ->
          let guide_support = Ast.Sites.dist_support guide_site.dist in
          let model_support = Ast.Sites.dist_support model_site.dist in
          if not (Ast.Sites.support_subset guide_support model_support) then
            raise
              (Support_mismatch
                 (guide_site.loc, Printf.sprintf
                    "guide support is not contained in model support at site '%s'"
                    guide_site.name)))
    guide_sites

let check_trace_compat_sites ?(slots = []) ~model_sites ~guide_sites () =
  let pp_frame frame =
    String.concat "," (List.map string_of_int (Array.to_list frame))
  in
  let slot_names = List.map (fun (name, _, _) -> name) slots in
  if List.length slot_names <>
     List.length (List.sort_uniq String.compare slot_names) then
    raise (Trace_mismatch "duplicate slot site name");
  List.iter
    (fun name ->
      match List.find_opt (fun site ->
        site.Ast.Sites.name = name) model_sites with
      | None ->
          raise
            (Trace_mismatch
               (Printf.sprintf "slotted site '%s' not found in model" name))
      | Some _ -> ())
    slot_names;
  List.iter
    (fun (guide_site : Ast.Sites.site) ->
      if List.mem guide_site.name slot_names then
        raise
          (Trace_mismatch
             (Printf.sprintf "slotted site '%s' also appears in guide"
                guide_site.name));
      match
        List.find_opt
          (fun site -> site.Ast.Sites.name = guide_site.name)
          model_sites
      with
      | None ->
          raise
            (Trace_mismatch
               (Printf.sprintf "guide site '%s' not found in model"
                  guide_site.name))
      | Some model_site ->
          if model_site.frame <> guide_site.frame then
            raise
              (Trace_mismatch
                 (Printf.sprintf
                    "site '%s' frame mismatch: model=[%s] guide=[%s]"
                    guide_site.name
                    (pp_frame model_site.frame)
                    (pp_frame guide_site.frame))))
    guide_sites;
  List.iter
    (fun (model_site : Ast.Sites.site) ->
      if List.mem model_site.name slot_names then () else match
        List.find_opt
          (fun site -> site.Ast.Sites.name = model_site.name)
          guide_sites
      with
      | None ->
          raise
            (Trace_mismatch
               (Printf.sprintf "model site '%s' not found in guide"
                  model_site.name))
      | Some _ -> ())
    model_sites

let check_trace_compat ~model ~guide =
  check_trace_compat_sites
    ~model_sites:(Ast.Sites.collect_sites model)
    ~guide_sites:(Ast.Sites.collect_sites guide) ()
