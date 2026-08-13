module Expand_rank = Expand_rank
module Desugar = Desugar
module Forward = Forward
module Unzip = Unzip
module Transpose = Transpose
module Reparam = Reparam
module Assess_expr = Assess_expr
open Ast.Types

(* --- discrete sample guard --- *)

exception Grad_error of loc * string

let check_no_samples (e : expr) =
  let rec walk = function
    | Sample (l, name, _, dist) -> (
        match dist with
        | D_categorical _ ->
            raise
              (Grad_error
                 ( l,
                   Printf.sprintf "discrete sample '%s' in differentiable path"
                     name ))
        | _ ->
            raise
              (Grad_error
                 ( l,
                   Printf.sprintf
                     "sample '%s' must be handled by reparam/assess_expr \
                      before grad"
                     name )))
    | Score (l, _) ->
        raise
          (Grad_error (l, "Score must be handled by assess_expr before grad"))
    | Const _ | Var _ -> ()
    | Prim (_, _, args) -> List.iter walk args
    | Let (_, _, e1, e2) ->
        walk e1;
        walk e2
    | Rank (_, _, _, args) -> List.iter walk args
  in
  walk e

type grad_program = {
  loss : expr;
  grads : (string * expr) list;
  (* Shared-evaluation form: primal computed once *)
  primal_bindings : (string * expr) list;
  loss_body : expr;
  grad_bindings : (string * expr) list;
  grad_bodies : (string * expr) list; (* (param_name, grad_body_expr) *)
}

type elbo_program = {
  elbo : expr;
  sites : Ast.Sites.site list;
  noise : (string * int array) list;
}

let noise_env (program : elbo_program) ~run_key =
  Ast.Sites.draw_noise ~namespace:Prng.Threefry.ns_guide ~run_key program.sites

let build_elbo ~model ~guide ~(env_shapes : (string * int array) list) :
    elbo_program =
  check_sites model;
  Reparam.check_guide guide;
  Reparam.check_trace_compat ~model ~guide;
  let sites = Ast.Sites.collect_sites guide in
  let noise =
    List.map
      (fun (site : Ast.Sites.site) -> (Ast.Sites.noise_name site, site.frame))
      sites
  in
  let slots =
    List.map
      (fun (site : Ast.Sites.site) ->
        (site.name, var (Ast.Sites.trace_name site)))
      sites
  in
  let guide_r = Reparam.reparam ~sites guide in
  let bindings, _ = Reparam.elim_samples ~sites guide_r in
  let model = Expand_rank.expand ~senv:env_shapes model in
  let guide = Expand_rank.expand ~senv:env_shapes guide in
  let trace_shapes =
    List.map
      (fun (site : Ast.Sites.site) -> (Ast.Sites.trace_name site, site.frame))
      sites
  in
  let assess_shapes = trace_shapes @ env_shapes in
  let model_ld =
    Assess_expr.assess_expr ~ns:"m." ~env_shapes:assess_shapes model slots
  in
  let guide_ld =
    Assess_expr.assess_expr ~ns:"g." ~env_shapes:assess_shapes guide slots
  in
  let body = prim Sub [ model_ld; guide_ld ] in
  let elbo = Forward.wrap_bindings bindings body in
  let elbo = Expand_rank.expand ~senv:(noise @ env_shapes) elbo in
  let elbo = Desugar.fuse_views elbo in
  { elbo; sites; noise }

let grad ~(param_shapes : (string * int array) list)
    ?(data_shapes : (string * int array) list = []) (e : expr) : grad_program =
  check_no_samples e;
  let all_shapes = param_shapes @ data_shapes in
  let e = Expand_rank.expand ~senv:all_shapes e in
  let e = Desugar.fuse_views e in
  Forward.reset_gensym ();
  let bs, primal_out, tangent_out = Forward.forward e in
  (* Some derivative rules contain scalar constants.  Keep their elementwise
     lifting explicit as Rank nodes, then expand the generated forward IR with
     both primal and tangent input shapes before unzip/transpose. *)
  let forward_input_shapes =
    all_shapes
    @ List.map (fun (s, sh) -> (Forward.tangent_name s, sh)) all_shapes
  in
  let bs, forward_shapes =
    List.fold_left
      (fun (acc, senv) (name, rhs) ->
        let rhs = Expand_rank.expand ~senv rhs in
        let shape = Expand_rank.infer_shape senv rhs in
        (acc @ [ (name, rhs) ], (name, shape) :: senv))
      ([], forward_input_shapes) bs
  in
  let primal_out = Expand_rank.expand ~senv:forward_shapes primal_out in
  let tangent_out = Expand_rank.expand ~senv:forward_shapes tangent_out in
  (* Zero data tangent variables: we differentiate wrt params, not data.
     This prevents data tangent refs from leaking into primal bindings. *)
  let data_tangent_zeros =
    List.map
      (fun (s, sh) -> (Forward.tangent_name s, const (View.Tensor.make sh)))
      data_shapes
  in
  let rec subst_data_tangents e =
    match e with
    | Var (_, s) -> (
        match List.assoc_opt s data_tangent_zeros with Some z -> z | None -> e)
    | Const _ -> e
    | Prim (l, p, args) -> Prim (l, p, List.map subst_data_tangents args)
    | Let (l, s, e1, e2) ->
        Let (l, s, subst_data_tangents e1, subst_data_tangents e2)
    | Rank _ ->
        failwith "grad: Rank remained after expand while zeroing data tangents"
    | Sample _ | Score _ -> e
  in
  let bs =
    if data_shapes = [] then bs
    else List.map (fun (n, e) -> (n, subst_data_tangents e)) bs
  in
  let tangent_out =
    if data_shapes = [] then tangent_out else subst_data_tangents tangent_out
  in
  let seeds = List.map (fun (s, _) -> Forward.tangent_name s) param_shapes in
  let uz = Unzip.unzip (bs, primal_out, tangent_out) ~seeds in
  let seed_shapes =
    List.map (fun (s, sh) -> (Forward.tangent_name s, sh)) param_shapes
  in
  let data_tangent_shapes =
    List.map (fun (s, sh) -> (Forward.tangent_name s, sh)) data_shapes
  in
  let input_shapes =
    param_shapes @ seed_shapes @ data_shapes @ data_tangent_shapes
  in
  let tr =
    Transpose.transpose ~primal_bindings:uz.primal_bindings
      ~tangent_bindings:uz.tangent_bindings ~tangent_out:uz.tangent_out ~seeds
      ~input_shapes ~cotangent_var:"%ct"
  in
  (* Loss: wrap primal bindings around primal output *)
  let loss = Forward.wrap_bindings uz.primal_bindings uz.primal_out in
  (* Scalar cotangent = 1.0 *)
  let loss_shape = Expand_rank.infer_shape input_shapes loss in
  let ct_tensor = View.Tensor.make loss_shape in
  let ct_numel = Array.fold_left ( * ) 1 loss_shape in
  for i = 0 to ct_numel - 1 do
    View.Buf.set ct_tensor.buf i 1.0
  done;
  (* Gradient exprs: primal bindings + %ct binding + grad bindings + grad expr *)
  let seed_to_param =
    List.map (fun (s, _) -> (Forward.tangent_name s, s)) param_shapes
  in
  let grad_bodies_and_wrapped =
    List.map
      (fun (seed, gexpr) ->
        let param = List.assoc seed seed_to_param in
        let grad_body =
          let_ "%ct" (const ct_tensor)
            (Forward.wrap_bindings tr.Transpose.grad_bindings gexpr)
        in
        let wrapped = Forward.wrap_bindings uz.primal_bindings grad_body in
        (param, grad_body, wrapped))
      tr.Transpose.grad_map
  in
  let grads = List.map (fun (p, _, w) -> (p, w)) grad_bodies_and_wrapped in
  let grad_bindings = ("%ct", const ct_tensor) :: tr.Transpose.grad_bindings in
  let grad_bodies = List.map (fun (seed, body) ->
    (List.assoc seed seed_to_param, body)) tr.Transpose.grad_map in
  {
    loss;
    grads;
    primal_bindings = uz.primal_bindings;
    loss_body = uz.primal_out;
    grad_bindings;
    grad_bodies;
  }
