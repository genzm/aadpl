module Expand_rank = Expand_rank
module Desugar = Desugar
module Forward = Forward
module Unzip = Unzip
module Transpose = Transpose
module Reparam = Reparam
module Assess_expr = Assess_expr
module Special = Special
module Quadrature = Quadrature
module Importance = Importance
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
    | Scan (_, scan, continuation) ->
        List.iter (fun (_, init, next) -> walk init; walk next) scan.carries;
        List.iter (fun (_, input) -> walk input) scan.inputs;
        walk continuation
    | Rank (_, _, _, args) -> List.iter walk args
  in
  walk e

type grad_program = {
  loss : expr;
  grads : (string * expr) list;
  (* Shared-evaluation form: primal computed once *)
  primal_bindings : binding list;
  loss_body : expr;
  grad_bindings : binding list;
  grad_bodies : (string * expr) list; (* (param_name, grad_body_expr) *)
}

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
      (fun (acc, senv) binding -> match binding with
      | Scan_binding (loc, scan) ->
        let output_name, _, _ = List.hd scan.carries in
        let expanded = Expand_rank.expand ~senv
          (Scan (loc, scan, var output_name)) in
        let scan = match expanded with
          | Scan (_, scan, _) -> scan
          | _ -> assert false in
        let output_shapes = List.map (fun (name, init, _) ->
          let shape = Expand_rank.infer_shape senv init in
          name, if scan.collect then Array.append [|scan.steps|] shape else shape)
          scan.carries in
        (acc @ [Scan_binding (loc, scan)], output_shapes @ senv)
      | Let_binding (name, rhs) ->
        let rhs = Expand_rank.expand ~senv rhs in
        let shape = Expand_rank.infer_shape senv rhs in
        (acc @ [ Let_binding (name, rhs) ], (name, shape) :: senv))
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
    | Scan (loc, scan, continuation) ->
        let carries = List.map (fun (name, init, next) ->
          name, subst_data_tangents init, subst_data_tangents next) scan.carries in
        let inputs = List.map (fun (name, input) ->
          name, subst_data_tangents input) scan.inputs in
        Scan (loc, {scan with carries; inputs}, subst_data_tangents continuation)
    | Rank _ ->
        failwith "grad: Rank remained after expand while zeroing data tangents"
    | Sample _ | Score _ -> e
  in
  let bs =
    if data_shapes = [] then bs
    else List.map (function
      | Let_binding (n, e) -> Let_binding (n, subst_data_tangents e)
      | Scan_binding (loc, scan) ->
        let expression = subst_data_tangents
          (Scan (loc, scan, var (let name, _, _ = List.hd scan.carries in name))) in
        match expression with
        | Scan (_, scan, _) -> Scan_binding (loc, scan)
        | _ -> assert false) bs
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
  let grad_bindings = Let_binding ("%ct", const ct_tensor)
    :: tr.Transpose.grad_bindings in
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
