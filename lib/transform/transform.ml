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
    | Sample (l, name, _, dist) ->
      (match dist with
       | D_categorical _ ->
         raise (Grad_error (l,
           Printf.sprintf "discrete sample '%s' in differentiable path" name))
       | _ ->
         raise (Grad_error (l,
           Printf.sprintf "sample '%s' must be handled by reparam/assess_expr before grad" name)))
    | Score (l, _) ->
      raise (Grad_error (l,
        "Score must be handled by assess_expr before grad"))
    | Const _ | Var _ -> ()
    | Prim (_, _, args) -> List.iter walk args
    | Let (_, _, e1, e2) -> walk e1; walk e2
    | Rank (_, _, _, args) -> List.iter walk args
  in
  walk e

type grad_program = {
  loss : expr;
  grads : (string * expr) list;
  (* Shared-evaluation form: primal computed once *)
  primal_bindings : (string * expr) list;
  loss_body : expr;
  grad_bodies : (string * expr) list;  (* (param_name, grad_body_expr) *)
}

let grad ~(param_shapes : (string * int array) list)
    ?(data_shapes : (string * int array) list = []) (e : expr) : grad_program =
  check_no_samples e;
  let all_shapes = param_shapes @ data_shapes in
  let e = Expand_rank.expand ~senv:all_shapes e in
  let e = Desugar.fuse_views e in
  Forward.reset_gensym ();
  let (bs, primal_out, tangent_out) = Forward.forward e in
  let seeds = List.map (fun (s, _) -> Forward.tangent_name s) param_shapes in
  let uz = Unzip.unzip (bs, primal_out, tangent_out) ~seeds in
  let seed_shapes =
    List.map (fun (s, sh) -> (Forward.tangent_name s, sh)) param_shapes in
  let data_tangent_shapes =
    List.map (fun (s, sh) -> (Forward.tangent_name s, sh)) data_shapes in
  let input_shapes = param_shapes @ seed_shapes @ data_shapes @ data_tangent_shapes in
  let tr = Transpose.transpose
    ~primal_bindings:uz.primal_bindings
    ~tangent_bindings:uz.tangent_bindings
    ~tangent_out:uz.tangent_out
    ~seeds
    ~input_shapes
    ~cotangent_var:"%ct" in
  (* Loss: wrap primal bindings around primal output *)
  let loss = Forward.wrap_bindings uz.primal_bindings uz.primal_out in
  (* Scalar cotangent = 1.0 *)
  let loss_shape = Expand_rank.infer_shape all_shapes loss in
  let ct_tensor = View.Tensor.make loss_shape in
  let ct_numel = Array.fold_left ( * ) 1 loss_shape in
  for i = 0 to ct_numel - 1 do View.Buf.set ct_tensor.buf i 1.0 done;
  (* Gradient exprs: primal bindings + %ct binding + grad bindings + grad expr *)
  let seed_to_param =
    List.map (fun (s, _) -> (Forward.tangent_name s, s)) param_shapes in
  let grad_bodies_and_wrapped = List.map (fun (seed, gexpr) ->
    let param = List.assoc seed seed_to_param in
    let grad_body =
      let_ "%ct" (const ct_tensor)
        (Forward.wrap_bindings tr.Transpose.grad_bindings gexpr) in
    let wrapped = Forward.wrap_bindings uz.primal_bindings grad_body in
    (param, grad_body, wrapped)
  ) tr.Transpose.grad_map in
  let grads = List.map (fun (p, _, w) -> (p, w)) grad_bodies_and_wrapped in
  let grad_bodies = List.map (fun (p, b, _) -> (p, b)) grad_bodies_and_wrapped in
  { loss; grads;
    primal_bindings = uz.primal_bindings;
    loss_body = uz.primal_out;
    grad_bodies }
