module Expand_rank = Expand_rank
module Desugar = Desugar
module Forward = Forward
module Unzip = Unzip
module Transpose = Transpose

open Ast.Types

type grad_program = {
  loss : expr;
  grads : (string * expr) list;
}

let grad ~(param_shapes : (string * int array) list) (e : expr) : grad_program =
  let e = Expand_rank.expand ~senv:param_shapes e in
  let e = Desugar.fuse_views e in
  Forward.reset_gensym ();
  let (bs, primal_out, tangent_out) = Forward.forward e in
  let seeds = List.map (fun (s, _) -> Forward.tangent_name s) param_shapes in
  let uz = Unzip.unzip (bs, primal_out, tangent_out) ~seeds in
  let seed_shapes =
    List.map (fun (s, sh) -> (Forward.tangent_name s, sh)) param_shapes in
  let input_shapes = param_shapes @ seed_shapes in
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
  let loss_shape = Expand_rank.infer_shape param_shapes loss in
  let ct_tensor = View.Tensor.make loss_shape in
  let ct_numel = Array.fold_left ( * ) 1 loss_shape in
  for i = 0 to ct_numel - 1 do View.Buf.set ct_tensor.buf i 1.0 done;
  (* Gradient exprs: primal bindings + %ct binding + grad bindings + grad expr *)
  let seed_to_param =
    List.map (fun (s, _) -> (Forward.tangent_name s, s)) param_shapes in
  let grads = List.map (fun (seed, gexpr) ->
    let param = List.assoc seed seed_to_param in
    let body =
      let_ "%ct" (const ct_tensor)
        (Forward.wrap_bindings tr.Transpose.grad_bindings gexpr) in
    (param, Forward.wrap_bindings uz.primal_bindings body)
  ) tr.Transpose.grad_map in
  { loss; grads }
