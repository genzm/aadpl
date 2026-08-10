(* Transpose: adjoint (reverse-mode) of the tangent-linear part.
   Input: unzipped tangent_bindings + tangent_out (all linear in tangent seeds).
   Output: for each seed, its cotangent (gradient) expression.

   Walk tangent_bindings in reverse. Maintain a cotangent map (var → expr).
   For each binding `let t = linear_expr`, distribute t's cotangent through
   the expression tree using adjoint rules.

   Fan-out: when a variable is used multiple times, cotangents are summed. *)

open Ast.Types

(* --- shape environment --- *)

type shape_env = (string * int array) list

let infer_shape = Expand_rank.infer_shape
let _output_shape = Expand_rank.output_shape

(* --- cotangent accumulation --- *)

type cotan_map = (string, expr list) Hashtbl.t

let add_cotan (m : cotan_map) (name : string) (ct : expr) =
  match Hashtbl.find_opt m name with
  | None -> Hashtbl.replace m name [ct]
  | Some cs -> Hashtbl.replace m name (ct :: cs)

let sum_cotans (cs : expr list) : expr =
  match cs with
  | [] -> failwith "sum_cotans: empty"
  | [c] -> c
  | c :: rest -> List.fold_left (fun acc e -> prim Add [acc; e]) c rest

(* --- transpose --- *)

type transposed = {
  grad_bindings : (string * expr) list;
  grad_map      : (string * expr) list;  (* seed → gradient expr *)
}

let transpose
    ~(primal_bindings : (string * expr) list)
    ~(tangent_bindings : (string * expr) list)
    ~(tangent_out : expr)
    ~(seeds : string list)
    ~(input_shapes : (string * int array) list)
    ~(cotangent_var : string)
    : transposed =
  let gensym_ctr = ref 0 in
  let gensym prefix =
    let n = !gensym_ctr in
    incr gensym_ctr;
    Printf.sprintf "%%ct_%s%d" prefix n
  in
  (* Build shape environment *)
  let senv = ref input_shapes in
  List.iter (fun (name, rhs) ->
    let sh = infer_shape !senv rhs in
    senv := (name, sh) :: !senv
  ) primal_bindings;
  let senv_full = ref !senv in
  List.iter (fun (name, rhs) ->
    let sh = infer_shape !senv_full rhs in
    senv_full := (name, sh) :: !senv_full
  ) tangent_bindings;
  let shape_of_var v =
    match List.assoc_opt v !senv_full with
    | Some s -> s
    | None -> failwith ("transpose: unknown shape for " ^ v)
  in
  let shape_of_expr e = infer_shape !senv_full e in
  (* tangent dependency *)
  let tangent_deps = ref (Unzip.SS.of_list seeds) in
  List.iter (fun (name, _) ->
    tangent_deps := Unzip.SS.add name !tangent_deps
  ) tangent_bindings;
  let is_dep e =
    not (Unzip.SS.is_empty (Unzip.SS.inter (Unzip.free_vars e) !tangent_deps))
  in
  (* Cotangent map and generated bindings *)
  let cotan : cotan_map = Hashtbl.create 32 in
  let grad_bs = ref [] in
  let add_grad_binding name rhs =
    grad_bs := (name, rhs) :: !grad_bs
  in
  let get_cotan (name : string) : expr =
    match Hashtbl.find_opt cotan name with
    | None -> failwith ("transpose: no cotangent for " ^ name)
    | Some cs ->
      let sum = sum_cotans cs in
      if List.length cs > 1 then begin
        let v = gensym "s" in
        add_grad_binding v sum;
        var v
      end else sum
  in
  (* Ensure cotangent expr is bound to a variable for reuse *)
  let ensure_ct_var (ct : expr) : expr =
    match ct with
    | Var _ | Const _ -> ct
    | _ ->
      let v = gensym "t" in
      add_grad_binding v ct;
      var v
  in
  (* Recursive adjoint: distribute cotangent ct through expression e.
     For each tangent-dependent Var leaf reached, accumulate cotangent. *)
  let rec adjoint (e : expr) (ct : expr) : unit =
    match e with
    | Var (_, s) ->
      add_cotan cotan s ct
    | Const _ ->
      (* const in tangent part = zero, no contribution *)
      ()
    | Prim (_, p, args) ->
      adjoint_prim p args ct
    | Let _ | Rank _ ->
      failwith "transpose: unexpected Let/Rank in tangent expression"

  and adjoint_prim (p : prim) (args : expr list) (ct : expr) : unit =
    match p, args with
    | Neg, [a] ->
      adjoint a (prim Neg [ct])

    | Add, [a; b] ->
      let ct = ensure_ct_var ct in
      if is_dep a then adjoint a ct;
      if is_dep b then adjoint b ct

    | Sub, [a; b] ->
      let ct = ensure_ct_var ct in
      if is_dep a then adjoint a ct;
      if is_dep b then adjoint b (prim Neg [ct])

    | Mul, [a; b] ->
      if is_dep a && not (is_dep b) then
        adjoint a (prim Mul [ct; b])
      else if not (is_dep a) && is_dep b then
        adjoint b (prim Mul [a; ct])
      else
        failwith "transpose: Mul both tangent-dependent"

    | Div, [a; b] ->
      (* a tangent-dep, b residual *)
      if is_dep a && not (is_dep b) then
        adjoint a (prim Div [ct; b])
      else
        failwith "transpose: Div denominator is tangent-dependent"

    | Sum_axis axis, [a] ->
      let a_shape = shape_of_expr a in
      let size = a_shape.(axis) in
      adjoint a (prim (Apply_view [Vbroadcast (axis, size)]) [ct])

    | Gather (axis, indices), [a] ->
      let orig_shape = shape_of_expr a in
      adjoint a (prim (Scatter_add (axis, indices, orig_shape)) [ct])

    | Select_axis axis, [a; idx] ->
      (* a is tangent-dep, idx is residual *)
      if is_dep a && not (is_dep idx) then begin
        let a_shape = shape_of_expr a in
        let target_axis_size = a_shape.(axis) in
        adjoint a (prim (Scatter_select_add (axis, target_axis_size)) [ct; idx])
      end else
        failwith "transpose: Select_axis index is tangent-dependent"

    | Apply_view spec, [a] ->
      let a_shape = shape_of_expr a in
      adjoint a (prim (Adjoint_view (spec, a_shape)) [ct])

    | Adjoint_view (spec, _target_shape), [a] ->
      adjoint a (prim (Apply_view spec) [ct])

    | Matmul, [a; b] ->
      if is_dep a && not (is_dep b) then begin
        (* x̄ += ct @ bᵀ *)
        let b_shape = shape_of_expr b in
        let rb = Array.length b_shape in
        let b_perm = Array.init rb (fun i ->
          if i < rb - 2 then i
          else if i = rb - 2 then rb - 1
          else rb - 2) in
        adjoint a (prim Matmul [ct; prim (Apply_view [Vtranspose b_perm]) [b]])
      end
      else if not (is_dep a) && is_dep b then begin
        (* x̄ += aᵀ @ ct *)
        let a_shape = shape_of_expr a in
        let ra = Array.length a_shape in
        let a_perm = Array.init ra (fun i ->
          if i < ra - 2 then i
          else if i = ra - 2 then ra - 1
          else ra - 2) in
        adjoint b (prim Matmul [prim (Apply_view [Vtranspose a_perm]) [a]; ct])
      end
      else
        failwith "transpose: Matmul both tangent-dependent"

    | _ ->
      failwith (Format.asprintf "transpose: unhandled prim %a" pp_prim p)
  in
  (* Process: tangent_out gets initial cotangent, then reverse walk bindings *)
  let out_binding_name = gensym "out" in
  let effective_bindings =
    tangent_bindings @ [(out_binding_name, tangent_out)] in
  add_cotan cotan out_binding_name (var cotangent_var);
  let rev_bindings = List.rev effective_bindings in
  List.iter (fun (name, rhs) ->
    match Hashtbl.find_opt cotan name with
    | None -> ()  (* dead tangent: no cotangent contribution *)
    | Some _ ->
      let ct = get_cotan name in
      adjoint rhs ct
  ) rev_bindings;
  (* Collect seed gradients *)
  let grad_map = List.map (fun seed ->
    let ct = match Hashtbl.find_opt cotan seed with
      | Some cs -> sum_cotans cs
      | None ->
        let sh = shape_of_var seed in
        const (View.Tensor.make sh)
    in
    (seed, ct)
  ) seeds in
  { grad_bindings = List.rev !grad_bs;
    grad_map }
