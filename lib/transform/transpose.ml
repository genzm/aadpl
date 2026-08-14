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
  grad_bindings : binding list;
  grad_map      : (string * expr) list;  (* seed → gradient expr *)
}

let rec transpose_many
    ~(primal_bindings : binding list)
    ~(tangent_bindings : binding list)
    ~(outputs : (expr * expr) list)
    ~(seeds : string list)
    ~(input_shapes : (string * int array) list)
    : transposed =
  let gensym prefix = Forward.gensym ("ct." ^ prefix) in
  (* Build shape environment *)
  let senv = ref input_shapes in
  let add_binding_shapes senv = function
    | Let_binding (name, rhs) ->
      let sh = infer_shape senv rhs in
      (name, sh) :: senv
    | Scan_binding (_, scan) ->
      List.fold_left (fun shapes (name, init, _) ->
        let shape = infer_shape senv init in
        let shape = if scan.collect then Array.append [|scan.steps|] shape
          else shape in
        (name, shape) :: shapes) senv scan.carries
  in
  List.iter (fun binding -> senv := add_binding_shapes !senv binding)
    primal_bindings;
  let senv_full = ref !senv in
  List.iter (fun binding ->
    senv_full := add_binding_shapes !senv_full binding) tangent_bindings;
  let shape_of_var v =
    match List.assoc_opt v !senv_full with
    | Some s -> s
    | None -> failwith ("transpose: unknown shape for " ^ v)
  in
  let shape_of_expr e = infer_shape !senv_full e in
  (* tangent dependency *)
  let tangent_deps = ref (Unzip.SS.of_list seeds) in
  List.iter (fun binding ->
    List.iter (fun name -> tangent_deps := Unzip.SS.add name !tangent_deps)
      (Unzip.binding_names binding)) tangent_bindings;
  let is_dep e =
    not (Unzip.SS.is_empty (Unzip.SS.inter (Unzip.free_vars e) !tangent_deps))
  in
  (* Cotangent map and generated bindings *)
  let cotan : cotan_map = Hashtbl.create 32 in
  let grad_bs = ref [] in
  let add_grad binding = grad_bs := binding :: !grad_bs in
  let add_grad_binding name rhs =
    add_grad (Let_binding (name, rhs))
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
  let zero shape = const (View.Tensor.make shape) in
  let select_last steps expression =
    prim (Sum_axis 0)
      [prim (Gather (0, [|steps - 1|])) [expression]] in
  let reverse_axis0 shape expression =
    let ranges = Array.mapi (fun axis size ->
      if axis = 0 then (size - 1, -1, -1) else (0, size, 1)) shape in
    prim (Apply_view [Vslice ranges]) [expression] in
  let rec peel_lets bindings = function
    | Let (_, name, rhs, body) -> peel_lets (Let_binding (name, rhs) :: bindings) body
    | body -> List.rev bindings, body in
  let rec adjoint (e : expr) (ct : expr) : unit =
    match e with
    | Var (_, s) ->
      add_cotan cotan s ct
    | Const _ ->
      (* const in tangent part = zero, no contribution *)
      ()
    | Prim (_, p, args) ->
      adjoint_prim p args ct
    | Let _ | Rank _ | Scan _ ->
      failwith "transpose: unexpected Let/Rank/Scan in tangent expression"
    | Sample _ -> failwith "transpose: Sample not supported"
    | Score _ -> failwith "transpose: Score not supported"

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

    | Mask, [a; mask] ->
      if is_dep a && not (is_dep mask) then
        adjoint a (prim Mask [ct; mask])
      else
        failwith "transpose: Mask mask is tangent-dependent"

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

  and adjoint_scan loc (scan : scan) : unit =
    let carry_shapes = List.map (fun (name, init, _) ->
      name, infer_shape !senv_full init) scan.carries in
    let input_shapes = List.map (fun (name, sequence) ->
      let shape = infer_shape !senv_full sequence in
      name, Array.sub shape 1 (Array.length shape - 1)) scan.inputs in
    let local_bindings, local_outputs =
      List.fold_left (fun (bindings, outputs) (name, _, next) ->
        let next_bindings, output = peel_lets [] next in
        bindings @ next_bindings, (name, output) :: outputs)
        ([], []) scan.carries in
    let local_outputs = List.rev local_outputs in
    let local_names =
      List.fold_left (fun names (name, _) -> Unzip.SS.add name names)
        Unzip.SS.empty (carry_shapes @ input_shapes)
      |> fun names -> List.fold_left (fun names binding ->
        List.fold_left (fun names name -> Unzip.SS.add name names) names
          (Unzip.binding_names binding)) names local_bindings in
    let body_variables = List.fold_left (fun variables (_, output) ->
      Unzip.SS.union variables (Unzip.free_vars output)) Unzip.SS.empty
        local_outputs
      |> fun variables -> List.fold_left (fun variables binding ->
        Unzip.SS.union variables (Unzip.binding_free_vars binding))
          variables local_bindings in
    let external_seeds = Unzip.SS.elements
      (Unzip.SS.diff (Unzip.SS.inter body_variables !tangent_deps) local_names) in
    let tangent_inputs = List.filter (fun (_, sequence) -> is_dep sequence)
      scan.inputs in
    let bar_names = List.map (fun (name, _) -> name, gensym "scan.bar.")
      carry_shapes in
    let output_inputs = if scan.collect then
      List.map (fun (name, shape) -> name, gensym "scan.out.", shape)
        carry_shapes else [] in
    let output_cotangent name =
      let bar = var (List.assoc name bar_names) in
      if scan.collect then prim Add [bar; var (let _, input, _ =
        List.find (fun (candidate, _, _) -> candidate = name) output_inputs
        in input)] else bar in
    let local_shape_env =
      carry_shapes @ input_shapes
      @ List.map (fun (name, bar) -> bar, List.assoc name carry_shapes) bar_names
      @ List.map (fun (_, input, shape) -> input, shape) output_inputs
      @ !senv_full in
    let local_seeds =
      List.map fst carry_shapes @ List.map fst tangent_inputs @ external_seeds in
    let local = transpose_many ~primal_bindings:[]
      ~tangent_bindings:local_bindings
      ~outputs:(List.map (fun (name, output) ->
        output, output_cotangent name) local_outputs)
      ~seeds:local_seeds ~input_shapes:local_shape_env in
    let local_gradient seed = List.assoc seed local.grad_map in
    let shared_names = List.map (fun seed -> seed, gensym "scan.shared.")
      external_seeds in
    let input_grad_names = List.map (fun (name, _) ->
      name, gensym "scan.input.") tangent_inputs in
    let wrap_local expression = Forward.wrap_bindings local.grad_bindings expression in
    let adjoint_carries =
      List.map (fun (name, shape) ->
        let downstream = match Hashtbl.find_opt cotan name with
          | Some cotangents -> sum_cotans cotangents
          | None -> zero (if scan.collect then Array.append [|scan.steps|] shape
              else shape) in
        let initial = if scan.collect then zero shape else downstream in
        List.assoc name bar_names, initial, wrap_local (local_gradient name))
        carry_shapes
      @ List.map (fun (seed, name) ->
        let shape = shape_of_var seed in
        name, zero shape,
          prim Add [var name; wrap_local (local_gradient seed)]) shared_names
      @ List.map (fun (seed, name) ->
        let shape = List.assoc seed input_shapes in
        name, zero shape, wrap_local (local_gradient seed)) input_grad_names in
    let output_cotangent_inputs = List.map (fun (name, input, shape) ->
      let full_shape = Array.append [|scan.steps|] shape in
      let downstream = match Hashtbl.find_opt cotan name with
        | Some cotangents -> sum_cotans cotangents
        | None -> zero full_shape in
      input, (if scan.reverse then reverse_axis0 full_shape downstream
        else downstream)) output_inputs in
    let adjoint_scan = {
      steps = scan.steps; carries = adjoint_carries;
      inputs = List.filter (fun (_, sequence) -> not (is_dep sequence)) scan.inputs
        @ output_cotangent_inputs;
      collect = true; reverse = not scan.reverse;
    } in
    add_grad (Scan_binding (loc, adjoint_scan));
    List.iter (fun (name, init, _) ->
      let trajectory = var (List.assoc name bar_names) in
      adjoint init (select_last scan.steps trajectory)) scan.carries;
    List.iter (fun (name, sequence) ->
      match List.assoc_opt name input_grad_names with
      | None -> ()
      | Some gradient_name ->
        let gradient = var gradient_name in
        let full_shape = infer_shape !senv_full sequence in
        let gradient = if scan.reverse then gradient
          else reverse_axis0 full_shape gradient in
        adjoint sequence gradient) scan.inputs;
    List.iter (fun (seed, name) ->
      adjoint (var seed) (select_last scan.steps (var name))) shared_names
  in
  (* Inject all output cotangents, then reverse the shared binding stream once. *)
  List.iter (fun (output, output_cotangent) ->
    adjoint output output_cotangent) outputs;
  let rev_bindings = List.rev tangent_bindings in
  List.iter (function
    | Scan_binding (loc, scan) ->
      if List.exists (fun (name, _, _) -> Hashtbl.mem cotan name) scan.carries
      then adjoint_scan loc scan
    | Let_binding (name, rhs) ->
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

let transpose ~primal_bindings ~tangent_bindings ~tangent_out ~seeds
    ~input_shapes ~cotangent_var =
  transpose_many ~primal_bindings ~tangent_bindings
    ~outputs:[tangent_out, var cotangent_var] ~seeds ~input_shapes
