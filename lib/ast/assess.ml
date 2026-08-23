(* assess: compute log-density of a trace under a probabilistic program.
   Given a trace (site name → value), computes:
   - The return value of the program (using traced values for Sample sites)
   - log p(trace) = Σ_site log p_site(value) + Σ Score contributions *)

open View

let scalar v =
  let t = Tensor.make [||] in
  Buf.set t.buf 0 v; t

let scalar_val (t : Tensor.t) = Buf.get t.buf 0

let sum_all (t : Tensor.t) =
  let n = Ndview.numel t.view in
  let data = Buf.create n in
  Tensor.read_view ~src:t.buf ~view:t.view ~dst:data;
  let total = ref 0.0 in
  for i = 0 to n - 1 do total := !total +. Buf.get data i done;
  !total

let frame_cell frame i (t : Tensor.t) =
  if t.view.Ndview.shape <> frame then t
  else begin
    let rank = Array.length frame in
    let index = Array.make rank 0 in
    let linear = ref i in
    for axis = rank - 1 downto 0 do
      index.(axis) <- !linear mod frame.(axis);
      linear := !linear / frame.(axis)
    done;
    scalar (Buf.get t.buf (Ndview.index_of t.view index))
  end

exception Support_error of Types.loc * string

let check_support ~(env : Eval.env) loc name dist (x : Tensor.t) =
  (* Here the weights can simply be evaluated, so learned weights resolve as
     easily as constant ones. *)
  let categorical_size weights =
    let value = Eval.eval env weights in
    let shape = value.view.Ndview.shape in
    if Array.length shape = 0 then None
    else Some shape.(Array.length shape - 1)
  in
  let support = Sites.dist_support ~categorical_size dist in
  let n = Ndview.numel x.view in
  for i = 0 to n - 1 do
    if not (Sites.support_contains support (Buf.get x.buf i)) then
      raise
        (Support_error
           (loc, Printf.sprintf "value outside support at site '%s'" name))
  done

(* Compute log density of a value under a distribution.
   For D_pushforward, uses JVP to compute the Jacobian of inv. *)
let rec log_density ?(frame = [||]) ?(frame_index = 0)
    (dist : Types.dist) (x : Tensor.t)
    (env : Eval.env) : float =
  match dist with
  | D_uniform ->
    (* Endpoints are an equivalent density version and absorb CDF roundoff in
       pushforwards.  A directly traced Uniform is still checked against (0,1). *)
    let v = scalar_val x in
    if v >= 0.0 && v <= 1.0 then 0.0 else neg_infinity

  | D_categorical weights_expr ->
    let weights = Eval.eval env weights_expr in
    let shape = weights.view.Ndview.shape in
    if Array.length shape = 0 then
      failwith "categorical weights must have a category axis";
    let n = shape.(Array.length shape - 1) in
    let per_cell = shape = Array.append frame [|n|] in
    if not (Array.length shape = 1 || per_cell) then
      failwith "categorical weights must have shape [C] or frame + [C]";
    let frame_coords = Array.make (Array.length frame) 0 in
    let rest = ref frame_index in
    for axis = Array.length frame - 1 downto 0 do
      frame_coords.(axis) <- !rest mod frame.(axis);
      rest := !rest / frame.(axis)
    done;
    let weight category =
      let index = if per_cell then Array.append frame_coords [|category|]
        else [|category|] in
      Buf.get weights.buf (Ndview.index_of weights.view index) in
    let k = int_of_float (scalar_val x) in
    if k < 0 || k >= n then neg_infinity
    else begin
      let sum = ref 0.0 in
      for i = 0 to n - 1 do
        sum := !sum +. weight i
      done;
      let wk = weight k in
      log (wk /. !sum)
    end

  | D_pushforward { inv_var; inv; base; _ } ->
    (* log p(x) = log p_base(inv(x)) + log |d(inv)/dx|
       For scalar→scalar: d(inv)/dx is computed by JVP with seed=1. *)
    let env' = (inv_var, x) :: env in
    let u = Eval.eval env' inv in
    (* Jacobian: JVP of inv at x with tangent=1 *)
    let seed = scalar 1.0 in
    let dual_env = List.map (fun (s, v) ->
      (s, (v, Simulate.scalar 0.0))) env in
    let dual_env' = (inv_var, (x, seed)) :: dual_env in
    let (_, dinv) = Jvp.jvp_eval dual_env' inv in
    let log_abs_jac = log (Float.abs (scalar_val dinv)) in
    let base_ld = log_density ~frame ~frame_index base u env in
    base_ld +. log_abs_jac

  | D_product (a, b) ->
    ignore (a, b, x);
    failwith "D_product log_density not yet implemented"

let assess (env : Eval.env) (e : Types.expr)
    (trace : Simulate.trace)
    : Types.value * Types.value =
  let log_density_acc = ref 0.0 in
  let rec go (env : Eval.env) (e : Types.expr) : Types.value =
    match e with
    | Const (_, v) -> v
    | Var (loc, s) ->
      (match List.assoc_opt s env with
       | Some v -> v
       | None -> raise (Eval.Eval_error (loc, "assess: unbound variable: " ^ s)))
    | Let (_, s, e1, e2) ->
      let v1 = go env e1 in
      go ((s, v1) :: env) e2
    | Scan _ -> Eval.eval env e
    | Prim (loc, p, args) ->
      let vs = List.map (go env) args in
      Eval.validate loc p vs;
      Eval.eval [] (Types.Prim (loc, p, List.map (fun v -> Types.Const (loc, v)) vs))
    | Rank (loc, _, _, _) ->
      raise (Eval.Eval_error (loc, "Rank node must be expanded before assess"))
    | Score (_, e) ->
      let v = go env e in
      log_density_acc := !log_density_acc +. sum_all v;
      scalar 0.0
    | Sample (loc, name, frame, dist) ->
      let v = List.assoc name trace in
      check_support ~env loc name dist v;
      if frame = [||] then begin
        let ld = log_density dist v env in
        log_density_acc := !log_density_acc +. ld
      end else begin
        (* Frame: element-wise log_density, summed (independent product measure) *)
        let n = Array.fold_left ( * ) 1 frame in
        for i = 0 to n - 1 do
          let xi = scalar (Buf.get v.buf i) in
          let cell_env = List.map (fun (name, value) ->
            (name, frame_cell frame i value)) env in
          let ld = log_density ~frame ~frame_index:i dist xi cell_env in
          log_density_acc := !log_density_acc +. ld
        done
      end;
      v
  in
  let result = go env e in
  (result, scalar !log_density_acc)
