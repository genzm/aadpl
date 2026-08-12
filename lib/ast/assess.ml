(* assess: compute log-density of a trace under a probabilistic program.
   Given a trace (site name → value), computes:
   - The return value of the program (using traced values for Sample sites)
   - log p(trace) = Σ_site log p_site(value) + Σ Score contributions *)

open View

let scalar v =
  let t = Tensor.make [||] in
  Buf.set t.buf 0 v; t

let scalar_val (t : Tensor.t) = Buf.get t.buf 0

(* Compute log density of a value under a distribution.
   For D_pushforward, uses JVP to compute the Jacobian of inv. *)
let rec log_density (dist : Types.dist) (x : Tensor.t)
    (env : Eval.env) : float =
  match dist with
  | D_uniform ->
    (* Uniform on (0,1): density = 1, log density = 0 *)
    let v = scalar_val x in
    if v > 0.0 && v < 1.0 then 0.0 else neg_infinity

  | D_categorical weights_expr ->
    let weights = Eval.eval env weights_expr in
    let n = weights.view.Ndview.shape.(0) in
    let k = int_of_float (scalar_val x) in
    if k < 0 || k >= n then neg_infinity
    else begin
      let sum = ref 0.0 in
      for i = 0 to n - 1 do
        sum := !sum +. Buf.get weights.buf (Ndview.index_of weights.view [|i|])
      done;
      let wk = Buf.get weights.buf (Ndview.index_of weights.view [|k|]) in
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
    let base_ld = log_density base u env in
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
    | Prim (loc, p, args) ->
      let vs = List.map (go env) args in
      Eval.validate loc p vs;
      Eval.eval [] (Types.Prim (loc, p, List.map (fun v -> Types.Const (loc, v)) vs))
    | Rank (loc, _, _, _) ->
      raise (Eval.Eval_error (loc, "Rank node must be expanded before assess"))
    | Score (_, e) ->
      let v = go env e in
      log_density_acc := !log_density_acc +. scalar_val v;
      scalar 0.0
    | Sample (_, name, frame, dist) ->
      let v = List.assoc name trace in
      if frame = [||] then begin
        let ld = log_density dist v env in
        log_density_acc := !log_density_acc +. ld
      end else begin
        (* Frame: element-wise log_density, summed (independent product measure) *)
        let n = Array.fold_left ( * ) 1 frame in
        for i = 0 to n - 1 do
          let xi = scalar (Buf.get v.buf i) in
          let ld = log_density dist xi env in
          log_density_acc := !log_density_acc +. ld
        done
      end;
      v
  in
  let result = go env e in
  (result, scalar !log_density_acc)
