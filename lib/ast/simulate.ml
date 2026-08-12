(* simulate: reference interpreter for probabilistic programs.
   Draws samples from distributions using Threefry PRNG,
   accumulates Score contributions as log weight. *)

open View

type trace = (string * Tensor.t) list

let scalar v =
  let t = Tensor.make [||] in
  Buf.set t.buf 0 v; t

let scalar_val (t : Tensor.t) = Buf.get t.buf 0

(* Assign consecutive site IDs based on traversal order *)
let assign_site_ids (e : Types.expr) : (string, int) Hashtbl.t =
  let tbl = Hashtbl.create 8 in
  let next = ref 0 in
  let rec walk = function
    | Types.Sample (_, name, _, dist) ->
      Hashtbl.replace tbl name !next;
      incr next;
      walk_dist dist
    | Score (_, e) -> walk e
    | Const _ | Var _ -> ()
    | Prim (_, _, args) -> List.iter walk args
    | Let (_, _, e, body) -> walk e; walk body
    | Rank (_, _, _, args) -> List.iter walk args
  and walk_dist = function
    | Types.D_uniform -> ()
    | D_categorical e -> walk e
    | D_pushforward { fwd; inv; base; _ } -> walk fwd; walk inv; walk_dist base
    | D_product (a, b) -> walk_dist a; walk_dist b
  in
  walk e; tbl

(* Sample from a distribution given PRNG key, site_id, and component path.
   component encodes D_product tree position: root=1, left=2k, right=2k+1. *)
let rec sample_dist ~key ~site_id ~component ~frame (dist : Types.dist)
    (env : Eval.env) : Tensor.t =
  match dist with
  | D_uniform ->
    if frame = [||] then begin
      let ctr = Prng.Threefry.make_ctr ~site_id ~component ~frame_index:0 in
      let (r0, _) = Prng.Threefry.threefry2x64 ~key ~ctr in
      scalar (Prng.Threefry.to_open_unit r0)
    end else begin
      let n = Array.fold_left ( * ) 1 frame in
      let t = Tensor.make frame in
      for i = 0 to n - 1 do
        let ctr = Prng.Threefry.make_ctr ~site_id ~component ~frame_index:i in
        let (r0, _) = Prng.Threefry.threefry2x64 ~key ~ctr in
        Buf.set t.buf i (Prng.Threefry.to_open_unit r0)
      done;
      t
    end

  | D_categorical weights_expr ->
    let weights = Eval.eval env weights_expr in
    let ws = weights.view.Ndview.shape in
    let n = ws.(0) in
    (* Draw a uniform random number *)
    let ctr = Prng.Threefry.make_ctr ~site_id ~component ~frame_index:0 in
    let (r0, _) = Prng.Threefry.threefry2x64 ~key ~ctr in
    let u = Prng.Threefry.to_open_unit r0 in
    (* Compute CDF and invert *)
    let sum = ref 0.0 in
    for i = 0 to n - 1 do
      sum := !sum +. Buf.get weights.buf (Ndview.index_of weights.view [|i|])
    done;
    let target = u *. !sum in
    let acc = ref 0.0 in
    let result = ref (n - 1) in
    let found = ref false in
    for i = 0 to n - 1 do
      if not !found then begin
        acc := !acc +. Buf.get weights.buf (Ndview.index_of weights.view [|i|]);
        if !acc >= target then begin
          result := i;
          found := true
        end
      end
    done;
    scalar (float_of_int !result)

  | D_pushforward { fwd_var; fwd; base; _ } ->
    let u = sample_dist ~key ~site_id ~component ~frame base env in
    if frame = [||] then begin
      let env' = (fwd_var, u) :: env in
      Eval.eval env' fwd
    end else begin
      (* Batch evaluation: broadcast all scalar env values and Const nodes
         to frame shape, then eval once — G elements processed in one call. *)
      let g = frame.(0) in
      let broadcast_to_frame (t : Tensor.t) : Tensor.t =
        if t.view.Ndview.shape = [||] then
          Tensor.broadcast t ~axis:0 ~size:g
        else t in
      let env' = (fwd_var, u) ::
        List.map (fun (s, v) -> (s, broadcast_to_frame v)) env in
      let rec lift_consts = function
        | Types.Const (l, v) -> Types.Const (l, broadcast_to_frame v)
        | Var _ as e -> e
        | Prim (l, p, args) -> Prim (l, p, List.map lift_consts args)
        | Let (l, s, e1, e2) -> Let (l, s, lift_consts e1, lift_consts e2)
        | Rank _ | Sample _ | Score _ ->
          failwith "batch fwd: non-elementwise construct" in
      Eval.eval env' (lift_consts fwd)
    end

  | D_product (a, b) ->
    let va = sample_dist ~key ~site_id ~component:(component * 2) ~frame a env in
    let vb = sample_dist ~key ~site_id ~component:(component * 2 + 1) ~frame b env in
    (* Stack along a new first axis *)
    ignore (va, vb);
    failwith "D_product sampling not yet implemented"

let simulate ~run_key (env : Eval.env) (e : Types.expr)
    : Types.value * trace * Types.value =
  let site_ids = assign_site_ids e in
  let key = Prng.Threefry.make_key ~run_key ~namespace:Prng.Threefry.ns_model in
  let trace = ref [] in
  let log_weight = ref 0.0 in
  let rec go (env : Eval.env) (e : Types.expr) : Types.value =
    match e with
    | Const (_, v) -> v
    | Var (loc, s) ->
      (match List.assoc_opt s env with
       | Some v -> v
       | None -> raise (Eval.Eval_error (loc, "simulate: unbound variable: " ^ s)))
    | Let (_, s, e1, e2) ->
      let v1 = go env e1 in
      go ((s, v1) :: env) e2
    | Prim (loc, p, args) ->
      let vs = List.map (go env) args in
      Eval.validate loc p vs;
      Eval.eval [] (Types.Prim (loc, p, List.map (fun v -> Types.Const (loc, v)) vs))
    | Rank (loc, _, _, _) ->
      raise (Eval.Eval_error (loc, "Rank node must be expanded before simulate"))
    | Score (_, e) ->
      let v = go env e in
      let s = scalar_val v in
      log_weight := !log_weight +. s;
      scalar 0.0
    | Sample (_, name, frame, dist) ->
      let site_id = Hashtbl.find site_ids name in
      let v = sample_dist ~key ~site_id ~component:1 ~frame dist env in
      trace := (name, v) :: !trace;
      v
  in
  let result = go env e in
  (result, List.rev !trace, scalar !log_weight)
