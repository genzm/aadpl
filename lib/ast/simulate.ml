(* simulate: reference interpreter for probabilistic programs.
   Draws samples from distributions using Threefry PRNG,
   accumulates Score contributions as log weight. *)

open View

type trace = (string * Tensor.t) list

let scalar v =
  let t = Tensor.make [||] in
  Buf.set t.buf 0 v;
  t

let scalar_val (t : Tensor.t) = Buf.get t.buf 0

let sum_all (t : Tensor.t) =
  let n = Ndview.numel t.view in
  let data = Buf.create n in
  Tensor.read_view ~src:t.buf ~view:t.view ~dst:data;
  let total = ref 0.0 in
  for i = 0 to n - 1 do total := !total +. Buf.get data i done;
  !total

(* Sample from a distribution given PRNG key, site_id, and component path.
   component encodes D_product tree position: root=1, left=2k, right=2k+1. *)
let rec sample_dist ~key ~site_id ~component ~frame (dist : Types.dist)
    (env : Eval.env) : Tensor.t =
  match dist with
  | D_uniform ->
      if frame = [||] then begin
        let ctr = Prng.Threefry.make_ctr ~site_id ~component ~frame_index:0 in
        let r0, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
        scalar (Prng.Threefry.to_open_unit r0)
      end
      else begin
        let n = Array.fold_left ( * ) 1 frame in
        let t = Tensor.make frame in
        for i = 0 to n - 1 do
          let ctr = Prng.Threefry.make_ctr ~site_id ~component ~frame_index:i in
          let r0, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
          Buf.set t.buf i (Prng.Threefry.to_open_unit r0)
        done;
        t
      end
  | D_categorical weights_expr ->
      let weights = Eval.eval env weights_expr in
      let ws = weights.view.Ndview.shape in
      if Array.length ws = 0 then
        failwith "categorical weights must have a category axis";
      let categories = ws.(Array.length ws - 1) in
      if categories = 0 then failwith "categorical weights must be nonempty";
      let per_cell = ws = Array.append frame [|categories|] in
      if not (Array.length ws = 1 || per_cell) then
        failwith "categorical weights must have shape [C] or frame + [C]";
      let cells = Array.fold_left ( * ) 1 frame in
      let result = Tensor.make frame in
      let frame_index linear =
        let index = Array.make (Array.length frame) 0 in
        let rest = ref linear in
        for axis = Array.length frame - 1 downto 0 do
          index.(axis) <- !rest mod frame.(axis);
          rest := !rest / frame.(axis)
        done;
        index
      in
      let weight cell category =
        let index = if per_cell then
          Array.append (frame_index cell) [|category|]
        else [|category|] in
        Buf.get weights.buf (Ndview.index_of weights.view index)
      in
      for cell = 0 to cells - 1 do
        let ctr = Prng.Threefry.make_ctr ~site_id ~component ~frame_index:cell in
        let r0, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
        let total = ref 0.0 in
        for category = 0 to categories - 1 do
          total := !total +. weight cell category
        done;
        let target = Prng.Threefry.to_open_unit r0 *. !total in
        let acc = ref 0.0 and selected = ref (categories - 1) in
        let found = ref false in
        for category = 0 to categories - 1 do
          if not !found then begin
            acc := !acc +. weight cell category;
            if !acc >= target then (selected := category; found := true)
          end
        done;
        Buf.set result.buf cell (float_of_int !selected)
      done;
      result
  | D_pushforward { fwd_var; fwd; base; _ } ->
      let u = sample_dist ~key ~site_id ~component ~frame base env in
      if frame = [||] then begin
        let env' = (fwd_var, u) :: env in
        Eval.eval env' fwd
      end
      else begin
        (* Only fwd's free variables are imported from env.  A referenced value
           must lead-agree with frame; unrelated bindings may have any shape. *)
        let broadcast_to_frame name (t : Tensor.t) : Tensor.t =
          let shape = t.view.Ndview.shape in
          if shape = frame then t
          else if Array.length shape <= Array.length frame
            && Array.for_all2 ( = ) shape (Array.sub frame 0 (Array.length shape))
          then
            let value = ref t in
            for axis = Array.length shape to Array.length frame - 1 do
              value := Tensor.broadcast !value ~axis ~size:frame.(axis)
            done;
            !value
          else
            failwith
              (Printf.sprintf
                 "batch fwd: variable '%s' does not lead-agree with frame" name)
        in
        let free_vars expression =
          let rec go bound variables = function
            | Types.Var (_, name) ->
                if List.mem name bound || List.mem name variables then variables
                else name :: variables
            | Const _ -> variables
            | Prim (_, _, args) | Rank (_, _, _, args) ->
                List.fold_left (go bound) variables args
            | Let (_, name, rhs, body) ->
                go (name :: bound) (go bound variables rhs) body
            | Sample _ | Score _ -> variables
          in
          go [] [] expression
        in
        let needed = List.filter (fun name -> name <> fwd_var) (free_vars fwd) in
        let env' =
          (fwd_var, u) :: List.filter_map (fun name ->
            Option.map (fun value -> name, broadcast_to_frame name value)
              (List.assoc_opt name env)) needed
        in
        let rec lift_consts = function
          | Types.Const (l, v) -> Types.Const (l, broadcast_to_frame "<const>" v)
          | Var _ as e -> e
          | Prim (l, p, args) -> Prim (l, p, List.map lift_consts args)
          | Let (l, s, e1, e2) -> Let (l, s, lift_consts e1, lift_consts e2)
          | Rank _ | Sample _ | Score _ ->
              failwith "batch fwd: non-elementwise construct"
        in
        Eval.eval env' (lift_consts fwd)
      end
  | D_product (a, b) ->
      let va =
        sample_dist ~key ~site_id ~component:(component * 2) ~frame a env
      in
      let vb =
        sample_dist ~key ~site_id ~component:((component * 2) + 1) ~frame b env
      in
      (* Stack along a new first axis *)
      ignore (va, vb);
      failwith "D_product sampling not yet implemented"

let simulate ?sites ~run_key (env : Eval.env) (e : Types.expr) :
    Types.value * trace * Types.value =
  let sites = Option.value sites ~default:(Sites.collect_sites e) in
  let key = Prng.Threefry.make_key ~run_key ~namespace:Prng.Threefry.ns_model in
  let trace = ref [] in
  let log_weight = ref 0.0 in
  let rec go (env : Eval.env) (e : Types.expr) : Types.value =
    match e with
    | Const (_, v) -> v
    | Var (loc, s) -> (
        match List.assoc_opt s env with
        | Some v -> v
        | None ->
            raise (Eval.Eval_error (loc, "simulate: unbound variable: " ^ s)))
    | Let (_, s, e1, e2) ->
        let v1 = go env e1 in
        go ((s, v1) :: env) e2
    | Prim (loc, p, args) ->
        let vs = List.map (go env) args in
        Eval.validate loc p vs;
        Eval.eval []
          (Types.Prim (loc, p, List.map (fun v -> Types.Const (loc, v)) vs))
    | Rank (loc, _, _, _) ->
        raise
          (Eval.Eval_error (loc, "Rank node must be expanded before simulate"))
    | Score (_, e) ->
        let v = go env e in
        let s = sum_all v in
        log_weight := !log_weight +. s;
        scalar 0.0
    | Sample (_, name, frame, dist) ->
        let site = Sites.find name sites in
        let v =
          sample_dist ~key ~site_id:site.id ~component:1 ~frame dist env
        in
        trace := (name, v) :: !trace;
        v
  in
  let result = go env e in
  (result, List.rev !trace, scalar !log_weight)
