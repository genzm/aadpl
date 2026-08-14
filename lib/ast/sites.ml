open Types
open View

type kind = [ `Cont | `Disc ]
type slot_role = [ `Condition | `Maximize ]
type slot = string * slot_role * expr
type site = {
  name : string;
  id : int;
  loc : loc;
  frame : int array;
  kind : kind;
  dist : dist;
}

let noise_name_of_name name = "%u." ^ name
let trace_name_of_name name = "%tr." ^ name
let noise_name site = noise_name_of_name site.name
let trace_name site = trace_name_of_name site.name

let rec dist_kind = function
  | D_uniform -> `Cont
  | D_pushforward { base; _ } -> dist_kind base
  | D_categorical _ -> `Disc
  | D_product (a, b) ->
      if dist_kind a = `Cont && dist_kind b = `Cont then `Cont else `Disc

let rec dist_support = function
  | D_uniform -> S_unit_interval
  | D_categorical (Const (_, weights)) ->
      let shape = weights.View.Tensor.view.View.Ndview.shape in
      if Array.length shape = 0 then
        failwith "dist_support: categorical weights need a category axis";
      S_finite shape.(Array.length shape - 1)
  | D_categorical _ ->
      failwith "dist_support: categorical size must be statically known"
  | D_pushforward { support; _ } -> support
  | D_product (a, b) -> S_product (dist_support a, dist_support b)

let rec support_subset left right =
  left = right
  || match left, right with
     | (S_unit_interval | S_positive), S_real -> true
     | S_unit_interval, S_positive -> true
     | S_finite n, S_finite m -> n <= m
     | S_product (la, lb), S_product (ra, rb) ->
         support_subset la ra && support_subset lb rb
     | _ -> false

let support_contains support value =
  match support with
  | S_real -> Float.is_finite value
  | S_positive -> Float.is_finite value && value > 0.0
  | S_unit_interval -> value > 0.0 && value < 1.0
  | S_finite n ->
      Float.is_finite value && value = Float.round value
      && value >= 0.0 && value < float_of_int n
  | S_product _ ->
      failwith "support_contains: D_product not supported (Phase 13)"

(* The sole definition of static site traversal and numbering. *)
let collect_sites (e : expr) : site list =
  (* Duplicate names would alias both the counter and generated %u/%tr names. *)
  check_sites e;
  let sites = ref [] in
  let next = ref 0 in
  let rec walk = function
    | Sample (loc, name, frame, dist) ->
        sites := { name; id = !next; loc; frame; kind = dist_kind dist; dist }
          :: !sites;
        incr next;
        walk_dist dist
    | Score (_, e) -> walk e
    | Const _ | Var _ -> ()
    | Prim (_, _, args) | Rank (_, _, _, args) -> List.iter walk args
    | Let (_, _, e1, e2) ->
        walk e1;
        walk e2
  and walk_dist = function
    | D_uniform -> ()
    | D_categorical weights -> walk weights
    | D_pushforward { fwd; inv; base; _ } ->
        walk fwd;
        walk inv;
        walk_dist base
    | D_product (a, b) ->
        walk_dist a;
        walk_dist b
  in
  walk e;
  List.rev !sites

let find name sites = List.find (fun site -> site.name = name) sites

let draw_noise ?(namespace = Prng.Threefry.ns_model) ~run_key
    (sites : site list) : (string * Tensor.t) list =
  let key = Prng.Threefry.make_key ~run_key ~namespace in
  List.map
    (fun site ->
      let n = Array.fold_left ( * ) 1 site.frame in
      let noise = Tensor.make site.frame in
      for frame_index = 0 to n - 1 do
        let ctr =
          Prng.Threefry.make_ctr ~site_id:site.id ~component:1 ~frame_index
        in
        let r0, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
        Buf.set noise.buf frame_index (Prng.Threefry.to_open_unit r0)
      done;
      (noise_name site, noise))
    sites
