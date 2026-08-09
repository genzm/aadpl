open View.Ndview

(* === Shadow model: stride-free reference === *)

type shadow = int array -> int

let sh_contiguous shape : shadow = fun idx ->
  let pos = ref 0 in
  Array.iteri (fun k j -> pos := !pos * shape.(k) + j) idx;
  !pos

let sh_broadcast (sh : shadow) ~axis : shadow = fun idx ->
  let r = Array.length idx in
  sh (Array.init (r - 1) (fun i -> if i < axis then idx.(i) else idx.(i + 1)))

let sh_transpose (sh : shadow) ~perm ~old_rank : shadow = fun idx ->
  let old_idx = Array.make old_rank 0 in
  Array.iteri (fun i p -> old_idx.(p) <- idx.(i)) perm;
  sh old_idx

let sh_slice (sh : shadow) ~ranges : shadow = fun idx ->
  sh (Array.mapi (fun k j ->
    let (start, _, step) = ranges.(k) in start + (j * step)) idx)

(* === Generators === *)

let gen_shape =
  let open QCheck.Gen in
  let* r = int_range 1 4 in
  let* dims = list_size (return r) (int_range 1 5) in
  return (Array.of_list dims)

let gen_contiguous =
  let open QCheck.Gen in
  let* shape = gen_shape in
  return (contiguous shape, sh_contiguous shape)

let gen_perm r =
  let open QCheck.Gen in
  let* js = list_size (return r) (int_range 0 99999) in
  return (
    let a = Array.init r Fun.id in
    List.iteri (fun i j ->
      let k = i + (j mod (r - i)) in
      let tmp = a.(i) in a.(i) <- a.(k); a.(k) <- tmp) js;
    a)

let gen_composed =
  let open QCheck.Gen in
  let rec apply_op (v, sh) =
    let r = rank v in
    let* op = oneof_weighted [
      3, return 0;   (* broadcast *)
      2, return 1;   (* transpose *)
      3, return 2;   (* slice positive *)
      2, return 3;   (* slice negative / partial reverse *)
    ] in
    match op with
    | 0 ->
      let* axis = int_range 0 r in
      let* size = int_range 1 4 in
      return (broadcast v ~axis ~size, sh_broadcast sh ~axis)
    | 1 ->
      if r < 2 then apply_op (v, sh)  (* retry instead of wasting op *)
      else
        let* perm = gen_perm r in
        return (transpose v ~perm, sh_transpose sh ~perm ~old_rank:r)
    | 2 ->
      if r = 0 then return (v, sh)
      else begin
        let ranges = Array.init r (fun k -> (0, v.shape.(k), 1)) in
        let* axis = int_range 0 (r - 1) in
        let n = v.shape.(axis) in
        if n <= 1 then
          return (slice v ~ranges, sh_slice sh ~ranges)
        else begin
          let* start = int_range 0 (n - 2) in
          let* stop = int_range (start + 1) n in
          ranges.(axis) <- (start, stop, 1);
          return (slice v ~ranges, sh_slice sh ~ranges)
        end
      end
    | _ ->
      if r = 0 then return (v, sh)
      else begin
        let* axis = int_range 0 (r - 1) in
        let n = v.shape.(axis) in
        if n <= 1 then return (v, sh)
        else begin
          let ranges = Array.init r (fun k -> (0, v.shape.(k), 1)) in
          (* partial reverse: start anywhere from 1..n-1, stop from -1..start-1 *)
          let* start = int_range 1 (n - 1) in
          let* stop = int_range (-1) (start - 1) in
          ranges.(axis) <- (start, stop, -1);
          return (slice v ~ranges, sh_slice sh ~ranges)
        end
      end
  in
  let* base = gen_contiguous in
  let* n_ops = int_range 1 4 in
  let rec go vs = function
    | 0 -> return vs
    | n ->
      let* vs' = apply_op vs in
      go vs' (n - 1)
  in
  go base n_ops

let gen_view = QCheck.Gen.map fst gen_composed
