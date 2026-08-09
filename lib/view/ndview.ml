type view = { shape : int array; strides : int array; offset : int }

let rank v = Array.length v.shape

let numel v = Array.fold_left ( * ) 1 v.shape

let contiguous shape =
  Array.iter (fun n -> assert (n > 0)) shape;
  let r = Array.length shape in
  let strides = Array.make r 1 in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * shape.(i + 1)
  done;
  { shape; strides; offset = 0 }

let scalar () = { shape = [||]; strides = [||]; offset = 0 }

(* --- index_of: reference implementation --- *)

let index_of v idx =
  let r = rank v in
  assert (Array.length idx = r);
  let pos = ref v.offset in
  for k = 0 to r - 1 do
    assert (0 <= idx.(k) && idx.(k) < v.shape.(k));
    pos := !pos + (idx.(k) * v.strides.(k))
  done;
  !pos

(* --- structural operations: all O(rank), no data --- *)

let broadcast v ~axis ~size =
  let r = rank v in
  assert (0 <= axis && axis <= r);
  assert (size > 0);
  let shape = Array.init (r + 1) (fun i ->
    if i < axis then v.shape.(i)
    else if i = axis then size
    else v.shape.(i - 1))
  in
  let strides = Array.init (r + 1) (fun i ->
    if i < axis then v.strides.(i)
    else if i = axis then 0
    else v.strides.(i - 1))
  in
  { shape; strides; offset = v.offset }

let transpose v ~perm =
  let r = rank v in
  assert (Array.length perm = r);
  (* Validate perm is a permutation *)
  let seen = Array.make r false in
  Array.iter (fun p ->
    assert (0 <= p && p < r);
    assert (not seen.(p));
    seen.(p) <- true) perm;
  let shape = Array.init r (fun i -> v.shape.(perm.(i))) in
  let strides = Array.init r (fun i -> v.strides.(perm.(i))) in
  { shape; strides; offset = v.offset }

let slice v ~ranges =
  let r = rank v in
  assert (Array.length ranges = r);
  let offset = ref v.offset in
  let shape = Array.make r 0 in
  let strides = Array.make r 0 in
  for k = 0 to r - 1 do
    let (start, stop, step) = ranges.(k) in
    assert (step <> 0);
    if step > 0 then begin
      assert (0 <= start && start < v.shape.(k));
      assert (start < stop && stop <= v.shape.(k));
      offset := !offset + (start * v.strides.(k));
      shape.(k) <- (stop - start + step - 1) / step;
      strides.(k) <- v.strides.(k) * step
    end else begin
      assert (0 <= start && start < v.shape.(k));
      assert (-1 <= stop && stop < start);
      offset := !offset + (start * v.strides.(k));
      shape.(k) <- (start - stop + (-step) - 1) / (-step);
      strides.(k) <- v.strides.(k) * step
    end
  done;
  { shape; strides; offset = !offset }

let is_contiguous v =
  let r = rank v in
  if r = 0 then true
  else begin
    let ok = ref true in
    let expected = ref 1 in
    for i = r - 1 downto 0 do
      (* size-1 axes don't constrain stride *)
      if v.shape.(i) > 1 && v.strides.(i) <> !expected then ok := false;
      expected := !expected * v.shape.(i)
    done;
    !ok
  end

let reshape v ~shape =
  let old_numel = numel v in
  let new_numel = Array.fold_left ( * ) 1 shape in
  if old_numel <> new_numel then None
  else if not (is_contiguous v) then None
  else begin
    Array.iter (fun n -> assert (n > 0)) shape;
    let r = Array.length shape in
    let strides = Array.make r 1 in
    for i = r - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * shape.(i + 1)
    done;
    Some { shape; strides; offset = v.offset }
  end

let is_injective v =
  let r = rank v in
  if r = 0 then true
  else begin
    (* Filter out size-1 axes: they don't move position *)
    let pairs =
      Array.init r (fun k -> (abs v.strides.(k), v.shape.(k)))
      |> Array.to_list
      |> List.filter (fun (_, n) -> n > 1)
      |> List.sort (fun (a, _) (b, _) -> compare a b)
      |> Array.of_list
    in
    let m = Array.length pairs in
    let ok = ref true in
    Array.iter (fun (s, _) -> if s = 0 then ok := false) pairs;
    if !ok then
      for k = 0 to m - 2 do
        let (s_k, n_k) = pairs.(k) in
        let (s_next, _) = pairs.(k + 1) in
        if n_k * s_k > s_next then ok := false
      done;
    !ok
  end

(* --- iteration (no allocation) --- *)

let iter_indices shape f =
  let r = Array.length shape in
  let n = Array.fold_left ( * ) 1 shape in
  let idx = Array.make r 0 in
  for i = 0 to n - 1 do
    f i idx;
    let k = ref (r - 1) in
    while !k >= 0 do
      idx.(!k) <- idx.(!k) + 1;
      if idx.(!k) < shape.(!k) then k := -1
      else (idx.(!k) <- 0; decr k)
    done
  done

(* kept for tests / shadow model only *)
let all_indices shape =
  let r = Array.length shape in
  let n = Array.fold_left ( * ) 1 shape in
  let result = Array.make n (Array.make 0 0) in
  let idx = Array.make r 0 in
  for i = 0 to n - 1 do
    result.(i) <- Array.copy idx;
    let k = ref (r - 1) in
    while !k >= 0 do
      idx.(!k) <- idx.(!k) + 1;
      if idx.(!k) < shape.(!k) then k := -1
      else (idx.(!k) <- 0; decr k)
    done
  done;
  result

(* --- base size: max reachable index + 1 --- *)

let base_size v =
  let r = rank v in
  if r = 0 then v.offset + 1
  else begin
    let max_off = ref v.offset in
    let min_off = ref v.offset in
    for k = 0 to r - 1 do
      let extent = (v.shape.(k) - 1) * v.strides.(k) in
      if extent >= 0 then
        max_off := !max_off + extent
      else
        min_off := !min_off + extent
    done;
    assert (!min_off >= 0);
    !max_off + 1
  end

(* --- pretty-printer --- *)

let pp fmt v =
  let pp_iarray fmt a =
    Format.fprintf fmt "[|";
    Array.iteri (fun i x ->
      if i > 0 then Format.fprintf fmt "; ";
      Format.fprintf fmt "%d" x) a;
    Format.fprintf fmt "|]"
  in
  Format.fprintf fmt "{shape=%a; strides=%a; offset=%d}"
    pp_iarray v.shape pp_iarray v.strides v.offset
