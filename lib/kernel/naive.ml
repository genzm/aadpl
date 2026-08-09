open View

let map1 ~f ~src ~view ~dst =
  assert (not dst.Buf.shared);
  assert (Buf.length src >= Ndview.base_size view);
  assert (Buf.length dst >= Ndview.numel view);
  Ndview.iter_indices view.Ndview.shape (fun i idx ->
    Buf.set dst i (f (Buf.get src (Ndview.index_of view idx))))

let map2 ~f ~src1 ~view1 ~src2 ~view2 ~dst =
  assert (not dst.Buf.shared);
  assert (view1.Ndview.shape = view2.Ndview.shape);
  assert (Buf.length src1 >= Ndview.base_size view1);
  assert (Buf.length src2 >= Ndview.base_size view2);
  assert (Buf.length dst >= Ndview.numel view1);
  Ndview.iter_indices view1.Ndview.shape (fun i idx ->
    Buf.set dst i (f (Buf.get src1 (Ndview.index_of view1 idx))
                     (Buf.get src2 (Ndview.index_of view2 idx))))

let sum_axis ~src ~view ~axis ~dst =
  assert (not dst.Buf.shared);
  let shape = view.Ndview.shape in
  let r = Array.length shape in
  assert (0 <= axis && axis < r);
  assert (Buf.length src >= Ndview.base_size view);
  let n_axis = shape.(axis) in
  let out_shape = Array.init (r - 1) (fun i ->
    if i < axis then shape.(i) else shape.(i + 1)) in
  assert (Buf.length dst >= Array.fold_left ( * ) 1 out_shape);
  Ndview.iter_indices out_shape (fun oi out_idx ->
    let full_idx = Array.init r (fun k ->
      if k < axis then out_idx.(k)
      else if k = axis then 0
      else out_idx.(k - 1)) in
    let s = ref 0.0 in
    for j = 0 to n_axis - 1 do
      full_idx.(axis) <- j;
      s := !s +. Buf.get src (Ndview.index_of view full_idx)
    done;
    Buf.set dst oi !s)

let max_axis ~src ~view ~axis ~dst ~dst_argmax =
  assert (not dst.Buf.shared);
  let shape = view.Ndview.shape in
  let r = Array.length shape in
  assert (0 <= axis && axis < r);
  assert (Buf.length src >= Ndview.base_size view);
  let n_axis = shape.(axis) in
  assert (n_axis > 0);
  let out_shape = Array.init (r - 1) (fun i ->
    if i < axis then shape.(i) else shape.(i + 1)) in
  let out_numel = Array.fold_left ( * ) 1 out_shape in
  assert (Buf.length dst >= out_numel);
  assert (Array.length dst_argmax >= out_numel);
  Ndview.iter_indices out_shape (fun oi out_idx ->
    let full_idx = Array.init r (fun k ->
      if k < axis then out_idx.(k)
      else if k = axis then 0
      else out_idx.(k - 1)) in
    let best_val = ref (Buf.get src (Ndview.index_of view full_idx)) in
    let best_j = ref 0 in
    for j = 1 to n_axis - 1 do
      full_idx.(axis) <- j;
      let v = Buf.get src (Ndview.index_of view full_idx) in
      if v > !best_val then (best_val := v; best_j := j)
    done;
    Buf.set dst oi !best_val;
    dst_argmax.(oi) <- !best_j)

let gather ~src ~view ~axis ~indices ~dst =
  assert (not dst.Buf.shared);
  let shape = view.Ndview.shape in
  let r = Array.length shape in
  assert (0 <= axis && axis < r);
  assert (Buf.length src >= Ndview.base_size view);
  let m = Array.length indices in
  Array.iter (fun j -> assert (0 <= j && j < shape.(axis))) indices;
  let out_shape = Array.init r (fun k ->
    if k = axis then m else shape.(k)) in
  assert (Buf.length dst >= Array.fold_left ( * ) 1 out_shape);
  Ndview.iter_indices out_shape (fun oi out_idx ->
    let full_idx = Array.copy out_idx in
    full_idx.(axis) <- indices.(out_idx.(axis));
    Buf.set dst oi (Buf.get src (Ndview.index_of view full_idx)))

let scatter_add ~src ~view ~axis ~indices ~acc =
  assert (not acc.Buf.shared);
  let shape = view.Ndview.shape in
  let r = Array.length shape in
  assert (0 <= axis && axis < r);
  let m = Array.length indices in
  let src_shape = Array.init r (fun k ->
    if k = axis then m else shape.(k)) in
  assert (Buf.length src >= Array.fold_left ( * ) 1 src_shape);
  assert (Buf.length acc >= Ndview.base_size view);
  Ndview.iter_indices src_shape (fun si src_idx ->
    let full_idx = Array.copy src_idx in
    full_idx.(axis) <- indices.(src_idx.(axis));
    let pos = Ndview.index_of view full_idx in
    Buf.set acc pos (Buf.get acc pos +. Buf.get src si))

let matmul ~a ~view_a ~b ~view_b ~dst =
  assert (not dst.Buf.shared);
  assert (Ndview.rank view_a = 2);
  assert (Ndview.rank view_b = 2);
  let m = view_a.Ndview.shape.(0) in
  let ka = view_a.Ndview.shape.(1) in
  let kb = view_b.Ndview.shape.(0) in
  let n = view_b.Ndview.shape.(1) in
  assert (ka = kb);
  assert (Buf.length a >= Ndview.base_size view_a);
  assert (Buf.length b >= Ndview.base_size view_b);
  assert (Buf.length dst >= m * n);
  let idx_a = [|0; 0|] in
  let idx_b = [|0; 0|] in
  for i = 0 to m - 1 do
    for j = 0 to n - 1 do
      let s = ref 0.0 in
      for k = 0 to ka - 1 do
        idx_a.(0) <- i; idx_a.(1) <- k;
        idx_b.(0) <- k; idx_b.(1) <- j;
        s := !s +. (Buf.get a (Ndview.index_of view_a idx_a)
                  *. Buf.get b (Ndview.index_of view_b idx_b))
      done;
      Buf.set dst (i * n + j) !s
    done
  done
