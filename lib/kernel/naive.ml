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

let matmul ~a ~view_a ~b ~view_b ~dst ~nframe =
  assert (not dst.Buf.shared);
  let ra = Ndview.rank view_a in
  let rb = Ndview.rank view_b in
  assert (ra = nframe + 2);
  assert (rb = nframe + 2);
  let sa = view_a.Ndview.shape in
  let sb = view_b.Ndview.shape in
  let m = sa.(nframe) in
  let ka = sa.(nframe + 1) in
  let n = sb.(nframe + 1) in
  assert (ka = sb.(nframe));
  assert (Buf.length a >= Ndview.base_size view_a);
  assert (Buf.length b >= Ndview.base_size view_b);
  let frame_shape = Array.sub sa 0 nframe in
  let out_shape = Array.append frame_shape [| m; n |] in
  let out_numel = Array.fold_left ( * ) 1 out_shape in
  assert (Buf.length dst >= out_numel);
  let out_view = Ndview.contiguous out_shape in
  let idx_a = Array.make ra 0 in
  let idx_b = Array.make rb 0 in
  let idx_out = Array.make (nframe + 2) 0 in
  let frame_numel = Array.fold_left ( * ) 1 frame_shape in
  let frame_idx = Array.make nframe 0 in
  for _fi = 0 to frame_numel - 1 do
    Array.blit frame_idx 0 idx_a 0 nframe;
    Array.blit frame_idx 0 idx_b 0 nframe;
    Array.blit frame_idx 0 idx_out 0 nframe;
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        let s = ref 0.0 in
        idx_a.(nframe) <- i;
        idx_b.(nframe + 1) <- j;
        idx_out.(nframe) <- i;
        idx_out.(nframe + 1) <- j;
        for k = 0 to ka - 1 do
          idx_a.(nframe + 1) <- k;
          idx_b.(nframe) <- k;
          s := !s +. (Buf.get a (Ndview.index_of view_a idx_a)
                    *. Buf.get b (Ndview.index_of view_b idx_b))
        done;
        Buf.set dst (Ndview.index_of out_view idx_out) !s
      done
    done;
    (* advance frame index *)
    let k = ref (nframe - 1) in
    while !k >= 0 do
      frame_idx.(!k) <- frame_idx.(!k) + 1;
      if frame_idx.(!k) < frame_shape.(!k) then k := -1
      else (frame_idx.(!k) <- 0; decr k)
    done
  done
