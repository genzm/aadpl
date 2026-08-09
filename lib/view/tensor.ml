type t = {
  buf : Buf.t;
  view : Ndview.view;
}

let of_buf buf view = { buf; view }

let make shape =
  let v = Ndview.contiguous shape in
  let buf = Buf.create (Ndview.numel v) in
  of_buf buf v

let make_random shape =
  let t = make shape in
  Buf.fill_random t.buf;
  t

(* All view operations go through here: marks buf as shared *)
let map_view t f =
  t.buf.Buf.shared <- true;
  { buf = t.buf; view = f t.view }

let broadcast t ~axis ~size = map_view t (fun v -> Ndview.broadcast v ~axis ~size)
let transpose t ~perm       = map_view t (fun v -> Ndview.transpose v ~perm)
let slice t ~ranges         = map_view t (fun v -> Ndview.slice v ~ranges)

let reshape t ~shape =
  match Ndview.reshape t.view ~shape with
  | None -> None
  | Some v -> t.buf.Buf.shared <- true; Some { buf = t.buf; view = v }

(* read_view: read src buffer through view into contiguous dst *)
let read_view ~src ~view ~dst =
  assert (Buf.length src >= Ndview.base_size view);
  assert (Buf.length dst >= Ndview.numel view);
  Ndview.iter_indices view.Ndview.shape (fun i idx ->
    let pos = Ndview.index_of view idx in
    Buf.set dst i (Buf.get src pos))

(* add_view: scatter-add src into acc buffer through view *)
let add_view ~src ~view ~acc =
  assert (not acc.Buf.shared);
  assert (Buf.length src >= Ndview.numel view);
  assert (Buf.length acc >= Ndview.base_size view);
  Ndview.iter_indices view.Ndview.shape (fun i idx ->
    let pos = Ndview.index_of view idx in
    Buf.set acc pos (Buf.get acc pos +. Buf.get src i))
