open Bigarray

type t = {
  data : (float, float64_elt, c_layout) Array1.t;
  mutable shared : bool;
}

let create n =
  let data = Array1.create float64 c_layout n in
  Array1.fill data 0.0;
  { data; shared = false }

let length b = Array1.dim b.data

let get b i = Array1.get b.data i
let set b i v = Array1.set b.data i v

let dot a b =
  let n = length a in
  assert (length b = n);
  let s = ref 0.0 in
  for i = 0 to n - 1 do
    s := !s +. (get a i *. get b i)
  done;
  !s

let fill_random b =
  let n = length b in
  for i = 0 to n - 1 do
    set b i (Random.float 2.0 -. 1.0)
  done

let copy b =
  let c = create (length b) in
  Array1.blit b.data c.data;
  c
