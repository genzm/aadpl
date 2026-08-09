open View.Ndview

let min_off v =
  let m = ref v.offset in
  Array.iteri (fun k s ->
    let e = (v.shape.(k) - 1) * s in
    if e < 0 then m := !m + e) v.strides;
  !m

let () =
  let n = 2000 in
  let rs = QCheck.Gen.generate ~n Gen_view.gen_view in
  let tbl = Hashtbl.create 64 in
  let min_off_positive = ref 0 in
  List.iter (fun v ->
    let s0 = Array.exists (fun s -> s = 0) v.strides in
    let neg = Array.exists (fun s -> s < 0) v.strides in
    let s0_count = Array.fold_left (fun acc s -> if s = 0 then acc + 1 else acc) 0 v.strides in
    if min_off v > 0 then incr min_off_positive;
    let k = Printf.sprintf "r%d/%s%s%s"
      (min (rank v) 4)
      (if s0 then (if s0_count > 1 then "00" else "0-") else "--")
      (if neg then "n" else "-")
      (if v.offset > 0 then "f" else "-")
    in
    Hashtbl.replace tbl k (1 + (try Hashtbl.find tbl k with Not_found -> 0)))
    rs;
  Printf.printf "=== Combination coverage (%d samples) ===\n" n;
  Printf.printf "key: r<rank>/<stride0><neg><offset>  (0=has0, 00=multi0, n=neg, f=off>0)\n\n";
  Hashtbl.fold (fun k c acc -> (k, c) :: acc) tbl []
  |> List.sort (fun (_, a) (_, b) -> compare b a)
  |> List.iter (fun (k, c) ->
    Printf.printf "  %-14s %4d (%5.1f%%)\n" k c (100. *. float c /. float n));
  Printf.printf "\nmin_off > 0:     %d (%5.1f%%)\n"
    !min_off_positive (100. *. float !min_off_positive /. float n)
