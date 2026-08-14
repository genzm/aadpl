(* Scalar special functions needed by Phase 13 tests and inference drivers. *)

let rec log_gamma z =
  if z <= 0.0 then invalid_arg "log_gamma: positive argument required";
  let coefficients = [|
    0.99999999999980993; 676.5203681218851; -1259.1392167224028;
    771.32342877765313; -176.61502916214059; 12.507343278686905;
    -0.13857109526572012; 9.9843695780195716e-6;
    1.5056327351493116e-7;
  |] in
  if z < 0.5 then
    log Float.pi -. log (sin (Float.pi *. z)) -. log_gamma (1.0 -. z)
  else
    let z = z -. 1.0 in
    let x = ref coefficients.(0) in
    for i = 1 to Array.length coefficients - 1 do
      x := !x +. coefficients.(i) /. (z +. float_of_int i)
    done;
    let t = z +. 7.5 in
    0.5 *. log (2.0 *. Float.pi) +. (z +. 0.5) *. log t -. t +. log !x

let gamma_series_p a x =
  let sum = ref (1.0 /. a) in
  let term = ref !sum in
  let ap = ref a in
  let iteration = ref 0 in
  while !iteration < 10_000 && Float.abs !term > Float.abs !sum *. 1e-15 do
    incr iteration;
    ap := !ap +. 1.0;
    term := !term *. x /. !ap;
    sum := !sum +. !term
  done;
  !sum *. exp (-.x +. a *. log x -. log_gamma a)

let gamma_continued_fraction_q a x =
  let tiny = 1e-300 in
  let b = ref (x +. 1.0 -. a) in
  let c = ref (1.0 /. tiny) in
  let d = ref (1.0 /. max tiny !b) in
  let h = ref !d in
  let iteration = ref 0 and converged = ref false in
  while !iteration < 10_000 && not !converged do
    incr iteration;
    let i = float_of_int !iteration in
    let an = -.i *. (i -. a) in
    b := !b +. 2.0;
    d := an *. !d +. !b;
    if Float.abs !d < tiny then d := tiny;
    c := !b +. an /. !c;
    if Float.abs !c < tiny then c := tiny;
    d := 1.0 /. !d;
    let delta = !d *. !c in
    h := !h *. delta;
    converged := Float.abs (delta -. 1.0) < 1e-15
  done;
  exp (-.x +. a *. log x -. log_gamma a) *. !h

let regularized_gamma_p a x =
  if a <= 0.0 || x < 0.0 || Float.is_nan x then
    invalid_arg "regularized_gamma_p: require a > 0 and x >= 0";
  if x = 0.0 then 0.0
  else if Float.is_infinite x then 1.0
  else if x < a +. 1.0 then gamma_series_p a x
  else 1.0 -. gamma_continued_fraction_q a x

let regularized_gamma_q a x =
  if a <= 0.0 || x < 0.0 || Float.is_nan x then
    invalid_arg "regularized_gamma_q: require a > 0 and x >= 0";
  if x = 0.0 then 1.0
  else if Float.is_infinite x then 0.0
  else if x < a +. 1.0 then 1.0 -. gamma_series_p a x
  else gamma_continued_fraction_q a x

let chi_square_1_cdf statistic =
  if statistic < 0.0 then invalid_arg "chi_square_1_cdf: negative statistic";
  regularized_gamma_p 0.5 (0.5 *. statistic)

let chi_square_1_sf statistic =
  if statistic < 0.0 then invalid_arg "chi_square_1_sf: negative statistic";
  regularized_gamma_q 0.5 (0.5 *. statistic)

let boundary_variance_component_p_value statistic =
  if statistic < 0.0 then
    invalid_arg "boundary_variance_component_p_value: negative statistic";
  if statistic = 0.0 then 1.0 else 0.5 *. chi_square_1_sf statistic
