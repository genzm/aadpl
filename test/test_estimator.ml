(* The Estimator IR boundary.

   An objective says WHICH statistical quantity is wanted; a lowering picks the
   estimator.  These tests pin the invariant that separates the two, so that a
   later lowering (enumeration, score function) can be added without disturbing
   the objective:

     - the objective's integrand is already a pure tensor expression;
       only the proposal carries Sample nodes,
     - a lowering removes every Sample and leaves the drawn randomness as free
       noise variables,
     - lowering carries no hidden state, so it is reproducible. *)

open Alcotest
open Ast.Types

let scalar value =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 value;
  t

let log_normal ~x ~mu ~sigma =
  let half = const (scalar 0.5) in
  let log_two_pi = const (scalar (log (2.0 *. Float.pi))) in
  let standardized = rank 0 Div [ rank 0 Sub [ x; mu ]; sigma ] in
  let normalizer =
    rank 0 Add [ rank 0 Mul [ half; log_two_pi ]; prim Log [ sigma ] ]
  in
  rank 0 Neg
    [
      rank 0 Add
        [
          normalizer;
          rank 0 Mul [ half; rank 0 Mul [ standardized; standardized ] ];
        ];
    ]

(* The conjugate Gaussian program of test_vi, kept small on purpose. *)
let conjugate samples =
  let model =
    let_ "z"
      (sample "z" [| samples |]
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_sigma"))
      (score
         (log_normal ~x:(var "x_obs") ~mu:(var "z") ~sigma:(var "obs_sigma")))
  in
  let guide =
    let_ "sigma_q"
      (prim Exp [ var "log_sigma_q" ])
      (sample "z" [| samples |] (Ast.Normal.normal ~mu:"mu_q" ~sigma:"sigma_q"))
  in
  let env_shapes =
    [
      ("prior_mu", [||]); ("prior_sigma", [||]); ("x_obs", [||]);
      ("obs_sigma", [||]); ("mu_q", [||]); ("log_sigma_q", [||]);
    ]
  in
  (model, guide, env_shapes)

let rec has_sample = function
  | Sample _ | Score _ -> true
  | Const _ | Var _ -> false
  | Prim (_, _, args) | Rank (_, _, _, args) -> List.exists has_sample args
  | Let (_, _, rhs, body) -> has_sample rhs || has_sample body
  | Scan (_, scan, continuation) ->
      List.exists
        (fun (_, init, next) -> has_sample init || has_sample next)
        scan.carries
      || List.exists (fun (_, input) -> has_sample input) scan.inputs
      || has_sample continuation

let site_names sites =
  List.map (fun (site : Ast.Sites.site) -> site.name) sites

(* The objective is model semantics only: densities are already symbolic, and
   the randomness still lives in the proposal where a lowering can find it. *)
let test_objective_separates_integrand_from_proposal () =
  let model, guide, env_shapes = conjugate 3 in
  match Estimator.elbo ~slots:[] ~model ~guide ~env_shapes with
  | Estimator.Types.Deterministic _ -> fail "ELBO must be an expectation"
  | Estimator.Types.Expect expectation ->
      check (list string) "proposal sites" [ "z" ]
        (site_names expectation.sites);
      check bool "integrand is a pure tensor expression" false
        (has_sample expectation.body);
      check bool "proposal still carries the Sample" true
        (has_sample expectation.proposal)

(* Pathwise lowering turns the expectation into a deterministic program whose
   only stochastic inputs are the free noise variables. *)
let test_pathwise_lowering_eliminates_samples () =
  let model, guide, env_shapes = conjugate 3 in
  let program =
    Estimator.lower_pathwise
    @@ Estimator.elbo ~slots:[] ~model ~guide ~env_shapes
  in
  check bool "lowered program has no Sample/Score" false
    (has_sample program.loss);
  check (list string) "noise variables" [ "%u.z" ] (List.map fst program.noise);
  check (list string) "sites survive lowering" [ "z" ]
    (site_names program.sites);
  let free = Transform.Assess_expr.free_vars program.loss in
  List.iter
    (fun (name, _) ->
      check bool (name ^ " is free in the lowered program") true
        (List.mem name free))
    program.noise

(* Lowering keeps no gensym or other hidden state: the same objective lowers to
   the same program every time.  This is what makes the objective safe to reuse
   across several lowerings. *)
let test_lowering_is_reproducible () =
  let model, guide, env_shapes = conjugate 3 in
  let objective = Estimator.elbo ~slots:[] ~model ~guide ~env_shapes in
  let first = Estimator.lower_pathwise objective in
  let second = Estimator.lower_pathwise objective in
  check string "lowering twice gives the same program"
    (Format.asprintf "%a" Ast.Types.pp first.loss)
    (Format.asprintf "%a" Ast.Types.pp second.loss)

let test_deterministic_objective_passes_through () =
  let expression = prim Add [ const (scalar 1.0); const (scalar 2.0) ] in
  let program =
    Estimator.lower_pathwise (Estimator.Types.Deterministic expression)
  in
  check (list string) "no sites" [] (site_names program.sites);
  check (list string) "no noise" [] (List.map fst program.noise);
  check (float 1e-12) "value preserved" 3.0
    (View.Buf.get (Ast.Eval.eval [] program.loss).View.Tensor.buf 0)

let () =
  run "Estimator"
    [
      ( "boundary",
        [
          test_case "objective separates integrand from proposal" `Quick
            test_objective_separates_integrand_from_proposal;
          test_case "deterministic objective passes through" `Quick
            test_deterministic_objective_passes_through;
        ] );
      ( "lower_pathwise",
        [
          test_case "eliminates samples" `Quick
            test_pathwise_lowering_eliminates_samples;
          test_case "is reproducible" `Quick test_lowering_is_reproducible;
        ] );
    ]
