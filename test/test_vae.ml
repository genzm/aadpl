(* Phase 11-5c preflight: the MNIST VAE shape pattern at tiny dimensions. *)

open Alcotest
open Ast.Types

let scalar value =
  let t = View.Tensor.make [||] in
  View.Buf.set t.buf 0 value;
  t

let tensor shape values =
  let t = View.Tensor.make shape in
  Array.iteri (View.Buf.set t.buf) values;
  t

let value t i = View.Buf.get t.View.Tensor.buf i
let numel t = View.Ndview.numel t.View.Tensor.view

let dense input weights bias =
  rank 1 Add [ rank 2 Matmul [ input; var weights ]; var bias ]

let vae ~batch ~latent ~obs =
  let model =
    let_ "z"
      (sample "z" [|batch; latent|]
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_sigma"))
      (let_ "decoder_h"
         (prim Relu [ dense (var "z") "w3" "b3" ])
         (let_ "eta" (dense (var "decoder_h") "w4" "b4")
            (score
               (rank 0 Add
                  [
                    rank 0 Mul [ var "x"; prim Logsigmoid [ var "eta" ] ];
                    rank 0 Mul
                      [
                        rank 0 Sub [ const (scalar 1.0); var "x" ];
                        prim Logsigmoid [ prim Neg [ var "eta" ] ];
                      ];
                  ]))))
  in
  let guide =
    let_ "encoder_h"
      (prim Relu [ dense (var "x") "w1" "b1" ])
      (let_ "mu_q" (dense (var "encoder_h") "w_mu" "b_mu")
         (let_ "rho_q" (dense (var "encoder_h") "w_rho" "b_rho")
            (let_ "sigma_q" (prim Exp [ var "rho_q" ])
               (sample "z" [|batch; latent|]
                  (Ast.Normal.normal ~mu:"mu_q" ~sigma:"sigma_q")))))
  in
  let param_shapes =
    [
      ("w1", [|obs; 3|]); ("b1", [|3|]);
      ("w_mu", [|3; latent|]); ("b_mu", [|latent|]);
      ("w_rho", [|3; latent|]); ("b_rho", [|latent|]);
      ("w3", [|latent; 3|]); ("b3", [|3|]);
      ("w4", [|3; obs|]); ("b4", [|obs|]);
    ]
  in
  let data_shapes =
    [("x", [|batch; obs|]); ("prior_mu", [||]); ("prior_sigma", [||])]
  in
  model, guide, param_shapes, data_shapes

let parameters () =
  [
    ("w1", tensor [|4; 3|] [|0.10; -0.05; 0.08; 0.04; 0.12; -0.03;
                              -0.07; 0.09; 0.05; 0.06; -0.02; 0.11|]);
    ("b1", tensor [|3|] [|0.3; 0.25; 0.2|]);
    ("w_mu", tensor [|3; 2|] [|0.1; -0.04; 0.03; 0.08; -0.06; 0.05|]);
    ("b_mu", tensor [|2|] [|0.02; -0.03|]);
    ("w_rho", tensor [|3; 2|] [|-0.03; 0.02; 0.04; -0.05; 0.01; 0.03|]);
    ("b_rho", tensor [|2|] [|-0.2; -0.1|]);
    ("w3", tensor [|2; 3|] [|0.08; -0.04; 0.06; -0.02; 0.07; 0.05|]);
    ("b3", tensor [|3|] [|0.25; 0.2; 0.3|]);
    ("w4", tensor [|3; 4|] [|0.05; -0.06; 0.04; 0.02; -0.03; 0.07;
                              0.01; -0.05; 0.06; 0.03; -0.04; 0.08|]);
    ("b4", tensor [|4|] [|-0.1; 0.05; -0.03; 0.08|]);
  ]

let data () =
  [
    ("x", tensor [|3; 4|] [|0.0; 1.0; 0.2; 0.8; 1.0; 0.0;
                             0.7; 0.1; 0.3; 0.9; 0.0; 1.0|]);
    ("prior_mu", scalar 0.0);
    ("prior_sigma", scalar 1.0);
  ]

let test_assess_two_frames () =
  let model, guide, param_shapes, data_shapes = vae ~batch:3 ~latent:2 ~obs:4 in
  let shapes = param_shapes @ data_shapes in
  let env = parameters () @ data () in
  let model = Transform.Expand_rank.expand ~senv:shapes model in
  let guide = Transform.Expand_rank.expand ~senv:shapes guide in
  let _, trace, _ = Ast.Simulate.simulate ~run_key:17L env guide in
  List.iter
    (fun (label, expr) ->
      let _, assessed = Ast.Assess.assess env expr trace in
      let slots = [("z", const (List.assoc "z" trace))] in
      let symbolic = Transform.Assess_expr.assess_expr ~env_shapes:shapes expr slots
                     |> Ast.Eval.eval env in
      check (float 1e-12) label (value assessed 0) (value symbolic 0))
    [("model assess", model); ("guide assess", guide)]

let test_all_parameter_fd () =
  let model, guide, param_shapes, data_shapes = vae ~batch:3 ~latent:2 ~obs:4 in
  let program = Transform.build_elbo
    ~observed:[] ~model ~guide ~env_shapes:(param_shapes @ data_shapes) in
  let gp = Transform.grad ~param_shapes
    ~data_shapes:(program.noise @ data_shapes) program.elbo in
  let params = parameters () in
  let fixed = Transform.noise_env program ~run_key:23L @ data () in
  let env = params @ fixed in
  let epsilon = 1e-5 in
  let eval ps = value (Ast.Eval.eval (ps @ fixed) gp.loss) 0 in
  List.iter
    (fun (name, parameter) ->
      let gradient = Ast.Eval.eval env (List.assoc name gp.grads) in
      for i = 0 to numel parameter - 1 do
        let plus = List.map (fun (n, t) ->
          if n <> name then n, t else
          let copy = tensor t.View.Tensor.view.View.Ndview.shape
            (Array.init (numel t) (value t)) in
          View.Buf.set copy.buf i (value copy i +. epsilon); n, copy) params in
        let minus = List.map (fun (n, t) ->
          if n <> name then n, t else
          let copy = tensor t.View.Tensor.view.View.Ndview.shape
            (Array.init (numel t) (value t)) in
          View.Buf.set copy.buf i (value copy i -. epsilon); n, copy) params in
        let fd = (eval plus -. eval minus) /. (2.0 *. epsilon) in
        let ad = value gradient i in
        let scale = max 1.0 (max (Float.abs fd) (Float.abs ad)) in
        check bool (Printf.sprintf "%s[%d] FD" name i) true
          (Float.abs (ad -. fd) /. scale < 1e-6)
      done)
    params

let () =
  run "VAE preflight"
    [
      ("shape", [test_case "two frame paths" `Quick test_assess_two_frames]);
      ("grad", [test_case "all parameters FD" `Slow test_all_parameter_fd]);
    ]
