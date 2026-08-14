(* Phase 11-5: MNIST VAE.  build_elbo and grad run once; each training step
   only draws guide noise, evaluates the transformed program, and updates Adam. *)

open View
open Ast.Types

let scalar value =
  let t = Tensor.make [||] in
  Buf.set t.buf 0 value;
  t

let scalar_value t = Buf.get t.Tensor.buf 0
let numel t = Ndview.numel t.Tensor.view

let read_int32_be ic =
  let b3 = input_byte ic and b2 = input_byte ic in
  let b1 = input_byte ic and b0 = input_byte ic in
  (b3 lsl 24) lor (b2 lsl 16) lor (b1 lsl 8) lor b0

let load_idx_images path =
  let ic = open_in_bin path in
  let _magic = read_int32_be ic in
  let n = read_int32_be ic in
  let rows = read_int32_be ic and cols = read_int32_be ic in
  let dim = rows * cols in
  let images = Tensor.make [|n; dim|] in
  for i = 0 to n * dim - 1 do
    Buf.set images.buf i (float_of_int (input_byte ic) /. 255.0)
  done;
  close_in ic;
  images

let find_data_dir () =
  List.find_opt
    (fun dir -> Sys.file_exists (Filename.concat dir "train-images-idx3-ubyte"))
    ["data"; "../data"; "../../data"]

let dense input weights bias =
  rank 1 Add [rank 2 Matmul [input; var weights]; var bias]

let make_program ~batch ~latent ~hidden ~pixels =
  let model =
    let_ "z"
      (sample "z" [|batch; latent|]
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_sigma"))
      (let_ "decoder_h" (prim Relu [dense (var "z") "w3" "b3"])
         (let_ "eta" (dense (var "decoder_h") "w4" "b4")
            (score
               (rank 0 Add
                  [
                    rank 0 Mul [var "x"; prim Logsigmoid [var "eta"]];
                    rank 0 Mul
                      [
                        rank 0 Sub [const (scalar 1.0); var "x"];
                        prim Logsigmoid [prim Neg [var "eta"]];
                      ];
                  ]))))
  in
  let guide =
    let_ "encoder_h" (prim Relu [dense (var "x") "w1" "b1"])
      (let_ "mu_q" (dense (var "encoder_h") "w_mu" "b_mu")
         (let_ "rho_q" (dense (var "encoder_h") "w_rho" "b_rho")
            (let_ "sigma_q" (prim Exp [var "rho_q"])
               (sample "z" [|batch; latent|]
                  (Ast.Normal.normal ~mu:"mu_q" ~sigma:"sigma_q")))))
  in
  let param_shapes =
    [
      ("w1", [|pixels; hidden|]); ("b1", [|hidden|]);
      ("w_mu", [|hidden; latent|]); ("b_mu", [|latent|]);
      ("w_rho", [|hidden; latent|]); ("b_rho", [|latent|]);
      ("w3", [|latent; hidden|]); ("b3", [|hidden|]);
      ("w4", [|hidden; pixels|]); ("b4", [|pixels|]);
    ]
  in
  let data_shapes =
    [("x", [|batch; pixels|]); ("prior_mu", [||]); ("prior_sigma", [||])]
  in
  let program = Transform.build_elbo
    ~slots:[] ~model ~guide ~env_shapes:(param_shapes @ data_shapes) in
  let average_elbo =
    prim Mul [const (scalar (1.0 /. float_of_int batch)); program.elbo]
  in
  let gradient = Transform.grad ~param_shapes
    ~data_shapes:(program.noise @ data_shapes) average_elbo in
  program, average_elbo, gradient, param_shapes, data_shapes

let random_tensor ~site_id ~fan_in shape =
  let result = Tensor.make shape in
  let key = Prng.Threefry.make_key ~run_key:0L
    ~namespace:Prng.Threefry.ns_init in
  let limit = sqrt (6.0 /. float_of_int fan_in) in
  for i = 0 to numel result - 1 do
    let ctr = Prng.Threefry.make_ctr ~site_id ~component:1 ~frame_index:i in
    let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
    let u = Prng.Threefry.to_open_unit bits in
    Buf.set result.buf i ((2.0 *. u -. 1.0) *. limit)
  done;
  result

let initial_params ~latent ~hidden ~pixels =
  [
    ("w1", random_tensor ~site_id:0 ~fan_in:pixels [|pixels; hidden|]);
    ("b1", Tensor.make [|hidden|]);
    ("w_mu", random_tensor ~site_id:1 ~fan_in:hidden [|hidden; latent|]);
    ("b_mu", Tensor.make [|latent|]);
    ("w_rho", random_tensor ~site_id:2 ~fan_in:hidden [|hidden; latent|]);
    ("b_rho", Tensor.make [|latent|]);
    ("w3", random_tensor ~site_id:3 ~fan_in:latent [|latent; hidden|]);
    ("b3", Tensor.make [|hidden|]);
    ("w4", random_tensor ~site_id:4 ~fan_in:hidden [|hidden; pixels|]);
    ("b4", Tensor.make [|pixels|]);
  ]

type adam_state = { m : Tensor.t; v : Tensor.t }

let adam_ascent ~step ~learning_rate states params gradients =
  let beta1 = 0.9 and beta2 = 0.999 and epsilon = 1e-8 in
  let correction1 = 1.0 -. beta1 ** float_of_int step in
  let correction2 = 1.0 -. beta2 ** float_of_int step in
  List.map
    (fun (name, (parameter : Tensor.t)) ->
      let gradient : Tensor.t = List.assoc name gradients in
      let state = List.assoc name states in
      let updated = Tensor.make parameter.view.Ndview.shape in
      for i = 0 to numel parameter - 1 do
        let g = Buf.get gradient.buf i in
        let m = beta1 *. Buf.get state.m.buf i +. (1.0 -. beta1) *. g in
        let v = beta2 *. Buf.get state.v.buf i +. (1.0 -. beta2) *. g *. g in
        Buf.set state.m.buf i m;
        Buf.set state.v.buf i v;
        Buf.set updated.buf i
          (Buf.get parameter.buf i +. learning_rate *. (m /. correction1)
             /. (sqrt (v /. correction2) +. epsilon))
      done;
      name, updated)
    params

let shuffle epoch n =
  let indices = Array.init n Fun.id in
  let key = Prng.Threefry.make_key ~run_key:(Int64.of_int epoch)
    ~namespace:Prng.Threefry.ns_data in
  for i = n - 1 downto 1 do
    let ctr = Prng.Threefry.make_ctr ~site_id:0 ~component:1 ~frame_index:i in
    let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
    let j = Int64.to_int (Int64.rem (Int64.logand bits Int64.max_int)
      (Int64.of_int (i + 1))) in
    let tmp = indices.(i) in
    indices.(i) <- indices.(j);
    indices.(j) <- tmp
  done;
  indices

(* Dynamic binarization: each epoch draws x_ij ~ Bernoulli(pixel_ij).
   The Bernoulli Score is therefore a normalized log density, so the reported
   objective is an actual ELBO.  site_id=1 separates these counters from the
   ns_data shuffle counters (site_id=0). *)
let batch_from ~run_key (images : Tensor.t) indices offset batch pixels =
  let result = Tensor.make [|batch; pixels|] in
  let key = Prng.Threefry.make_key ~run_key
    ~namespace:Prng.Threefry.ns_data in
  for row = 0 to batch - 1 do
    let source = indices.(offset + row) in
    for col = 0 to pixels - 1 do
      let ctr = Prng.Threefry.make_ctr ~site_id:1 ~component:1
        ~frame_index:(source * pixels + col) in
      let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr in
      let draw = Prng.Threefry.to_open_unit bits in
      Buf.set result.buf (row * pixels + col)
        (if Buf.get images.buf (source * pixels + col) > draw then 1.0 else 0.0)
    done
  done;
  result

let sequential_batch ~run_key (images : Tensor.t) offset batch pixels =
  batch_from ~run_key images
    (Array.init images.view.Ndview.shape.(0) Fun.id) offset batch pixels

let reconstruction_expr ~param_shapes ~data_shapes =
  let expression =
    let_ "encoder_h" (prim Relu [dense (var "x") "w1" "b1"])
      (let_ "z_mean" (dense (var "encoder_h") "w_mu" "b_mu")
         (let_ "decoder_h" (prim Relu [dense (var "z_mean") "w3" "b3"])
            (let_ "eta" (dense (var "decoder_h") "w4" "b4")
               (prim Exp [prim Logsigmoid [var "eta"]]))))
  in
  Transform.Desugar.fuse_views
    (Transform.Expand_rank.expand ~senv:(param_shapes @ data_shapes) expression)

let latent_kl_expr ~batch ~param_shapes ~data_shapes =
  let half = const (scalar 0.5) and one = const (scalar 1.0) in
  let two = const (scalar 2.0) in
  let expression =
    let_ "encoder_h" (prim Relu [dense (var "x") "w1" "b1"])
      (let_ "mu_q" (dense (var "encoder_h") "w_mu" "b_mu")
         (let_ "rho_q" (dense (var "encoder_h") "w_rho" "b_rho")
            (let_ "kl_cells"
               (rank 0 Mul
                  [
                    half;
                    rank 0 Sub
                      [
                        rank 0 Add
                          [
                            rank 0 Mul [var "mu_q"; var "mu_q"];
                            prim Exp [rank 0 Mul [two; var "rho_q"]];
                          ];
                        rank 0 Add [one; rank 0 Mul [two; var "rho_q"]];
                      ];
                  ])
               (rank 0 Mul
                  [
                    const (scalar (1.0 /. float_of_int batch));
                    prim (Sum_axis 0) [var "kl_cells"];
                  ]))))
  in
  Transform.Desugar.fuse_views
    (Transform.Expand_rank.expand ~senv:(param_shapes @ data_shapes) expression)

let generator_expr ~batch ~latent ~param_shapes ~data_shapes =
  let expression =
    let_ "z"
      (sample "z" [|batch; latent|]
         (Ast.Normal.normal ~mu:"prior_mu" ~sigma:"prior_sigma"))
      (let_ "decoder_h" (prim Relu [dense (var "z") "w3" "b3"])
         (let_ "eta" (dense (var "decoder_h") "w4" "b4")
            (prim Exp [prim Logsigmoid [var "eta"]])))
  in
  Transform.Expand_rank.expand ~senv:(param_shapes @ data_shapes) expression

let write_ppm path ~rows ~cols (images : Tensor.t) =
  let oc = open_out_bin path in
  Printf.fprintf oc "P6\n%d %d\n255\n" (cols * 28) (rows * 28);
  for grid_row = 0 to rows - 1 do
    for pixel_row = 0 to 27 do
      for grid_col = 0 to cols - 1 do
        let image = grid_row * cols + grid_col in
        for pixel_col = 0 to 27 do
          let index = image * 784 + pixel_row * 28 + pixel_col in
          let shade = int_of_float
            (255.0 *. max 0.0 (min 1.0 (Buf.get images.Tensor.buf index))) in
          output_byte oc shade; output_byte oc shade; output_byte oc shade
        done
      done
    done
  done;
  close_out oc

let ensure_dir path =
  if Sys.file_exists path then begin
    if not (Sys.is_directory path) then failwith (path ^ " is not a directory")
  end else Unix.mkdir path 0o755

let () =
  let epochs = if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1) else 5 in
  let output_dir = if Array.length Sys.argv > 2 then Sys.argv.(2) else "vae-output" in
  let batch_limit =
    if Array.length Sys.argv > 3 then Some (int_of_string Sys.argv.(3)) else None in
  let profile = Sys.getenv_opt "VAE_PROFILE" = Some "1" in
  let data_dir = match find_data_dir () with
    | Some dir -> dir
    | None -> failwith "MNIST data not found"
  in
  let train = load_idx_images (Filename.concat data_dir "train-images-idx3-ubyte") in
  let test = load_idx_images (Filename.concat data_dir "t10k-images-idx3-ubyte") in
  let batch = 128 and latent = 20 and hidden = 400 and pixels = 784 in
  let program, average_elbo, gradient, param_shapes, data_shapes =
    make_program ~batch ~latent ~hidden ~pixels in
  let params = ref (initial_params ~latent ~hidden ~pixels) in
  let states = List.map (fun (name, shape) ->
    name, {m = Tensor.make shape; v = Tensor.make shape}) param_shapes in
  let fixed = [("prior_mu", scalar 0.0); ("prior_sigma", scalar 1.0)] in
  let step = ref 0 in
  let evaluate (images : Tensor.t) n_batches run_key =
    let total = ref 0.0 in
    for bi = 0 to n_batches - 1 do
      let batch_key = Int64.add run_key (Int64.of_int bi) in
      let x = sequential_batch ~run_key:batch_key images
        (bi * batch) batch pixels in
      let env = Transform.noise_env program
        ~run_key:batch_key
        @ (("x", x) :: !params @ fixed) in
      total := !total +. scalar_value (Ast.Eval.eval env average_elbo)
    done;
    !total /. float_of_int n_batches
  in
  let initial_holdout = evaluate test 10 1000000L in
  Printf.printf "MNIST VAE 784-%d-%d, batch %d\n" hidden latent batch;
  Printf.printf "initial holdout ELBO: %.3f\n%!" initial_holdout;
  for epoch = 1 to epochs do
    let indices = shuffle epoch train.view.Ndview.shape.(0) in
    let available = train.view.Ndview.shape.(0) / batch in
    let n_batches = match batch_limit with
      | None -> available | Some limit -> min limit available in
    let epoch_elbo = ref 0.0 in
    if profile then begin Ast.Eval.reset_stats (); Ast.Eval.enable_stats () end;
    let started = Unix.gettimeofday () in
    for bi = 0 to n_batches - 1 do
      incr step;
      let x = batch_from ~run_key:(Int64.of_int epoch) train indices
        (bi * batch) batch pixels in
      let env = Transform.noise_env program ~run_key:(Int64.of_int !step)
        @ (("x", x) :: !params @ fixed) in
      let elbo, gradients = Ast.Eval.eval_grad env
        ~primal_bindings:gradient.primal_bindings
        ~loss_body:gradient.loss_body ~grad_bindings:gradient.grad_bindings
        ~grad_bodies:gradient.grad_bodies in
      epoch_elbo := !epoch_elbo +. scalar_value elbo;
      params := adam_ascent ~step:!step ~learning_rate:0.001
        states !params gradients
    done;
    let elapsed = Unix.gettimeofday () -. started in
    if profile then begin Ast.Eval.disable_stats (); Ast.Eval.report () end;
    let train_elbo = !epoch_elbo /. float_of_int n_batches in
    let holdout = evaluate test 10 (Int64.of_int (1000000 + epoch * 100)) in
    Printf.printf
      "epoch %d: train ELBO %.3f, holdout %.3f, %.1fs, %.1f ms/step\n%!"
      epoch train_elbo holdout elapsed
      (elapsed *. 1000.0 /. float_of_int n_batches)
  done;
  ensure_dir output_dir;
  let x = sequential_batch ~run_key:2000000L test 0 batch pixels in
  let reconstruction = reconstruction_expr ~param_shapes ~data_shapes
    |> Ast.Eval.eval (("x", x) :: !params @ fixed) in
  write_ppm (Filename.concat output_dir "reconstruction.ppm")
    ~rows:8 ~cols:8 reconstruction;
  let generator = generator_expr ~batch ~latent ~param_shapes ~data_shapes in
  let generated, _, _ = Ast.Simulate.simulate ~run_key:9000000L
    (!params @ fixed) generator in
  write_ppm (Filename.concat output_dir "prior.ppm")
    ~rows:8 ~cols:8 generated;
  let kl_expr = latent_kl_expr ~batch ~param_shapes ~data_shapes in
  let kl_sums = Array.make latent 0.0 in
  for bi = 0 to 9 do
    let x = sequential_batch ~run_key:(Int64.of_int (3000000 + bi)) test
      (bi * batch) batch pixels in
    let kl = Ast.Eval.eval (("x", x) :: !params @ fixed) kl_expr in
    for dim = 0 to latent - 1 do
      kl_sums.(dim) <- kl_sums.(dim) +. Buf.get kl.buf dim
    done
  done;
  let active = ref 0 in
  Array.iter (fun total -> if total /. 10.0 > 0.01 then incr active) kl_sums;
  Printf.printf "active latent dimensions (mean KL > 0.01): %d/%d\n"
    !active latent;
  Printf.printf "wrote %s and %s\n"
    (Filename.concat output_dir "reconstruction.ppm")
    (Filename.concat output_dir "prior.ppm")
