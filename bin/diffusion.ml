(* Phase 14 S-5: a 2D diffusion model.  Training is ordinary transformed
   array code; only generation uses reverse Scan, with all noise supplied as
   a [time; sample; coordinate] input and the complete trajectory collected. *)

open View
open Ast.Types

let scalar value =
  let tensor = Tensor.make [||] in
  Buf.set tensor.buf 0 value;
  tensor

let numel tensor = Ndview.numel tensor.Tensor.view
let scalar_value tensor = Buf.get tensor.Tensor.buf 0

let dense input weights bias =
  rank 1 Add [prim Matmul [input; var weights]; var bias]

let epsilon_network input time =
  let_ "eps.h1"
    (prim Relu [rank 1 Add [dense input "w1" "b1";
      prim Matmul [time; var "wt"]]])
    (let_ "eps.h2" (prim Relu [dense (var "eps.h1") "w2" "b2"])
      (dense (var "eps.h2") "w3" "b3"))

let uniform ~run_key ~site_id ~frame_index =
  let key = Prng.Threefry.make_key ~run_key
    ~namespace:Prng.Threefry.ns_data in
  let counter = Prng.Threefry.make_ctr ~site_id ~component:1 ~frame_index in
  let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr:counter in
  Prng.Threefry.to_open_unit bits

let normal_pair ~run_key ~site_id ~frame_index =
  let key = Prng.Threefry.make_key ~run_key
    ~namespace:Prng.Threefry.ns_data in
  let counter = Prng.Threefry.make_ctr ~site_id ~component:1 ~frame_index in
  let first, second = Prng.Threefry.threefry2x64 ~key ~ctr:counter in
  let radius = sqrt (-2.0 *. log (Prng.Threefry.to_open_unit first)) in
  let angle = 2.0 *. Float.pi *. Prng.Threefry.to_open_unit second in
  radius *. cos angle, radius *. sin angle

let schedule steps =
  let beta = Tensor.make [|steps|] in
  for step = 0 to steps - 1 do
    let fraction = float_of_int step /. float_of_int (steps - 1) in
    (* With only 100 steps, beta=0.02 would leave alpha_bar_T around 0.36,
       inconsistent with generation from N(0,I).  This schedule ends below
       0.01 while keeping every transition well-conditioned. *)
    Buf.set beta.buf step (1e-4 +. fraction *. (0.1 -. 1e-4))
  done;
  let alpha_expression = rank 0 Sub [const (scalar 1.0); const beta] in
  let alpha = Transform.Expand_rank.expand alpha_expression |> Ast.Eval.eval [] in
  let alpha_bar_expression =
    scan ~steps
      ~carries:[("alpha_bar", const (scalar 1.0),
        prim Mul [var "alpha_bar"; var "alpha"])]
      ~inputs:[("alpha", const alpha)] ~collect:true ~reverse:false
      (var "alpha_bar") in
  let alpha_bar = Ast.Eval.eval [] alpha_bar_expression in
  beta, alpha, alpha_bar

let time_features time =
  let result = Tensor.make [|4|] in
  Buf.set result.buf 0 time;
  Buf.set result.buf 1 (time *. time);
  Buf.set result.buf 2 (sin (2.0 *. Float.pi *. time));
  Buf.set result.buf 3 (cos (2.0 *. Float.pi *. time));
  result

let make_training_program ~batch ~hidden =
  let loss = let_ "predicted" (epsilon_network (var "x_t") (var "time"))
    (let_ "difference" (rank 0 Sub [var "predicted"; var "epsilon"])
      (rank 0 Mul [const (scalar (1.0 /. float_of_int (batch * 2)));
        prim (Sum_axis 0) [prim (Sum_axis 0)
          [rank 0 Mul [var "difference"; var "difference"]]]])) in
  let parameter_shapes = [
    ("w1", [|2; hidden|]); ("wt", [|4; hidden|]); ("b1", [|hidden|]);
    ("w2", [|hidden; hidden|]); ("b2", [|hidden|]);
    ("w3", [|hidden; 2|]); ("b3", [|2|]);
  ] in
  let data_shapes = [("x_t", [|batch; 2|]); ("time", [|batch; 4|]);
    ("epsilon", [|batch; 2|])] in
  let gradient = Transform.grad ~param_shapes:parameter_shapes ~data_shapes loss in
  gradient, parameter_shapes

let random_tensor ~site_id ~fan_in shape =
  let result = Tensor.make shape in
  let key = Prng.Threefry.make_key ~run_key:0L
    ~namespace:Prng.Threefry.ns_init in
  let limit = sqrt (6.0 /. float_of_int fan_in) in
  for index = 0 to numel result - 1 do
    let counter = Prng.Threefry.make_ctr ~site_id ~component:1
      ~frame_index:index in
    let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr:counter in
    let value = 2.0 *. Prng.Threefry.to_open_unit bits -. 1.0 in
    Buf.set result.buf index (value *. limit)
  done;
  result

let initial_parameters hidden = [
  ("w1", random_tensor ~site_id:32 ~fan_in:2 [|2; hidden|]);
  ("wt", random_tensor ~site_id:33 ~fan_in:4 [|4; hidden|]);
  ("b1", Tensor.make [|hidden|]);
  ("w2", random_tensor ~site_id:34 ~fan_in:hidden [|hidden; hidden|]);
  ("b2", Tensor.make [|hidden|]);
  ("w3", random_tensor ~site_id:35 ~fan_in:hidden [|hidden; 2|]);
  ("b3", Tensor.make [|2|]);
]

type adam_state = { mean : Tensor.t; variance : Tensor.t }

let adam_descent ~step ~rate states parameters gradients =
  let beta1 = 0.9 and beta2 = 0.999 and epsilon = 1e-8 in
  let correction1 = 1.0 -. beta1 ** float_of_int step
  and correction2 = 1.0 -. beta2 ** float_of_int step in
  List.map (fun (name, (parameter : Tensor.t)) ->
    let gradient : Tensor.t = List.assoc name gradients in
    let state = List.assoc name states in
    let updated = Tensor.make parameter.Tensor.view.Ndview.shape in
    for index = 0 to numel parameter - 1 do
      let g = Buf.get gradient.buf index in
      let mean = beta1 *. Buf.get state.mean.buf index +. (1.0 -. beta1) *. g in
      let variance = beta2 *. Buf.get state.variance.buf index
        +. (1.0 -. beta2) *. g *. g in
      Buf.set state.mean.buf index mean;
      Buf.set state.variance.buf index variance;
      Buf.set updated.buf index (Buf.get parameter.buf index
        -. rate *. (mean /. correction1)
          /. (sqrt (variance /. correction2) +. epsilon))
    done;
    name, updated) parameters

let training_batch ~run_key ~batch ~diffusion_steps alpha_bar =
  let x_t = Tensor.make [|batch; 2|]
  and epsilon = Tensor.make [|batch; 2|]
  and time = Tensor.make [|batch; 4|] in
  for row = 0 to batch - 1 do
    let position = uniform ~run_key ~site_id:3 ~frame_index:row in
    let angle = 4.0 *. Float.pi *. position in
    let radius = 0.15 +. 1.85 *. position in
    let jitter_x, jitter_y = normal_pair ~run_key ~site_id:4 ~frame_index:row in
    let x0 = radius *. cos angle +. 0.04 *. jitter_x
    and y0 = radius *. sin angle +. 0.04 *. jitter_y in
    let timestep = min (diffusion_steps - 1)
      (int_of_float (uniform ~run_key ~site_id:5 ~frame_index:row
        *. float_of_int diffusion_steps)) in
    let noise_x, noise_y = normal_pair ~run_key ~site_id:6 ~frame_index:row in
    let alpha_bar = Buf.get alpha_bar.Tensor.buf timestep in
    let signal = sqrt alpha_bar and noise = sqrt (1.0 -. alpha_bar) in
    Buf.set x_t.buf (row * 2) (signal *. x0 +. noise *. noise_x);
    Buf.set x_t.buf (row * 2 + 1) (signal *. y0 +. noise *. noise_y);
    Buf.set epsilon.buf (row * 2) noise_x;
    Buf.set epsilon.buf (row * 2 + 1) noise_y;
    let features = time_features
      (float_of_int timestep /. float_of_int (diffusion_steps - 1)) in
    for feature = 0 to 3 do
      Buf.set time.buf (row * 4 + feature) (Buf.get features.buf feature)
    done
  done;
  x_t, time, epsilon

let generation_program ~steps ~samples ~hidden beta alpha alpha_bar =
  let coefficient shape value_at =
    let result = Tensor.make shape in
    for step = 0 to steps - 1 do
      for sample = 0 to samples - 1 do
        for coordinate = 0 to 1 do
          Buf.set result.buf ((step * samples + sample) * 2 + coordinate)
            (value_at step)
        done
      done
    done;
    result in
  let inv_sqrt_alpha = coefficient [|steps; samples; 2|]
    (fun step -> 1.0 /. sqrt (Buf.get alpha.Tensor.buf step))
  and beta_over_noise = coefficient [|steps; samples; 2|]
    (fun step -> Buf.get beta.Tensor.buf step
      /. sqrt (1.0 -. Buf.get alpha_bar.Tensor.buf step))
  and sigma = coefficient [|steps; samples; 2|]
    (fun step -> if step = 0 then 0.0 else
      let beta_t = Buf.get beta.Tensor.buf step
      and alpha_bar_t = Buf.get alpha_bar.Tensor.buf step
      and alpha_bar_previous = Buf.get alpha_bar.Tensor.buf (step - 1) in
      sqrt (beta_t *. (1.0 -. alpha_bar_previous) /. (1.0 -. alpha_bar_t))) in
  let time = Tensor.make [|steps; samples; 4|] in
  for step = 0 to steps - 1 do
    let features = time_features
      (float_of_int step /. float_of_int (steps - 1)) in
    for sample = 0 to samples - 1 do
      for feature = 0 to 3 do
        Buf.set time.buf ((step * samples + sample) * 4 + feature)
          (Buf.get features.buf feature)
      done
    done
  done;
  let noise = Tensor.make [|steps; samples; 2|] in
  for step = 0 to steps - 1 do
    for sample = 0 to samples - 1 do
      let first, second = normal_pair ~run_key:9000000L ~site_id:7
        ~frame_index:(step * samples + sample) in
      Buf.set noise.buf ((step * samples + sample) * 2) first;
      Buf.set noise.buf ((step * samples + sample) * 2 + 1) second
    done
  done;
  let initial = Tensor.make [|samples; 2|] in
  for sample = 0 to samples - 1 do
    let first, second = normal_pair ~run_key:9000001L ~site_id:8
      ~frame_index:sample in
    Buf.set initial.buf (sample * 2) first;
    Buf.set initial.buf (sample * 2 + 1) second
  done;
  let predicted = epsilon_network (var "state") (var "time.cell") in
  let mean = rank 0 Mul [var "inv.sqrt.alpha";
    rank 0 Sub [var "state";
      rank 0 Mul [var "beta.over.noise"; predicted]]] in
  let next = rank 0 Add [mean; rank 0 Mul [var "sigma"; var "noise"]] in
  let expression = scan ~steps
    ~carries:[("state", const initial, next)]
    ~inputs:[
      ("inv.sqrt.alpha", const inv_sqrt_alpha);
      ("beta.over.noise", const beta_over_noise);
      ("sigma", const sigma);
      ("time.cell", const time);
      ("noise", const noise);
    ] ~collect:true ~reverse:true (var "state") in
  let parameter_shapes = [
    ("w1", [|2; hidden|]); ("wt", [|4; hidden|]); ("b1", [|hidden|]);
    ("w2", [|hidden; hidden|]); ("b2", [|hidden|]);
    ("w3", [|hidden; 2|]); ("b3", [|2|]);
  ] in
  Transform.Expand_rank.expand ~senv:parameter_shapes expression

let target_points count =
  let result = Tensor.make [|count; 2|] in
  for row = 0 to count - 1 do
    let position = uniform ~run_key:7000000L ~site_id:3 ~frame_index:row in
    let angle = 4.0 *. Float.pi *. position in
    let radius = 0.15 +. 1.85 *. position in
    let jitter_x, jitter_y = normal_pair ~run_key:7000000L ~site_id:4
      ~frame_index:row in
    Buf.set result.buf (row * 2) (radius *. cos angle +. 0.04 *. jitter_x);
    Buf.set result.buf (row * 2 + 1) (radius *. sin angle +. 0.04 *. jitter_y)
  done;
  result

let write_scatter path panels =
  let panel_size = 256 and margin = 12 in
  let width = panel_size * List.length panels and height = panel_size in
  let pixels = Bytes.make (width * height * 3) '\255' in
  let set_pixel x y red green blue =
    if x >= 0 && x < width && y >= 0 && y < height then begin
      let index = (y * width + x) * 3 in
      Bytes.set pixels index (Char.chr red);
      Bytes.set pixels (index + 1) (Char.chr green);
      Bytes.set pixels (index + 2) (Char.chr blue)
    end in
  List.iteri (fun panel points ->
    let count = points.Tensor.view.Ndview.shape.(0) in
    for row = 0 to count - 1 do
      let x = Buf.get points.buf (row * 2)
      and y = Buf.get points.buf (row * 2 + 1) in
      let px = panel * panel_size + margin
        + int_of_float ((x +. 2.6) /. 5.2 *. float_of_int (panel_size - 2 * margin))
      and py = margin
        + int_of_float ((2.6 -. y) /. 5.2 *. float_of_int (panel_size - 2 * margin)) in
      for dx = -1 to 1 do for dy = -1 to 1 do
        set_pixel (px + dx) (py + dy) 25 65 130
      done done
    done) panels;
  let channel = open_out_bin path in
  Printf.fprintf channel "P6\n%d %d\n255\n" width height;
  output_bytes channel pixels;
  close_out channel

let trajectory_cell trajectory step =
  let samples = trajectory.Tensor.view.Ndview.shape.(1) in
  let result = Tensor.make [|samples; 2|] in
  for index = 0 to samples * 2 - 1 do
    Buf.set result.buf index (Buf.get trajectory.buf (step * samples * 2 + index))
  done;
  result

let ensure_directory path =
  if Sys.file_exists path then begin
    if not (Sys.is_directory path) then failwith (path ^ " is not a directory")
  end else Unix.mkdir path 0o755

let () =
  let training_steps = if Array.length Sys.argv > 1
    then int_of_string Sys.argv.(1) else 3000 in
  let output_directory = if Array.length Sys.argv > 2
    then Sys.argv.(2) else "diffusion-output" in
  let batch = 256 and hidden = 128 and diffusion_steps = 100 in
  let beta, alpha, alpha_bar = schedule diffusion_steps in
  Printf.printf "schedule alpha_bar_T %.6f\n%!"
    (Buf.get alpha_bar.buf (diffusion_steps - 1));
  let gradient, parameter_shapes = make_training_program ~batch ~hidden in
  let parameters = ref (initial_parameters hidden) in
  let states = List.map (fun (name, shape) ->
    name, {mean = Tensor.make shape; variance = Tensor.make shape})
    parameter_shapes in
  let started = Unix.gettimeofday () and initial_loss = ref nan in
  for step = 1 to training_steps do
    let x_t, time, epsilon = training_batch ~run_key:(Int64.of_int step)
      ~batch ~diffusion_steps alpha_bar in
    let loss, gradients = Ast.Eval.eval_grad
      (("x_t", x_t) :: ("time", time) :: ("epsilon", epsilon) :: !parameters)
      ~primal_bindings:gradient.primal_bindings ~loss_body:gradient.loss_body
      ~grad_bindings:gradient.grad_bindings ~grad_bodies:gradient.grad_bodies in
    if step = 1 then initial_loss := scalar_value loss;
    parameters := adam_descent ~step ~rate:0.001 states !parameters gradients;
    if step = 1 || step mod 250 = 0 || step = training_steps then
      Printf.printf "step %d/%d loss %.5f\n%!" step training_steps
        (scalar_value loss)
  done;
  let elapsed = Unix.gettimeofday () -. started in
  let samples = 2048 in
  let generator = generation_program ~steps:diffusion_steps ~samples ~hidden
    beta alpha alpha_bar in
  let generation_started = Unix.gettimeofday () in
  let trajectory = Ast.Eval.eval !parameters generator in
  let generation_elapsed = Unix.gettimeofday () -. generation_started in
  let generated = trajectory_cell trajectory (diffusion_steps - 1)
  and target = target_points samples in
  ensure_directory output_directory;
  write_scatter (Filename.concat output_directory "spiral.ppm")
    [target; generated];
  let indices = [0; diffusion_steps / 5; 2 * diffusion_steps / 5;
    3 * diffusion_steps / 5; 4 * diffusion_steps / 5; diffusion_steps - 1] in
  write_scatter (Filename.concat output_directory "trajectory.ppm")
    (List.map (trajectory_cell trajectory) indices);
  Printf.printf "training %.2fs (%.2f ms/step), initial loss %.5f\n"
    elapsed (elapsed *. 1000.0 /. float_of_int training_steps) !initial_loss;
  Printf.printf "generation %.3fs (%d steps x %d samples)\n"
    generation_elapsed diffusion_steps samples;
  Printf.printf "wrote %s and %s\n"
    (Filename.concat output_directory "spiral.ppm")
    (Filename.concat output_directory "trajectory.ppm")
