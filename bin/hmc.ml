(* Phase 14 S-7: fixed-step HMC as two bounded sequential axes.  The inner
   Scan performs leapfrog integration; the outer Scan performs Metropolis
   transitions and collects all chains.  Momentum and acceptance uniforms
   are supplied outside both scans.  The sampler itself is not differentiated. *)

open View
open Ast.Types

let scalar value =
  let result = Tensor.make [||] in
  Buf.set result.buf 0 value;
  result

let tensor_of_array shape values =
  let result = Tensor.make shape in
  Array.iteri (fun index value -> Buf.set result.buf index value) values;
  result

let potential_cells ~observed ~noise position =
  let inverse_variance = 1.0 /. (noise *. noise) in
  rank 0 Add [
    rank 0 Mul [const (scalar 0.5); rank 0 Mul [position; position]];
    rank 0 Mul [const (scalar (0.5 *. inverse_variance));
      let difference = rank 0 Sub [const (scalar observed); position] in
      rank 0 Mul [difference; difference]];
  ]

let potential_gradient ~chains ~observed ~noise =
  let loss = prim (Sum_axis 0)
    [potential_cells ~observed ~noise (var "position")] in
  let program = Transform.grad ~param_shapes:[("position", [|chains|])] loss in
  List.assoc "position" program.grads

let hmc_program ~chains ~iterations ~leapfrog_steps ~step_size
    ~observed ~noise initial momenta uniforms =
  let gradient = potential_gradient ~chains ~observed ~noise in
  let gradient_at position = let_ "position" position gradient in
  let half_step = 0.5 *. step_size in
  let factors = Tensor.make [|leapfrog_steps|] in
  for step = 0 to leapfrog_steps - 1 do
    Buf.set factors.buf step
      (if step = leapfrog_steps - 1 then half_step else step_size)
  done;
  let next_position = rank 0 Add [var "lf.position";
    rank 0 Mul [const (scalar step_size); var "lf.momentum"]] in
  let next_momentum = rank 0 Sub [var "lf.momentum";
    rank 0 Mul [var "lf.gradient.step"; gradient_at next_position]] in
  let initial_momentum = rank 0 Sub [var "proposal.momentum";
    rank 0 Mul [const (scalar half_step); gradient]] in
  let proposal =
    scan ~steps:leapfrog_steps ~carries:[
        ("lf.position", var "position", next_position);
        ("lf.momentum", initial_momentum, next_momentum);
      ] ~inputs:[("lf.gradient.step", const factors)]
      ~collect:false ~reverse:false
      (let old_energy = rank 0 Add [
         potential_cells ~observed ~noise (var "position");
         rank 0 Mul [const (scalar 0.5);
           rank 0 Mul [var "proposal.momentum"; var "proposal.momentum"]];
       ] in
       let new_energy = rank 0 Add [
         potential_cells ~observed ~noise (var "lf.position");
         rank 0 Mul [const (scalar 0.5);
           rank 0 Mul [var "lf.momentum"; var "lf.momentum"]];
       ] in
       let accept = prim Step [rank 0 Sub [
         rank 0 Sub [old_energy; new_energy]; prim Log [var "accept.uniform"]]] in
       rank 0 Add [
         rank 0 Mul [accept; var "lf.position"];
         rank 0 Mul [rank 0 Sub [const (scalar 1.0); accept]; var "position"];
       ]) in
  let expression = scan ~steps:iterations
    ~carries:[("position", const initial, proposal)]
    ~inputs:[
      ("proposal.momentum", const momenta);
      ("accept.uniform", const uniforms);
    ] ~collect:true ~reverse:false (var "position") in
  Transform.Expand_rank.expand expression

let random_inputs ~iterations ~chains =
  let momenta = Tensor.make [|iterations; chains|]
  and uniforms = Tensor.make [|iterations; chains|] in
  let key = Prng.Threefry.make_key ~run_key:1407000L
    ~namespace:Prng.Threefry.ns_data in
  for iteration = 0 to iterations - 1 do
    for chain = 0 to chains - 1 do
      let index = iteration * chains + chain in
      let counter = Prng.Threefry.make_ctr ~site_id:11 ~component:1
        ~frame_index:index in
      let first, second = Prng.Threefry.threefry2x64 ~key ~ctr:counter in
      let radius = sqrt (-2.0 *. log (Prng.Threefry.to_open_unit first)) in
      let angle = 2.0 *. Float.pi *. Prng.Threefry.to_open_unit second in
      Buf.set momenta.buf index (radius *. cos angle);
      let accept_counter = Prng.Threefry.make_ctr ~site_id:12 ~component:1
        ~frame_index:index in
      let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr:accept_counter in
      Buf.set uniforms.buf index (Prng.Threefry.to_open_unit bits)
    done
  done;
  momenta, uniforms

let diagnostics trajectory ~burn_in =
  let shape = trajectory.Tensor.view.Ndview.shape in
  let iterations = shape.(0) and chains = shape.(1) in
  let samples = iterations - burn_in in
  let chain_means = Array.make chains 0.0
  and chain_variances = Array.make chains 0.0 in
  for chain = 0 to chains - 1 do
    for iteration = burn_in to iterations - 1 do
      chain_means.(chain) <- chain_means.(chain)
        +. Buf.get trajectory.buf (iteration * chains + chain)
    done;
    chain_means.(chain) <- chain_means.(chain) /. float_of_int samples;
    for iteration = burn_in to iterations - 1 do
      let difference = Buf.get trajectory.buf (iteration * chains + chain)
        -. chain_means.(chain) in
      chain_variances.(chain) <- chain_variances.(chain)
        +. difference *. difference
    done;
    chain_variances.(chain) <- chain_variances.(chain)
      /. float_of_int (samples - 1)
  done;
  let mean = Array.fold_left ( +. ) 0.0 chain_means /. float_of_int chains in
  let within = Array.fold_left ( +. ) 0.0 chain_variances
    /. float_of_int chains in
  let between = ref 0.0 in
  Array.iter (fun chain_mean ->
    between := !between +. (chain_mean -. mean) *. (chain_mean -. mean))
    chain_means;
  let between = float_of_int samples *. !between /. float_of_int (chains - 1) in
  let variance_hat = float_of_int (samples - 1) /. float_of_int samples *. within
    +. between /. float_of_int samples in
  let pooled_variance = ref 0.0 in
  for chain = 0 to chains - 1 do
    for iteration = burn_in to iterations - 1 do
      let difference = Buf.get trajectory.buf (iteration * chains + chain) -. mean in
      pooled_variance := !pooled_variance +. difference *. difference
    done
  done;
  let pooled_variance = !pooled_variance
    /. float_of_int (samples * chains - 1) in
  let accepted = ref 0 in
  for chain = 0 to chains - 1 do
    for iteration = 1 to iterations - 1 do
      if Buf.get trajectory.buf (iteration * chains + chain)
         <> Buf.get trajectory.buf ((iteration - 1) * chains + chain)
      then incr accepted
    done
  done;
  mean, pooled_variance, sqrt (variance_hat /. within),
  float_of_int !accepted /. float_of_int ((iterations - 1) * chains)

let () =
  let chains = 4 and iterations = 2000 and burn_in = 500
  and leapfrog_steps = 6 and step_size = 0.25 in
  let observed = 1.2 and noise = 0.7 in
  let initial = tensor_of_array [|chains|] [|-2.0; 0.0; 2.0; 4.0|] in
  let momenta, uniforms = random_inputs ~iterations ~chains in
  let expression = hmc_program ~chains ~iterations ~leapfrog_steps ~step_size
    ~observed ~noise initial momenta uniforms in
  let started = Unix.gettimeofday () in
  let trajectory = Ast.Eval.eval [] expression in
  let elapsed = Unix.gettimeofday () -. started in
  let mean, variance, rhat, acceptance = diagnostics trajectory ~burn_in in
  let expected_mean = observed /. (1.0 +. noise *. noise)
  and expected_variance = noise *. noise /. (1.0 +. noise *. noise) in
  Printf.printf "HMC %d chains x %d iterations, %d leapfrog: %.3fs\n"
    chains iterations leapfrog_steps elapsed;
  Printf.printf "mean %.6f (closed %.6f), variance %.6f (closed %.6f)\n"
    mean expected_mean variance expected_variance;
  Printf.printf "R-hat %.6f, acceptance %.3f\n" rhat acceptance;
  if Float.abs (mean -. expected_mean) > 0.03 then
    failwith "HMC posterior mean does not match the conjugate posterior";
  if Float.abs (variance -. expected_variance) > 0.03 then
    failwith "HMC posterior variance does not match the conjugate posterior";
  if rhat > 1.02 then failwith "HMC chains did not mix"
