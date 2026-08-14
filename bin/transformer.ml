(* Phase 14 S-6: a character-level causal Transformer.  Teacher-forced
   training is an ordinary differentiable array program.  Autoregressive
   generation is a separate, non-differentiated Scan with two carries: the
   one-hot token buffer and the write position.  All uniforms are supplied
   outside the Scan. *)

open View
open Ast.Types

type config = {
  batch : int;
  context : int;
  dim : int;
  heads : int;
  blocks : int;
  vocab : int;
}

let scalar value =
  let result = Tensor.make [||] in
  Buf.set result.buf 0 value;
  result

let numel tensor = Ndview.numel tensor.Tensor.view
let scalar_value tensor = Buf.get tensor.Tensor.buf 0

let tensor_of_array shape values =
  let result = Tensor.make shape in
  Array.iteri (fun index value -> Buf.set result.buf index value) values;
  result

let bind name value continuation = let_ name value (continuation (var name))

let dense input weights bias =
  rank 1 Add [rank 2 Matmul [input; var weights]; var bias]

let linear input weights = rank 2 Matmul [input; var weights]

let layer_norm config prefix input continuation =
  bind (prefix ^ ".sum") (prim (Sum_axis 2) [input]) (fun sum ->
  bind (prefix ^ ".mean")
    (rank 0 Div [sum; const (scalar (float_of_int config.dim))]) (fun mean ->
  bind (prefix ^ ".centered")
    (prim Sub [input;
      prim (Apply_view [Vbroadcast (2, config.dim)]) [mean]]) (fun centered ->
  bind (prefix ^ ".sq") (prim Mul [centered; centered]) (fun squared ->
  bind (prefix ^ ".var")
    (rank 0 Div [prim (Sum_axis 2) [squared];
      const (scalar (float_of_int config.dim))]) (fun variance ->
  bind (prefix ^ ".scale")
    (prim Sqrt [rank 0 Add [variance; const (scalar 1e-5)]]) (fun scale ->
  let normalized = prim Div [centered;
    prim (Apply_view [Vbroadcast (2, config.dim)]) [scale]] in
  continuation
    (rank 1 Add [rank 1 Mul [normalized; var (prefix ^ ".gamma")];
      var (prefix ^ ".beta")])))))))

let causal_bias context =
  let result = Tensor.make [|context; context|] in
  for row = 0 to context - 1 do
    for column = 0 to context - 1 do
      Buf.set result.buf (row * context + column)
        (if column <= row then 0.0 else -1e9)
    done
  done;
  result

let softmax_last4 config prefix scores continuation =
  bind (prefix ^ ".max") (prim (Max_axis 3) [scores]) (fun maximum ->
  bind (prefix ^ ".exp")
    (prim Exp [prim Sub [scores;
      prim (Apply_view [Vbroadcast (3, config.context)]) [maximum]]])
    (fun exponentials ->
  bind (prefix ^ ".denom") (prim (Sum_axis 3) [exponentials])
    (fun denominator ->
  continuation (prim Div [exponentials;
    prim (Apply_view [Vbroadcast (3, config.context)]) [denominator]]))))

let attention config prefix input continuation =
  let head_dim = config.dim / config.heads in
  let split value =
    prim (Apply_view [
      Vreshape [|config.batch; config.context; config.heads; head_dim|];
      Vtranspose [|0; 2; 1; 3|];
    ]) [value] in
  bind (prefix ^ ".q") (split (linear input (prefix ^ ".wq"))) (fun query ->
  bind (prefix ^ ".k") (split (linear input (prefix ^ ".wk"))) (fun key ->
  bind (prefix ^ ".v") (split (linear input (prefix ^ ".wv"))) (fun value ->
  bind (prefix ^ ".scores")
    (rank 0 Mul [
      rank 2 Matmul [query;
        prim (Apply_view [Vtranspose [|0; 1; 3; 2|]]) [key]];
      const (scalar (1.0 /. sqrt (float_of_int head_dim)));
    ]) (fun raw_scores ->
  bind (prefix ^ ".masked")
    (rank 2 Add [raw_scores; const (causal_bias config.context)])
    (fun masked_scores ->
  softmax_last4 config (prefix ^ ".softmax") masked_scores (fun weights ->
  bind (prefix ^ ".heads") (rank 2 Matmul [weights; value]) (fun heads ->
  let merged = prim (Apply_view [
    Vtranspose [|0; 2; 1; 3|];
    Vreshape [|config.batch; config.context; config.dim|];
  ]) [heads] in
  continuation (linear merged (prefix ^ ".wo")))))))))

let block config index input continuation =
  let prefix = Printf.sprintf "block%d" index in
  layer_norm config (prefix ^ ".ln1") input (fun normalized ->
  attention config (prefix ^ ".attn") normalized (fun attended ->
  bind (prefix ^ ".residual1") (prim Add [input; attended]) (fun residual1 ->
  layer_norm config (prefix ^ ".ln2") residual1 (fun normalized2 ->
  bind (prefix ^ ".ff.hidden")
    (prim Relu [dense normalized2 (prefix ^ ".ff.w1")
      (prefix ^ ".ff.b1")]) (fun hidden ->
  bind (prefix ^ ".ff.out")
    (dense hidden (prefix ^ ".ff.w2") (prefix ^ ".ff.b2")) (fun output ->
  continuation (prim Add [residual1; output])))))))

let transformer config tokens continuation =
  bind "transformer.embedding" (linear tokens "token.embedding") (fun embedded ->
  bind "transformer.input"
    (rank 2 Add [embedded; var "position.embedding"]) (fun input ->
  let rec blocks index value =
    if index = config.blocks then
      layer_norm config "final.ln" value (fun normalized ->
        continuation (dense normalized "head.w" "head.b"))
    else block config index value (blocks (index + 1)) in
  blocks 0 input))

let log_softmax config logits continuation =
  bind "loss.max" (prim (Max_axis 2) [logits]) (fun maximum ->
  bind "loss.shifted" (prim Sub [logits;
    prim (Apply_view [Vbroadcast (2, config.vocab)]) [maximum]])
    (fun shifted ->
  bind "loss.log.denom"
    (prim Log [prim (Sum_axis 2) [prim Exp [shifted]]]) (fun log_denom ->
  continuation (prim Sub [shifted;
    prim (Apply_view [Vbroadcast (2, config.vocab)]) [log_denom]]))))

let parameter_shapes config =
  let feed_forward = 4 * config.dim in
  let common = [
    ("token.embedding", [|config.vocab; config.dim|]);
    ("position.embedding", [|config.context; config.dim|]);
    ("final.ln.gamma", [|config.dim|]);
    ("final.ln.beta", [|config.dim|]);
    ("head.w", [|config.dim; config.vocab|]);
    ("head.b", [|config.vocab|]);
  ] in
  let per_block index =
    let prefix = Printf.sprintf "block%d" index in [
      (prefix ^ ".ln1.gamma", [|config.dim|]);
      (prefix ^ ".ln1.beta", [|config.dim|]);
      (prefix ^ ".attn.wq", [|config.dim; config.dim|]);
      (prefix ^ ".attn.wk", [|config.dim; config.dim|]);
      (prefix ^ ".attn.wv", [|config.dim; config.dim|]);
      (prefix ^ ".attn.wo", [|config.dim; config.dim|]);
      (prefix ^ ".ln2.gamma", [|config.dim|]);
      (prefix ^ ".ln2.beta", [|config.dim|]);
      (prefix ^ ".ff.w1", [|config.dim; feed_forward|]);
      (prefix ^ ".ff.b1", [|feed_forward|]);
      (prefix ^ ".ff.w2", [|feed_forward; config.dim|]);
      (prefix ^ ".ff.b2", [|config.dim|]);
    ] in
  common @ List.concat (List.init config.blocks per_block)

let training_program config =
  let loss = transformer config (var "tokens") (fun logits ->
    log_softmax config logits (fun log_probabilities ->
      prim Neg [rank 0 Mul [
        const (scalar (1.0 /. float_of_int (config.batch * config.context)));
        prim (Sum_axis 0) [prim (Sum_axis 0) [prim (Sum_axis 0)
          [prim Mul [var "targets"; log_probabilities]]]];
      ]])) in
  let parameters = parameter_shapes config in
  let data = [("tokens", [|config.batch; config.context; config.vocab|]);
    ("targets", [|config.batch; config.context; config.vocab|])] in
  Transform.grad ~param_shapes:parameters ~data_shapes:data loss, parameters

let random_tensor ~site_id ~fan_in ~fan_out shape =
  let result = Tensor.make shape in
  let key = Prng.Threefry.make_key ~run_key:0L
    ~namespace:Prng.Threefry.ns_init in
  let limit = sqrt (6.0 /. float_of_int (fan_in + fan_out)) in
  for index = 0 to numel result - 1 do
    let counter = Prng.Threefry.make_ctr ~site_id ~component:1
      ~frame_index:index in
    let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr:counter in
    Buf.set result.buf index
      ((2.0 *. Prng.Threefry.to_open_unit bits -. 1.0) *. limit)
  done;
  result

let filled shape value =
  let result = Tensor.make shape in
  for index = 0 to numel result - 1 do Buf.set result.buf index value done;
  result

let initial_parameters config =
  let site = ref 64 in
  List.map (fun (name, shape) ->
    let result =
      if Filename.check_suffix name ".gamma" then filled shape 1.0
      else if Filename.check_suffix name ".beta"
           || Filename.check_suffix name ".b1"
           || Filename.check_suffix name ".b2"
           || name = "head.b" then Tensor.make shape
      else
        let rank = Array.length shape in
        let fan_in = if rank < 2 then max 1 shape.(0) else shape.(rank - 2)
        and fan_out = if rank < 2 then max 1 shape.(0) else shape.(rank - 1) in
        let value = random_tensor ~site_id:!site ~fan_in ~fan_out shape in
        incr site;
        value in
    name, result) (parameter_shapes config)

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

let read_file path =
  let channel = open_in_bin path in
  let length = in_channel_length channel in
  let contents = really_input_string channel length in
  close_in channel;
  contents

let vocabulary text =
  let present = Array.make 256 false in
  String.iter (fun character -> present.(Char.code character) <- true) text;
  Array.init 256 Fun.id |> Array.to_list
  |> List.filter (fun code -> present.(code))
  |> List.map Char.chr |> Array.of_list

let encode vocabulary text =
  let inverse = Array.make 256 (-1) in
  Array.iteri (fun index character -> inverse.(Char.code character) <- index)
    vocabulary;
  Array.init (String.length text) (fun index -> inverse.(Char.code text.[index]))

let uniform ~run_key ~site_id ~frame_index =
  let key = Prng.Threefry.make_key ~run_key
    ~namespace:Prng.Threefry.ns_data in
  let counter = Prng.Threefry.make_ctr ~site_id ~component:1 ~frame_index in
  let bits, _ = Prng.Threefry.threefry2x64 ~key ~ctr:counter in
  Prng.Threefry.to_open_unit bits

let training_batch config encoded ~run_key =
  let tokens = Tensor.make [|config.batch; config.context; config.vocab|]
  and targets = Tensor.make [|config.batch; config.context; config.vocab|] in
  let bound = Array.length encoded - config.context - 1 in
  if bound <= 0 then failwith "corpus is shorter than the context window";
  for row = 0 to config.batch - 1 do
    let start = int_of_float (uniform ~run_key ~site_id:9 ~frame_index:row
      *. float_of_int bound) in
    for column = 0 to config.context - 1 do
      let base = (row * config.context + column) * config.vocab in
      Buf.set tokens.buf (base + encoded.(start + column)) 1.0;
      Buf.set targets.buf (base + encoded.(start + column + 1)) 1.0
    done
  done;
  tokens, targets

let equality_mask indices value =
  prim Sub [
    prim Step [rank 0 Add [rank 0 Sub [indices; value]; const (scalar 0.5)]];
    prim Step [rank 0 Sub [rank 0 Sub [indices; value]; const (scalar 0.5)]];
  ]

let generation_program config ~prompt ~new_tokens ~temperature =
  if Array.length prompt + new_tokens > config.context then
    invalid_arg "prompt plus generated tokens exceeds context";
  let initial_buffer = Tensor.make [|config.context; config.vocab|] in
  Array.iteri (fun position token ->
    Buf.set initial_buffer.buf (position * config.vocab + token) 1.0) prompt;
  let positions = tensor_of_array [|config.context|]
    (Array.init config.context float_of_int)
  and token_indices = tensor_of_array [|config.vocab|]
    (Array.init config.vocab float_of_int) in
  let cumulative = Tensor.make [|config.vocab; config.vocab|] in
  for row = 0 to config.vocab - 1 do
    for column = row to config.vocab - 1 do
      Buf.set cumulative.buf (row * config.vocab + column) 1.0
    done
  done;
  let buffer_batch = prim (Apply_view [Vbroadcast (0, 1)]) [var "buffer"] in
  let next_buffer = transformer {config with batch = 1} buffer_batch (fun logits ->
    bind "generate.logits" (prim (Sum_axis 0) [logits]) (fun logits ->
    bind "generate.read.mask"
      (equality_mask (const positions)
        (prim Sub [var "position"; const (scalar 1.0)])) (fun read_mask ->
    bind "generate.row" (prim (Sum_axis 0) [prim Mul [logits;
      prim (Apply_view [Vbroadcast (1, config.vocab)]) [read_mask]]]) (fun row ->
    bind "generate.scaled"
      (rank 0 Div [row; const (scalar temperature)]) (fun scaled ->
    bind "generate.max" (prim (Max_axis 0) [scaled]) (fun maximum ->
    bind "generate.exp" (prim Exp [rank 0 Sub [scaled; maximum]])
      (fun exponentials ->
    bind "generate.probabilities"
      (rank 0 Div [exponentials; prim (Sum_axis 0) [exponentials]])
      (fun probabilities ->
    bind "generate.cdf"
      (prim (Apply_view [Vreshape [|config.vocab|]]) [
        prim Matmul [
          prim (Apply_view [Vreshape [|1; config.vocab|]]) [probabilities];
          const cumulative;
        ]]) (fun cdf ->
    bind "generate.token"
      (prim (Sum_axis 0) [prim Step [rank 0 Sub [var "uniform"; cdf]]])
      (fun token ->
    bind "generate.onehot" (equality_mask (const token_indices) token)
      (fun onehot ->
    bind "generate.write.mask"
      (equality_mask (const positions) (var "position")) (fun write_mask ->
    let mask = prim (Apply_view [Vbroadcast (1, config.vocab)]) [write_mask] in
    rank 0 Add [
      rank 0 Mul [var "buffer";
        rank 0 Sub [const (scalar 1.0); mask]];
      rank 0 Mul [
        prim (Apply_view [Vbroadcast (0, config.context)]) [onehot];
        mask];
    ])))))))))))) in
  let uniforms = Tensor.make [|new_tokens|] in
  for step = 0 to new_tokens - 1 do
    Buf.set uniforms.buf step
      (uniform ~run_key:9000000L ~site_id:10 ~frame_index:step)
  done;
  let expression = scan ~steps:new_tokens ~carries:[
      ("buffer", const initial_buffer, next_buffer);
      ("position", const (scalar (float_of_int (Array.length prompt))),
        prim Add [var "position"; const (scalar 1.0)]);
    ] ~inputs:[("uniform", const uniforms)] ~collect:false ~reverse:false
    (var "buffer") in
  Transform.Expand_rank.expand ~senv:(parameter_shapes config) expression

let decode_buffer vocabulary buffer length =
  String.init length (fun position ->
    let best = ref 0 and best_value = ref neg_infinity in
    for token = 0 to Array.length vocabulary - 1 do
      let value = Buf.get buffer.Tensor.buf
        (position * Array.length vocabulary + token) in
      if value > !best_value then begin best := token; best_value := value end
    done;
    vocabulary.(!best))

let getenv_int name default =
  match Sys.getenv_opt name with Some value -> int_of_string value | None -> default

let () =
  let training_steps = if Array.length Sys.argv > 1
    then int_of_string Sys.argv.(1) else 1000 in
  let corpus_path = if Array.length Sys.argv > 2
    then Sys.argv.(2) else "data/tinyshakespeare.txt" in
  let batch = getenv_int "TRANSFORMER_BATCH" 32
  and context = getenv_int "TRANSFORMER_CONTEXT" 64
  and dim = getenv_int "TRANSFORMER_DIM" 128
  and heads = getenv_int "TRANSFORMER_HEADS" 4
  and blocks = getenv_int "TRANSFORMER_BLOCKS" 2 in
  if dim mod heads <> 0 then invalid_arg "dimension must be divisible by heads";
  let text = read_file corpus_path in
  let characters = vocabulary text in
  let encoded = encode characters text in
  let config = {batch; context; dim; heads; blocks;
    vocab = Array.length characters} in
  Printf.printf "corpus %d chars, vocab %d, batch %d, context %d, dim %d, heads %d, blocks %d\n%!"
    (String.length text) config.vocab batch context dim heads blocks;
  let gradient, shapes = training_program config in
  let parameters = ref (initial_parameters config) in
  let states = List.map (fun (name, shape) ->
    name, {mean = Tensor.make shape; variance = Tensor.make shape}) shapes in
  let started = Unix.gettimeofday () and initial_loss = ref nan in
  for step = 1 to training_steps do
    let tokens, targets = training_batch config encoded
      ~run_key:(Int64.of_int step) in
    let loss, gradients = Ast.Eval.eval_grad
      (("tokens", tokens) :: ("targets", targets) :: !parameters)
      ~primal_bindings:gradient.primal_bindings ~loss_body:gradient.loss_body
      ~grad_bindings:gradient.grad_bindings ~grad_bodies:gradient.grad_bodies in
    if step = 1 then initial_loss := scalar_value loss;
    parameters := adam_descent ~step ~rate:0.0003 states !parameters gradients;
    if step = 1 || step mod 50 = 0 || step = training_steps then
      Printf.printf "step %d/%d loss %.5f\n%!" step training_steps
        (scalar_value loss)
  done;
  let elapsed = Unix.gettimeofday () -. started in
  let full_prompt = "ROMEO:\n" in
  let prompt_text = String.sub full_prompt 0
    (min (String.length full_prompt) (max 1 (context / 4))) in
  let prompt = encode characters prompt_text in
  let new_tokens = min (context - Array.length prompt) 56 in
  let generation_config = {config with batch = 1} in
  let generator = generation_program generation_config ~prompt ~new_tokens
    ~temperature:0.8 in
  let generation_started = Unix.gettimeofday () in
  let buffer = Ast.Eval.eval !parameters generator in
  let generation_elapsed = Unix.gettimeofday () -. generation_started in
  let generated = decode_buffer characters buffer (Array.length prompt + new_tokens) in
  Printf.printf "training %.2fs (%.2f ms/step), initial loss %.5f\n"
    elapsed (elapsed *. 1000.0 /. float_of_int training_steps) !initial_loss;
  Printf.printf "generation %.3fs (%d tokens)\n%s\n"
    generation_elapsed new_tokens generated
