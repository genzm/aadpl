(* Scalar literals, shared by the lowerings.  Each call allocates, so no two
   constants in a program share a buffer. *)

let tensor value =
  let result = View.Tensor.make [||] in
  View.Buf.set result.buf 0 value;
  result

let scalar value = Ast.Types.const (tensor value)
let zero () = scalar 0.0

(* Sum a list of scalar-or-frame terms with leading agreement. *)
let sum = function
  | [] -> zero ()
  | first :: rest ->
      List.fold_left
        (fun acc term -> Ast.Types.rank 0 Ast.Types.Add [ acc; term ])
        first rest
