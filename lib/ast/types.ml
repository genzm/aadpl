type loc = { file : string; line : int; col : int }

let dummy_loc = { file = "<none>"; line = 0; col = 0 }

type prim =
  (* map1 *)
  | Neg | Exp | Log | Sqrt | Relu
  (* map2 — require shape match *)
  | Add | Sub | Mul | Div | Max2
  (* reduce *)
  | Sum_axis of int | Max_axis of int
  (* structural *)
  | Transpose of int array
  | Reshape of int array
  | Broadcast of int * int   (* axis, size *)
  | Slice of (int * int * int) array
  | Gather of int * int array (* axis, indices — integer, no tangent (§3.1) *)
  (* linear algebra *)
  | Matmul

type expr =
  | Const of loc * value
  | Var   of loc * string
  | Prim  of loc * prim * expr list
  | Let   of loc * string * expr * expr

and value = View.Tensor.t   (* future: variant with dtype *)

let loc_of = function
  | Const (l, _) | Var (l, _) | Prim (l, _, _) | Let (l, _, _, _) -> l

(* --- constructors with dummy_loc for tests --- *)

let const v       = Const (dummy_loc, v)
let var s         = Var (dummy_loc, s)
let prim p args   = Prim (dummy_loc, p, args)
let let_ s e body = Let (dummy_loc, s, e, body)

(* --- pretty-printer --- *)

let pp_prim fmt = function
  | Neg -> Format.fprintf fmt "neg"
  | Exp -> Format.fprintf fmt "exp"
  | Log -> Format.fprintf fmt "log"
  | Sqrt -> Format.fprintf fmt "sqrt"
  | Relu -> Format.fprintf fmt "relu"
  | Add -> Format.fprintf fmt "add"
  | Sub -> Format.fprintf fmt "sub"
  | Mul -> Format.fprintf fmt "mul"
  | Div -> Format.fprintf fmt "div"
  | Max2 -> Format.fprintf fmt "max2"
  | Sum_axis a -> Format.fprintf fmt "sum_axis(%d)" a
  | Max_axis a -> Format.fprintf fmt "max_axis(%d)" a
  | Transpose perm ->
    Format.fprintf fmt "transpose(%a)"
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
         Format.pp_print_int)
      (Array.to_list perm)
  | Reshape shape ->
    Format.fprintf fmt "reshape(%a)"
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
         Format.pp_print_int)
      (Array.to_list shape)
  | Broadcast (axis, size) -> Format.fprintf fmt "broadcast(%d,%d)" axis size
  | Slice ranges ->
    Format.fprintf fmt "slice(%a)"
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ";")
         (fun fmt (start, stop, step) ->
            Format.fprintf fmt "%d:%d:%d" start stop step))
      (Array.to_list ranges)
  | Gather (axis, indices) ->
    Format.fprintf fmt "gather(%d,[%a])" axis
      (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ",")
         Format.pp_print_int)
      (Array.to_list indices)
  | Matmul -> Format.fprintf fmt "matmul"

let rec pp fmt = function
  | Const (_, v) ->
    Format.fprintf fmt "(const %a)" View.Ndview.pp v.View.Tensor.view
  | Var (_, s) -> Format.fprintf fmt "%s" s
  | Prim (_, p, args) ->
    Format.fprintf fmt "(%a %a)" pp_prim p
      (Format.pp_print_list ~pp_sep:Format.pp_print_space pp) args
  | Let (_, s, e, body) ->
    Format.fprintf fmt "(let %s = %a in@;<1 2>%a)" s pp e pp body
