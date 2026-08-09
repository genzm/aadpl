module type S = sig
  val map1 :
    f:(float -> float) ->
    src:View.Buf.t -> view:View.Ndview.view -> dst:View.Buf.t -> unit

  val map2 :
    f:(float -> float -> float) ->
    src1:View.Buf.t -> view1:View.Ndview.view ->
    src2:View.Buf.t -> view2:View.Ndview.view ->
    dst:View.Buf.t -> unit

  val sum_axis :
    src:View.Buf.t -> view:View.Ndview.view -> axis:int ->
    dst:View.Buf.t -> unit

  val max_axis :
    src:View.Buf.t -> view:View.Ndview.view -> axis:int ->
    dst:View.Buf.t -> dst_argmax:int array -> unit

  val gather :
    src:View.Buf.t -> view:View.Ndview.view -> axis:int ->
    indices:int array -> dst:View.Buf.t -> unit

  val scatter_add :
    src:View.Buf.t -> view:View.Ndview.view -> axis:int ->
    indices:int array -> acc:View.Buf.t -> unit

  val matmul :
    a:View.Buf.t -> view_a:View.Ndview.view ->
    b:View.Buf.t -> view_b:View.Ndview.view ->
    dst:View.Buf.t -> nframe:int -> unit
end

module Naive = Naive
