module Types = Types
module Elbo = Elbo
module Lower_pathwise = Lower_pathwise

(* The two halves of what Transform.build_elbo used to be, in the intended order
   of use: state the objective, then choose the estimator.

     Estimator.elbo ~slots ~model ~guide ~env_shapes
     |> Estimator.lower_pathwise *)

let elbo = Elbo.elbo
let lower_pathwise = Lower_pathwise.lower
let noise_env = Lower_pathwise.noise_env
