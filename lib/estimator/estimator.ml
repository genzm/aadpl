module Types = Types
module Strategy = Strategy
module Elbo = Elbo
module Lower_pathwise = Lower_pathwise
module Lower_enumerate = Lower_enumerate
module Lower_score = Lower_score
module Rename = Rename
module Decompose = Decompose
module Lit = Lit
module Batch = Batch

(* State the objective, then choose the estimator:

     Estimator.elbo ~slots ~model ~guide ~env_shapes
     |> Estimator.lower_pathwise

   Which lowering an objective admits is reported by Estimator.strategy.  There
   is deliberately no `Auto`: picking the estimator silently is how a program
   ends up quietly estimating something other than what it says. *)

let elbo = Elbo.elbo
let lower_pathwise = Lower_pathwise.lower
let lower_enumerate = Lower_enumerate.lower
let lower_score = Lower_score.lower
let draw = Lower_score.draw
let noise_env = Lower_pathwise.noise_env

let strategy (objective : Types.t) =
  match objective with
  | Types.Deterministic _ -> Strategy.Pathwise
  | Types.Expect { sites; _ } -> Strategy.of_sites sites
