(* LogNormal(mu, sigma) as exp pushed forward over Normal(mu, sigma). *)

open Types

let log_normal ~mu ~sigma : dist =
  D_pushforward
    {
      fwd_var = "%d.log_u";
      fwd = prim Exp [var "%d.log_u"];
      inv_var = "%d.log_x";
      inv = prim Log [var "%d.log_x"];
      support = S_positive;
      base = Normal.normal ~mu ~sigma;
    }
