# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
dune build                          # build everything
dune runtest                        # run all tests
dune test test/test_view.exe        # run a single test suite
dune exec bin/main.exe              # run the CLI
```

OCaml 5.4+, dune 3.0+. Tests use Alcotest + QCheck.

Release builds use `-O3 -unsafe`; dev builds enable `-w +a-4-9-40-41-42-44-45-70`.

## Architecture

This is an **array language** with automatic differentiation and probabilistic programming, implemented as an OCaml compiler/interpreter. The design follows **BQN-style leading-axis agreement** (frame is leading, cell rank is trailing).

### Library layers (bottom → top)

| Library | Purpose |
|---|---|
| `view` | Dense arrays via affine index maps. `Buf.t` (bigarray float64), `Ndview.view` (shape/strides/offset), `Tensor.t` (buf+view). All structural ops are O(rank), no data copy. Broadcast uses stride=0. |
| `kernel` | Bulk ops interface (`Kernel.S`) + pure-OCaml reference impl (`Naive`). Ops: map1/map2, sum/max axis, gather, scatter_add, matmul. |
| `blas_stub` | C FFI to macOS Accelerate. Pluggable BLAS backend. |
| `prng` | Threefry-2x64 counter-based PRNG. Counter encodes site_id, component (D_product tree), frame_index. |
| `ast` | Core types + interpreters. `Types` (expr, prim, viewspec, dist), `Eval` (deterministic), `Jvp` (forward-mode AD), `Simulate` (probabilistic with Threefry sampling), `Sites` (static site enumeration), `Normal` (Gaussian via D_pushforward). |
| `transform` | AST→AST passes (see pipeline below). |
| `estimator` | Estimator IR: which statistical quantity is wanted (`Elbo`), separated from how it is estimated (`Lower_pathwise`, `Lower_enumerate`). `Strategy` classifies sites. Depends on `transform`. |

### Layering

```
Model AST  --(Estimator.elbo)-->  Estimator IR  --(lowering)-->  Tensor AST
                                                                     |
                                        Transform.grad: Forward → Unzip → Transpose
```

`estimator` states the objective (`E_{z~q}[log p − log q]`); a lowering picks the estimator and
produces an ordinary tensor program. `transform` never depends on `estimator` — the dune library
boundary enforces the direction.

| Lowering | Applies to | Result |
|---|---|---|
| `lower_pathwise` | `D_pushforward` / `D_uniform` sites | noise lifted to free `%u.<site>` vars; `grad` with them in `~data_shapes` is the pathwise gradient |
| `lower_enumerate` | scalar `D_categorical` sites | `noise = []`; sums `q(z)·f(z)` over the joint support, so `grad` gives the **exact** gradient of an expectation |
| `lower_score` | any sampled site | draws `%tr.<site>` supplied by `Estimator.draw`; emits a surrogate whose value is `f` and whose gradient is the REINFORCE estimator |

The surrogate is `f + Σᵢ stopgrad(mᵢ − b)·(log qᵢ − stopgrad(log qᵢ))`. Each bracket evaluates to zero, so
the program still **reports the objective** while differentiating like the estimator; a bare
`f + stopgrad(f)·log q` would differentiate correctly and then report a number nobody asked for.
`Lower_score.options` sets `mᵢ`: `loss_to_go` attributes to site *i* only the cost drawn at or after it
(valid because `E[c·∇log qᵢ | z_<ᵢ] = 0` for earlier `c`), and `baselines` subtracts a per-site control
variate. A baseline may be a constant, a parameter, or an expression in the draws made *before* its site —
never one that reads its own site or a later one, which `lower_score` checks rather than assumes. Note that
a baseline sits inside `stopgrad`, so `dL/db = 0`: fitting one is a separate objective, written by the caller.

`lower_enumerate` comes in two forms. `replicate` (the default) gives each assignment its own copy of the
body — simple, but the program grows with the product of the supports, so it refuses past `max_assignments`.
`~batched:true` binds each trace to the column of its coordinates and lifts the body **once** to that axis
via `Batch.lift`, which reuses `Expand_rank.shift_prim` — the batch axis *is* a frame axis, so there is no
second implementation of the axis arithmetic. The replicating form is kept as the reference the batched one
is checked against (value and every gradient, 1e-12), on the principle that keeps `Kernel.Naive` next to BLAS.

`lower_enumerate` is the golden reference for `lower_score`: on a finite support, `Σ_z q(z)·grad_score(z)`
must equal `grad_enumerate` exactly, so unbiasedness is a **deterministic** test rather than a statistical
one — and so is variance, which makes "loss-to-go reduces variance" a checkable claim rather than a hope.
That is why enumeration was built first.

`Expect` carries `log_density` (the proposal's own `log q`) alongside `body`, because an estimator that
differentiates the *measure* needs it and one that differentiates the *path* does not. It also carries
`supports`, resolved once by the objective — a categorical's size needs shape inference over the proposal,
including its `Let`-bound locals (`Reparam.local_shapes`), which no lowering can redo on its own.

`Decompose.split` recovers the additive terms `assess_expr` summed, hoisting its gensymmed `Let`s; that is
what lets a score term drop the cost it cannot have influenced. A lowering that uses a sub-expression twice
must `Rename.binders` one copy — `Unzip` requires every binding in a program to be unique.

`Estimator.strategy` reports which lowering an objective admits, derived from the *constructors*
of the distribution algebra (`Strategy.of_dist`) rather than from a per-distribution table — so
Normal/LogNormal/HalfNormal are all covered by the one `D_pushforward → Pathwise` line. There is
deliberately no `Auto`: the caller names the estimator.

Preconditions follow the split. A discrete site is a valid *objective*; only `lower_pathwise`
rejects it (via `Reparam.check_guide`). `Estimator.elbo` checks only what makes the proposal a
distribution (no `Score`, distinct sites).

A categorical's support size is the last axis of its weights. `ast` cannot call shape inference
(that lives in `transform`), so `Ast.Sites.dist_support` takes an optional `?categorical_size`
resolver; `Reparam.categorical_size env_shapes` supplies it from `infer_shape`, and `Ast.Assess`
supplies it by evaluating the weights. Without a resolver the size stays unknown and
`dist_support` refuses rather than guessing — so `Categorical(exp logits)` works, while weights
bound by a `Let` inside the model (invisible to `env_shapes`) are still rejected.

### Transform pipeline

The `Transform.grad` function chains these passes for reverse-mode AD:

1. **Expand_rank** — Eliminates `Rank(k, prim, args)` nodes. Infers shapes, checks leading agreement, inserts broadcasts, shifts axis parameters by frame size.
2. **Desugar** — Fuses adjacent `Apply_view` nodes via viewspec composition.
3. **Forward** — Forward-mode AD AST transform. Emits `(bindings, primal, tangent)`. Captures residuals for nonlinear ops. Tangent vars named `%<var>.t`.
4. **Unzip** — Separates bindings into primal-only vs tangent-dependent. Verifies linearity of tangent part.
5. **Transpose** — Reverse-mode from tangent-linear part. Walks tangent bindings in reverse, propagates cotangents using adjoint rules. Fan-out sums cotangents.

For probabilistic programs, the `estimator` layer additionally uses:
- **Assess_expr** — Builds symbolic log-density expressions for model/guide, handling Jacobian determinants for pushforwards. Used by `Estimator.elbo` to form the integrand.
- **Reparam** — Eliminates `D_pushforward` from guide samples, substituting base distributions + forward transforms. Used by `Estimator.lower_pathwise`, not by the objective.

### Key type: `expr` (in `lib/ast/types.ml`)

Eight node types: `Const | Var | Prim | Let | Scan | Rank | Sample | Score`. No control flow or recursion. `Rank(k, prim, args)` is the rank-polymorphism node — it specifies cell rank `k` and gets eliminated by `Expand_rank`. `Scan` is the bounded sequential axis (static `steps`).

`Stop_gradient` is a `prim`: the identity on values, zero on tangents. `Forward` keeps the node in the primal, so differentiating twice keeps stopping rather than silently resuming. It is what lets a surrogate objective be written as ordinary array code.

### Leading-axis semantics

Shape = frame (leading) ++ cell (trailing). For `Rank(k, ...)`, cell rank is `k`. Agreement: shorter frames must be a prefix of the longest frame; missing leading axes are broadcast. Example: `add⎉1` on shapes `[N]` and `[B,N]` broadcasts `[N]` to `[B,N]` by inserting axis 0.

### AD conventions

- Forward tangent vars: `%<name>.t` for user vars, gensym'd `%<ns><prefix><counter>` for intermediates
- `Forward.reset_gensym()` before each transform pass
- `assess_expr` inlines tangent variables (no Let bindings) to avoid collisions with the outer Forward pass
- Adjoint of `Apply_view` is `Adjoint_view`; adjoint of broadcast (stride=0) is `Sum_axis`
