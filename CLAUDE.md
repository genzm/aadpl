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

### Transform pipeline

The `Transform.grad` function chains these passes for reverse-mode AD:

1. **Expand_rank** — Eliminates `Rank(k, prim, args)` nodes. Infers shapes, checks leading agreement, inserts broadcasts, shifts axis parameters by frame size.
2. **Desugar** — Fuses adjacent `Apply_view` nodes via viewspec composition.
3. **Forward** — Forward-mode AD AST transform. Emits `(bindings, primal, tangent)`. Captures residuals for nonlinear ops. Tangent vars named `%<var>.t`.
4. **Unzip** — Separates bindings into primal-only vs tangent-dependent. Verifies linearity of tangent part.
5. **Transpose** — Reverse-mode from tangent-linear part. Walks tangent bindings in reverse, propagates cotangents using adjoint rules. Fan-out sums cotangents.

For probabilistic programs, `Transform.build_elbo` additionally uses:
- **Reparam** — Eliminates `D_pushforward` from guide samples, substituting base distributions + forward transforms.
- **Assess_expr** — Builds symbolic log-density expressions for model/guide, handling Jacobian determinants for pushforwards.

### Key type: `expr` (in `lib/ast/types.ml`)

Seven node types: `Const | Var | Prim | Let | Rank | Sample | Score`. No control flow or recursion. `Rank(k, prim, args)` is the rank-polymorphism node — it specifies cell rank `k` and gets eliminated by `Expand_rank`.

### Leading-axis semantics

Shape = frame (leading) ++ cell (trailing). For `Rank(k, ...)`, cell rank is `k`. Agreement: shorter frames must be a prefix of the longest frame; missing leading axes are broadcast. Example: `add⎉1` on shapes `[N]` and `[B,N]` broadcasts `[N]` to `[B,N]` by inserting axis 0.

### AD conventions

- Forward tangent vars: `%<name>.t` for user vars, gensym'd `%<ns><prefix><counter>` for intermediates
- `Forward.reset_gensym()` before each transform pass
- `assess_expr` inlines tangent variables (no Let bindings) to avoid collisions with the outer Forward pass
- Adjoint of `Apply_view` is `Adjoint_view`; adjoint of broadcast (stride=0) is `Sum_axis`
