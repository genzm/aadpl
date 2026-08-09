# 実装ロードマップ

**状態**: 着手可能
**前提資料**: 意味論 第4稿 / 実装設計 第1枚（view 層）/ VAE 紙上構成 / 階層モデル紙上構成

---

## 0. 資料の全体像

| 資料 | 役割 | 状態 |
|---|---|---|
| 元の設計文書（21章） | 最初の構想 | 固定。参照のみ |
| **意味論 第4稿** | 何を計算しているかの定義 | 固まった。未決2件 |
| **実装設計 第1枚（view 層）** | $\sigma$ の実装設計 | 固まった |
| VAE 紙上構成 | 意味論の検証① | 完了。穴は反映済み |
| 階層モデル紙上構成 | 意味論の検証② | 完了。穴は反映済み |
| **本書** | これから何をするか | — |

**原則**: 意味論は妥協しない。実装はその限定された断片でよい。覆っていない部分はエラーとして拒否する（黙って近似しない）。

---

## 1. 目標

| | 内容 |
|---|---|
| **最終目標** | 配列指向で AD と PPL がネイティブな言語。機械学習が実際に回る |
| **北極星①** | MNIST 上の VAE（ML 側） |
| **北極星②** | 階層ロジスティック回帰 + 尤度比検定（頻度論側） |
| **北極星③** | 同じモデルでベイズファクター（ベイズ側） |

②と③が**同じモデル記述**を使うことが要点。頻度論とベイズの違いは、trace 型の各部分への操作の違いに帰着する。

---

## 2. 貫く規律

各 Phase で必ず守る。**これが最も重要な部分。**

### (a) 各 Phase の終わりに厳密に検査できるものがあること

確率的プログラムは間違っていても動く。検査できない状態を積み上げない。

### (b) 内積等式を四度使う

$$\langle L(v),\, u\rangle = \langle v,\, L^{*}(u)\rangle$$

| Phase | 対象 |
|---|---|
| 2 | view と `add_view` |
| 3 | gather / scatter_add、sum / broadcast |
| 7 | プログラム全体（JVP と VJP） |
| 12 | `Restrict` を含む確率プログラム |

**最初に検査コードを丁寧に書けば、以後ほぼ流用できる。**

### (c) Naive 実装を捨てない

Phase 3 の素朴なカーネルは、Phase 9 で Owl 版の正しさを検査する基準になる。

### (d) 早い段階で入れるもの（後からは高くつく）

- AST ノードのソース位置（型定義に1フィールド）
- 所有の印（`mutable shared` を base に）
- pretty-printer と `.ocamlinit` の `#install_printer`
- 名前空間つき乱数キー（`init` / `model` / `data`）

### (e) 妥協してはいけない一点

**Phase 5 の `⎉` をループで実装しない。** frame 軸をプリミティブに押し込む書き換えとして実装する。ループにすると Phase 8 までは動くが Phase 9 で性能が出ず、原因特定が困難になる。

---

## 3. 環境構築

```bash
# 一度だけ
brew install opam gpatch direnv
opam init --bare -a -y

# プロジェクトのスイッチ（flambda は事実上必須）
opam switch create lang ocaml-variants.5.4.1+options ocaml-option-flambda
eval $(opam env --switch=lang)
opam install dune ocaml-lsp-server ocamlformat utop \
             alcotest qcheck qcheck-alcotest ppx_expect

# プロジェクト
mkdir lang && cd lang
dune init proj lang
git init
```

`.envrc`:
```bash
export OPAMSWITCH=lang
eval $(opam env --switch=lang --set-switch)
```
```bash
direnv allow
```

**Owl はまだ入れない**（Phase 9 まで不要。型が染み出す誘惑を断つ）。

### 判断の記録

- **5.5.0 ではなく 5.4.1**: 周辺ツール（Merlin/LSP）の安定性。5.4.1 は macOS の `-pack` と文字列操作の誤コンパイル修正を含む
- **`dune pkg` は使わない**: プレビュー段階。`dune install` 未対応。Phase 9 以降に再評価
- **flambda**: 素朴なカーネルループのインライン展開に効く。後から切り替えるとビルド設定を触ることになる

### 構成

```
lang/
├── dune-project
├── .ocamlformat        # profile = default
├── .ocamlinit          # #install_printer
├── .envrc
├── lib/
│   ├── view/           # Phase 1-2
│   ├── kernel/         # Phase 3   KERNEL sig + Naive
│   ├── kernel_owl/     # Phase 9   ← 別ライブラリ。core は依存しない
│   ├── ast/            # Phase 4
│   └── transform/      # Phase 6-7
├── bin/
└── test/
```

`dune-project` のプロファイル:
```
(env
 (dev     (flags (:standard -w +a-4-9-40-41-42-44-45-70)))
 (release (ocamlopt_flags (:standard -O3 -unsafe))))
```

**`-unsafe` は release のみ。** dev で境界検査を落とすと、添字バグがメモリ破壊として現れて特定が困難になる。

---

## 4. Phase 一覧

### 基盤（Phase 0–3）

---

#### Phase 0 — 足場（半日）

**作る**: dune プロジェクト、テストが走る状態、git

**完了条件**: `dune build && dune runtest` が通る

---

#### Phase 1 — view の代数（2〜3日）

**作る**

```ocaml
type view = { shape : int array; strides : int array; offset : int }

val broadcast    : t -> axis:int -> size:int -> t   (* stride 0 を挿入 *)
val transpose    : t -> perm:int array -> t
val slice        : t -> ranges:(int*int*int) array -> t
val reshape      : t -> shape:int array -> t
val is_injective : view -> bool
val index_of     : view -> int array -> int          (* 参照実装 *)
```

すべて O(rank) の算術。**データには触らない。**

**同時に入れる**: 所有の印（base 側に `mutable shared`）、pretty-printer、`.ocamlinit`

**検査**

- `index_of` を基準に、materialize した結果と照合（view をランダム生成、数百通り）
- `is_injective` を小さい shape で全数検査（像の重複を数える）

**完了条件**: QCheck の property が 500 通り通る

---

#### Phase 2 — `add_view` と随伴性（2日）

**最重要ステップ。** ここで AD の土台が確定する。

**作る**

```ocaml
val read_view : t -> t -> unit    (* 実体化 *)
val add_view  : t -> t -> unit    (* 唯一の随伴規則 *)
```

`add_view` は一般の scatter-add ループのみ。**高速路は書かない。**

**検査**

$$\langle \texttt{read\_view}\ v\ x,\ g\rangle = \langle x,\ \texttt{add\_view}\ v\ g\rangle$$

broadcast・transpose・slice・reshape とその合成をランダム生成して 500 通り。

**完了条件**: 上が通る。**この時点で、AD を一行も書かずに AD の正しさの半分が保証される。**

**確認できること**: 意味論 §4.5 の「唯一の規則」が実装で成立している。§4.6（rank 下の共有）が stride 0 の随伴として得られる。

---

#### Phase 3 — 数値カーネル Naive（3〜4日）

**作る**: `map1` / `map2` / `sum_axis` / `max_axis` / `gather` / `scatter_add` / `matmul`

全部素朴な OCaml ループ。**速度は完全に無視。**

**検査**

- `gather` ↔ `scatter_add` の随伴性（内積等式）
- `sum_axis` ↔ `broadcast` の随伴性（内積等式）
- `matmul` を小行列で手計算と照合

**完了条件**: 上が通る。**この実装は永久に残す**（Phase 9 の検査基準）

---

### 言語（Phase 4–5）

---

#### Phase 4 — AST と参照インタプリタ（3〜4日）

**作る**

```ocaml
type expr =
  | Const of value
  | Var   of string
  | Prim  of prim * expr list
  | Let   of string * expr * expr
  | Rank  of int * expr * expr list
```

プリミティブ 10〜15 個: `add`, `mul`, `matmul`, `sum`, `max`, `exp`, `log`, `transpose`, `reshape`, `broadcast`, `gather`

**分布も確率もまだ入れない。** 決定的な配列言語として動かす。

**必ず入れる**: ノードにソース位置

**検査**: 手書きの式が期待値を返す。線形代数の恒等式

---

#### Phase 5 — `⎉` の展開（2〜3日）

**ここで妥協しない。**

rank を「frame 軸をプリミティブに押し込む書き換え」として実装する。共有される値には frame 軸に stride 0 を挿入する。

**確認すべき性質**: Dense 層の rank は**単一の gemm（M = batch）**になる。バッチ行列積は不要。

**検査**

$$\texttt{F⎉r} \text{ の結果} = \text{各セルに素朴に適用して結合した結果}$$

素朴なループ版を**テスト専用**に別途書き、厳密一致を確認する。

---

### AD（Phase 6–8）

---

#### Phase 6 — 前進モード AD（3日）

**作る**: 全プリミティブの JVP 規則

整数・真偽値には tangent を作らない（意味論 §3.1、$T(\mathtt{Z}) = \mathbf{0}$）。

**検査**: 有限差分照合（$\epsilon = 10^{-5}$、相対誤差 $10^{-6}$）

---

#### Phase 7 — unzip と転置（4〜5日）

**このプロジェクトで最も難しい部分。**

意味論 §4.2 の三分解:

$$\text{前進モード AD} \to \text{unzip} \to \text{転置}$$

**転置規則が必要なのは線形プリミティブのみ。** 総数 6 個（view, gather, sum, broadcast, matmul, conv）

線形性の印は当面 AST のフラグでよい。**Linear A 相当の部分構造型は後回し**（正しさは検査で担保）。

**検査**

$$\langle \texttt{jvp}\ f\ x\ v,\ u\rangle = \langle v,\ \texttt{vjp}\ f\ x\ u\rangle$$

Phase 2 と同じ形の検査が、プログラム全体に効く。加えて勾配と有限差分の照合。

---

#### Phase 8 — 線形回帰（1〜2日）

**第一の到達点。**

損失が下がり、閉形式解に収束する。Naive カーネルなので遅いが、小さい問題なら動く。

---

### 性能（Phase 9）

---

#### Phase 9 — Owl 導入と MLP（3〜4日）

**第二の到達点。ここで性能の仮説が検証される。**

> 厚いプリミティブ + BLAS なら、素朴な木歩きインタプリタでも MNIST は回る

**手順**

1. `KERNEL` シグネチャを切り、既存を `module Naive : KERNEL` に
2. `module Owl_backed : KERNEL`（`gemm` と要素ごと）
3. **Naive と Owl_backed の出力一致を検査**（乱数固定、厳密比較）
4. MNIST ローダ（idx 形式、約50行）
5. PPM 出力（約30行）
6. MLP を学習

**注意**

- buffer 型は Bigarray のまま。**Owl の型を使わない**
- 要素ごと演算は手書き OCaml ループではなく Owl の C 実装を使う
- `matmul` の入口で gemm 適合性を検査（$\min(|s_0|,|s_1|)=1$ かつ stride 正）。不適合なら実体化。**最初は常に実体化でよい**

**仮説が外れた場合**: 設計を見直す。早めに来るのが良い理由がこれ。

---

### 確率（Phase 10–13）

---

#### Phase 10 — 分布代数と trace（4〜5日）

**作る**

- 原始分布 2 つ: `Uniform[0,1)`, `Categorical(w)`
- 構成子: `Pushforward`, `Product`（`Restrict` はまだ）
- `Normal = Pushforward(Φ⁻¹, Uniform)`（`Φ⁻¹` は Owl の `erfinv` から）
- アドレス（構造化タプル、文字列にしない）
- **trace 型**
- 名前空間つき乱数キー（counter-based PRNG。Threefry 相当）
- `?` と `Score`

**確認すべき性質**: `?¨` で $n$ 個引くとアドレスが $n$ 個できる。`Product` はプログラム層では使わない

**検査**

- 低次元で `logDensity` を数値積分して 1 になるか
- 同じ `runKey` で同じ trace が出るか
- アドレスが $\theta$ に依存しないか

---

#### Phase 11 — `_reparam` と VAE（1週間）

**第三の到達点。**

- `_reparam`（定義の展開として実装）
- キー引き継ぎ規則
- ELBO（trace 型を介してモデルとガイドを対応させる）
- レコードへの map（modifier として）
- `logsigmoid` などの融合規則

**確認すべき性質**: VAE では戦略の注釈が一切不要（全サイトが構成から pathwise と判定される）

**検査**

- `_reparam` 前後で、同じ `runKey` に対し**同じ値と同じ密度**
- ELBO の勾配と有限差分の照合
- 再構成画像が読める（PPM）

---

#### Phase 12 — `Restrict` と階層モデル（1週間）

**作る**

- `Restrict(μ, A, Z)` 構成子
- 固定予算 $K$ の棄却サンプラ（$K$ 試行は**ランク1の配列演算**）
- `HalfNormal = Restrict(Normal(0,σ), x>0, 1/2)`
- 固定反復の optimizer（modifier）
- ragged のパディング + マスク

**検査**

- 内積等式（`Restrict` を含むプログラム）
- **SBC**（事前から引く → データ生成 → 推論 → 順位の一様性）
- 受容率の実測と $Z$ の照合

---

#### Phase 13 — 求積と検定（1週間）

**作る**

- `_quadrature`（Gauss–Hermite。$K$ 節点の固定軸）
- 特殊関数（Owl の `gammainc` / `betainc`）
- 尤度比検定（境界なので $\tfrac12\chi^2_0 + \tfrac12\chi^2_1$）
- パラメトリックブートストラップ（二重 rank）
- `_importance`（**重みの配列を返す**。$\log Z$ ではない）
- ベイズファクター（`logsumexp` の差）

**確認すべき性質**

- 頻度論とベイズが**同じモデル記述**から出る
- $N \times M \times K$ の三重が全て静的に有界で、完全にベクトル化される

---

## 5. 全体の形

```
Phase 0–3    基盤     約 1.5 週   ← 内積等式
Phase 4–5    言語     約 1.5 週   ← 素朴版との一致
Phase 6–8    AD       約 2 週     ← 内積等式・有限差分
Phase 9      性能     約 1 週     ← Naive との一致
─────────────────────────────────────────
             ここまでで MLP が MNIST で学習（約 6 週相当）

Phase 10–11  確率     約 2 週     ← VAE
Phase 12–13  統計     約 2 週     ← 検定
```

個人プロジェクトの実時間なら 2〜3 倍を見るべきだが、**順序と検査の構造は変わらない**。

---

## 6. 到達点とリスク

| 到達点 | Phase | 検証されること |
|---|---|---|
| 線形回帰の学習 | 8 | AD が正しい |
| **MNIST で MLP** | **9** | **性能の仮説** |
| **VAE** | **11** | **意味論 §21 の主張（配列・AD・PPL・学習の連続性）** |
| 尤度比検定 | 13 | 頻度論とベイズの統一 |

### 主なリスク

| リスク | 顕在化する Phase | 対処 |
|---|---|---|
| インタプリタが遅い | 9 | Phase 5 を正しく作れば回避できる |
| unzip/転置が複雑になる | 7 | 内積等式で早期に検出。線形型は後回しでよい |
| Owl が 5.4 で動かない | 9 | 別スイッチで 4.14.3 に落とす（opam の利点） |
| trace 型の設計が足りない | 11 | 紙上構成で二度検証済み。残余リスクは低い |

---

## 7. 保留事項

### 意味論の未決（2件）

| | 内容 | 影響 |
|---|---|---|
| $T(M\,\tau)$ の定義 | 測度の接空間。自然勾配に必要 | **VAE・階層モデルには不要**。書き換え規則で済む見込み |
| 特殊関数 | 不完全ガンマ・ベータ | Phase 13。**Owl で解決済み** |

### 実装の未決

- conv の view 表現（MLP には不要）
- `reshape` が実体化を要する条件の正確な判定
- buffer の再利用（liveness 解析）— **AD の後に行う**。optimizer の状態だけなら解析不要で 8 割の利益

### 再考条件（採用しなかったものを再検討する条件）

| 選択肢 | 再考条件 |
|---|---|
| Stochastic Probabilities | 密度が扱えない変分族 / 真に再帰的な入れ子推論 / **MC による周辺化** |
| quasi-Borel spaces | 関数を値にする（一階性の放棄） |
| Dex の Accum 効果 | コア IR を index comprehension まで分解する |
| `Maximize` を構成子に | 離散決定変数の分枝限定など、最適化の意味論的解析が必要になったとき |
| `dune pkg` | Phase 9 以降に再評価 |
| 記号的簡約器 | 合成分布（$\exp_* N$ 等）の密度計算が遅くなったとき |

---

## 8. 明日やること

1. 環境構築（§3 のコマンド）
2. `lib/view/` に `view.ml` の型定義
3. `broadcast` / `transpose` を書く
4. `index_of` を書く
5. QCheck の view ジェネレータを書く
6. Phase 1 の property を通す

**5 に時間をかける価値がある。** view のランダム生成器は Phase 1・2・3・7 の四度使う。縮小（shrinking）が効くので、失敗時に最小の反例が読める形で出てくる。
