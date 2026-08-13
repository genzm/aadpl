# 意味論 第2枚：変換のパイプライン

**状態**: 第1稿
**位置づけ**: 言語仕様。第1枚（値・型・接空間・測度 rev4）が「何を計算しているか」を定めるのに対し、この文書は**変換が互いにどう噛み合うか**を定める。
**根拠**: Phase 0–11 の実装。紙の上で先に決めたのではなく、動いている実装から抽出して固定する。

---

## 0. この文書が存在する理由

第1枚 rev4 の直後に「第2枚は変換のパイプライン——各変換の事前条件・事後条件・順序の固定。**silent な誤りの最大の温床**」と予告して、そのまま実装に入った。Phase 10–11 で見つかった欠陥を並べる。

| 欠陥 | 種類 |
|---|---|
| `dist_kind` が `D_pushforward` の base を見ない | 不変条件（構成からの導出）の破れ |
| `wrap_rank0` が rank $k>0$ を黙って rank 0 に落とす | **事前条件が守られていない** |
| `elim_samples` が α 一意性を暗黙に仮定 | **暗黙の事前条件** |
| `Score` の frame 和が simulate / assess / assess_expr でずれる | **規約が三箇所に散在** |
| `draw_noise` の名前空間が model 固定 | 規約に定義点がない |
| `lift_consts` / `subst_data_tangents` が非要素ごと構文を素通し | 事前条件の不在 |
| Q3（$c_{\text{elem}}$）を「解決済み」と記録したが gemm のみ | **判定の記録が事後条件を伴っていない** |

**全部この文書の欠落である。** 六つ目までは値が静かに間違う種類、七つ目は判断が静かに間違う種類。

この文書の要求は一つ。

> **すべての変換は、事前条件・事後条件・保存する不変条件を持つ。事前条件は入口で loc 付きに落とし、事後条件は述語として表現し、次の変換はそれを仮定してよい。**

「黙って間違える」を「その場で落ちる」に置き換えることが、意味論の自然さと同じ順位の要求である（第1枚 §0 の「覆っていない部分はエラーとして拒否する。黙って近似しない」の、変換層への適用）。

---

## 1. 中間表現の階層

式の型は一つ（`expr`）だが、**通過した変換によって満たす性質が異なる**。この性質の集まりを層と呼ぶ。

```
表層                    -- 未実装。パーサの出力
  │ desugar
  ▼
コア                    -- Rank あり / Sample・Score あり / 形状は未確定
  │ reparam                    ← ここで動く変換がある
  │ expand_rank(senv)
  ▼
展開済みコア            -- Rank なし / 形状が全て確定
  │ assess_expr                ← ここでしか動かない変換がある
  │ fuse_views
  ▼
確定形                  -- Sample・Score なし / 純粋な数値式
  │ forward
  ▼
接続形                  -- primal と tangent が混在
  │ unzip
  ▼
分離形                  -- primal 束縛 / tangent 束縛
  │ transpose
  ▼
勾配プログラム          -- 余接から入力への線形写像
```

各層に述語を対応させる。

| 層 | 述語 | 意味 |
|---|---|---|
| コア | `is_desugared` | 糖衣が展開されている |
| 展開済みコア | `is_expanded` | `Rank` ノードが存在しない |
| — | `is_reparammed` | `D_pushforward` を dist に持つ `Sample` が存在しない |
| 確定形 | `has_no_samples` | `Sample` / `Score` が存在しない |

**述語は事後条件の実体である。** 変換 $T$ の事後条件が述語 $P$ なら、$T$ の直後で `assert (P e)` が真であること、および $P$ を事前条件とする変換だけが $T$ の後に来られることを意味する。

### 1.1 層は全順序ではない

**重要**: `reparam` は**展開前**に動き、`assess_expr` は**展開後**にしか動かない。

- `reparam` は `wrap_rank0` によって `Rank(0, …)` を**生成する**。したがってその出力は展開を必要とする
- `assess_expr` は `| Rank _ -> failwith "assess_expr: Rank must be expanded first"` で `Rank` を拒否する

同じ guide に対して両方を適用するため、`build_elbo` は**別々の系統で処理し、最後に合流させる**（§3.1）。この非直線性は設計であり、事故ではない。しかし文書化されていなければ、次に変換を足す人（半年後の自分を含む）が必ず踏む。

---

## 2. 変換の目録

記法: $\Pr$ = 事前条件、$\Post$ = 事後条件、$\Inv$ = 保存する不変条件。

### 2.1 静的検査（式を変えない）

| 名 | 対象 | 落とす例外 |
|---|---|---|
| `check_sites` | 任意の式 | `Duplicate_site (loc, name)` |
| `check_guide` | guide 候補 | `Guide_error (loc, msg)` |
| `check_trace_compat` | (model, guide) | `Trace_mismatch msg` |
| `check_support_compat` | (model, guide) | `Support_mismatch (loc, msg)` |
| `check_no_samples` | grad の入力 | `Grad_error (loc, msg)` |

**規約**: 検査は式を返さない（`unit`）。検査を通ったことは型に現れないので、**呼ぶ場所を仕様で固定する**（§3）。

#### `check_sites`

- $\Pr$: なし
- $\Post$: サイト名が式全体で一意
- **なぜ必要か**: `Sites.find` は `List.find` であり、同名サイトが二つあると二つ目が一つ目の `site_id` で乱数を引く。同じ $u$、同じ `%tr` 名。**値が静かに相関する**
- **置き場所**: `collect_sites` の**採番より前**。採番後に検査すると、既に不正な表が作られている

#### `check_guide`

guide 部分言語の妥当性を**一つの述語に集約する**。個別の変換が実行時に一つずつ蹴る形にしない。

- $\Pr$: なし
- $\Post$: 以下がすべて成り立つ
  - `Score` を含まない
  - すべての `Sample` の dist が、base まで再帰して `D_categorical` を含まない
  - すべての `Sample` の dist が `D_product` を含まない
  - サイト名が一意（`check_sites` を内包）
- $\Inv$: model 側には別の述語が対応する（model は `Score` 可、離散可）。**二つの述語が並ぶことで、model と guide の役割の差がコードの構造に現れる**

#### `check_trace_compat`

- $\Pr$: `check_sites model`、`check_sites guide`
- $\Post$: model のサイト集合と guide のサイト集合が名前で全単射、かつ対応するサイトの `frame` が一致
- **将来**: 部分 trace（一部を guide、残りを事前から）を許すとき、この全単射は「guide ⊆ model」に緩む。**そのときも非対称性は明示的な引数（`?partial:`）で表現し、既定は全単射のまま**にする

#### `check_no_samples`

- $\Pr$: なし
- $\Post$: `has_no_samples`
- **メッセージを三種に分ける**: 離散 `Sample` / 連続 `Sample` / `Score`。それぞれ対処が違う（前者は $T(\mathrm{Fin}\,n) = \mathbf 0$ で原理的に不可能、後二者は変換の呼び忘れ）

#### `check_support_compat`

- $\Pr$: `check_trace_compat model guide`
- $\Post$: 各 guide site $n$ について $\mathrm{supp}(q_n) \subseteq \mathrm{supp}(p_n)$
- 台は分布の構成木から `Sites.dist_support` が導出する。`D_pushforward` は意味情報として像の台を保持し、Normal は `Real`、HalfNormal は `Positive` を与える
- 違反は guide site の loc で落とす。特に model の HalfNormal site に Normal guide を当てることを拒否する

### 2.2 式変換

#### `desugar` / `fuse_views`

- $\Pr$: なし
- $\Post$: `is_desugared`
- $\Inv$: 評価結果を変えない（**浮動小数まで**ではない。view の融合は丸めを変えうる）

#### `expand_rank ~senv`

- $\Pr$: `senv` が式中の自由変数すべての形状を含む
- $\Post$: `is_expanded`
- $\Inv$: 評価結果を変えない
- **副関数** `infer_shape senv e` は、`is_expanded` を仮定せず形状を返す
- **落とし穴**: `senv` に載っていない自由変数があると、形状推論が失敗するか誤った形状を返す。**`senv` の完全性は呼び出し側の責務**であり、これは現在検査されていない（§7）

#### `reparam ?sites`

$$\texttt{Sample}(l, n, f, \texttt{D\_pushforward}\{v, \mathit{fwd}, \mathit{base}\})
\;\Longrightarrow\;
\texttt{Let}(\texttt{\%u.}n,\ \texttt{Sample}(l,n,f,\mathit{base}),\ \texttt{Let}(\texttt{\%tr.}n,\ \mathit{fwd}[v := \texttt{\%u.}n],\ \texttt{\%tr.}n))$$

- $\Pr$: `check_sites`。`sites` を渡す場合、同じ式から得た表であること
- $\Post$: `is_reparammed`。かつ**すべてのサイト $n$ について `%tr.`$n$ が束縛されている**
- $\Inv$: **鍵引き継ぎ規則**。base の `Sample` は元の site 名を保持するため、Threefry の counter $(\mathit{site\_id}, \mathit{component}, \mathit{frame\_index})$ が変換前後で一致する。したがって同じ `run_key` に対し `simulate(e)` と `eval(reparam(e))` は**ビット一致**する（第1枚 §6.7）
- **意味論上の位置づけ**: 定義の展開。新しい意味を与えない
- **`inv` は破棄する**。密度は変換前の式に対する `assess` が担う

`%tr.`$n$ の束縛を `reparam` の責務に含めることが要点である。これにより:

- guide のどこに `Sample` が埋まっていても、その値は**予測可能な名前**で式中に存在する
- ELBO のスロットは常に `Var("%tr." ^ name)`。複数サイトでも同じ
- 「guide の返り値が潜在値」という規約が**不要になる**（この規約は複数サイトで破綻する）

**部分関数 `wrap_rank0`**

- $\Pr$: 引数が要素ごとの構文のみ（`Const` / `Var` / `Let` / `Prim` / `Rank 0`）
- $\Post$: すべての `Prim` が `Rank(0, …)` に包まれている
- **`Rank` $k>0$、`Sample`、`Score` は failwith**。ここを素通しにすると、rank 2 の `Matmul` が黙って rank 0 になり**意味が変わる**

#### `elim_samples ~sites`

reparam 済みの guide から原始 `Sample` を消し、`Let` を平坦な束縛リストにする。

- $\Pr$: `is_reparammed`。かつ **α 一意性**——式全体で束縛変数名が重複しない
- $\Post$: 返る束縛リストが評価順で、`%u.`$n$ が自由変数として残る
- **α 一意性が破れると値が静かに間違う**:

  ```
  Prim(Add, [Let(h, e₁, h); Let(h, e₂, h)])
    ⟹ bindings = [(h,e₁); (h,e₂)], body = Add[h; h]
    ⟹ let h = e₁ in let h = e₂ in h + h     (* = 2·e₂ *)
  ```

  理想は gensym による α 変換だが、投資として過大。**前提を検査して落ちる**（`Elim_error`）で足りる

#### `assess_expr ~ns ~env_shapes e slots`

trace 上の対数密度を**式として**構成する。

- $\Pr$: `is_expanded`。`slots` が式中の全サイトを覆う。`env_shapes` が自由変数と slot 変数の形状を含む
- $\Post$: 返る式は `Sample` / `Score` を含まない
- $\Inv$: $\texttt{eval}(\texttt{assess\_expr}(e, \{n \mapsto v\})) = \texttt{assess}(e, \{n \mapsto v\})$ ——**値レベル実装との一致**（1e-12）
- **`ns` は gensym の名前空間**。model と guide を同じ式に合流させるため、生成変数の衝突を規約で防ぐ（`"m."` / `"g."`）
- **`Sample` / `Score` は `Let` 束縛に限る**。`Prim` の引数に埋めると密度が落ちるため、`assert_no_sample_in_args` で loc 付きに落とす

**同じ `Sample` ノードの三つの読み**（第1枚 §6.3 と §6.7 の実装対応物）:

| 変換 | `Sample` を何と読むか |
|---|---|
| `simulate` | 乱数を引く |
| `assess` / `assess_expr` | 密度に読む |
| `elim_samples` | 入力（自由変数）に読む |

#### `forward`

- $\Pr$: `is_expanded`、`is_desugared`、`has_no_samples`
- $\Post$: `(bindings, primal_out, tangent_out)`。tangent は primal に線形
- $\Inv$: **JVP の正しさ**——有限差分との一致
- **注**: 導関数規則がスカラー定数を含むため、生成 IR には `Rank` ノードが再び現れる。したがって `forward` の出力は**再度 `expand_rank` を要する**。これは層が単調でない二つ目の例である

#### `unzip ~seeds`

- $\Pr$: `forward` の出力形式
- $\Post$: primal 束縛と tangent 束縛が分離され、tangent 束縛は `seeds` について線形
- $\Inv$: **`let` の分配の自由度がそのまま checkpointing である**（第1枚 付録A）

#### `transpose ~seeds ~input_shapes ~cotangent_var`

- $\Pr$: `unzip` の出力形式。tangent 束縛が線形
- $\Post$: 各 seed に対する勾配式
- $\Inv$: **内積等式** $\langle L(v), u\rangle = \langle v, L^{*}(u)\rangle$。これが AD の正しさのほぼ全部を捕捉する

---

## 3. パイプラインの固定

### 3.1 推論構築（`build_elbo`）

```
入力: model, guide, env_shapes

  1. check_sites model
  2. check_guide guide                    (check_sites guide を内包)
  3. check_trace_compat ~model ~guide
  ─────────────────────────────── 以降、事前条件は満たされている
  4. sites   ← collect_sites guide        (採番の唯一の定義点)
  5. noise   ← [(%u.n, frame) | n ∈ sites]
     slots   ← [(n, Var %tr.n) | n ∈ sites]

     ┌── 値の系統（展開前） ──────────┐   ┌── 密度の系統（展開後） ────┐
  6. │ guide_r  ← reparam ~sites guide │ 7.│ model_e ← expand model      │
     │ bindings ← elim_samples guide_r │   │ guide_e ← expand guide      │
     └─────────────────────────────────┘   │ m_ld ← assess_expr ~ns:"m." │
                                           │ g_ld ← assess_expr ~ns:"g." │
                                           └─────────────────────────────┘
  8. elbo ← wrap_bindings bindings (Sub [m_ld; g_ld])
  9. elbo ← expand ~senv:(noise @ env_shapes) elbo
 10. elbo ← fuse_views elbo

出力: { elbo; sites; noise }
```

**6 と 7 が別系統であることが、この設計の非自明な点である**（§1.1）。6 は `Rank(0,…)` を生成し、7 は `Rank` を拒否する。8 で合流し、9 で全体を再展開して整合する。

$\Post$: `elbo` は `Sample` / `Score` を含まず、自由変数は `noise ∪ env_shapes` の名前のみ。

### 3.2 微分（`grad`）

```
入力: e, param_shapes, data_shapes

  1. check_no_samples e
  2. e ← expand ~senv:(param_shapes @ data_shapes) e
  3. e ← fuse_views e
  4. reset_gensym ()
  5. (bs, primal_out, tangent_out) ← forward e
  6. bs, primal_out, tangent_out を再展開       ← forward が Rank を生成するため
  7. data の接ベクトルを零に置換                 ← data について微分しない
  8. unzip ~seeds:(param の接ベクトル名)
  9. transpose ~cotangent_var:"%ct"

出力: { loss; grads; primal_bindings; loss_body; grad_bodies }
```

**4 の `reset_gensym` は決定性の要求である。** 同じ入力から同じ変数名が出ることを保証しないと、生成された式が実行ごとに変わり、回帰検査ができない。

### 3.3 学習ループ

$\Post$ **ループ内で変換を呼ばない。**

```
build_elbo と grad はループの前に一度だけ。
ループ内は:
    データの切り出し  →  noise_env  →  eval_grad  →  optimizer
```

これは性能上の要求ではなく、**「推論はプログラム変換である」という主張の実体**である。ループが変換を呼んだ瞬間、変換ではなくインタプリタの機能になる。**検査可能な形にしておくこと**——ループ本体から到達できる関数の集合に `Reparam` / `Assess_expr` / `Expand_rank` が含まれないこと。

---

## 4. 名前の規約

### 4.1 変数名（予約領域）

| 形 | 生成者 | 意味 |
|---|---|---|
| `%u.`$n$ | `Sites` | サイト $n$ の基底乱数（一様） |
| `%tr.`$n$ | `Reparam` | サイト $n$ の trace 値 |
| `%ct` | `Transpose` | 余接 |
| tangent 名 | `Forward` | 接ベクトル |
| `p`$k$ / `r`$k$ | `Forward.gensym` | 中間値・残差 |
| `m.` / `g.` + `a.`… | `Assess_expr` | model / guide の密度中間値 |

$\Inv$: **ユーザ由来の変数名はこの領域に入らない。** 現在これは検査されていない（§7）。表層構文を作る時点で、`%` で始まる識別子を字句レベルで禁じるのが最も安い。

`%u.` と `%tr.` の対応は `Sites.noise_name_of_name` / `trace_name_of_name` の**二関数が唯一の定義点**であり、文字列連結を他所に書かない。

### 4.2 乱数の名前空間

$$\texttt{key} = (\text{名前空間},\ \texttt{run\_key}), \qquad \texttt{ctr} = (\mathit{site\_id},\ \mathit{component},\ \mathit{frame\_index})$$

| 名前空間 | 用途 |
|---|---|
| `ns_init` | パラメータ初期化 |
| `ns_model` | model の `Sample`（`simulate`） |
| `ns_guide` | guide の基底乱数（`noise_env`） |
| `ns_data` | ミニバッチのシャッフル、データの確率的前処理 |

**`ns_model` と `ns_guide` を分けることが必須である。** 同一にすると、同じ `run_key`・同じ `site_id` に対して model の事前抽出と guide の基底乱数が**同一の一様乱数になる**。ELBO を回すだけなら model の乱数を引かないので無害だが、**SBC（事前から引く → データ生成 → 推論 → 順位の一様性）で順位が壊れる**。しかもバグではなく「相関」として出るため、原因特定が最悪の部類になる。

$\Inv$ **counter の構造的単射性**: $(\mathit{site\_id}, \mathit{component}, \mathit{frame\_index}) \mapsto \texttt{ctr}$ が単射。各フィールドに割り当てたビット幅を仕様として書き、**実際に使う最大値がその範囲に収まることを検査する**。

ビット配分を次で固定する。

- `ctr[0]` 上位 32 bit: unsigned `site_id`（$0 \le i < 2^{32}$）
- `ctr[0]` 下位 32 bit: unsigned `component`（$0 \le c < 2^{32}$）。root は 1、積の左右は $2c,2c+1$ なので深さ上限は 31
- `ctr[1]` 64 bit: `frame_index`。現在の API は非負の OCaml `int` を受けるため、実装上の上限は `Stdlib.max_int`（64-bit runtime では $2^{62}-1$）

`make_ctr` はこの範囲を入口で検査する。境界テストは MNIST 動的二値化の最大値付近 `47_039_999` / `47_040_000` と、各フィールドの実装上限を含む。

> 現に、MNIST の動的二値化は $\mathit{frame\_index} = \mathit{source} \times 784 + \mathit{col}$ を使い、最大 $4.7\times10^7$（26 ビット）に達する。単射性のテストが $10^6$ までしか見ていないなら、この用途は仕様の外にある。

$\Inv$ **coupling**: $\theta$（`fwd` の係数）を摂動しても、引かれる一様乱数列が変わらない。これが pathwise 勾配と共通乱数による有限差分照合の根拠である。

---

## 5. site 表

$$\texttt{site} = \{\ \texttt{name};\ \texttt{id};\ \texttt{frame};\ \texttt{kind}\ \}$$

**`collect_sites` が静的サイト走査と採番の唯一の定義点である。** `simulate` / `reparam` / `elim_samples` / `build_elbo` / `draw_noise` / 学習ループは、すべてこの表を引く。

- `id` は走査順に 0 から。Threefry の $\mathit{site\_id}$ に直結するため、**走査順を変えると乱数列が変わる**
- `frame` は plate の宣言そのもの。`Sample` ノードに明示的に載る
- `kind` は $T(\mathrm{Fin}\,n) = \mathbf 0$ の判定に使う。**`D_pushforward` は base の `kind` を継承する**——押し出しは可微分性を作らない

$\Inv$: **`kind` は構成から導出する。手で持たせない**（第1枚 (D3)）。

意味論上、この表は trace 型 $\mathcal T$ の実装対応物である。Phase 12（`Restrict` の試行番号軸）と Phase 13（`Marginalize`$\,S\,P$ の $S$ = 表の部分集合）は**この上に乗る**。

---

## 6. 検査の分類

| 段階 | 何を守るか | 落ち方 |
|---|---|---|
| 静的検査（§2.1） | 変換の事前条件 | loc 付き例外。**利用者に見せるエラー** |
| 述語 assert | 変換の事後条件 | 開発時の内部矛盾。利用者には見えないはず |
| 不変条件テスト | 意味の保存 | 回帰検査 |

**不変条件テストの目録**（機械的に照合できるもののみ）:

| 不変条件 | 検査 |
|---|---|
| 内積等式 | view 層 / カーネル / プログラム全体 / 確率プログラム（**四度**） |
| JVP の正しさ | 中心差分との相対照合 |
| 密度の正規化 | 低次元の数値積分が 1 |
| 密度の閉形式一致 | Normal 等で 1e-12 |
| `assess` ≡ `assess_expr` | 同一式・同一 trace で 1e-12 |
| coupling | `simulate` ≡ `eval ∘ reparam` が**ビット一致** |
| `ns_model` ≠ `ns_guide` | 同 `run_key`・同 `site_id` で異なる値 |
| 変換の決定性 | `reset_gensym` 後、同一入力 → 同一式 |
| **閉形式との収束一致** | 共役ガウス VI（変分族が真の事後を含む場合、gap = 0） |
| **SBC** | 事前から引く → データ生成 → 推論 → 順位の一様性（Phase 12） |

最後の二つが「絵を見て納得する」に代わるものである。VAE までは再構成画像が最終検収になったが、**Phase 12 以降にそれは無い**。

### 6.1 Phase 12 の検査ラベル（着手前に固定）

| ラベル | 検査 | 解釈 |
|---|---|---|
| **必ず通る（実装不変条件）** | HalfNormal の逆写像・閉形式密度・正規化・FD | 落ちれば分布定義または AD の実装バグ |
| **必ず通る（実装不変条件）** | 階層 site の `assess ≡ assess_expr`、全パラメータ FD、coupling、内積等式 | 落ちれば変換・shape・乱数対応の実装バグ |
| **必ず通る（静的安全性）** | support 包含、潜在⊎観測、ragged のマスク外ビット不変 | 落ちれば事前条件検査または密度実装のバグ |
| **必ず通る（較正）** | SBC 1a（閉形式）、SBC 1b（独立共役・平均場が厳密） | 落ちれば SBC 機械または言語実装のバグ |
| **近似品質の診断** | SBC 2（funnel を含む階層平均場 VI）、真値の信用区間被覆 | 正しい実装でも平均場近似により落ちうる。完了条件にしない |

結果を見てからラベルを変更しない。とくに SBC 2 の失敗を実装バグの証拠にせず、SBC 1a/1b の失敗を「統計的に難しい」で済ませない。

---

## 7. 現在守られていない前提

正直に列挙する。すべて「検査が無い」であって「間違っている」ではない。

| 前提 | 現状 | 対処の重さ |
|---|---|---|
| `expand_rank` の `senv` が自由変数を網羅 | 未検査 | 自由変数を集めて差分を取る。中 |
| ユーザ変数が `%` 領域に入らない | **予約を宣言済み**。強制は未実装 | 字句レベルで禁止。表層構文と同時。小 |
| counter の $\mathit{frame\_index}$ が範囲内 | **仕様・入口検査・境界テストあり**（§4.2） | — |
| `elim_samples` の α 一意性 | 検査あり（`Elim_error`） | — |
| ループが変換を呼ばない | **§3.3 の規約として固定**。現在は目視検査 | 将来、到達可能性の静的検査 |
| `subst_data_tangents` が `Rank` を受ける | **failwith で拒否**（展開事後条件の破れ） | — |
| $c_{\text{elem}}$ が許容内（Q3 の要素ごと側） | **未解決**。約 20 ns/要素、ハードウェア限界の 5–10 倍 | §8 |

---

## 8. 性能判定の記録（Q3 の分割）

第1枚は性能を扱わないが、**判定の記録が事後条件を伴っていなかった**という点でこの文書の主題に属する。

Phase 9 の性能モデル:

$$\text{総時間} = (\text{ノード数}) \times c_{\text{node}} + \sum_{\text{node}} (\text{要素数}) \times c_{\text{elem}}$$

三つの問い Q1（rank 展開は要素数を稼げているか）/ Q2（$c_{\text{node}}$）/ Q3（$c_{\text{elem}}$）のうち、**Q3 は gemm についてのみ解決していた**。MNIST の MLP は要素ごと演算が $[64,400]$ 中心で量が少なく、露出しなかった。

VAE の実測（batch 128、784-400-20、release、1 step）:

| | |
|---|---|
| 総時間 | 116.0 ms |
| `matmul` | 22 回、15.6 ms（**13%**、約 32 GFLOP/s） |
| `logsigmoid` + `mul` + `add` | 56.3 ms |
| `Buf.create` | 238 回、45.5 MB |

逆算すると $c_{\text{elem}} \approx 20$ ns/要素。倍精度の `exp` はハードウェアで 5 ns 程度、`mul`/`add` は 1 ns 未満なので、これは演算コストではなく**要素ごとカーネルのクロージャ間接呼び出し**である。確保と零埋めは 45.5 MB / 10 GB·s⁻¹ ≈ 5 ms で支配項ではない。

**判定を書き換える**:

> Q3 を二つに分ける。**gemm**: BLAS で解決（wall time の 13%）。**要素ごと**: 未解決。$c_{\text{elem}}$ がハードウェア限界の 5–10 倍。原因は要素ごとカーネルの実装であり、インタプリタのノード固定費（Q2）でも数値演算そのものでもない。
>
> **Phase 12–13 では対処しない。** 53 秒/エポックで実験は回る。将来速度が問題になったときの探索対象が確定していることに価値がある。

**教訓としての一般則**: 判定を記録するときは、$\Post$ を伴わせる。「Q3 解決」ではなく「Q3 は $X$ の範囲で解決、$Y$ については未測定」。

---

## 9. Phase 12–13 で増える変換の置き場所

新しい原理は増えない。すべて既存の階層に配置できる。

| 追加 | 層 | $\Pr$ / $\Post$ |
|---|---|---|
| `Restrict` の展開（固定予算 $K$ の棄却） | コア → コア。`reparam` と同格 | $\Post$: `Restrict` を持つ `Sample` が残らない。**site 表に試行番号軸が増える**（§5 が受け止める） |
| ragged のパディング + マスク | 展開済みコア | $\Inv$: マスクされた位置が密度に寄与しない |
| `_quadrature`（$K$ 節点の固定軸） | コア → コア | `_enumerate` と同型。$\Post$: 対象サイトが表から消え、重み付き軸に置き換わる |
| `_importance` | 密度の系統 | $\Post$: **重みの配列を返す**（$\log Z$ に潰さない） |
| 固定反復 optimizer | 学習ループ側 | 式に $M$ 反復を展開しない（AST サイズの爆発を避ける） |

$\Inv$ **有界作業量の原則**: 棄却法の $K$、内側最適化の $M$、列挙の台、求積の節点——すべて静的に有界。この原則が破れる変換は、この言語に置けない。

`Restrict` を入れる時点で `check_guide` に条件が一つ増える（guide に `Restrict` を許すか）。**その答えを書く場所が §2.1 に既に在る**——これが述語を一箇所に集約した理由である。

---

## 10. 未決

| | 内容 | 影響 |
|---|---|---|
| 表層構文とパーサ | 未着手 | §4.1 の予約領域の強制は、ここで最も安く入る |
| **印字の対象となる層** | 未決 | 設計文書18章「印字したコードは必ず再入力できる」は**一度も検査されていない**。展開済みコアは形状が焼き込まれ再利用できず、コアを印字するなら明示 broadcast が表層言語の一部でなければならない |
| $T(M\,\tau)$ | 第1枚から持ち越し | 自然勾配のみ |
| 特殊関数 | Phase 13 | Owl で解決済み |

**二つ目が最大の未決である。** 変換のパイプラインを固定したことで、「どの層を印字するか」が答えられる問いになった。第3枚の主題になるはずである。

---

## 付録: 一行要約

$$\text{すべての変換は } (\Pr, \Post, \Inv) \text{ の三つ組を持ち、} \Pr \text{ は入口で loc 付きに落ち、} \Post \text{ は述語であり、} \Inv \text{ は機械的に検査される。}$$
