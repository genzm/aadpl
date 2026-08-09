# 実装設計 第1枚：shape / stride / view 層

**位置づけ**: 意味論 §4.5（添字写像の随伴）の直接の実装。他のすべてが乗る土台。
**ホスト**: OCaml。buffer は Bigarray、カーネルは差し替え可能（Naive → Owl → 直接）。

---

## 1. この層が満たすべき要求

意味論から逆算すると、要求は四つ。

| 要求 | 出どころ |
|---|---|
| 構造操作を添字写像 $\sigma$ の引き戻しとして統一的に表す | §4.5 |
| 随伴 $\sigma_*$ が**単一の bulk 演算**になる（累積効果を使わない） | §4.5, §10.2 |
| shape を動的に持ち回る（転置の入力データ） | §4.5 |
| rank 下の共有が明示 broadcast になり、随伴が自動的に和になる | §4.6 |

さらに実装側の要求として、

| 要求 | 出どころ |
|---|---|
| 行列積が BLAS に直結する（厚いプリミティブ） | 二層 IR の方針 |
| Owl 依存が葉のカーネルに閉じる | ホスト選定の判断 |

---

## 2. 中心的な決定：$\sigma$ は view である

**$\sigma$ を一般の関数として持たない。** 一般の $j \mapsto \sigma(j)$ では $\sigma^{-1}(i)$ が計算できず、BLAS にも届かない。

**$\sigma$ を構成子の列挙にもしない。** broadcast・transpose・slice・reshape をそれぞれ別の構成子にすると、随伴規則が構成子ごとに必要になり、「唯一の規則」が崩れる。

採るのは第三の道である。

> **$\sigma$ を、基底バッファへの添字写像として (shape, strides, offset) で表す。**

$$
\sigma(j_0,\dots,j_{r-1}) = \text{offset} + \sum_k j_k \cdot s_k
$$

これが効く理由は、**構造操作の大半がこの形に収まり、しかも一つの随伴規則を共有するから**である。

| 操作 | view での表現 |
|---|---|
| broadcast | **stride = 0** |
| transpose / 軸の並べ替え | (shape, stride) の対を置換 |
| slice | offset をずらし、stride を倍加 |
| reshape（連続時） | shape を組み替え、stride を再計算 |
| rank 下の共有 | frame 軸に stride 0 |

**broadcast が stride 0 であること**が要石である。§4.6 の「rank 下の共有を明示 broadcast として表現すれば、随伴は通常の broadcast の規則で和になる」が、実装では「stride 0 の軸は随伴で和を取る軸である」という一文になる。

---

## 3. 表現

```ocaml
type dtype = F64 | I32 | Bool

type buffer =
  | F64 of (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  | I32 of (int32, Bigarray.int32_elt,   Bigarray.c_layout) Bigarray.Array1.t
  | Bool of (int,  Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type view = {
  shape   : int array;   (* 論理 shape J *)
  strides : int array;   (* 基底バッファへの stride *)
  offset  : int;
}

type t = {
  base : buffer;         (* 実データ *)
  view : view;
}
```

**基底は 1 次元の Bigarray に統一する。** 多次元性は view が持つ。これにより

- Owl の型が染み出さない（buffer は Bigarray のまま）
- view の合成が単なる算術になる
- 実体化（materialize）が「新しい連続バッファへコピー」という一つの操作になる

`dtype` を分けるのは §3.1 の $T(\mathtt{Z}) = T(\mathtt{B}) = \mathbf{0}$ に対応する。**整数・真偽値の配列には接ベクトルを作らない。** AD 層はこの区別を見て、tangent を生成するかどうかを決める。

---

## 4. 唯一の随伴規則

view $v$ を通した読み出しの随伴は、**view を通した書き戻し（scatter-add）**である。

```
transpose_view v ḡ acc =
    forall j ∈ v.shape:  acc[v(j)] += ḡ[j]
```

`acc` は基底バッファと同じ大きさで、**ゼロ初期化されている**こと。

これが全部を覆うことを確認する。

| view | 随伴の挙動 |
|---|---|
| stride 0（broadcast） | 同じ位置に複数回加算 → **その軸方向の和** |
| 全単射（transpose, 連続 reshape） | 各位置にちょうど1回 → **逆置換** |
| 単射だが非全射（slice） | 一部の位置のみ加算 → **ゼロ埋め + 配置** |
| 一般 | scatter-add |

**規則は一つ。上の四行は「規則の場合分け」ではなく「同じ規則の実行時の帰結」である。**

### 高速路（規則ではなくディスパッチ）

素朴な scatter-add ループは遅い。しかし**高速路は最適化であって、別の随伴規則ではない**。

| 条件 | 高速路 |
|---|---|
| $v$ が単射（重なりなし） | 加算不要。ゼロ埋め + strided copy |
| stride 0 の軸のみが多対一 | その軸で `sum_axis` してから配置。ベクトル化可能 |
| それ以外 | 一般の scatter-add |

**単射判定**（重なりの検査）は標準的な手続きで実装できる。

```
strides を |s_k| で昇順にソートし、
  すべての k について  n_k * |s_k| ≤ |s_{k+1}|
かつ すべての s_k ≠ 0
```

この判定を一箇所に置き、随伴の実装がそこを見て分岐する。**判定が誤っても結果は正しい**（一般路に落ちるだけ）のが良い性質。まず一般路だけ書き、後から高速路を足せる。

---

## 5. データ依存の $\sigma$：gather と filter

view で表せない $\sigma$ が一種類ある。**添字配列による gather** である。

```ocaml
type index_map =
  | View   of view                          (* affine *)
  | Gather of { src : view; idx : t; axis : int }   (* data-dependent *)
```

`filter`（`/`）もここに入る。**残す位置の添字配列を作り、gather する**と考える。したがって

$$\texttt{filter} = \texttt{（マスクから添字配列を作る）} \circ \texttt{gather}$$

随伴は `scatter_add`。**単一の bulk 演算**であり、§10.2 で累積効果を採らなかった判断が保たれる。

### 二種類でよい理由

$\sigma$ の性質は二つに分かれる。

$$
\begin{aligned}
\textbf{affine} &: \text{ファイバー } \sigma^{-1}(i) \text{ が軸ごとのファイバーの積} \\
\textbf{任意} &: \text{ファイバーが実行時にしか分からない}
\end{aligned}
$$

前者は shape と stride で完全に決まり、随伴が軸方向の演算に落ちる。後者は添字配列が必要。**この二分は数学的な区別であって、実装の都合ではない。**

---

## 6. reduce と broadcast は双対ペアである

`sum_axis` は $\sigma^*$ ではなく $\sigma_*$ そのものである。したがって

$$
\texttt{broadcast}^{*} = \texttt{sum\_axis}, \qquad
\texttt{sum\_axis}^{*} = \texttt{broadcast}
$$

**両方とも線形プリミティブ**なので、§4.2 の「転置規則が必要なのは線形プリミティブのみ」に従い、この一対が転置規則を持つ。

必要な転置規則の総数を数えると：

| 線形プリミティブ | 転置 |
|---|---|
| view を通した読み出し | view を通した scatter-add |
| gather | scatter_add |
| sum_axis | broadcast |
| broadcast | sum_axis |
| matmul | 転置した引数での matmul |
| conv | 転置畳み込み |

**6 個。** 非線形プリミティブ（`exp`, `log`, `max`, `Φ⁻¹`, …）には JVP 規則のみ書けばよい。

---

## 7. rank はループではなく書き換えである

これが性能を決める。

`F⎉r` の素朴な実装は「frame の各セルについて `F` を呼ぶ」だが、**これはノードあたりの仕事量を $O(1)$ にして BLAS の恩恵を消す**。前に議論した「配列言語だからインタプリタでも速い」という論理が崩れる。

正しい実装は：

> **`⎉` は frame 軸をプリミティブの引数に押し込む。プリミティブがバッチ軸を扱う。**

これが設計文書18章の「Rank や agreement の暗黙処理を明示する」変換の中身である。

### 具体例：Dense 層

```
Dense⎉1 W‿b‿xs      -- xs : [B, 784], W : [400, 784]
```

- `xs` は frame 軸 `B` を持つ
- `W` は frame 上で**変化しない** → **frame 軸に stride 0 を挿入**
- `matmul` は `[B,784] × [784,400]` を **単一の gemm（M = B）** として実行

**バッチ行列積は要らない。** frame 軸が gemm の `M` に畳み込まれるだけ。

そして §4.6 が自動的に成立する。

$$
\underbrace{W \text{ の frame 軸が stride 0}}_{\text{共有}}
\quad\xrightarrow{\ \text{随伴}\ }\quad
\underbrace{\text{その軸で和}}_{\text{セル間の勾配の合算}}
$$

**§4.6 の「rank 専用の逆伝播規則は存在しない」が、実装では「stride 0 の随伴は和である」の一例になる。**

---

## 8. 実体化と BLAS

view は任意の stride を持てるが、**BLAS は特定の形しか受け付けない**。

CBLAS の `dgemm` は `(M, N, K, A, lda, B, ldb, C, ldc)` を取るので、2 次元で

$$\min(|s_0|, |s_1|) = 1 \quad\text{かつ}\quad \text{stride が正}$$

なら直接渡せる（`trans` フラグで転置は吸収できる）。満たさない場合は**実体化**する。

```ocaml
val materialize : t -> t   (* 連続な新バッファへコピー *)
```

方針：

- `matmul` の入口で gemm 適合性を検査し、不適合なら実体化
- **それ以外では実体化しない**（view のまま持ち回る）
- 実体化はコピーなのでコストが見える。ログに出せるようにしておく

**最初は「常に実体化」で書いてよい。** 正しさが先。適合性検査は後から足せる最適化で、しかも足しても意味は変わらない。

---

## 9. カーネルのインタフェース

```ocaml
module type KERNEL = sig
  type buf

  (* 線形代数（BLAS 直結） *)
  val gemm : trans_a:bool -> trans_b:bool ->
             m:int -> n:int -> k:int ->
             a:buf -> lda:int -> b:buf -> ldb:int -> c:buf -> ldc:int -> unit

  (* 縮約 *)
  val sum_axis  : src:t -> axis:int -> dst:t -> unit
  val max_axis  : src:t -> axis:int -> dst:t -> unit

  (* 要素ごと（view 経由で読み、連続に書く） *)
  val map1 : unop  -> t -> t -> unit
  val map2 : binop -> t -> t -> t -> unit

  (* 添字操作 *)
  val gather      : src:t -> idx:t -> axis:int -> dst:t -> unit
  val scatter_add : src:t -> idx:t -> axis:int -> acc:t -> unit
  val copy_view   : src:t -> dst:t -> unit          (* 実体化 *)
  val add_view    : src:t -> acc:t -> unit          (* 唯一の随伴規則 *)

  (* 特殊関数（スカラー、Owl に残してよい） *)
  val erfinv   : float -> float
  val gammainc : float -> float -> float
  val betainc  : float -> float -> float -> float
end
```

**約 12 個。** 前に「15〜20 個」と見積もったが、view を導入したことで減った。構造操作が `copy_view` / `add_view` の二つに集約されたため。

実装の三段階：

| 段階 | 実装 | 目的 |
|---|---|---|
| 0 | `Naive`（純 OCaml ループ） | 正しさの基準。**捨てない** |
| 1 | `Owl_backed` | `gemm` と要素ごとを Owl へ |
| 2 | `Blas_direct`（ctypes） | 必要になったら |

段階 1 以降は、**段階 0 との一致で検査する**（乱数を固定して比較）。

---

## 10. 最初に書くもの

順序を提案する。各段階で検査可能な状態を保つ。

### (1) view の代数

```ocaml
val broadcast : t -> axis:int -> size:int -> t   (* stride 0 を挿入 *)
val transpose : t -> perm:int array -> t
val slice     : t -> ranges:(int*int*int) array -> t
val reshape   : t -> shape:int array -> t        (* 不可能なら実体化 *)
val is_injective : view -> bool
```

**すべて O(rank) の算術**でデータを触らない。ここが正しく書けているかは、`materialize` してから素朴に添字を計算した結果と比較すれば検査できる。

### (2) `add_view`（唯一の随伴規則）

一般の scatter-add ループのみ。高速路はまだ書かない。

### (3) 随伴性の検査

**ここを他より先に書く。** §9.1 の内積検査を、この層に対して直接適用する。

```
任意の view v、乱数 x（基底サイズ）、乱数 g（v.shape）について

    ⟨ read_view v x , g ⟩  =  ⟨ x , add_view v g ⟩
```

左辺は view を通して読んで内積、右辺は書き戻してから内積。**これが乱数 100 通りで一致すれば、view と随伴の実装はほぼ確実に正しい。**

broadcast・transpose・slice・reshape・その合成をランダムに生成して検査する。**AD を書く前に、AD の土台が正しいことを保証できる。**

### (4) 要素ごと演算と reduce

### (5) matmul（まず実体化 + Naive、次に gemm）

---

## 11. 決めたこと

| | 判断 |
|---|---|
| $\sigma$ の表現 | **(shape, strides, offset) の view + データ依存の gather の二種** |
| broadcast | **stride 0** |
| 随伴 | **`add_view`（view を通した scatter-add）ただ一つ。** 高速路はディスパッチであって規則ではない |
| §4.6（rank 下の共有） | **stride 0 の随伴が和であることの一例。専用規則なし** |
| rank の実装 | **ループではなく、frame 軸をプリミティブに押し込む書き換え** |
| バッチ行列積 | **不要。** frame 軸が gemm の `M` に畳み込まれる |
| 転置規則の総数 | **6 個**（view, gather, sum, broadcast, matmul, conv） |
| カーネル数 | **約 12 個** |
| 基底バッファ | **1 次元 Bigarray。** Owl の型を使わない |
| 実体化 | 最初は常時。gemm 適合性検査は後から |
| 最初の検査 | **随伴性の内積検査。AD より先に書く** |

---

## 12. 未決

- **conv の view 表現**。im2col で行列積に帰着させるなら gather で書けるが、メモリを食う。直接実装するなら専用カーネル。MNIST の MLP には不要なので後回し
- **`reshape` が実体化を要する条件**の正確な判定。連続性の検査だが、stride 0 が混ざるときの扱いに注意が要る
- **メモリの再利用**。純関数的に書くと Adam の更新で毎回コピーが出る。参照カウントによる in-place 化は後から入れにくいので、`t` に可変性の印を持たせるかを早めに決める価値がある
