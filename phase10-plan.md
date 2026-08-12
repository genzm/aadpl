# Phase 10 作業計画(確定版)

**目的**: 分布代数・trace・乱数の基盤。`?` と `Score` が動き、密度が計算できる状態にする。
**作らないもの**: `_reparam`(Phase 11)、`Restrict`(Phase 12)、dtype variant(再考条件つき先送り)、勾配との接続(Phase 11)。

---

## 決定事項(再掲・確定)

| 項目 | 決定 |
|---|---|
| dtype | 入れない。離散値は float 埋め込み(範囲検査済みの実績方式)。再考条件: `_enumerate` 着手時 or silent バグ 1 件 |
| PRNG | Threefry-2x64。counter-based(評価順序に依存しない = coupling の基盤) |
| counter | ハッシュではなく**構造的に単射**: (site 番号, frame 内線形添字)。runKey と namespace(`Init/Model/Data`)は key 側 |
| 一様乱数 | **開区間 (0,1)**: `((x >> 11) + 0.5) * 2^-53`。§8 の表に「意味論は [0,1)、実装は (0,1)、a.e. 一致」を一行追加 |
| `Sample` | **frame(plate)を明示**: `Sample of loc * string * int array * dist`。frame が添字パス、値は frame shape の 1 スロット |
| `Normal` | ライブラリ関数(OCaml 側で構成子を合成)。prim には `Erf`・`Erfinv` の**両方**を追加(密度に inv = Φ が要る) |
| `D_pushforward` | 逆写像 inv を**構文で**持つ。inv は証明義務((D3) の例外、`Restrict` の Z と同格)で、`inv(f(u))=u` の数値照合を決定的テストに |
| インタプリタ | 変換より先に参照実装 2 本: `simulate` / `assess`(jvp.ml が forward の基準になったのと同じ構図) |

## AST(確定形)

```ocaml
type dist =
  | D_uniform
  | D_categorical of expr                      (* weights : [n]R *)
  | D_pushforward of {
      fwd_var : string;  fwd : expr;           (* f: 束縛変数と本体 *)
      inv_var : string;  inv : expr;           (* f⁻¹: 証明義務 *)
      base : dist;
    }
  | D_product of dist * dist

type expr = ...
  | Sample of loc * string * int array * dist   (* site名, frame, dist *)
  | Score  of loc * expr
```

- site 名の重複はプログラム走査で loc 付きエラー(unzip のシャドーイング検査と同型、5 行)
- アドレス = (site_id, frame 内添字)。呼び出しスタックは空で予約
- trace = (site 名 → Tensor.t) の連想。frame shape がスロットの shape

## インタプリタの型

```ocaml
val simulate : runKey:int64 -> env -> expr -> value * trace * value
  (* 返り値, t ~ p(事前), Σ Score = log w *)
val assess   : env -> expr -> trace -> value * value
  (* 返り値, log(dμ/dλ)(t) = Σ_site log p_site + Σ Score *)
```

- guide(Score なし)の assess = log q。model の assess = log p + log w
- **ELBO = assess(model, t) − assess(guide, t)** がこの 2 本だけで書ける(Phase 11 の検収基準)
- `Score` は log 重みに加算。frame 下(Sample の frame 経由で配列が来る場合)はセル方向に和(§6.3)
- 密度の frame 方向も和(独立積参照測度)

## D_pushforward の評価規則

- sample: base から u を引き、fwd を **rank 0 で frame に持ち上げて**評価(体は要素ごとの式なので既存 expand_rank がそのまま効く)
- log_density(z): `u = inv(z)` を評価し、`log p_base(u) + log |d inv/dz|`。**ヤコビアンは inv 式に既存 jvp を適用**して得る(§5.4「ヤコビアンは AD が計算する」の実装)
- 当面 fwd/inv は**スカラー→スカラーの単調関数に限定**(Normal/LogNormal/affine で足りる)。多変量は Phase 12 以降

## 作業順序

```
10-1  Threefry-2x64                                          1日
      - 20 ラウンド + rotation 定数
      - 【必須】原論文の既知入出力ベクタとの一致テスト
      - counter 符号化(構造的単射)、(0,1) 変換
      - 検査: 再現性 / namespace・添字の 1 差で値が変わる / 10^6 標本のモーメント
10-2  AST + pp + site 重複検査                                半日
10-3  simulate + 検査                                        1日
      - 同 runKey → trace が浮動小数まで一致
      - θ(fwd の係数)を変えても引かれる一様乱数列が不変(coupling / pathwise の根拠)
10-4  Erf・Erfinv prim + Normal ライブラリ                     1日
      - Giles(2010) の erfinv 近似 + A&S 系 erf(自前 ~50 行、依存ゼロ維持)
      - JVP: erf → (2/√π)e^{-x²}、erfinv → (√π/2)e^{(erfinv x)²}
      - 端点近く(|x|→1)の相対誤差テスト
      - inv∘fwd = id の数値照合(証明義務の検査)
10-5  assess + 検査                                          1-2日
      - 低次元数値積分で正規化(Uniform / Normal / Categorical)
      - Normal の log_density を閉形式と照合(ヤコビアン=AD の最初の検証)
10-6  frame 付き Sample の一括化 + density の frame 和         1日
      - frame [|G|] で G 個の値が 1 回の持ち上げ評価で出ること(計装で確認)
      - 検査: frame 版 = スカラー版を G 回(素朴ループ、テスト専用)と一致
```

## Phase 10 の完了条件

```
□ Threefry が参照ベクタと一致、モーメント検査通過
□ simulate / assess が上の型で動く
□ Normal(ライブラリ)の密度が閉形式と 1e-12 で一致
□ 正規化積分が 1(3 分布)
□ coupling: θ 摂動で乱数列不変
□ frame 付き Sample が一括評価される(計装の呼び出し回数で確認)
□ ELBO が simulate + assess の 2 呼び出しで書けることを確認する smoke test
  (勾配はまだ取らない — Phase 11 の入口)
```

## Phase 11 への引き継ぎメモ

- `_reparam` = D_pushforward の定義展開(fwd を式にインライン)。simulate との一致が検収
- 離散 seed 禁止の検査(T(Fin n)=0)は grad 側に置く
- trace 型(モデル/ガイドの対応)はここで導入
- Erfinv の JVP が seed 経路に入るので、FD 照合を Normal サイト込みで一本
```
