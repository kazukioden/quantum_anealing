# ナップサック問題のハミルトニアン定義

## 問題設定
- N個のアイテムがあり、各アイテムiは重さw_i、価値v_iを持つ
- 容量Cのナップサックに入れる
- 価値の合計を最大化しつつ、重量制約を満たす

## 変数定義

### バイナリ変数
- `x_i ∈ {0, 1}`: アイテムiを選ぶかどうか

### 整数変数（Integerクラス使用時）
- `total_weight ∈ [0, C]`: 選んだアイテムの総重量を表す整数変数
- LogEncInteger型で効率的にエンコード

## ハミルトニアンの構成

### 1. Integerクラスを使用した定式化

#### 目的関数
価値の最大化（符号反転で最小化問題に）：
```
f = -Σ_i v_i·x_i
```

#### 制約条件
重量の不等式制約：
```
Σ_i w_i·x_i ≤ C
```

Integerクラスを使った巧妙な表現：
- `total_weight ∈ [0, C]` という整数変数を導入
- 等式制約 `Σ_i w_i·x_i = total_weight` をペナルティ項として追加
- `total_weight` の定義域により、自動的に `Σ_i w_i·x_i ≤ C` が保証される

ペナルティ項：
```
H_constraint = (Σ_i w_i·x_i - total_weight)²
```

#### 完全なハミルトニアン
```
H = -Σ_i v_i·x_i + λ·(Σ_i w_i·x_i - total_weight)²
```

ここで、`λ = max(v_i)·N` はペナルティの強さ

### 2. バイナリ変数のみの定式化（比較用）

#### スラック変数の導入
容量制約 `Σ_i w_i·x_i ≤ C` を等式制約にするため、スラック変数sを導入：
```
Σ_i w_i·x_i + s = C
```

ここで `s = Σ_j 2^j·s_j` （バイナリ展開）

#### ハミルトニアン
```
H = -Σ_i v_i·x_i + λ·(Σ_i w_i·x_i + s - C)²
```

## 実装上の工夫

### 1. Integerクラスの利点
- **変数削減**: LogEncIntegerにより、容量Cを表現するのに`log₂(C)`個の変数で済む
- **自動的な範囲制約**: total_weight ∈ [0, C]が自動的に保証される
- **効率的なエンコーディング**: 対数エンコーディングによりQUBO変数数を削減

### 2. ペナルティ係数の設定
```python
λ = max(values) * n
```
- 全アイテムの価値の合計よりも大きい値に設定
- 制約違反のペナルティが目的関数を上回るように調整

### 3. 計算量の比較
| 手法 | バイナリ変数数 | 計算量 |
|------|---------------|--------|
| Integerクラス使用 | N + log₂(C) | O(N·log(C)) |
| バイナリ変数のみ | N + log₂(C) | O(N·log(C)) |

両手法とも変数数は同程度だが、Integerクラスの方が：
- コードが簡潔
- 制約の表現が自然
- デバッグが容易

### 4. LogEncIntegerの内部動作
```python
total_weight = LogEncInteger('total_weight', (0, capacity))
```

内部では以下のようにバイナリ変数に展開：
- `total_weight = Σ_k 2^k·b_k` （b_kはバイナリ変数）
- 必要なビット数: `⌈log₂(capacity+1)⌉`

## 実装コードとの対応

```python
# 目的関数（価値の最大化）
total_value = sum(values[i] * x[i] for i in range(n))
objective = -total_value

# 重量制約
weight_sum = sum(weights[i] * x[i] for i in range(n))
weight_constraint = Constraint((weight_sum - total_weight) ** 2, label='weight_equality')

# 完全なハミルトニアン
H = objective + lambda_weight * weight_constraint
```

## 物理的解釈
- 各バイナリ変数x_iはスピン状態に対応
- エネルギー最小状態が最適な選択を表現
- 制約違反はエネルギーペナルティとして現れる