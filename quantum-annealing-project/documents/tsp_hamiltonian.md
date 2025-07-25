# TSP（巡回セールスマン問題）のハミルトニアン定義

## 問題設定
N個の都市を全て1回ずつ訪問して出発地に戻る最短経路を見つける問題。

## 変数定義
バイナリ変数を使った順列行列表現：
- `x[i][t] ∈ {0, 1}`: 時刻tに都市iを訪問するかどうか
- i = 0, 1, ..., N-1 (都市インデックス)
- t = 0, 1, ..., N-1 (訪問順序)

## ハミルトニアンの構成

### 制約条件

#### 1. 時間制約：各時刻に訪問する都市は1つ
```
H_time = Σ_t (Σ_i x[i][t] - 1)²
```

#### 2. 都市制約：各都市を1回だけ訪問
```
H_city = Σ_i (Σ_t x[i][t] - 1)²
```

### 目的関数
総移動距離の最小化：
```
H_distance = Σ_t Σ_i Σ_j d[i][j] × x[i][t] × x[j][t+1]
```
ここで、d[i][j]は都市i,j間の距離、t+1はmod Nで計算（巡回）

### 完全なハミルトニアン
```
H = H_distance + λ × (H_time + H_city)
```

λはペナルティ係数で、通常は最大距離×都市数程度に設定。

## 実装上の工夫

### 1. 順列行列表現の利点
- 各都市の訪問順序を明確に表現
- 制約条件が単純な等式制約として記述可能
- 対称性の破れを自然に回避

### 2. 計算量
- バイナリ変数数: N²
- 制約項の数: 2N
- 距離項の数: N³（最悪ケース）
- QUBO項数: O(N⁴)

### 3. ペナルティ係数の設定
```python
λ = max(d[i][j]) × N
```
制約違反のペナルティが距離の改善を上回るように設定。

### 4. 効率化のポイント
- 自己ループ（i=j）を除外して項数を削減
- 対称な距離行列の場合は重複計算を避ける
- 大規模問題では近傍のみを考慮する近似も可能

## 15都市前後での実装指針

### アニーリングパラメータ
```python
num_reads = 100      # 実行回数
num_sweeps = 3000    # 各実行のスイープ数
beta_range = (0.1, 10.0)  # 逆温度の範囲
```

### 問題サイズと計算時間
| 都市数 | 変数数 | QUBO項数（概算） | 計算時間（目安） |
|--------|--------|------------------|------------------|
| 10     | 100    | ~10,000          | 数秒             |
| 15     | 225    | ~50,000          | 10-30秒         |
| 20     | 400    | ~160,000         | 1-2分            |

### 解の品質向上のための工夫
1. **初期解の設定**: 貪欲法などで良い初期解を与える
2. **温度スケジュール**: 幾何的冷却が効果的
3. **複数実行**: num_readsを増やして最良解を選択

## コード実装との対応

```python
# 制約項（各時刻・各都市）
for t in range(n_cities):
    constraints += (sum(x[i][t] for i in range(n_cities)) - 1) ** 2
for i in range(n_cities):
    constraints += (sum(x[i][t] for t in range(n_cities)) - 1) ** 2

# 距離項
for t in range(n_cities):
    t_next = (t + 1) % n_cities
    for i in range(n_cities):
        for j in range(n_cities):
            distance_obj += distance_matrix[i][j] * x[i][t] * x[j][t_next]

# ハミルトニアン
H = distance_obj + λ * constraints
```

## 物理的解釈
- エネルギー最小状態が最短経路に対応
- 制約違反は高エネルギー状態として排除
- アニーリング過程で局所最適解から脱出