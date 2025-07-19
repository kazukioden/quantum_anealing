# pyquboでシミュレーテッドアニーリングを解く基本フロー

## 概要
pyquboを使った量子アニーリング/シミュレーテッドアニーリングの実装には、以下の普遍的な流れがあります。

## 基本的な実装フロー

### 1. 変数定義
問題に応じて適切な変数タイプを選択します。

```python
# バイナリ変数（0 or 1）
x = Binary('x')
y = [Binary(f'y{i}') for i in range(n)]

# スピン変数（-1 or +1）- 自動でバイナリに変換される
s = Spin('s')

# 整数変数 - 内部で複数のバイナリ変数に展開
z = LogEncInteger('z', (0, 10))  # 対数エンコーディング（メモリ効率的）
w = OneHotEncInteger('w', (1, 5))  # ワンホットエンコーディング
```

### 2. ハミルトニアン定義
目的関数と制約条件を組み合わせてハミルトニアンを構築します。

```python
# 目的関数（最小化したい関数）
objective = sum(c[i] * x[i] for i in range(n))

# 制約条件（ペナルティ法で表現）
constraint = (sum(x[i] for i in range(n)) - k) ** 2

# 完全なハミルトニアン
lambda_penalty = 100  # ペナルティの強さ
H = objective + lambda_penalty * constraint
```

### 3. モデルのコンパイル
定義したハミルトニアンをコンパイルし、内部表現を最適化します。

```python
model = H.compile()
```

この段階で行われること：
- 式の展開と簡約化
- 変数の依存関係の解析
- 内部表現の最適化

### 4. QUBO変換
ハミルトニアンをQUBO（Quadratic Unconstrained Binary Optimization）形式に変換します。

```python
qubo, offset = model.to_qubo()
```

- すべての変数を{0,1}のバイナリに変換
- 二次形式の係数行列を生成
- 定数項（offset）を分離

### 5. サンプラーで解く
QUBOをシミュレーテッドアニーリングまたは量子アニーリングで解きます。

```python
# シミュレーテッドアニーリング（ローカル環境）
from dwave.samplers import SimulatedAnnealingSampler
sampler = SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo, num_reads=1000)

# D-Wave実機を使う場合
from dwave.system import DWaveSampler, EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(qubo, num_reads=100)
```

### 6. 結果の取得と解釈
得られた解を解釈し、元の問題の解として復元します。

```python
# 最良解の取得
best_sample = response.first.sample
best_energy = response.first.energy

# デコード（元の変数表現に戻す）
decoded = model.decode_sample(best_sample, vartype='BINARY')

# 制約違反のチェック
broken = decoded.constraints(only_broken=True)
if broken:
    print("制約違反:", broken)
```

## QUBOの数学的形式

QUBOは以下の形式で表現されます：

```
E(x) = Σᵢ aᵢxᵢ + Σᵢ<ⱼ bᵢⱼxᵢxⱼ + c
```

- 一次項と二次項のみ（高次項は自動的に二次に変換）
- xᵢ ∈ {0, 1}のバイナリ変数
- 最小化問題として定式化

## 高度な使用方法

### プレースホルダーを使った柔軟な定義
パラメータを後から調整可能にします。

```python
from pyqubo import Placeholder

# プレースホルダーでパラメータを定義
lambda_ph = Placeholder('lambda')
H = objective + lambda_ph * constraint

# コンパイル
model = H.compile()

# 実行時にパラメータを指定
qubo, offset = model.to_qubo(feed_dict={'lambda': 10.0})
```

### 複数の解の分析
上位の解を分析して、解の多様性を確認します。

```python
# 上位5個の解を表示
for idx, (sample, energy, occur) in enumerate(response.data(['sample', 'energy', 'num_occurrences'])):
    if idx < 5:
        print(f"Energy: {energy}, Occurrences: {occur}")
        print(f"Sample: {sample}")
        
        # デコードして元の変数値を確認
        decoded = model.decode_sample(sample, vartype='BINARY')
        print(f"Decoded: {decoded.sample}")
```

### Constraintクラスの活用
制約違反を明示的に管理します。

```python
from pyqubo import Constraint

# 名前付き制約
constraint1 = Constraint((sum(x) - 5)**2, label='sum_constraint')
constraint2 = Constraint((x[0] + x[1] - 1)**2, label='pair_constraint')

H = objective + constraint1 + constraint2
```

## よくあるパターン

### 1. 組合せ最適化
```python
# 変数定義
x = [Binary(f'x{i}') for i in range(n)]

# 目的関数と制約
H = sum(profit[i] * x[i] for i in range(n)) + λ * constraints
```

### 2. グラフ問題
```python
# エッジの選択
edges = [(i, j) for i in range(n) for j in range(i+1, n)]
x = {e: Binary(f'x{e}') for e in edges}
```

### 3. スケジューリング
```python
# 時間スロットへの割り当て
x = [[Binary(f'x_{i}_{t}') for t in range(T)] for i in range(n)]
```

## デバッグのヒント

1. **小規模問題で検証**: まず小さい問題サイズで動作確認
2. **制約の強さを調整**: ペナルティ係数λを段階的に増加
3. **エネルギー値を確認**: 制約違反時は高いエネルギー値になるはず
4. **複数回実行**: 確率的アルゴリズムなので複数回試行

この基本フローを理解していれば、様々な組合せ最適化問題をpyquboで実装できます。