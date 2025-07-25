# Quantum Annealing TSP Project

量子アニーリングを用いた巡回セールスマン問題（TSP）の実装と、シミュレーテッド・アニーリング（SA）とシミュレーテッド量子アニーリング（SQA）の比較研究。

## 概要

本プロジェクトでは、PyQuboとOpenJijを使用して、TSPの様々な定式化と解法を実装しています。特に、制約違反を最小化するための係数最適化と、量子揺らぎの影響を定量的に評価することに焦点を当てています。

## 主な機能

### 1. TSP実装バリエーション
- **単純実装** (`tsp_simple.py`): 基本的な順列表現
- **エッジ表現** (`tsp_edge.py`): 二値変数によるエッジ選択
- **部分巡回除去** (`tsp_edge_subtour.py`): 三角形・四角形ペナルティ
- **量子アニーリング対応** (`tsp_edge_quantum.py`): SA/SQA比較実装

### 2. 最適化機能
- **Optuna統合**: 制約係数の自動最適化
- **トロッター数スケーリング**: SQA用パラメータの理論的調整
- **制約違反最小化**: 次数制約 + 三角形ループ + 単一ツアー

### 3. 実験・分析ツール
- **SA vs SQA比較**: 横磁場の有無による性能差
- **sweep数依存性分析** (`sweep_comparison.py`): 収束性評価

## 環境構築

```bash
# uvを使用（推奨）
uv venv
uv sync

# または pip を使用
pip install -r pyproject.toml
```

### 必要な依存関係
- Python >= 3.11
- pyqubo >= 1.5.0
- openjij >= 0.10.14
- optuna >= 3.0.0
- matplotlib >= 3.10.3

## 使い方

### 基本的なTSP実行
```python
# 20都市のTSP（シンプル版）
uv run python tsp_simple.py

# エッジ表現 + 部分巡回除去
uv run python tsp_edge_subtour.py
```

### SA/SQA比較実験
```python
# Optuna最適化 + 比較実験
uv run python tsp_edge_quantum.py
# → 選択肢2を選ぶと自動最適化実行

# sweep数依存性分析
uv run python sweep_comparison.py
```

## 実験結果の要点

### SA vs SQA の制約満足性
- **SA（横磁場なし）**: 適切な係数で20%の成功率
- **SQA（横磁場=0）**: 同じ係数でも0%の成功率
- **原因**: トロッター分解による制約バリアの希釈

### 最適化された係数例（15都市）
- **SA**: degree=8.83, triangle=0.18
- **SQA**: trotter=32, degree=370.24, triangle=20.8
  - 約40倍の係数強化が必要

## プロジェクト構造

```
quantum-annealing-project/
├── documents/           # 理論的背景・数式ドキュメント
│   ├── tsp_hamiltonian.md
│   ├── integer_partition_hamiltonian.md
│   └── knapsack_hamiltonian.md
├── tsp_*.py            # TSP実装各種
├── sweep_comparison.py  # 実験用スクリプト
├── config.py           # D-Wave API設定（使用時）
└── pyproject.toml      # 依存関係管理
```

## 技術的詳細

### エッジ表現のハミルトニアン

```
H = Σ d_ij * e_ij + λ_degree * Σ (Σ e_ij - 2)² + λ_triangle * Σ triangle_penalty
```

- 第1項: 総距離最小化
- 第2項: 次数制約（各都市から2本）
- 第3項: 三角形抑制ペナルティ

詳細は[documents/tsp_hamiltonian.md](documents/tsp_hamiltonian.md)を参照。

## 参考文献

- OpenJij公式ドキュメント
- Suzuki-Trotter分解とSQAの理論（arXiv:2501.03518）
- 制約最適化における量子アニーリング（Physical Review X 8, 031016）

## ライセンス

MIT License