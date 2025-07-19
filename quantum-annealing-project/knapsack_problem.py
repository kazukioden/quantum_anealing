from pyqubo import Binary, LogEncInteger, UnaryEncInteger, Constraint
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np


def solve_knapsack_integer(weights, values, capacity):
    """
    ナップサック問題をIntegerクラスを使って解く
    
    Args:
        weights: 各アイテムの重さのリスト
        values: 各アイテムの価値のリスト
        capacity: ナップサックの容量
    
    Returns:
        解の辞書
    """
    n = len(weights)
    
    # バイナリ変数：各アイテムを選ぶかどうか
    x = [Binary(f'x{i}') for i in range(n)]
    
    # 整数変数：選んだアイテムの総重量（LogEncIntegerで効率的に表現）
    # 容量の範囲は0からcapacityまで
    max_weight_bits = int(np.log2(capacity)) + 1
    total_weight = LogEncInteger('total_weight', (0, capacity))
    
    # 制約1: 選んだアイテムの重量の合計 = total_weight
    weight_sum = sum(weights[i] * x[i] for i in range(n))
    weight_constraint = Constraint((weight_sum - total_weight) ** 2, label='weight_equality')
    
    # 制約2: 総重量が容量以下（ペナルティ方式）
    # total_weight <= capacity は自動的に満たされる（変数の定義域による）
    
    # 目的関数：価値の最大化（符号を反転して最小化問題に）
    total_value = sum(values[i] * x[i] for i in range(n))
    objective = -total_value
    
    # ハミルトニアン（目的関数 + ペナルティ項）
    lambda_weight = max(values) * n  # ペナルティの強さ
    H = objective + lambda_weight * weight_constraint
    
    # コンパイルとQUBO変換
    model = H.compile()
    qubo, offset = model.to_qubo()
    
    # シミュレーテッドアニーリングで解く
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_reads=1000)
    
    # 最良解を取得
    best_sample = response.first.sample
    decoded_sample = model.decode_sample(best_sample, vartype='BINARY')
    
    # 結果を解釈
    selected_items = []
    total_weight_actual = 0
    total_value_actual = 0
    
    for i in range(n):
        if best_sample.get(f'x{i}', 0) == 1:
            selected_items.append(i)
            total_weight_actual += weights[i]
            total_value_actual += values[i]
    
    # 制約違反をチェック
    broken_constraints = decoded_sample.constraints(only_broken=True)
    
    return {
        'selected_items': selected_items,
        'total_weight': total_weight_actual,
        'total_value': total_value_actual,
        'capacity': capacity,
        'energy': response.first.energy,
        'valid': len(broken_constraints) == 0,
        'broken_constraints': broken_constraints
    }


def solve_knapsack_simple(weights, values, capacity):
    """
    ナップサック問題をバイナリ変数のみで解く（比較用）
    """
    n = len(weights)
    
    # バイナリ変数：各アイテムを選ぶかどうか
    x = [Binary(f'x{i}') for i in range(n)]
    
    # 容量制約をペナルティ項として追加
    weight_sum = sum(weights[i] * x[i] for i in range(n))
    
    # 容量超過分にペナルティ
    # 二次形式にするため、スラック変数を使用
    slack_bits = int(np.log2(capacity)) + 1
    slack = [Binary(f's{j}') for j in range(slack_bits)]
    slack_value = sum(2**j * slack[j] for j in range(slack_bits))
    
    # weight_sum + slack_value = capacity の制約
    capacity_constraint = (weight_sum + slack_value - capacity) ** 2
    
    # 目的関数：価値の最大化
    total_value = sum(values[i] * x[i] for i in range(n))
    
    # ハミルトニアン
    lambda_capacity = max(values) * n
    H = -total_value + lambda_capacity * capacity_constraint
    
    # コンパイルとQUBO変換
    model = H.compile()
    qubo, offset = model.to_qubo()
    
    # 解く
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_reads=1000)
    
    # 結果を解釈
    best_sample = response.first.sample
    selected_items = []
    total_weight = 0
    total_value_result = 0
    
    for i in range(n):
        if best_sample.get(f'x{i}', 0) == 1:
            selected_items.append(i)
            total_weight += weights[i]
            total_value_result += values[i]
    
    return {
        'selected_items': selected_items,
        'total_weight': total_weight,
        'total_value': total_value_result,
        'capacity': capacity,
        'energy': response.first.energy,
        'valid': total_weight <= capacity
    }


if __name__ == "__main__":
    # テスト問題1: 小規模
    weights1 = [2, 1, 3, 2]
    values1 = [12, 10, 20, 15]
    capacity1 = 5
    
    print("=== ナップサック問題（Integerクラス使用）===")
    print(f"アイテムの重さ: {weights1}")
    print(f"アイテムの価値: {values1}")
    print(f"容量: {capacity1}")
    
    result1 = solve_knapsack_integer(weights1, values1, capacity1)
    
    print(f"\n選択されたアイテム: {result1['selected_items']}")
    print(f"総重量: {result1['total_weight']} / {result1['capacity']}")
    print(f"総価値: {result1['total_value']}")
    print(f"制約を満たしているか: {result1['valid']}")
    if not result1['valid']:
        print(f"違反した制約: {result1['broken_constraints']}")
    
    # 比較のためバイナリ変数のみの解法も実行
    print("\n=== ナップサック問題（バイナリ変数のみ）===")
    result2 = solve_knapsack_simple(weights1, values1, capacity1)
    
    print(f"選択されたアイテム: {result2['selected_items']}")
    print(f"総重量: {result2['total_weight']} / {result2['capacity']}")
    print(f"総価値: {result2['total_value']}")
    print(f"制約を満たしているか: {result2['valid']}")
    
    # テスト問題2: より大きな問題
    print("\n" + "="*50)
    weights2 = [10, 20, 30, 40, 50]
    values2 = [60, 100, 120, 140, 160]
    capacity2 = 80
    
    print("=== より大きなナップサック問題 ===")
    print(f"アイテムの重さ: {weights2}")
    print(f"アイテムの価値: {values2}")
    print(f"容量: {capacity2}")
    
    result3 = solve_knapsack_integer(weights2, values2, capacity2)
    
    print(f"\n選択されたアイテム: {result3['selected_items']}")
    print(f"総重量: {result3['total_weight']} / {result3['capacity']}")
    print(f"総価値: {result3['total_value']}")
    print(f"制約を満たしているか: {result3['valid']}")