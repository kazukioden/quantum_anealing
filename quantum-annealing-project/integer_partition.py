from pyqubo import Spin
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler


def solve_integer_partition(numbers):
    """
    整数分割問題を量子アニーリングで解く
    
    Args:
        numbers: 分割したい整数のリスト
    
    Returns:
        解の辞書
    """
    n = len(numbers)
    
    # スピン変数を定義 (sᵢ = +1: 集合A, sᵢ = -1: 集合B)
    x = [Spin(f'x{i}') for i in range(n)]
    
    # 目的関数: 集合の和の差の二乗を最小化
    # スピン変数により S_A - S_B = ∑ nᵢ·sᵢ と表現
    # (S_A - S_B)² を直接最小化（制約不要）
    sum_diff = sum(numbers[i] * x[i] for i in range(n))  # S_A - S_B
    objective = sum_diff ** 2
    
    # QUBOモデルにコンパイル
    model = objective.compile()
    qubo, offset = model.to_qubo()
    
    # シミュレーテッドアニーリングで解く
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(qubo, num_reads=1000)
    
    # 最良解を取得
    best_sample = response.first.sample
    
    # 結果を解釈
    group_a = []
    group_b = []
    
    for i in range(n):
        if best_sample.get(f'x{i}', 0) == 1:
            group_a.append(numbers[i])
        else:
            group_b.append(numbers[i])
    
    return {
        'group_a': group_a,
        'group_b': group_b,
        'sum_a': sum(group_a),
        'sum_b': sum(group_b),
        'difference': abs(sum(group_a) - sum(group_b)),
        'energy': response.first.energy
    }


if __name__ == "__main__":
    # サンプル問題: [3, 1, 1, 2, 2, 1] を2つの集合に分割
    numbers = [3, 1, 1, 2, 2, 1]
    
    print(f"分割する数値: {numbers}")
    print(f"総和: {sum(numbers)}")
    
    result = solve_integer_partition(numbers)
    
    print("\n=== 解 ===")
    print(f"グループA: {result['group_a']} (和: {result['sum_a']})")
    print(f"グループB: {result['group_b']} (和: {result['sum_b']})")
    print(f"差: {result['difference']}")
    print(f"エネルギー: {result['energy']}")
    
    # 別の例も試す
    print("\n" + "="*50)
    numbers2 = [10, 8, 7, 6, 5]
    print(f"分割する数値: {numbers2}")
    print(f"総和: {sum(numbers2)}")
    
    result2 = solve_integer_partition(numbers2)
    
    print("\n=== 解 ===")
    print(f"グループA: {result2['group_a']} (和: {result2['sum_a']})")
    print(f"グループB: {result2['group_b']} (和: {result2['sum_b']})")
    print(f"差: {result2['difference']}")
    print(f"エネルギー: {result2['energy']}")