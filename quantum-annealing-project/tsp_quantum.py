from pyqubo import Binary
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import matplotlib.pyplot as plt
import time


def solve_tsp_quantum(distance_matrix, num_reads=100):
    """TSPを量子アニーリング（D-Wave）で解く"""
    n_cities = len(distance_matrix)
    print(f"都市数: {n_cities}")
    
    # バイナリ変数
    x = [[Binary(f'x_{i}_{t}') for t in range(n_cities)] for i in range(n_cities)]
    
    # 制約項
    constraints = 0
    for t in range(n_cities):
        constraints += (sum(x[i][t] for i in range(n_cities)) - 1) ** 2
    for i in range(n_cities):
        constraints += (sum(x[i][t] for t in range(n_cities)) - 1) ** 2
    
    # 目的関数（経路の総距離）
    distance_obj = 0
    for t in range(n_cities):
        t_next = (t + 1) % n_cities
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:  # 自己ループを除外
                    distance_obj += distance_matrix[i][j] * x[i][t] * x[j][t_next]
    
    # ハミルトニアン（ペナルティ係数を適度に設定）
    lambda_constraint = np.max(distance_matrix) * n_cities / 50
    H = distance_obj + lambda_constraint * constraints
    
    print(f"  最大距離: {np.max(distance_matrix):.2f}")
    print(f"  ペナルティ係数: {lambda_constraint:.2f}")
    
    # コンパイル
    print("コンパイル中...")
    start = time.time()
    model = H.compile()
    qubo, offset = model.to_qubo()
    compile_time = time.time() - start
    print(f"コンパイル時間: {compile_time:.2f}秒")
    print(f"QUBO変数数: {len(qubo)}")
    
    # 量子アニーリング（D-Wave）
    print(f"量子アニーリング実行中... (num_reads={num_reads})")
    try:
        sampler = EmbeddingComposite(DWaveSampler())
        start = time.time()
        response = sampler.sample_qubo(qubo, num_reads=num_reads)
        solve_time = time.time() - start
        print(f"量子アニーリング時間: {solve_time:.2f}秒")
        
        # QPU情報を表示
        if hasattr(response, 'info'):
            timing = response.info.get('timing', {})
            print(f"QPU実行時間: {timing.get('qpu_access_time', 0)/1000:.3f}秒")
    
    except Exception as e:
        print(f"D-Wave接続エラー: {e}")
        print("代替としてシミュレーテッドアニーリングを使用します")
        from dwave.samplers import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
        start = time.time()
        response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=3000)
        solve_time = time.time() - start
        print(f"シミュレーテッドアニーリング時間: {solve_time:.2f}秒")
    
    # 結果を取得
    best_sample = response.first.sample
    route = []
    for t in range(n_cities):
        for i in range(n_cities):
            if best_sample.get(f'x_{i}_{t}', 0) == 1:
                route.append(i)
                break
    
    # 制約チェック
    valid_route = len(route) == n_cities and len(set(route)) == n_cities and -1 not in route
    
    # 総距離計算
    total_distance = 0
    if valid_route:
        for i in range(len(route)):
            total_distance += distance_matrix[route[i]][route[(i+1)%len(route)]]
    else:
        total_distance = float('inf')
        print(f"警告: 無効な経路 {route}")
    
    # 制約違反の計算
    constraint_violation = 0
    # 時間制約違反
    for t in range(n_cities):
        cities_at_time_t = sum(1 for i in range(n_cities) if best_sample.get(f'x_{i}_{t}', 0) == 1)
        constraint_violation += (cities_at_time_t - 1) ** 2
    # 都市制約違反
    for i in range(n_cities):
        times_for_city_i = sum(1 for t in range(n_cities) if best_sample.get(f'x_{i}_{t}', 0) == 1)
        constraint_violation += (times_for_city_i - 1) ** 2
    
    # デバッグ情報
    print(f"  経路の長さ: {len(route)}, ユニーク都市数: {len(set(route))}")
    print(f"  有効な経路: {valid_route}")
    print(f"  総距離: {total_distance:.2f}")
    print(f"  制約違反: {constraint_violation}")
    print(f"  エネルギー: {response.first.energy:.2f}")
    print(f"  距離成分: {total_distance:.2f}, ペナルティ成分: {constraint_violation * lambda_constraint:.2f}")
    
    return route, total_distance, response.first.energy, response, compile_time, solve_time


def run_quantum_reads_experiment(n_cities=30, trials=10):
    """num_readsを変えた量子アニーリング実験"""
    print(f"\n=== 量子アニーリング実験 ({n_cities}都市, {trials}回試行) ===")
    
    # 固定問題インスタンス（シミュレーテッドアニーリングと同じ）
    cities = create_cities(n_cities, seed=42)
    distance_matrix = calculate_distances(cities)
    
    # 様々なnum_readsで実験
    num_reads_list = [10, 50, 100, 500, 1000]
    results = []
    
    for num_reads in num_reads_list:
        print(f"\nnum_reads={num_reads}での実験...")
        
        times = []
        energies = []
        distances = []
        
        for trial in range(trials):
            # 各トライアルで異なるシード値を使用
            np.random.seed(42 + trial)
            route, distance, energy, response, compile_time, annealing_time = solve_tsp_quantum(
                distance_matrix, num_reads=num_reads
            )
            
            # 量子アニーリング時間のみを記録
            times.append(annealing_time)
            energies.append(energy)
            distances.append(distance)
        
        # 最小値を取得（最適化問題では最小値が重要）
        min_energy_idx = np.argmin(energies)
        min_energy = energies[min_energy_idx]
        min_distance = distances[min_energy_idx]
        min_time = times[min_energy_idx]
        
        avg_time = np.mean(times)
        
        results.append({
            'num_reads': num_reads,
            'times': times,
            'energies': energies,
            'distances': distances,
            'min_time': min_time,
            'min_energy': min_energy,
            'min_distance': min_distance,
            'avg_time': avg_time
        })
        
        print(f"  最小エネルギー: {min_energy:.2f} (時間: {min_time:.2f}秒)")
        print(f"  最小距離: {min_distance:.2f}")
        print(f"  平均時間: {avg_time:.2f}秒")
    
    # グラフ描画
    plot_quantum_results(results)
    
    return results


def plot_quantum_results(results):
    """量子アニーリング結果をプロット"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    num_reads = [r['num_reads'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    min_energies = [r['min_energy'] for r in results]
    min_distances = [r['min_distance'] for r in results]
    
    # 1. Time vs Reads (平均時間)
    ax1.errorbar(num_reads, avg_times, 
                yerr=[np.std(r['times']) for r in results],
                fmt='bo-', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Number of Reads')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Quantum Annealing Time vs Number of Reads')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Energy vs Reads (最小エネルギー)
    ax2.plot(num_reads, min_energies, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Reads')
    ax2.set_ylabel('Best Hamiltonian Value (Energy)')
    ax2.set_title('Best Solution Quality vs Number of Reads')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Distance vs Reads (最小距離)
    ax3.plot(num_reads, min_distances, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Reads')
    ax3.set_ylabel('Best Total Distance')
    ax3.set_title('Best Route Distance vs Number of Reads')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. 有効解の割合
    valid_rates = []
    for r in results:
        valid_count = sum(1 for d in r['distances'] if d != float('inf'))
        valid_rates.append(valid_count / len(r['distances']) * 100)
    
    ax4.plot(num_reads, valid_rates, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Reads')
    ax4.set_ylabel('Valid Solution Rate (%)')
    ax4.set_title('Constraint Satisfaction vs Number of Reads')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.show()


def create_cities(n_cities, seed=None):
    """都市座標を生成"""
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100


def calculate_distances(cities):
    """距離行列を計算"""
    n = len(cities)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i][j] = np.linalg.norm(cities[i] - cities[j])
    return distances


def plot_route(cities, route):
    """経路を可視化"""
    plt.figure(figsize=(10, 8))
    
    # 都市
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=200, zorder=3)
    for i, city in enumerate(cities):
        plt.annotate(str(i), (city[0], city[1]), fontsize=12, ha='center', va='center')
    
    # 経路
    for i in range(len(route)):
        start = cities[route[i]]
        end = cities[route[(i + 1) % len(route)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2, alpha=0.7)
    
    total_distance = sum(np.linalg.norm(cities[route[i]] - cities[route[(i+1)%len(route)]]) for i in range(len(route)))
    plt.title(f"Quantum TSP Route - Total Distance: {total_distance:.2f}")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    # 30都市で実行
    n = 30
    cities = create_cities(n)
    distances = calculate_distances(cities)
    
    print(f"=== {n}都市TSP（量子アニーリング）===")
    route, distance, energy, response, compile_time, annealing_time = solve_tsp_quantum(distances)
    
    print(f"\n結果:")
    print(f"経路: {route}")
    print(f"総距離: {distance:.2f}")
    print(f"エネルギー: {energy:.2f}")
    
    plot_route(cities, route)
    
    # 実験実行
    print("\n\n量子アニーリング実験を実行しますか？ (y/n): ", end="")
    if input().lower() == 'y':
        run_quantum_reads_experiment(n_cities=30, trials=10)