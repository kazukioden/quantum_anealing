from pyqubo import Binary
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import permutations


def solve_tsp(distance_matrix, num_reads=100, num_sweeps=3000):
    """TSPをシミュレーテッドアニーリングで解く"""
    n_cities = len(distance_matrix)
    print(f"都市数: {n_cities}")
    
    # 20都市以上の場合はパラメータを調整
    if n_cities >= 20:
        num_reads = 50
        num_sweeps = 5000
    
    # バイナリ変数: x[i][t] = 1 なら時刻tに都市iを訪問
    x = [[Binary(f'x_{i}_{t}') for t in range(n_cities)] for i in range(n_cities)]
    
    # 制約項
    constraints = 0
    
    # 各時刻に1都市
    for t in range(n_cities):
        constraints += (sum(x[i][t] for i in range(n_cities)) - 1) ** 2
    
    # 各都市を1回訪問
    for i in range(n_cities):
        constraints += (sum(x[i][t] for t in range(n_cities)) - 1) ** 2
    
    # 目的関数: 総移動距離
    distance_obj = 0
    for t in range(n_cities):
        t_next = (t + 1) % n_cities
        for i in range(n_cities):
            for j in range(n_cities):
                distance_obj += distance_matrix[i][j] * x[i][t] * x[j][t_next]
    
    # ハミルトニアン
    lambda_constraint = np.max(distance_matrix) * n_cities
    H = distance_obj + lambda_constraint * constraints
    
    # コンパイル
    print("コンパイル中...")
    start = time.time()
    model = H.compile()
    qubo, offset = model.to_qubo()
    print(f"コンパイル時間: {time.time() - start:.2f}秒")
    
    # アニーリング
    print(f"アニーリング実行中... (num_reads={num_reads}, num_sweeps={num_sweeps})")
    sampler = SimulatedAnnealingSampler()
    start = time.time()
    response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps)
    print(f"実行時間: {time.time() - start:.2f}秒")
    
    # 最良解から経路を復元
    best_sample = response.first.sample
    route = []
    for t in range(n_cities):
        for i in range(n_cities):
            if best_sample.get(f'x_{i}_{t}', 0) == 1:
                route.append(i)
                break
    
    # 総距離計算
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i]][route[(i+1)%len(route)]]
    
    return route, total_distance, response.first.energy


def create_cities(n_cities, seed=None):
    """ランダムな都市座標を生成"""
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
    
    plt.title(f"TSP Route - Total Distance: {sum(np.linalg.norm(cities[route[i]] - cities[route[(i+1)%len(route)]]) for i in range(len(route))):.2f}")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    # 20都市で実行
    n = 20
    cities = create_cities(n)
    distances = calculate_distances(cities)
    
    print(f"=== {n}都市TSP ===")
    route, distance, energy = solve_tsp(distances)
    
    print(f"\n結果:")
    print(f"経路: {route}")
    print(f"総距離: {distance:.2f}")
    print(f"エネルギー: {energy:.2f}")
    
    plot_route(cities, route)


def find_optimal_solution(distance_matrix, max_cities=8):
    """総当たり法で最適解を求める（小規模問題のみ）"""
    n = len(distance_matrix)
    if n > max_cities:
        return None, float('inf')
    
    min_distance = float('inf')
    best_route = None
    
    # 都市0を起点に固定してpermuteする
    for perm in permutations(range(1, n)):
        route = [0] + list(perm)
        distance = 0
        for i in range(n):
            distance += distance_matrix[route[i]][route[(i+1)%n]]
        
        if distance < min_distance:
            min_distance = distance
            best_route = route
    
    return best_route, min_distance


def run_time_to_solution_experiment(n_cities=10, trials=20):
    """Time to solution実験"""
    print(f"\n=== Time to Solution実験 ({n_cities}都市, {trials}回試行) ===")
    
    # 固定された問題インスタンス
    cities = create_cities(n_cities, seed=42)
    distance_matrix = calculate_distances(cities)
    
    # 最適解を求める（小規模問題のみ）
    optimal_route, optimal_distance = find_optimal_solution(distance_matrix)
    if optimal_route is None:
        print("問題が大きすぎて最適解を求められません")
        return
    
    print(f"最適解: 距離={optimal_distance:.2f}")
    
    # 様々なnum_readsで実験
    num_reads_list = [10, 20, 50, 100, 200, 500]
    results = []
    
    for num_reads in num_reads_list:
        print(f"\nnum_reads={num_reads}での実験...")
        
        times = []
        gaps = []
        success_count = 0
        
        for trial in range(trials):
            start_time = time.time()
            route, distance, energy = solve_tsp(distance_matrix, num_reads=num_reads, num_sweeps=1000)
            solve_time = time.time() - start_time
            
            # Gap = (found_solution - optimal) / optimal * 100
            gap = (distance - optimal_distance) / optimal_distance * 100
            
            times.append(solve_time)
            gaps.append(gap)
            
            if gap < 0.01:  # 0.01%以内なら成功
                success_count += 1
        
        avg_time = np.mean(times)
        avg_gap = np.mean(gaps)
        success_rate = success_count / trials * 100
        
        results.append({
            'num_reads': num_reads,
            'avg_time': avg_time,
            'avg_gap': avg_gap,
            'success_rate': success_rate,
            'times': times,
            'gaps': gaps
        })
        
        print(f"  平均時間: {avg_time:.2f}秒")
        print(f"  平均Gap: {avg_gap:.2f}%")
        print(f"  成功率: {success_rate:.1f}%")
    
    # グラフを描画
    plot_time_to_solution(results)
    
    return results


def plot_time_to_solution(results):
    """Time to solutionとGapのグラフを描画"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    num_reads = [r['num_reads'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    avg_gaps = [r['avg_gap'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    # 1. Time to Solution
    ax1.plot(num_reads, avg_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Reads')
    ax1.set_ylabel('Average Time (seconds)')
    ax1.set_title('Time to Solution')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Gap vs num_reads
    ax2.plot(num_reads, avg_gaps, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Reads')
    ax2.set_ylabel('Average Gap (%)')
    ax2.set_title('Solution Quality (Gap)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Success Rate
    ax3.plot(num_reads, success_rates, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Reads')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate (Gap < 0.01%)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Time vs Gap散布図
    all_times = []
    all_gaps = []
    colors = []
    color_map = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        all_times.extend(result['times'])
        all_gaps.extend(result['gaps'])
        colors.extend([color_map[i]] * len(result['times']))
    
    scatter = ax4.scatter(all_times, all_gaps, c=colors, alpha=0.6)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Gap (%)')
    ax4.set_title('Time vs Gap')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_scaling_experiment():
    """都市数に対するスケーリング実験"""
    print("\n=== スケーリング実験 ===")
    
    city_sizes = [6, 8, 10, 12, 15]
    results = []
    
    for n_cities in city_sizes:
        print(f"\n{n_cities}都市での実験...")
        
        # 固定問題インスタンス
        cities = create_cities(n_cities, seed=42)
        distance_matrix = calculate_distances(cities)
        
        # 最適解（小規模のみ）
        optimal_route, optimal_distance = find_optimal_solution(distance_matrix)
        
        if optimal_route is None:
            print(f"  {n_cities}都市: 最適解計算不可")
            continue
        
        # アニーリング実行
        times = []
        gaps = []
        
        for trial in range(10):
            start_time = time.time()
            route, distance, energy = solve_tsp(distance_matrix, num_reads=100, num_sweeps=2000)
            solve_time = time.time() - start_time
            
            gap = (distance - optimal_distance) / optimal_distance * 100
            
            times.append(solve_time)
            gaps.append(gap)
        
        avg_time = np.mean(times)
        avg_gap = np.mean(gaps)
        
        results.append({
            'n_cities': n_cities,
            'avg_time': avg_time,
            'avg_gap': avg_gap,
            'optimal_distance': optimal_distance
        })
        
        print(f"  平均時間: {avg_time:.2f}秒")
        print(f"  平均Gap: {avg_gap:.2f}%")
    
    # スケーリングプロット
    plot_scaling_results(results)
    
    return results


def plot_scaling_results(results):
    """スケーリング結果をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    n_cities = [r['n_cities'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    avg_gaps = [r['avg_gap'] for r in results]
    
    # 計算時間のスケーリング
    ax1.plot(n_cities, avg_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Cities')
    ax1.set_ylabel('Average Time (seconds)')
    ax1.set_title('Computation Time Scaling')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 解の品質のスケーリング
    ax2.plot(n_cities, avg_gaps, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Cities')
    ax2.set_ylabel('Average Gap (%)')
    ax2.set_title('Solution Quality Scaling')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 20都市で実行
    n = 20
    cities = create_cities(n)
    distances = calculate_distances(cities)
    
    print(f"=== {n}都市TSP ===")
    route, distance, energy = solve_tsp(distances)
    
    print(f"\n結果:")
    print(f"経路: {route}")
    print(f"総距離: {distance:.2f}")
    print(f"エネルギー: {energy:.2f}")
    
    plot_route(cities, route)
    
    # 実験実行
    print("\n\n実験を実行しますか？")
    print("1. Time to Solution実験 (10都市)")
    print("2. スケーリング実験 (6-15都市)")
    print("3. 両方実行")
    choice = input("選択 (1/2/3): ")
    
    if choice in ['1', '3']:
        run_time_to_solution_experiment(n_cities=10, trials=20)
    
    if choice in ['2', '3']:
        run_scaling_experiment()