from pyqubo import Binary
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
import matplotlib.pyplot as plt
import time


def solve_tsp_edge_encoding(distance_matrix, num_reads=100, num_sweeps=3000):
    """TSPをエッジ表現で解く"""
    n_cities = len(distance_matrix)
    print(f"都市数: {n_cities}")
    
    # エッジ変数: e[i][j] = 1 なら都市i,j間のエッジを使用
    e = {}
    for i in range(n_cities):
        for j in range(i+1, n_cities):  # 無向グラフなのでi<jのみ
            e[(i, j)] = Binary(f'e_{i}_{j}')
    
    # 制約1: 各都市の次数は2（各都市から2本のエッジ）
    degree_constraints = 0
    for i in range(n_cities):
        degree = 0
        for j in range(n_cities):
            if i < j and (i, j) in e:
                degree += e[(i, j)]
            elif i > j and (j, i) in e:
                degree += e[(j, i)]
        degree_constraints += (degree - 2) ** 2
    
    # 制約2: 部分循環の排除（Miller-Tucker-Zemlin制約の代替）
    # シンプルな近似：エッジ数がちょうどn本
    total_edges = sum(e[(i, j)] for i, j in e.keys())
    edge_count_constraint = (total_edges - n_cities) ** 2
    
    # 目的関数: 総距離の最小化
    distance_obj = 0
    for (i, j), edge_var in e.items():
        distance_obj += distance_matrix[i][j] * edge_var
    
    # ハミルトニアン
    lambda_degree = np.max(distance_matrix) * n_cities / 10
    lambda_edge = np.max(distance_matrix) * n_cities / 20
    H = distance_obj + lambda_degree * degree_constraints + lambda_edge * edge_count_constraint
    
    print(f"  最大距離: {np.max(distance_matrix):.2f}")
    print(f"  エッジ変数数: {len(e)}")
    print(f"  次数制約係数: {lambda_degree:.2f}")
    print(f"  エッジ数制約係数: {lambda_edge:.2f}")
    
    # コンパイル
    print("コンパイル中...")
    start = time.time()
    model = H.compile()
    qubo, offset = model.to_qubo()
    compile_time = time.time() - start
    print(f"コンパイル時間: {compile_time:.2f}秒")
    print(f"QUBO変数数: {len(qubo)}")
    
    # アニーリング
    print(f"アニーリング実行中... (num_reads={num_reads}, num_sweeps={num_sweeps})")
    sampler = SimulatedAnnealingSampler()
    start = time.time()
    seed = np.random.randint(0, 2**31)
    response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    solve_time = time.time() - start
    print(f"アニーリング時間: {solve_time:.2f}秒")
    
    # 結果を取得
    best_sample = response.first.sample
    selected_edges = []
    for (i, j), edge_var in e.items():
        if best_sample.get(f'e_{i}_{j}', 0) == 1:
            selected_edges.append((i, j))
    
    print(f"  選択されたエッジ数: {len(selected_edges)}")
    print(f"  選択されたエッジ: {selected_edges}")
    
    # エッジから経路を構築（可能であれば）
    route, total_distance = construct_route_from_edges(selected_edges, distance_matrix, n_cities)
    
    # 制約チェック
    degree_violations = 0
    for i in range(n_cities):
        degree = 0
        for (u, v) in selected_edges:
            if u == i or v == i:
                degree += 1
        degree_violations += (degree - 2) ** 2
    
    edge_count_violation = (len(selected_edges) - n_cities) ** 2
    
    print(f"  構築された経路: {route}")
    print(f"  総距離: {total_distance:.2f}")
    print(f"  次数制約違反: {degree_violations}")
    print(f"  エッジ数制約違反: {edge_count_violation}")
    print(f"  エネルギー: {response.first.energy:.2f}")
    
    return route, total_distance, response.first.energy, selected_edges, compile_time, solve_time


def construct_route_from_edges(edges, distance_matrix, n_cities):
    """エッジリストから経路を構築"""
    if len(edges) != n_cities:
        return [], float('inf')
    
    # 隣接リストを構築
    adj = {i: [] for i in range(n_cities)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # 各頂点の次数をチェック
    for i in range(n_cities):
        if len(adj[i]) != 2:
            return [], float('inf')
    
    # 経路を構築（0から開始）
    route = [0]
    current = 0
    prev = -1
    
    for _ in range(n_cities - 1):
        neighbors = adj[current]
        next_node = None
        for neighbor in neighbors:
            if neighbor != prev:
                next_node = neighbor
                break
        
        if next_node is None:
            return [], float('inf')  # 経路が構築できない
        
        route.append(next_node)
        prev = current
        current = next_node
    
    # 最後のノードから最初のノードに戻れるかチェック
    if 0 not in adj[current]:
        return [], float('inf')
    
    # 距離計算
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i]][route[(i+1) % len(route)]]
    
    return route, total_distance


def run_edge_experiment(n_cities=20, trials=10):
    """エッジ表現での実験"""
    print(f"\n=== エッジ表現実験 ({n_cities}都市, {trials}回試行) ===")
    
    # 固定問題インスタンス
    cities = create_cities(n_cities, seed=42)
    distance_matrix = calculate_distances(cities)
    
    # 様々なnum_sweepsで実験
    num_sweeps_list = [100, 500, 1000, 2000, 5000]
    results = []
    
    for num_sweeps in num_sweeps_list:
        print(f"\nnum_sweeps={num_sweeps}での実験...")
        
        times = []
        energies = []
        distances = []
        valid_solutions = 0
        
        for trial in range(trials):
            np.random.seed(42 + trial)
            route, distance, energy, edges, compile_time, annealing_time = solve_tsp_edge_encoding(
                distance_matrix, num_reads=50, num_sweeps=num_sweeps
            )
            
            times.append(annealing_time)
            energies.append(energy)
            distances.append(distance)
            
            if distance != float('inf'):
                valid_solutions += 1
        
        # 有効解のみで統計を計算
        valid_distances = [d for d in distances if d != float('inf')]
        if valid_distances:
            min_distance = min(valid_distances)
            avg_distance = np.mean(valid_distances)
        else:
            min_distance = float('inf')
            avg_distance = float('inf')
        
        min_energy = min(energies)
        avg_time = np.mean(times)
        valid_rate = valid_solutions / trials * 100
        
        results.append({
            'num_sweeps': num_sweeps,
            'times': times,
            'energies': energies,
            'distances': distances,
            'min_energy': min_energy,
            'min_distance': min_distance,
            'avg_distance': avg_distance,
            'avg_time': avg_time,
            'valid_rate': valid_rate
        })
        
        print(f"  最小エネルギー: {min_energy:.2f}")
        print(f"  最小距離: {min_distance:.2f}")
        print(f"  平均時間: {avg_time:.2f}秒")
        print(f"  有効解率: {valid_rate:.1f}%")
    
    # グラフ描画
    plot_edge_results(results)
    
    return results


def plot_edge_results(results):
    """エッジ表現の結果をプロット"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    num_sweeps = [r['num_sweeps'] for r in results]
    avg_times = [r['avg_time'] for r in results]
    min_energies = [r['min_energy'] for r in results]
    min_distances = [r['min_distance'] for r in results]
    valid_rates = [r['valid_rate'] for r in results]
    
    # 1. Time vs Sweeps
    ax1.plot(num_sweeps, avg_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Sweeps')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Edge Encoding: Time vs Sweeps')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Energy vs Sweeps
    ax2.plot(num_sweeps, min_energies, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Sweeps')
    ax2.set_ylabel('Best Energy')
    ax2.set_title('Edge Encoding: Best Energy vs Sweeps')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Distance vs Sweeps
    valid_distances = [d if d != float('inf') else None for d in min_distances]
    valid_sweeps = [s for s, d in zip(num_sweeps, valid_distances) if d is not None]
    valid_distances = [d for d in valid_distances if d is not None]
    
    if valid_distances:
        ax3.plot(valid_sweeps, valid_distances, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Sweeps')
    ax3.set_ylabel('Best Distance')
    ax3.set_title('Edge Encoding: Best Distance vs Sweeps')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Valid Solution Rate
    ax4.plot(num_sweeps, valid_rates, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Sweeps')
    ax4.set_ylabel('Valid Solution Rate (%)')
    ax4.set_title('Edge Encoding: Valid Solution Rate')
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


def plot_route(cities, route, edges=None):
    """経路を可視化"""
    plt.figure(figsize=(10, 8))
    
    # 都市
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=200, zorder=3)
    for i, city in enumerate(cities):
        plt.annotate(str(i), (city[0], city[1]), fontsize=12, ha='center', va='center')
    
    # 経路またはエッジ
    if route and len(route) > 1:
        for i in range(len(route)):
            start = cities[route[i]]
            end = cities[route[(i + 1) % len(route)]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2, alpha=0.7)
        
        total_distance = sum(np.linalg.norm(cities[route[i]] - cities[route[(i+1)%len(route)]]) for i in range(len(route)))
        plt.title(f"Edge TSP Route - Total Distance: {total_distance:.2f}")
    elif edges:
        for (i, j) in edges:
            start = cities[i]
            end = cities[j]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2, alpha=0.7)
        plt.title("Selected Edges (Edge Encoding)")
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    # 20都市で実行
    n = 20
    cities = create_cities(n)
    distances = calculate_distances(cities)
    
    print(f"=== {n}都市TSP（エッジ表現）===")
    route, distance, energy, edges, compile_time, annealing_time = solve_tsp_edge_encoding(distances)
    
    print(f"\n結果:")
    print(f"経路: {route}")
    print(f"総距離: {distance:.2f}")
    print(f"エネルギー: {energy:.2f}")
    
    plot_route(cities, route, edges)
    
    # 実験実行
    print("\n\nエッジ表現実験を実行しますか？ (y/n): ", end="")
    if input().lower() == 'y':
        run_edge_experiment(n_cities=20, trials=10)