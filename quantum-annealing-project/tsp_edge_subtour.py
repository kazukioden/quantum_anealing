from pyqubo import Binary
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# Fix matplotlib font for English display
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def solve_tsp_edge_with_subtour_elimination(distance_matrix, num_reads=100, num_sweeps=3000, 
                                          enable_triangle=True, enable_quad=False, 
                                          tri_penalty_factor=1.5, quad_penalty_factor=1.0):
    """小ループ抑制ペナルティ付きエッジ表現TSP"""
    n_cities = len(distance_matrix)
    print(f"都市数: {n_cities}")
    print(f"三角形ペナルティ: {enable_triangle}, 四角形ペナルティ: {enable_quad}")
    
    # エッジ変数: e[i][j] = 1 なら都市i,j間のエッジを使用
    e = {}
    for i in range(n_cities):
        for j in range(i+1, n_cities):  # 無向グラフなのでi<jのみ
            e[(i, j)] = Binary(f'e_{i}_{j}')
    
    print(f"エッジ変数数: {len(e)}")
    
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
    
    # 目的関数: 総距離の最小化
    distance_obj = 0
    for (i, j), edge_var in e.items():
        distance_obj += distance_matrix[i][j] * edge_var
    
    # 基本ペナルティ係数
    lambda_degree = np.max(distance_matrix) * n_cities / 10
    
    # ハミルトニアンの構築開始
    H = distance_obj + lambda_degree * degree_constraints
    
    # === 小ループ抑制ペナルティ ===
    
    # 三角形ペナルティ (k=3)
    triangle_count = 0
    if enable_triangle:
        P_TRI = lambda_degree * tri_penalty_factor * 0.1  # Much smaller penalty
        print(f"三角形ペナルティ係数: {P_TRI:.2f}")
        
        for i, j, k in itertools.combinations(range(n_cities), 3):
            # 三角形の3つのエッジを取得
            ij = e[(min(i,j), max(i,j))]
            jk = e[(min(j,k), max(j,k))]
            ik = e[(min(i,k), max(i,k))]
            
            # (y_ij + y_jk + y_ik - 3)^2 を展開
            # 真のツアーでは合計=2なので penalty=1、閉ループでは合計=3なので penalty=0
            # 実際は (sum - 3)^2 だが、真のツアーで合計=2のときにペナルティ=1になる
            # より正確には、3つ全部選んだときだけペナルティを課したいので
            # 3つ全部が1のときにペナルティが最大になるような項を追加
            
            # 3つのエッジがすべて選ばれることを抑制
            # ij * jk + jk * ik + ij * ik の項でペナルティ
            triangle_penalty = P_TRI * (ij * jk + jk * ik + ij * ik)
            H += triangle_penalty
            triangle_count += 1
    
    print(f"三角形組み合わせ数: {triangle_count}")
    
    # 四角形ペナルティ (k=4)
    quad_count = 0
    if enable_quad:
        P_QUAD = lambda_degree * quad_penalty_factor * 0.01  # Much weaker penalty
        print(f"四角形ペナルティ係数: {P_QUAD:.2f}")
        
        # 四角形サイクルのみを対象にする（4つのエッジで閉ループ）
        for i, j, k, l in itertools.combinations(range(n_cities), 4):
            # 四角形サイクルを形成する4つのエッジのみ
            # 例: i-j-k-l-i のサイクル
            cycle_edges = [(i,j), (j,k), (k,l), (l,i)]
            cycle_vars = []
            
            for a, b in cycle_edges:
                min_node, max_node = min(a, b), max(a, b)
                if (min_node, max_node) in e:
                    cycle_vars.append(e[(min_node, max_node)])
            
            if len(cycle_vars) == 4:  # 4つのエッジが全て存在する場合のみ
                # 4つ全部選ばれたらペナルティ（サイクル形成の抑制）
                cycle_sum = sum(cycle_vars)
                quad_penalty = P_QUAD * cycle_sum * (cycle_sum - 1) * (cycle_sum - 2) * (cycle_sum - 3) / 6
                H += quad_penalty
                quad_count += 1
    
    print(f"四角形組み合わせ数: {quad_count}")
    
    print(f"  最大距離: {np.max(distance_matrix):.2f}")
    print(f"  次数制約係数: {lambda_degree:.2f}")
    
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
    
    # エッジから経路を構築
    route, total_distance, is_single_tour = construct_route_from_edges_advanced(selected_edges, distance_matrix, n_cities)
    
    # 制約チェック
    degree_violations = 0
    for i in range(n_cities):
        degree = 0
        for (u, v) in selected_edges:
            if u == i or v == i:
                degree += 1
        degree_violations += (degree - 2) ** 2
    
    # 小ループの検出
    triangle_violations, quad_violations = count_small_loops(selected_edges, n_cities)
    
    print(f"  構築された経路: {route}")
    print(f"  総距離: {total_distance:.2f}")
    print(f"  単一ツアー: {is_single_tour}")
    print(f"  次数制約違反: {degree_violations}")
    print(f"  三角形ループ数: {triangle_violations}")
    print(f"  四角形ループ数: {quad_violations}")
    print(f"  エネルギー: {response.first.energy:.2f}")
    
    return route, total_distance, response.first.energy, selected_edges, compile_time, solve_time, is_single_tour


def count_small_loops(edges, n_cities):
    """小ループの数を数える"""
    # 隣接リストを構築
    adj = {i: [] for i in range(n_cities)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    triangle_count = 0
    quad_count = 0
    
    # 三角形の検出
    for i, j, k in itertools.combinations(range(n_cities), 3):
        if (j in adj[i] and k in adj[j] and i in adj[k]):
            triangle_count += 1
    
    # 四角形の検出（簡易版）
    for i, j, k, l in itertools.combinations(range(n_cities), 4):
        edges_in_quad = 0
        nodes = [i, j, k, l]
        for a in range(4):
            for b in range(a+1, 4):
                if nodes[b] in adj[nodes[a]]:
                    edges_in_quad += 1
        if edges_in_quad == 4:  # 4つのエッジで四角形
            quad_count += 1
    
    return triangle_count, quad_count


def construct_route_from_edges_advanced(edges, distance_matrix, n_cities):
    """エッジリストから経路を構築（改良版）"""
    if len(edges) == 0:
        return [], float('inf'), False
    
    # 隣接リストを構築
    adj = {i: [] for i in range(n_cities)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # 連結成分を見つける
    visited = set()
    components = []
    
    for start in range(n_cities):
        if start not in visited:
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            if component:
                components.append(component)
    
    print(f"  連結成分数: {len(components)}")
    print(f"  連結成分: {components}")
    
    # 単一の連結成分かチェック
    is_single_tour = len(components) == 1 and len(components[0]) == n_cities
    
    if not is_single_tour:
        return [], float('inf'), False
    
    # 単一ツアーの場合、経路を構築
    component = components[0]
    
    # 次数2でない頂点があるかチェック
    for node in component:
        if len(adj[node]) != 2:
            return [], float('inf'), False
    
    # 経路を構築（0から開始）
    if 0 not in component:
        return [], float('inf'), False
    
    route = [0]
    current = 0
    prev = -1
    
    for _ in range(len(component) - 1):
        neighbors = adj[current]
        next_node = None
        for neighbor in neighbors:
            if neighbor != prev:
                next_node = neighbor
                break
        
        if next_node is None:
            return [], float('inf'), False
        
        route.append(next_node)
        prev = current
        current = next_node
    
    # 距離計算
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance_matrix[route[i]][route[(i+1) % len(route)]]
    
    return route, total_distance, True


def run_subtour_elimination_experiment(n_cities=20, trials=10):
    """小ループ抑制実験"""
    print(f"\n=== 小ループ抑制実験 ({n_cities}都市, {trials}回試行) ===")
    
    # 固定問題インスタンス
    cities = create_cities(n_cities, seed=42)
    distance_matrix = calculate_distances(cities)
    
    # 実験設定
    configs = [
        {"name": "Baseline (No Penalty)", "tri": False, "quad": False},
        {"name": "Triangle Penalty Only", "tri": True, "quad": False},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}での実験...")
        
        single_tour_count = 0
        distances = []
        times = []
        
        for trial in range(trials):
            np.random.seed(42 + trial)
            route, distance, energy, edges, compile_time, annealing_time, is_single_tour = \
                solve_tsp_edge_with_subtour_elimination(
                    distance_matrix, 
                    num_reads=50, 
                    num_sweeps=2000,
                    enable_triangle=config["tri"],
                    enable_quad=config["quad"]
                )
            
            if is_single_tour:
                single_tour_count += 1
                distances.append(distance)
            else:
                distances.append(float('inf'))
            
            times.append(annealing_time)
        
        # 統計計算
        success_rate = single_tour_count / trials * 100
        valid_distances = [d for d in distances if d != float('inf')]
        min_distance = min(valid_distances) if valid_distances else float('inf')
        avg_distance = np.mean(valid_distances) if valid_distances else float('inf')
        avg_time = np.mean(times)
        
        results.append({
            'name': config['name'],
            'success_rate': success_rate,
            'min_distance': min_distance,
            'avg_distance': avg_distance,
            'avg_time': avg_time,
            'distances': distances,
            'times': times
        })
        
        print(f"  単一ツアー成功率: {success_rate:.1f}%")
        print(f"  最短距離: {min_distance:.2f}")
        print(f"  平均時間: {avg_time:.2f}秒")
    
    # 結果表示
    plot_subtour_elimination_results(results)
    
    return results


def plot_subtour_elimination_results(results):
    """Plot subtour elimination experiment results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    names = [r['name'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    min_distances = [r['min_distance'] if r['min_distance'] != float('inf') else 0 for r in results]
    avg_times = [r['avg_time'] for r in results]
    
    # Short names for x-axis to avoid clipping
    short_names = ['Baseline', 'Triangle']
    
    # 1. Success rate comparison
    x_pos = range(len(names))
    bars1 = ax1.bar(x_pos, success_rates, color=['red', 'orange'], alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Single Tour Success Rate')
    ax1.set_ylim(0, 105)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars1, success_rates)):
        ax1.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
    
    # 2. Best distance comparison (only for valid solutions)
    valid_indices = [i for i, d in enumerate(min_distances) if d > 0]
    valid_distances = [min_distances[i] for i in valid_indices]
    valid_short_names = [short_names[i] for i in valid_indices]
    
    if valid_distances:
        x_pos_valid = range(len(valid_distances))
        colors_valid = ['orange'][:len(valid_distances)]
        bars2 = ax2.bar(x_pos_valid, valid_distances, color=colors_valid, alpha=0.7)
        ax2.set_xticks(x_pos_valid)
        ax2.set_xticklabels(valid_short_names)
        ax2.set_ylabel('Best Distance')
        ax2.set_title('Best Solution Quality')
    
    # 3. Computation time comparison
    bars3 = ax3.bar(x_pos, avg_times, color=['red', 'orange'], alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(short_names)
    ax3.set_ylabel('Average Time (seconds)')
    ax3.set_title('Computation Time')
    
    # 4. Distance distribution (valid tours only)
    colors = ['red', 'orange']
    for i, result in enumerate(results):
        valid_dists = [d for d in result['distances'] if d != float('inf')]
        if valid_dists:
            ax4.hist(valid_dists, alpha=0.6, label=short_names[i], 
                    color=colors[i], bins=10)
    
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution (Valid Tours Only)')
    ax4.legend()
    
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


def plot_route(cities, route, edges=None, title_suffix=""):
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
        plt.title(f"Subtour Elimination TSP{title_suffix} - Distance: {total_distance:.2f}")
    elif edges:
        for (i, j) in edges:
            start = cities[i]
            end = cities[j]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2, alpha=0.7)
        plt.title(f"Selected Edges{title_suffix}")
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    # 20都市で実行
    n = 20
    cities = create_cities(n)
    distances = calculate_distances(cities)
    
    print(f"=== {n}都市TSP（小ループ抑制ペナルティ）===")
    route, distance, energy, edges, compile_time, annealing_time, is_single_tour = \
        solve_tsp_edge_with_subtour_elimination(distances, enable_triangle=True, enable_quad=False)
    
    print(f"\n結果:")
    print(f"経路: {route}")
    print(f"総距離: {distance:.2f}")
    print(f"エネルギー: {energy:.2f}")
    print(f"単一ツアー: {is_single_tour}")
    
    plot_route(cities, route, edges, " (Triangle Penalty)")
    
    # 実験実行
    print("\n\n小ループ抑制実験を実行しますか？ (y/n): ", end="")
    if input().lower() == 'y':
        run_subtour_elimination_experiment(n_cities=20, trials=10)