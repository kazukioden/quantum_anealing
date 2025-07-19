from pyqubo import Binary
import openjij as oj
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# Fix matplotlib font for English display
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def solve_tsp_edge_openjij_with_subtour_elimination(distance_matrix, num_reads=100, 
                                                   enable_triangle=True, use_sqa=True,
                                                   degree_coeff=None, triangle_coeff=None,
                                                   num_sweeps=3000, trotter=None, gamma=None):
    """小ループ抑制ペナルティ付きエッジ表現TSPをOpenJijで解く"""
    n_cities = len(distance_matrix)
    
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
    
    # 目的関数: 総距離の最小化
    distance_obj = 0
    for (i, j), edge_var in e.items():
        distance_obj += distance_matrix[i][j] * edge_var
    
    # アルゴリズム別基本ペナルティ係数
    base_penalty = np.max(distance_matrix) * n_cities / 10
    
    # 制約係数設定
    lambda_degree = base_penalty * degree_coeff
    lambda_triangle = base_penalty * triangle_coeff
    
    # ハミルトニアンの構築開始
    H = distance_obj + lambda_degree * degree_constraints
    
    # === 小ループ抑制ペナルティ ===
    
    # 三角形ペナルティ (k=3) - アルゴリズム別調整
    triangle_count = 0
    if enable_triangle:
        for i, j, k in itertools.combinations(range(n_cities), 3):
            # 三角形の3つのエッジを取得
            ij = e[(min(i,j), max(i,j))]
            jk = e[(min(j,k), max(j,k))]
            ik = e[(min(i,k), max(i,k))]
            
            # 3つのエッジがすべて選ばれることを抑制
            triangle_penalty = lambda_triangle * (ij * jk + jk * ik + ij * ik)
            H += triangle_penalty
            triangle_count += 1
    
    # コンパイル
    model = H.compile()
    qubo, offset = model.to_qubo()
    
    # アニーリング
    if use_sqa:
        # SQAパラメータ設定
        sqa_trotter = trotter if trotter is not None else 16  # デフォルトL=16
        sqa_gamma = gamma if gamma is not None else 1.5       # デフォルトγ=1.5
        
        sampler = oj.SQASampler()
        response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps, 
                                     trotter=sqa_trotter, gamma=sqa_gamma)
    else:
        sampler = oj.SASampler()
        response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps)
    
    # 結果を取得
    best_sample = response.first.sample
    selected_edges = []
    for (i, j) in e.keys():
        if best_sample.get(f'e_{i}_{j}', 0) == 1:
            selected_edges.append((i, j))
    
    # エッジから経路を構築
    route, total_distance, is_single_tour = construct_route_from_edges_advanced(selected_edges, distance_matrix, n_cities)
    
    return route, total_distance, response.first.energy, selected_edges, is_single_tour


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


def sweep_comparison_experiment(n_cities=15, trials_per_sweep=10):
    """num_sweepsを変化させてSA vs SQAを比較"""
    print(f"=== num_sweeps比較実験 ({n_cities}都市) ===")
    
    # 固定問題インスタンス
    cities = create_cities(n_cities, seed=42)
    distance_matrix = calculate_distances(cities)
    
    # 最適化された係数
    sa_degree_coeff = 8.83
    sa_triangle_coeff = 0.18
    
    sqa_trotter = 32
    sqa_gamma = 1.905
    sqa_degree_multiplier = 11.57
    sqa_triangle_multiplier = 0.65
    sqa_degree_coeff = sqa_trotter * sqa_degree_multiplier  # 370.24
    sqa_triangle_coeff = sqa_trotter * sqa_triangle_multiplier  # 20.8
    
    # num_sweepsの範囲
    sweep_values = [10, 100, 1000, 3000, 5000, 10000]
    
    sa_results = []
    sqa_results = []
    
    for num_sweeps in sweep_values:
        print(f"\nnum_sweeps = {num_sweeps}")
        
        # SA実験
        sa_distances = []
        for trial in range(trials_per_sweep):
            np.random.seed(42 + trial)
            route, distance, energy, edges, is_single_tour = solve_tsp_edge_openjij_with_subtour_elimination(
                distance_matrix, 
                num_reads=30,
                enable_triangle=True,
                use_sqa=False,
                degree_coeff=sa_degree_coeff,
                triangle_coeff=sa_triangle_coeff,
                num_sweeps=num_sweeps
            )
            
            if is_single_tour:
                sa_distances.append(distance)
            else:
                sa_distances.append(float('inf'))
        
        # SA最良値
        valid_sa_distances = [d for d in sa_distances if d != float('inf')]
        sa_best = min(valid_sa_distances) if valid_sa_distances else float('inf')
        sa_success_rate = len(valid_sa_distances) / trials_per_sweep * 100
        
        # SQA実験
        sqa_distances = []
        for trial in range(trials_per_sweep):
            np.random.seed(42 + trial)
            route, distance, energy, edges, is_single_tour = solve_tsp_edge_openjij_with_subtour_elimination(
                distance_matrix, 
                num_reads=30,
                enable_triangle=True,
                use_sqa=True,
                degree_coeff=sqa_degree_coeff,
                triangle_coeff=sqa_triangle_coeff,
                num_sweeps=num_sweeps,
                trotter=sqa_trotter,
                gamma=sqa_gamma
            )
            
            if is_single_tour:
                sqa_distances.append(distance)
            else:
                sqa_distances.append(float('inf'))
        
        # SQA最良値
        valid_sqa_distances = [d for d in sqa_distances if d != float('inf')]
        sqa_best = min(valid_sqa_distances) if valid_sqa_distances else float('inf')
        sqa_success_rate = len(valid_sqa_distances) / trials_per_sweep * 100
        
        sa_results.append({
            'num_sweeps': num_sweeps,
            'best_distance': sa_best,
            'success_rate': sa_success_rate
        })
        
        sqa_results.append({
            'num_sweeps': num_sweeps,
            'best_distance': sqa_best,
            'success_rate': sqa_success_rate
        })
        
        print(f"  SA  - 最良距離: {sa_best:.2f}, 成功率: {sa_success_rate:.1f}%")
        print(f"  SQA - 最良距離: {sqa_best:.2f}, 成功率: {sqa_success_rate:.1f}%")
    
    # 結果をプロット
    plot_sweep_comparison(sa_results, sqa_results)
    
    return sa_results, sqa_results


def plot_sweep_comparison(sa_results, sqa_results):
    """num_sweeps比較結果をプロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # データ準備
    sweeps = [r['num_sweeps'] for r in sa_results]
    sa_distances = [r['best_distance'] if r['best_distance'] != float('inf') else None for r in sa_results]
    sqa_distances = [r['best_distance'] if r['best_distance'] != float('inf') else None for r in sqa_results]
    sa_success_rates = [r['success_rate'] for r in sa_results]
    sqa_success_rates = [r['success_rate'] for r in sqa_results]
    
    # 1. 最良距離の比較
    ax1.semilogx(sweeps, sa_distances, 'bo-', label='SA', alpha=0.7, linewidth=2, markersize=6)
    ax1.semilogx(sweeps, sqa_distances, 'ro-', label='SQA', alpha=0.7, linewidth=2, markersize=6)
    ax1.set_xlabel('num_sweeps')
    ax1.set_ylabel('Best Distance')
    ax1.set_title('Best Distance vs num_sweeps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 成功率の比較
    ax2.semilogx(sweeps, sa_success_rates, 'bo-', label='SA', alpha=0.7, linewidth=2, markersize=6)
    ax2.semilogx(sweeps, sqa_success_rates, 'ro-', label='SQA', alpha=0.7, linewidth=2, markersize=6)
    ax2.set_xlabel('num_sweeps')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate vs num_sweeps')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # num_sweeps比較実験を実行
    sa_results, sqa_results = sweep_comparison_experiment(n_cities=15, trials_per_sweep=10)