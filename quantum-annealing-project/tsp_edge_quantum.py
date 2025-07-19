from pyqubo import Binary
import openjij as oj
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import optuna

# Fix matplotlib font for English display
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def solve_tsp_edge_openjij_with_subtour_elimination(distance_matrix, num_reads=100, 
                                                   enable_triangle=True, use_sqa=True,
                                                   degree_coeff=None, triangle_coeff=None,
                                                   num_sweeps=3000, trotter=None, gamma=None):
    """小ループ抑制ペナルティ付きエッジ表現TSPをOpenJijで解く"""
    n_cities = len(distance_matrix)
    print(f"都市数: {n_cities}")
    print(f"三角形ペナルティ: {enable_triangle}")
    print(f"アルゴリズム: {'SQA' if use_sqa else 'SA'}")
    
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
    
    # アルゴリズム別基本ペナルティ係数
    base_penalty = np.max(distance_matrix) * n_cities / 10
    
    if degree_coeff is not None:
        # Optuna最適化時は指定された係数を使用
        lambda_degree = base_penalty * degree_coeff
    else:
        if use_sqa:
            # SQAは基本制約を大幅強化（量子ゆらぎに対抗）
            lambda_degree = base_penalty*10   # SQA用: 10000倍強化
        else:
            # SAは従来通り
            lambda_degree = base_penalty*10  # SA用: 標準
    
    print(f"  次数制約係数: {lambda_degree:.2f} (base_penalty * {lambda_degree/base_penalty:.1f})")
    
    # ハミルトニアンの構築開始
    H = distance_obj + lambda_degree * degree_constraints
    
    # === 小ループ抑制ペナルティ ===
    
    # 三角形ペナルティ (k=3) - アルゴリズム別調整
    triangle_count = 0
    if enable_triangle:
        if triangle_coeff is not None:
            # Optuna最適化時は指定された係数を使用
            lambda_triangle = base_penalty * triangle_coeff
        else:
            if use_sqa:
                # SQAは量子ゆらぎによる三角形形成を抑制するため強化
                lambda_triangle = base_penalty*0.1     # SQA用: 強い三角形制約
            else:
                # SAは標準的な三角形制約
                lambda_triangle = base_penalty*0.1   # SA用: 弱い三角形制約
        
        print(f"  三角形ペナルティ係数: {lambda_triangle:.2f} (base_penalty * {lambda_triangle/base_penalty:.1f})")
        
        for i, j, k in itertools.combinations(range(n_cities), 3):
            # 三角形の3つのエッジを取得
            ij = e[(min(i,j), max(i,j))]
            jk = e[(min(j,k), max(j,k))]
            ik = e[(min(i,k), max(i,k))]
            
            # 3つのエッジがすべて選ばれることを抑制
            triangle_penalty = lambda_triangle * (ij * jk + jk * ik + ij * ik)
            H += triangle_penalty
            triangle_count += 1
    
    print(f"三角形組み合わせ数: {triangle_count}")
    print(f"  最大距離: {np.max(distance_matrix):.2f}")
    
    # コンパイル
    print("コンパイル中...")
    start = time.time()
    model = H.compile()
    qubo, offset = model.to_qubo()
    compile_time = time.time() - start
    print(f"コンパイル時間: {compile_time:.2f}秒")
    print(f"QUBO変数数: {len(qubo)}")
    
    # アニーリング
    if use_sqa:
        # SQAパラメータ設定
        sqa_trotter = trotter if trotter is not None else 16  # デフォルトL=16
        sqa_gamma = gamma if gamma is not None else 1.5       # デフォルトγ=1.5
        
        print(f"SQA実行中... (num_reads={num_reads}, num_sweeps={num_sweeps}, trotter={sqa_trotter}, gamma={sqa_gamma})")
        sampler = oj.SQASampler()
        start = time.time()
        response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps, 
                                     trotter=sqa_trotter, gamma=sqa_gamma)
        solve_time = time.time() - start
        print(f"SQA時間: {solve_time:.2f}秒")
    else:
        print(f"SA実行中... (num_reads={num_reads}, num_sweeps={num_sweeps})")
        sampler = oj.SASampler()
        start = time.time()
        response = sampler.sample_qubo(qubo, num_reads=num_reads, num_sweeps=num_sweeps)
        solve_time = time.time() - start
        print(f"SA時間: {solve_time:.2f}秒")
    
    # 結果を取得
    best_sample = response.first.sample
    selected_edges = []
    for (i, j) in e.keys():
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
    triangle_violations = count_triangles(selected_edges, n_cities)
    
    print(f"  構築された経路: {route}")
    print(f"  総距離: {total_distance:.2f}")
    print(f"  単一ツアー: {is_single_tour}")
    print(f"  次数制約違反: {degree_violations}")
    print(f"  三角形ループ数: {triangle_violations}")
    print(f"  エネルギー: {response.first.energy:.2f}")
    
    return route, total_distance, response.first.energy, selected_edges, compile_time, solve_time, is_single_tour


def count_triangles(edges, n_cities):
    """三角形ループの数を数える"""
    # 隣接リストを構築
    adj = {i: [] for i in range(n_cities)}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    triangle_count = 0
    
    # 三角形の検出
    for i, j, k in itertools.combinations(range(n_cities), 3):
        if (j in adj[i] and k in adj[j] and i in adj[k]):
            triangle_count += 1
    
    return triangle_count


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


def run_sa_vs_sqa_experiment(n_cities=15, trials=5):
    """SA vs SQA の比較実験"""
    print(f"\n=== SA vs SQA 比較 ({n_cities}都市, {trials}回試行) ===")
    
    # 固定問題インスタンス
    cities = create_cities(n_cities, seed=42)
    distance_matrix = calculate_distances(cities)
    
    # 実験設定
    configs = [
        {"name": "Simulated Annealing (SA)", "sqa": False, "triangle": True},
        {"name": "Simulated Quantum Annealing (SQA)", "sqa": True, "triangle": True},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}での実験...")
        
        single_tour_count = 0
        distances = []
        times = []
        qpu_times = []
        
        for trial in range(trials):
            np.random.seed(42 + trial)
            route, distance, energy, edges, compile_time, annealing_time, is_single_tour = \
                solve_tsp_edge_openjij_with_subtour_elimination(
                    distance_matrix, 
                    num_reads=50,
                    enable_triangle=config["triangle"],
                    use_sqa=config["sqa"]
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
    plot_sa_vs_sqa_results(results)
    
    return results


def plot_sa_vs_sqa_results(results):
    """SA vs SQA の結果をプロット"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    names = [r['name'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    min_distances = [r['min_distance'] if r['min_distance'] != float('inf') else 0 for r in results]
    avg_times = [r['avg_time'] for r in results]
    
    # Short names for x-axis
    short_names = ['SA', 'SQA']
    
    # 1. Success rate comparison
    x_pos = range(len(names))
    bars1 = ax1.bar(x_pos, success_rates, color=['blue', 'red'], alpha=0.7)
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
        colors_valid = ['blue', 'red'][:len(valid_distances)]
        bars2 = ax2.bar(x_pos_valid, valid_distances, color=colors_valid, alpha=0.7)
        ax2.set_xticks(x_pos_valid)
        ax2.set_xticklabels(valid_short_names)
        ax2.set_ylabel('Best Distance')
        ax2.set_title('Best Solution Quality')
    
    # 3. Computation time comparison
    bars3 = ax3.bar(x_pos, avg_times, color=['blue', 'red'], alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(short_names)
    ax3.set_ylabel('Average Time (seconds)')
    ax3.set_title('Computation Time')
    
    # 4. Distance distribution (valid tours only)
    colors = ['blue', 'red']
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
        plt.title(f"OpenJij Edge TSP{title_suffix} - Distance: {total_distance:.2f}")
    elif edges:
        for (i, j) in edges:
            start = cities[i]
            end = cities[j]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2, alpha=0.7)
        plt.title(f"Selected Edges{title_suffix}")
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def optimize_coefficients_with_optuna(distance_matrix, use_sqa=True, n_trials=50):
    """Optunaを使って制約係数を最適化する"""
    print(f"\n=== Optuna最適化 ({'SQA' if use_sqa else 'SA'}) ===")
    
    def objective(trial):
        # 係数の探索範囲（実験結果ベース）
        if use_sqa:
            # SQAアニーリングパラメータ
            trotter = trial.suggest_int('trotter', 8, 32, step=8)  # 8, 16, 24, 32
            gamma = trial.suggest_float('gamma', 1.0, 2.5)
            
            # SQA用：倍率を独立最適化してからトロッター数でスケール
            degree_multiplier = trial.suggest_float('degree_multiplier', 5, 20)  # SA成功値10の±100%
            triangle_multiplier = trial.suggest_float('triangle_multiplier', 0.5, 2)  # SA成功値1の±100%
            
            # 最終係数 = base_penalty * (trotter * multiplier)
            degree_coeff = trotter * degree_multiplier
            triangle_coeff = trotter * triangle_multiplier
        else:
            # SA用：成功パターン周辺を探索
            trotter = None
            gamma = None
            degree_coeff = trial.suggest_float('degree_coeff', 5, 20)    # 10±50%
            triangle_coeff = trial.suggest_float('triangle_coeff', 0.05, 0.2)  # 0.1±50%
        
        # 複数回実行して平均をとる
        total_score = 0
        valid_runs = 0
        
        for run in range(3):  # 3回実行
            try:
                route, distance, energy, edges, compile_time, annealing_time, is_single_tour = \
                    solve_tsp_edge_openjij_with_subtour_elimination(
                        distance_matrix, 
                        num_reads=30,  # 最適化時は中程度の精度
                        enable_triangle=True,
                        use_sqa=use_sqa,
                        degree_coeff=degree_coeff,
                        triangle_coeff=triangle_coeff,
                        num_sweeps=1000,  # 最適化時は高速化
                        trotter=trotter,
                        gamma=gamma
                    )
                
                # スコア計算: 制約違反数を最小化
                # 次数制約違反を計算
                degree_violations = 0
                for i in range(len(distance_matrix)):
                    degree = 0
                    for (u, v) in edges:
                        if u == i or v == i:
                            degree += 1
                    degree_violations += (degree - 2) ** 2
                
                # 三角形ループ数を計算
                triangle_violations = count_triangles(edges, len(distance_matrix))
                
                # 単一ツアー失敗ペナルティを追加
                single_tour_penalty = 0 if is_single_tour else 10
                
                # 制約違反総数 + 単一ツアー失敗ペナルティ
                total_violations = degree_violations + triangle_violations + single_tour_penalty
                total_score += total_violations
                valid_runs += 1
                
            except Exception as e:
                print(f"  実行{run+1}でエラー: {e}")
                total_score += 10000  # エラー時は大きなペナルティ
                valid_runs += 1
        
        avg_violations = total_score / valid_runs if valid_runs > 0 else 10000
        return avg_violations  # 制約違反数を最小化
    
    # Optuna最適化を実行
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n最適化結果:")
    print(f"最小制約違反数: {study.best_value:.1f}")
    print(f"最良パラメータ:")
    if use_sqa:
        # SQA用の詳細表示
        trotter = study.best_params['trotter']
        gamma = study.best_params['gamma']
        degree_mult = study.best_params['degree_multiplier']
        triangle_mult = study.best_params['triangle_multiplier']
        print(f"  trotter: {trotter}")
        print(f"  gamma: {gamma:.3f}")
        print(f"  degree_multiplier: {degree_mult:.2f}")
        print(f"  triangle_multiplier: {triangle_mult:.2f}")
        print(f"  → 最終係数:")
        print(f"    degree_coeff: {trotter * degree_mult:.2f} (trotter×{degree_mult:.2f})")
        print(f"    triangle_coeff: {trotter * triangle_mult:.2f} (trotter×{triangle_mult:.2f})")
    else:
        # SA用の表示
        print(f"  degree_coeff: {study.best_params['degree_coeff']:.2f}")
        print(f"  triangle_coeff: {study.best_params['triangle_coeff']:.2f}")
    
    return study.best_params


def run_optimized_comparison(n_cities=15, trials=5):
    """最適化された係数でSA vs SQAを比較"""
    print(f"\n=== 最適化係数でのSA vs SQA比較 ({n_cities}都市) ===")
    
    # 固定問題インスタンス
    cities = create_cities(n_cities, seed=42)
    distance_matrix = calculate_distances(cities)
    
    # SA用係数を最適化
    print("\nSA用係数を最適化中...")
    sa_params = optimize_coefficients_with_optuna(distance_matrix, use_sqa=False, n_trials=100)
    
    # SQA用係数を最適化
    print("\nSQA用係数を最適化中...")
    sqa_params = optimize_coefficients_with_optuna(distance_matrix, use_sqa=True, n_trials=100)
    
    # 最適化された係数で比較実験
    configs = [
        {"name": "SA (Optimized)", "sqa": False, "params": sa_params},
        {"name": "SQA (Optimized)", "sqa": True, "params": sqa_params},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}での実験...")
        
        single_tour_count = 0
        distances = []
        times = []
        
        for trial in range(trials):
            np.random.seed(42 + trial)
            # SQAの場合は倍率から係数を計算
            if config["sqa"]:
                trotter = config["params"]["trotter"]
                gamma = config["params"]["gamma"]
                degree_coeff = trotter * config["params"]["degree_multiplier"]
                triangle_coeff = trotter * config["params"]["triangle_multiplier"]
                
                route, distance, energy, edges, compile_time, annealing_time, is_single_tour = \
                    solve_tsp_edge_openjij_with_subtour_elimination(
                        distance_matrix, 
                        num_reads=50,
                        enable_triangle=True,
                        use_sqa=config["sqa"],
                        degree_coeff=degree_coeff,
                        triangle_coeff=triangle_coeff,
                        trotter=trotter,
                        gamma=gamma
                    )
            else:
                route, distance, energy, edges, compile_time, annealing_time, is_single_tour = \
                    solve_tsp_edge_openjij_with_subtour_elimination(
                        distance_matrix, 
                        num_reads=50,
                        enable_triangle=True,
                        use_sqa=config["sqa"],
                        degree_coeff=config["params"]["degree_coeff"],
                        triangle_coeff=config["params"]["triangle_coeff"]
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
            'times': times,
            'params': config['params']
        })
        
        print(f"  単一ツアー成功率: {success_rate:.1f}%")
        print(f"  最短距離: {min_distance:.2f}")
        print(f"  平均時間: {avg_time:.2f}秒")
        
        # パラメータ表示
        if config["sqa"]:
            trotter = config["params"]["trotter"]
            gamma = config["params"]["gamma"]
            degree_mult = config["params"]["degree_multiplier"]
            triangle_mult = config["params"]["triangle_multiplier"]
            print(f"  使用パラメータ - trotter: {trotter}, gamma: {gamma:.3f}")
            print(f"  倍率 - degree: {degree_mult:.2f}, triangle: {triangle_mult:.2f}")
            print(f"  最終係数 - degree: {trotter*degree_mult:.2f}, triangle: {trotter*triangle_mult:.2f}")
        else:
            print(f"  使用係数 - degree: {config['params']['degree_coeff']:.2f}, triangle: {config['params']['triangle_coeff']:.2f}")
    
    # 結果表示
    plot_optimized_results(results)
    
    return results


def plot_optimized_results(results):
    """最適化結果をプロット"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    names = [r['name'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    min_distances = [r['min_distance'] if r['min_distance'] != float('inf') else 0 for r in results]
    avg_times = [r['avg_time'] for r in results]
    
    short_names = ['SA (Opt)', 'SQA (Opt)']
    
    # 1. Success rate comparison
    x_pos = range(len(names))
    bars1 = ax1.bar(x_pos, success_rates, color=['blue', 'red'], alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate (Optimized Coefficients)')
    ax1.set_ylim(0, 105)
    
    for i, (bar, rate) in enumerate(zip(bars1, success_rates)):
        ax1.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom')
    
    # 2. Best distance comparison
    valid_indices = [i for i, d in enumerate(min_distances) if d > 0]
    valid_distances = [min_distances[i] for i in valid_indices]
    valid_short_names = [short_names[i] for i in valid_indices]
    
    if valid_distances:
        x_pos_valid = range(len(valid_distances))
        colors_valid = ['blue', 'red'][:len(valid_distances)]
        bars2 = ax2.bar(x_pos_valid, valid_distances, color=colors_valid, alpha=0.7)
        ax2.set_xticks(x_pos_valid)
        ax2.set_xticklabels(valid_short_names)
        ax2.set_ylabel('Best Distance')
        ax2.set_title('Best Solution Quality (Optimized)')
    
    # 3. Coefficient comparison
    degree_coeffs = []
    triangle_coeffs = []
    
    for r in results:
        if 'degree_coeff' in r['params']:
            # SA用
            degree_coeffs.append(r['params']['degree_coeff'])
            triangle_coeffs.append(r['params']['triangle_coeff'])
        else:
            # SQA用：倍率×トロッター数で計算
            trotter = r['params']['trotter']
            degree_mult = r['params']['degree_multiplier']
            triangle_mult = r['params']['triangle_multiplier']
            degree_coeffs.append(trotter * degree_mult)
            triangle_coeffs.append(trotter * triangle_mult)
    
    x = np.arange(len(short_names))
    width = 0.35
    
    ax3.bar(x - width/2, degree_coeffs, width, label='Degree Coeff', alpha=0.7)
    ax3.bar(x + width/2, triangle_coeffs, width, label='Triangle Coeff', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names)
    ax3.set_ylabel('Coefficient Value')
    ax3.set_title('Optimized Coefficients')
    ax3.legend()
    ax3.set_yscale('log')
    
    # 4. Distance distribution
    colors = ['blue', 'red']
    for i, result in enumerate(results):
        valid_dists = [d for d in result['distances'] if d != float('inf')]
        if valid_dists:
            ax4.hist(valid_dists, alpha=0.6, label=short_names[i], 
                    color=colors[i], bins=10)
    
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution (Optimized)')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 15都市で実行（量子アニーリング用に小さめ）
    n = 15
    cities = create_cities(n)
    distances = calculate_distances(cities)
    
    print(f"=== {n}都市TSP（OpenJij SQA + 小ループ抑制）===")
    route, distance, energy, edges, compile_time, annealing_time, is_single_tour = \
        solve_tsp_edge_openjij_with_subtour_elimination(distances, enable_triangle=True, use_sqa=True)
    
    print(f"\n結果:")
    print(f"経路: {route}")
    print(f"総距離: {distance:.2f}")
    print(f"エネルギー: {energy:.2f}")
    print(f"単一ツアー: {is_single_tour}")
    
    plot_route(cities, route, edges, " (OpenJij SQA Triangle Penalty)")
    
    # 実験選択
    print("\n実験を選択してください:")
    print("1. 通常のSA vs SQA比較")
    print("2. Optuna最適化済みSA vs SQA比較")
    choice = input("選択 (1 or 2): ")
    
    if choice == "1":
        run_sa_vs_sqa_experiment(n_cities=15, trials=5)
    elif choice == "2":
        run_optimized_comparison(n_cities=15, trials=5)