import numpy as np
import matplotlib.pyplot as plt
import time

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def buffon_needle_simulation(n, needle_length=1, line_spacing=2):
    """
    模拟 Buffon 投针实验
    :param n: 实验次数（投针次数）
    :param needle_length: 针的长度 (l)
    :param line_spacing: 平行线间距 (d)，满足 l <= d
    :return: π 的估计值, 相交次数
    """
    # 记录相交次数
    hits = 0
    
    # 使用向量化操作提高性能
    y_positions = np.random.uniform(0, line_spacing, n)
    angles = np.random.uniform(0, np.pi/2, n)
    
    # 计算针在垂直方向上的投影长度
    vertical_projections = (needle_length / 2) * np.sin(angles)
    
    # 判断针是否与平行线相交
    hits = np.sum((y_positions <= vertical_projections) | 
                 (y_positions >= (line_spacing - vertical_projections)))
    
    # 计算 π 的估计值
    if hits == 0:  # 避免除以零
        return 0, 0
    probability = hits / n
    pi_estimate = (2 * needle_length) / (probability * line_spacing)
    return pi_estimate, hits

def run_experiments(trial_counts):
    """
    运行不同次数的实验并记录结果
    :param trial_counts: 实验次数列表
    :return: 结果列表 (实验次数, π估计值, 相交次数, 绝对误差, 运行时间)
    """
    results = []
    true_pi = np.pi
    
    for n in trial_counts:
        start_time = time.time()
        pi_est, hits = buffon_needle_simulation(n)
        end_time = time.time()
        
        error = abs(pi_est - true_pi)
        runtime = end_time - start_time
        
        results.append((n, pi_est, hits, error, runtime))
        print(f"实验次数: {n:>7}, π估计值: {pi_est:.6f}, 相交次数: {hits}, 绝对误差: {error:.6f}, 耗时: {runtime:.4f}秒")
    
    return results

def visualize_results(results, save_path=None):
    """
    可视化实验结果
    :param results: 实验结果列表
    :param save_path: 图片保存路径
    """
    trial_counts = [r[0] for r in results]
    pi_estimates = [r[1] for r in results]
    errors = [r[3] for r in results]
    true_pi = np.pi
    
    plt.figure(figsize=(15, 5))
    
    # 绘制估计值变化
    plt.subplot(1, 2, 1)
    plt.semilogx(trial_counts, pi_estimates, 'bo-', label='估计值')
    plt.axhline(y=true_pi, color='r', linestyle='--', label='真实π值')
    plt.xlabel('实验次数 (对数坐标)')
    plt.ylabel('π估计值')
    plt.title('π估计值随实验次数的变化')
    plt.legend()
    plt.grid(True)
    
    # 绘制误差变化
    plt.subplot(1, 2, 2)
    plt.loglog(trial_counts, errors, 'go-')
    plt.xlabel('实验次数 (对数坐标)')
    plt.ylabel('绝对误差 (对数坐标)')
    plt.title('误差随实验次数的变化 (双对数坐标)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存为 '{save_path}'")
    else:
        plt.show()

def main():
    # 设置不同的实验次数
    trial_counts = [100, 1000, 10000, 100000, 1000000]
    
    print("Buffon 投针实验 - π 估计")
    print("=" * 50)
    
    # 运行实验
    results = run_experiments(trial_counts)
    
    # 可视化结果
    visualize_results(results, save_path='buffon_results.png')
    
    # 输出最终结果
    print("\n最终结果汇总:")
    print("=" * 50)
    print("实验次数 | π估计值 | 相交次数 | 绝对误差 | 运行时间")
    print("-" * 50)
    for n, pi_est, hits, error, runtime in results:
        print(f"{n:>7} | {pi_est:>8.6f} | {hits:>8} | {error:>8.6f} | {runtime:>8.4f}秒")

if __name__ == "__main__":
    main()
