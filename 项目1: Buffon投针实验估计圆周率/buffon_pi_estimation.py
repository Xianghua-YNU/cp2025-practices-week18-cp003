import numpy as np
import matplotlib.pyplot as plt

def buffon_needle_simulation(n, needle_length=1, line_spacing=2):
    """
    模拟 Buffon 投针实验
    :param n: 实验次数（投针次数）
    :param needle_length: 针的长度 (l)
    :param line_spacing: 平行线间距 (d)，满足 l <= d
    :return: π 的估计值
    """
    # 记录相交次数
    hits = 0
    
    for _ in range(n):
        # 随机生成针的中点位置 y（在 [0, d] 范围内均匀分布）
        y = np.random.uniform(0, line_spacing)
        # 随机生成针的角度 θ（在 [0, π/2] 范围内均匀分布）
        theta = np.random.uniform(0, np.pi/2)
        
        # 计算针到最近平行线的垂直距离
        # 针在垂直方向上的投影长度
        vertical_projection = (needle_length / 2) * np.sin(theta)
        
        # 判断针是否与平行线相交
        if y <= vertical_projection or y >= (line_spacing - vertical_projection):
            hits += 1
    
    # 计算 π 的估计值
    if hits == 0:  # 避免除以零
        return 0
    probability = hits / n
    pi_estimate = (2 * needle_length) / (probability * line_spacing)
    return pi_estimate

def analyze_impact_of_trials(max_trials=100000, step=1000):
    """
    Analyze the impact of number of trials on estimation accuracy
    """
    # 存储结果
    trial_counts = []
    pi_estimates = []
    errors = []
    
    # 真实 π 值
    true_pi = np.pi
    
    # 进行多次实验
    current_trials = step
    while current_trials <= max_trials:
        pi_est = buffon_needle_simulation(current_trials)
        error = abs(pi_est - true_pi)
        
        trial_counts.append(current_trials)
        pi_estimates.append(pi_est)
        errors.append(error)
        
        current_trials += step
    
    # 输出最终结果
    print(f"Final estimate (n={max_trials}): {pi_estimates[-1]}")
    print(f"Absolute error: {errors[-1]}")
    print(f"Relative error: {errors[-1]/true_pi*100:.2f}%")
    
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    # 绘制估计值变化
    plt.subplot(1, 2, 1)
    plt.plot(trial_counts, pi_estimates, 'b-', label='Estimate')
    plt.axhline(y=true_pi, color='r', linestyle='--', label='True π')
    plt.xlabel('Number of Trials')
    plt.ylabel('π Estimate')
    plt.title('π Estimate vs. Number of Trials')
    plt.legend()
    plt.grid(True)
    
    # 绘制误差变化
    plt.subplot(1, 2, 2)
    plt.loglog(trial_counts, errors, 'g-')
    plt.xlabel('Number of Trials (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error vs. Number of Trials (Log-Log)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('buffon_simulation.png')
    print("Plot saved as 'buffon_simulation.png'")

# 主程序
if __name__ == "__main__":
    # 设置实验次数
    n = 100000
    
    # 单次实验估计 π
    pi_estimate = buffon_needle_simulation(n)
    print(f"经过 {n} 次投针实验，π 的估计值为: {pi_estimate}")
    print(f"与真实 π 值的绝对误差: {abs(pi_estimate - np.pi)}")
    
    # 分析不同实验次数对精度的影响
    analyze_impact_of_trials(max_trials=100000, step=5000)
