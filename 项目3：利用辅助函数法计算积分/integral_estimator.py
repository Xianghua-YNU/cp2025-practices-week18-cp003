import numpy as np
import matplotlib.pyplot as plt

def generate_random_numbers(N):
    """
    生成满足权重函数 p(x) = 1/(2√x) 分布的随机数
    参数:
        N: 生成的随机数数量
    返回:
        满足 p(x) 分布的随机数数组
    """
    u = np.random.rand(N)  # 生成 [0,1] 上的均匀分布随机数
    x = u**2               # 通过逆变换法得到满足 p(x) 分布的随机数
    return x

def estimate_integral(N):
    """
    估计积分值并计算统计误差
    参数:
        N: 使用的随机点数
    返回:
        integral: 积分估计值
        error: 统计误差
    """
    # 生成满足 p(x) 分布的随机数
    x = generate_random_numbers(N)
    
    # 计算被积函数 f(x) = (x^{-1/2}/(e^x + 1)) / p(x) = 2/(e^x + 1)
    f = 2 / (np.exp(x) + 1)
    
    # 计算积分估计值
    integral = np.mean(f)
    
    # 计算方差和统计误差
    variance = np.var(f)
    error = np.sqrt(variance) / np.sqrt(N)
    
    return integral, error

def plot_distribution(N=10000):
    """
    可视化生成的随机数分布与理论分布的比较
    参数:
        N: 用于绘图的随机数数量
    """
    x = generate_random_numbers(N)
    
    # 理论概率密度函数
    x_vals = np.linspace(0.001, 1, 1000)
    p_x = 1 / (2 * np.sqrt(x_vals))
    
    # 归一化直方图
    plt.hist(x, bins=50, density=True, alpha=0.6, label='生成随机数的分布')
    plt.plot(x_vals, p_x, 'r-', lw=2, label='理论分布 p(x) = 1/(2√x)')
    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('随机数分布与理论分布比较')
    plt.legend()
    plt.show()

def main():
    # 设置随机数种子以确保结果可重复
    np.random.seed(42)
    
    # 绘制分布图
    plot_distribution()
    
    # 计算积分估计值
    N = 1000000
    integral, error = estimate_integral(N)
    
    print(f"积分估计值: {integral:.6f}")
    print(f"统计误差: {error:.6f}")
    print(f"95% 置信区间: [{integral - 1.96*error:.6f}, {integral + 1.96*error:.6f}]")
    
    # 进行多次独立实验观察收敛性
    num_experiments = 10
    results = []
    for _ in range(num_experiments):
        res, _ = estimate_integral(N)
        results.append(res)
    
    print(f"\n{num_experiments} 次独立实验结果:")
    print(f"平均值: {np.mean(results):.6f}")
    print(f"标准差: {np.std(results):.6f}")

if __name__ == "__main__":
    main()
