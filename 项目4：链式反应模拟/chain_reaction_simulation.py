import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class ChainReactionSimulator:
    def __init__(self, initial_neutrons=1, capture_prob=0.3, fission_prob=0.5, nu=2.5, 
                 max_generations=20, max_neutrons=1e6):
        """
        初始化链式反应模拟器
        
        参数:
            initial_neutrons: 初始中子数
            capture_prob: 中子被俘获的概率(不引发裂变)
            fission_prob: 中子引发裂变的概率(需要满足 capture_prob + fission_prob <= 1)
            nu: 每次裂变释放的平均中子数
            max_generations: 最大模拟代际数
            max_neutrons: 中子数上限(防止数值爆炸)
        """
        self.initial_neutrons = initial_neutrons
        self.capture_prob = capture_prob
        self.fission_prob = fission_prob
        self.nu = nu
        self.max_generations = max_generations
        self.max_neutrons = max_neutrons
        
        # 验证概率总和不超过1
        assert (capture_prob + fission_prob) <= 1.0, "俘获概率和裂变概率总和不能超过1"
        
        # 逃逸概率 = 1 - capture_prob - fission_prob
        self.escape_prob = 1 - capture_prob - fission_prob
    
    def simulate(self):
        """运行链式反应模拟"""
        neutron_counts = []
        current_neutrons = self.initial_neutrons
        neutron_counts.append(current_neutrons)
        
        for generation in range(1, self.max_generations + 1):
            new_neutrons = 0
            
            # 对当前代每个中子进行模拟
            for _ in range(current_neutrons):
                fate = random.random()  # 决定中子的命运
                
                if fate < self.capture_prob:
                    # 中子被俘获，不产生新中子
                    pass
                elif fate < self.capture_prob + self.fission_prob:
                    # 中子引发裂变，产生nu个新中子(取整数)
                    new_neutrons += int(round(self.nu))
                # else: 中子逃逸，不产生新中子
            
            # 更新中子数
            current_neutrons = new_neutrons
            neutron_counts.append(current_neutrons)
            
            # 检查终止条件
            if current_neutrons <= 0:
                print(f"链式反应在第{generation}代终止")
                break
            if current_neutrons >= self.max_neutrons:
                print(f"链式反应在第{generation}代达到中子数上限")
                break
        
        return neutron_counts
    
    def multi_simulation(self, num_simulations=100):
        """多次运行模拟，返回统计结果"""
        all_results = []
        for _ in range(num_simulations):
            result = self.simulate()
            all_results.append(result)
        
        # 找到最长的结果以便对齐
        max_length = max(len(r) for r in all_results)
        
        # 填充短的结果使其长度一致
        padded_results = []
        for r in all_results:
            if len(r) < max_length:
                padded = r + [0] * (max_length - len(r))
                padded_results.append(padded)
            else:
                padded_results.append(r)
        
        # 计算统计量
        mean_counts = np.mean(padded_results, axis=0)
        std_counts = np.std(padded_results, axis=0)
        
        return {
            'all_results': all_results,
            'mean_counts': mean_counts,
            'std_counts': std_counts,
            'max_length': max_length
        }


def plot_single_simulation(counts):
    """绘制单次模拟结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(counts, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Generation')
    plt.ylabel('Number of Neutrons')
    plt.title('Chain Reaction Simulation (Single Run)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.show()


def plot_multi_simulation(results):
    """绘制多次模拟的统计结果"""
    mean_counts = results['mean_counts']
    std_counts = results['std_counts']
    generations = np.arange(len(mean_counts))
    
    plt.figure(figsize=(12, 7))
    
    # 绘制均值
    plt.plot(generations, mean_counts, 'b-', linewidth=2, label='Mean Neutron Count')
    
    # 绘制标准差范围
    plt.fill_between(generations, 
                    mean_counts - std_counts, 
                    mean_counts + std_counts, 
                    color='blue', alpha=0.2, label='±1 Standard Deviation')
    
    plt.xlabel('Generation')
    plt.ylabel('Number of Neutrons')
    plt.title('Chain Reaction Simulation (Multiple Runs)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.legend()
    plt.show()


def analyze_parameters():
    """分析不同参数对链式反应的影响"""
    # 基准参数
    base_params = {
        'initial_neutrons': 1,
        'capture_prob': 0.3,
        'fission_prob': 0.5,
        'nu': 2.5,
        'max_generations': 20,
        'num_simulations': 50
    }
    
    # 测试不同参数
    param_variations = [
        {'capture_prob': 0.2, 'fission_prob': 0.7},  # 更易裂变
        {'capture_prob': 0.4, 'fission_prob': 0.4},  # 更难裂变
        {'nu': 2.0},  # 每次裂变产生较少中子
        {'nu': 3.0},  # 每次裂变产生较多中子
        {'capture_prob': 0.35, 'fission_prob': 0.35, 'nu': 2.8},  # 综合变化
    ]
    
    plt.figure(figsize=(14, 8))
    
    # 绘制基准情况
    simulator = ChainReactionSimulator(**base_params)
    base_results = simulator.multi_simulation(base_params['num_simulations'])
    plt.plot(base_results['mean_counts'], 'k-', linewidth=3, label='Base Case')
    
    # 绘制参数变化情况
    for i, params in enumerate(param_variations):
        # 合并参数
        current_params = base_params.copy()
        current_params.update(params)
        
        # 运行模拟
        simulator = ChainReactionSimulator(
            initial_neutrons=current_params['initial_neutrons'],
            capture_prob=current_params['capture_prob'],
            fission_prob=current_params['fission_prob'],
            nu=current_params['nu'],
            max_generations=current_params['max_generations']
        )
        results = simulator.multi_simulation(current_params['num_simulations'])
        
        # 生成标签
        label_parts = []
        for k, v in params.items():
            if k in ['capture_prob', 'fission_prob']:
                label_parts.append(f"{k}={v:.2f}")
            else:
                label_parts.append(f"{k}={v}")
        label = ", ".join(label_parts)
        
        # 绘制结果
        plt.plot(results['mean_counts'], '--', linewidth=2, label=label)
    
    plt.xlabel('Generation')
    plt.ylabel('Number of Neutrons (Mean)')
    plt.title('Chain Reaction Simulation with Different Parameters')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def criticality_analysis():
    """分析临界条件"""
    # 定义参数范围
    fission_probs = np.linspace(0.1, 0.7, 20)
    nus = np.linspace(1.5, 3.5, 20)
    
    # 固定其他参数
    initial_neutrons = 1
    capture_prob = 0.3
    max_generations = 20
    num_simulations = 20
    
    # 存储结果
    results = np.zeros((len(fission_probs), len(nus)))
    
    for i, fp in enumerate(fission_probs):
        for j, nu in enumerate(nus):
            # 计算有效增殖因子 k = fission_prob * nu / (capture_prob + fission_prob)
            # 但我们将通过模拟来观察实际行为
            
            simulator = ChainReactionSimulator(
                initial_neutrons=initial_neutrons,
                capture_prob=capture_prob,
                fission_prob=fp,
                nu=nu,
                max_generations=max_generations
            )
            
            # 运行多次模拟
            multi_results = simulator.multi_simulation(num_simulations)
            
            # 计算最终代的中子数均值
            final_counts = [r[-1] for r in multi_results['all_results']]
            mean_final = np.mean(final_counts)
            
            # 判断反应是否持续(临界状态)
            if mean_final > 1.0:
                results[i, j] = 1  # 超临界
            elif mean_final > 0.1:
                results[i, j] = 0.5  # 临界
            else:
                results[i, j] = 0  # 次临界
    
    # 绘制临界状态图
    plt.figure(figsize=(10, 8))
    plt.imshow(results, extent=[nus[0], nus[-1], fission_probs[-1], fission_probs[0]], 
              cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Reaction State (0=subcritical, 0.5=critical, 1=supercritical)')
    plt.xlabel('Neutrons per fission (nu)')
    plt.ylabel('Fission probability')
    plt.title('Criticality Analysis of Chain Reaction')
    
    # 计算理论临界线 k = fission_prob * nu / (capture_prob + fission_prob) = 1
    theoretical_nu = (capture_prob + fission_probs) / fission_probs
    plt.plot(theoretical_nu, fission_probs, 'k--', linewidth=2, label='Theoretical Criticality')
    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("链式反应模拟程序")
    
    # 示例模拟
    simulator = ChainReactionSimulator(
        initial_neutrons=1,
        capture_prob=0.3,
        fission_prob=0.5,
        nu=2.5,
        max_generations=20
    )
    
    # 单次模拟
    print("\n运行单次模拟...")
    counts = simulator.simulate()
    plot_single_simulation(counts)
    
    # 多次模拟
    print("\n运行多次模拟并计算统计量...")
    multi_results = simulator.multi_simulation(100)
    plot_multi_simulation(multi_results)
    
    # 参数分析
    print("\n分析不同参数对链式反应的影响...")
    analyze_parameters()
    
    # 临界分析
    print("\n进行临界条件分析...")
    criticality_analysis()
