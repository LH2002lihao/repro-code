import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置中文字体 (尝试使用常见中文字体，如果报错请根据您的系统修改)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


def draw_final_figure():
    # 1. 设置画布 (16:9 宽屏)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=120)
    plt.subplots_adjust(wspace=0.15, left=0.05, right=0.95, top=0.9, bottom=0.1)

    # 2. 定义高级语义色板
    c_bg_left = '#FFF5F5'  # 左背景：淡红灰
    c_bg_right = '#F0F8FF'  # 右背景：淡蓝
    c_chaos = '#FF4D4F'  # 混乱：警示红
    c_season = '#5C8C8C'  # 伪变化/季节：浑浊青
    c_logic = '#1890FF'  # 逻辑流：科技蓝
    c_gold = '#FAAD14'  # 核心控制：金色
    c_input_T1 = '#95DE64'  # T1夏：草绿
    c_input_T2 = '#FFEC3D'  # T2秋：枯黄

    # ==========================================
    # 左图：(a) 现有方法：不受控的纠缠 (幻觉)
    # ==========================================
    ax1.set_facecolor(c_bg_left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("(a) 现有方法：隐式纠缠与不受控激活\n(伪变化导致幻觉)", fontsize=18, fontweight='bold', color='#555',
                  pad=20)

    # 输入 T1
    ax1.add_patch(patches.Rectangle((1, 7), 2, 2, facecolor=c_input_T1, edgecolor='#333', lw=2))
    ax1.text(2, 9.3, "影像 T1 (夏)", ha='center', fontsize=12, fontweight='bold')

    # 输入 T2
    ax1.add_patch(patches.Rectangle((1, 4), 2, 2, facecolor=c_input_T2, edgecolor='#333', lw=2))
    ax1.text(2, 3.7, "影像 T2 (秋)", ha='center', fontsize=12, fontweight='bold')

    # 中间：混乱区域
    chaos_circle = patches.Circle((5.5, 6.5), 1.8, facecolor='#FFEBEB', edgecolor=c_chaos, linestyle='--', lw=2,
                                  alpha=0.5)
    ax1.add_patch(chaos_circle)

    # 绘制红色乱线 (模拟Attention)
    np.random.seed(42)
    for _ in range(40):
        x = np.linspace(3.2, 7.8, 20)
        y = np.linspace(np.random.uniform(7, 8), np.random.uniform(4, 5), 20) + np.random.normal(0, 0.5, 20)
        ax1.plot(x, y, color=c_chaos, alpha=0.4, lw=1.5)

    ax1.text(5.5, 8.5, "伪变化被激活!", ha='center', color=c_chaos, fontsize=12, fontweight='bold')

    # Decoder & 结果
    ax1.add_patch(patches.Rectangle((7.5, 5), 2, 1.5, facecolor='white', edgecolor='#333', boxstyle="Round,pad=0.2"))
    ax1.text(8.5, 5.75, "解码器", ha='center', va='center', fontsize=14)

    # 错误气泡
    ax1.text(5.5, 1.5, "❌ 幻觉生成:\n\"绿色的树木被移除了...\"", ha='center', fontsize=15, color='red',
             bbox=dict(boxstyle="darrow,pad=0.5", fc="#FFF1F0", ec="red", lw=2))
    ax1.arrow(8.5, 5, -1, -2.5, head_width=0.2, fc='red', ec='red', alpha=0.3)

    # ==========================================
    # 右图：(b) 本文方法：基于推理的逻辑拒止
    # ==========================================
    ax2.set_facecolor(c_bg_right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("(b) 本文方法：认知先验驱动的推理范式\n(逻辑门控抑制伪变化)", fontsize=18, fontweight='bold',
                  color='#333', pad=20)

    # 输入
    ax2.add_patch(patches.Rectangle((0.5, 7.5), 1.5, 1.5, facecolor=c_input_T1, edgecolor='#333', lw=2))
    ax2.add_patch(patches.Rectangle((0.5, 5.5), 1.5, 1.5, facecolor=c_input_T2, edgecolor='#333', lw=2))
    ax2.text(1.25, 9.3, "输入影像", ha='center', fontsize=12)

    # Step 1: 认知先验 (大脑)
    brain_box = patches.Circle((3.5, 7.25), 0.9, facecolor='white', edgecolor=c_gold, lw=3)
    ax2.add_patch(brain_box)
    ax2.text(3.5, 7.25, "认知先验\n(CPPM)", ha='center', va='center', fontsize=14, fontweight='bold', color=c_gold)

    # 信号框
    ax2.text(3.5, 8.5, "前提判定:\n无语义变化", ha='center', fontsize=11, color='white', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc=c_gold, ec=c_gold))

    # Step 2: 逻辑管道与闸门
    # 上管道 (背景流)
    ax2.add_patch(
        patches.Rectangle((5.5, 7.5), 3, 0.8, facecolor='#E6F7FF', edgecolor=c_season, linestyle='--', hatch='///',
                          alpha=0.5))
    ax2.text(7, 8.0, "背景/季节流 (被分离)", ha='center', va='center', fontsize=10, color=c_season)

    # 下管道 (变化流) - 空心
    ax2.add_patch(patches.Rectangle((5.5, 5), 3, 0.8, facecolor='white', edgecolor=c_logic, linestyle='-'))

    # 闸门 (Gate) - 核心
    gate = patches.Rectangle((5.8, 4.6), 0.4, 1.6, facecolor=c_chaos, edgecolor='#820014', lw=2)
    ax2.add_patch(gate)
    ax2.text(6.0, 4.2, "🔒 逻辑锁", ha='center', fontsize=12, color=c_chaos, fontweight='bold')

    # 金色光束 (Guidance)
    ax2.annotate("", xy=(6.0, 6.2), xytext=(4.2, 7.2),
                 arrowprops=dict(arrowstyle="->", color=c_gold, lw=5, linestyle='-'))
    ax2.text(5.0, 6.5, "逻辑阻断指令", color=c_gold, fontsize=12, fontweight='bold', rotation=-25,
             bbox=dict(fc='white', ec='none', alpha=0.7))

    # 管道内的伪变化被挡住
    ax2.text(5.3, 5.4, "伪变化\n(季节)", ha='right', va='center', color=c_season, fontsize=10)
    ax2.text(6.5, 5.4, "(此处为空)", ha='center', va='center', color='#CCC', fontsize=10)

    # Step 3: Decoder
    ax2.add_patch(patches.Rectangle((9, 6), 1, 1.5, facecolor='white', edgecolor='#333', boxstyle="Round,pad=0.2"))
    ax2.text(9.5, 6.75, "解码器", ha='center', fontsize=12)

    # 结果 (正确)
    ax2.text(5.5, 1.5, "✅ 正确推理:\n\"场景未发生语义变化\"", ha='center', fontsize=15, color='green',
             bbox=dict(boxstyle="round,pad=0.5", fc="#F6FFED", ec="green", lw=2))

    # 连接线
    ax2.arrow(8.5, 5.4, 0.5, 0.8, head_width=0.2, fc=c_logic, ec=c_logic)  # 下路进Decoder
    ax2.arrow(9.5, 6, 0, -3.5, head_width=0.3, fc='green', ec='green')  # Decoder出结果

    # 中间分隔线
    line = plt.Line2D([0.5, 0.5], [0.1, 0.9], transform=fig.transFigure, color='black', linestyle=':', linewidth=2,
                      alpha=0.3)
    fig.add_artist(line)

    plt.suptitle("冲动 vs. 克制：本文方法与现有方法的逻辑范式对比", fontsize=22, y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_final_figure()