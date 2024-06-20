import numpy as np
from matplotlib import pyplot as plt


def draw(datasets, Datasets_name):
    # 设置图形大小
    plt.figure(figsize=(18, 12))

    for idx, data in enumerate(datasets):
        # 计算当前子图的位置
        plt.subplot(2, 3, idx + 1)

        # 使用numpy计算四分位数
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1  # 四分位距

        # 可以基于IQR来决定bins的大小或数量，这里仅作为示例
        # 这里我们简单地将整个数据范围分成更细的部分
        bin_width = iqr / 4
        n_bins = int((max(data) - min(data)) / bin_width)

        # 生成bins
        bins = np.linspace(min(data), max(data), n_bins)

        # 绘制直方图
        plt.hist(data, bins=bins, edgecolor='white')

        # 添加标题和标签
        plt.title(Datasets_name[idx])
        plt.xlabel('Length')
        plt.ylabel('Count')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    plt.show()


# 示例数据集
datasets = [
    np.random.normal(loc=0, scale=1, size=1000),
    np.random.normal(loc=5, scale=2, size=1000),
    np.random.normal(loc=-3, scale=1, size=1000),
    np.random.uniform(low=-3, high=3, size=1000),
    np.random.exponential(scale=1.0, size=1000),
    np.random.binomial(n=10, p=0.5, size=1000)
]

# 数据集名称
Datasets_name = ['Normal(0,1)', 'Normal(5,2)', 'Normal(-3,1)', 'Uniform(-3,3)', 'Exponential(1.0)', 'Binomial(10,0.5)']

# 调用绘图函数
draw(datasets, Datasets_name)
