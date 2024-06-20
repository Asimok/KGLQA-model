import json
import matplotlib.pyplot as plt

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y = [49.15, 48.92, 49.04, 51.08, 48.96, 49.15, 49.19, 48.15, 47.61, 46.57, 46.61]
y = [49.65, 49.58, 49.54, 49.46, 49.54, 49.5, 49.65, 48.42, 47.8, 46.73, 46.53]
# 图片宽度1000
plt.figure(figsize=(10, 4), dpi=300)
# 绘制折线图
plt.plot(x, y, 'ro-', linewidth=2)
# 添加星号点
plt.plot(0.3, 51.08, 'b*', markersize=10)
# 显示数值
for i in range(len(x)):
    plt.text(x[i], y[i], f"{y[i]}", ha='center', va='bottom', color='gray')
plt.text(0.3, 51.08, "51.08", ha='center', va='bottom', color='blue')

plt.xlabel("score threshold")
plt.ylabel("accuracy")
# 保存
plt.savefig('threshold_analyze.png')
plt.show()
# 2 4 6
print()
