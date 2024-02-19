import json
import matplotlib.pyplot as plt

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y = [54.61, 54.95, 55.37, 55.07, 55.22, 55.14, 53.00, 52.12, 50.94, 50.59, 50.55]
# 图片宽度1000
plt.figure(figsize=(10, 4),dpi=300)
# 绘制折线图
plt.plot(x, y, 'ro-', linewidth=2)
# 添加星号点
plt.plot(0.25, 55.33, 'b*', markersize=10)
# 显示数值
for i in range(len(x)):
    plt.text(x[i]+0.01, y[i] - 0.3, f"{y[i]}", ha='center', va='bottom', color='gray')
plt.text(0.28, 55.33, "55.33", ha='center', va='bottom', color='blue')

plt.xlabel("summary proportion")
plt.ylabel("accuracy")
# 保存
plt.savefig('proportion_analyze.png')
plt.show()
# 2 4 6
print()
