import json
import matplotlib.pyplot as plt

out_file = '/data0/maqi/KGLQA-data/datasets/NCR/len_analyze/result.json'
with open(out_file, 'r') as f:
    answer_dict = json.load(f)

x = [int(k) for k in answer_dict.keys()]
x.sort()
y = [answer_dict[str(k)] * 100 for k in x]
# 图片宽度1000
plt.figure(figsize=(10, 4))
# 绘制折线图
plt.plot(x, y, 'ro-', linewidth=2)

# 添加星号点
plt.plot(1400, 54.12, 'b*', markersize=10)
plt.text(1400, 54.12, "54.12", ha='center', va='bottom', color='blue')

plt.xlabel("Key Information length: l")
plt.ylabel("accuracy")
# 保存
plt.savefig('len_analyze.png')
plt.show()

print()
