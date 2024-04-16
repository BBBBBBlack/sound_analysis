import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 生成一些随机的实际标签和预测结果
actual = np.random.randint(0, 2, size=100)
predicted = np.random.randint(0, 2, size=100)

# 计算混淆矩阵
cm = confusion_matrix(actual, predicted)

# 创建混淆矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')

plt.show()
