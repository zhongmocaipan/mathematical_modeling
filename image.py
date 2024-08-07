import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS

# 模拟数据（这里假设你已经有数据）
# 假设X是基因表达数据，y是核黄素产量
# 替换为你的数据
n_obs = 56
n_genes = 40
np.random.seed(0)
X = np.random.rand(n_obs, n_genes)
y = 7 + np.dot(X, np.random.rand(n_genes)) + np.random.normal(size=n_obs)

# 回归模型
X = sm.add_constant(X)  # 添加常数项
model = OLS(y, X).fit()

# 提取回归系数和p值
coef = model.params
p_values = model.pvalues

# 1. 回归系数及其显著性图
plt.figure(figsize=(12, 6))
sns.barplot(x=np.arange(len(coef)), y=coef, palette='viridis')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Gene Index')
plt.ylabel('Coefficient')
plt.title('Regression Coefficients and Their Significance')
for i in range(len(coef)):
    plt.text(i, coef[i], f'{p_values[i]:.3f}', ha='center', va='bottom' if coef[i] > 0 else 'top')
plt.show()

# 2. 实际值与预测值的对比图
y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# 3. 残差图
residuals = y - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# 4. QQ图
sm.qqplot(residuals, line='45')
plt.title('QQ Plot of Residuals')
plt.show()
