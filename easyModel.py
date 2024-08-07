# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Lasso, LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# import statsmodels.api as sm

# # 加载数据集
# data = pd.read_csv('X.csv')

# # 假设最后一列是目标变量 y，其他列是特征 X
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# # 标准化特征
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 将数据分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # 使用 Lasso 回归进行特征选择
# lasso = Lasso(alpha=0.01)
# lasso.fit(X_train, y_train)

# # 获取 Lasso 模型的系数
# coef = lasso.coef_

# # 选择系数不为零的特征
# selected_features = np.where(coef != 0)[0]
# X_train_selected = X_train[:, selected_features]
# X_test_selected = X_test[:, selected_features]

# # 在选择的特征上拟合线性回归模型
# lin_reg = LinearRegression()
# lin_reg.fit(X_train_selected, y_train)

# # 评估模型性能
# y_pred = lin_reg.predict(X_test_selected)
# mse = mean_squared_error(y_test, y_pred)
# print(f'均方误差: {mse}')

# # 统计检验所选变量的显著性
# X_train_selected = sm.add_constant(X_train_selected)  # 为预测变量添加常数项
# model = sm.OLS(y_train, X_train_selected)
# results = model.fit()

# # 显示回归模型摘要
# print(results.summary())
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# 加载数据集
data = pd.read_csv('X.csv')

# 假设最后一列是目标变量 y，其他列是特征 X
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将非数值型列转换为数值型
X = pd.get_dummies(X, drop_first=True)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=72)

# 使用 Lasso 回归进行特征选择
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# 获取 Lasso 模型的系数
coef = lasso.coef_

# 选择系数不为零的特征
selected_features = np.where(coef != 0)[0]
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# 在选择的特征上拟合线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_train_selected, y_train)

# 评估模型性能
y_pred = lin_reg.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差: {mse}')

# 统计检验所选变量的显著性
X_train_selected = sm.add_constant(X_train_selected)  # 为预测变量添加常数项
model = sm.OLS(y_train, X_train_selected)
results = model.fit()

# 显示回归模型摘要
print(results.summary())
