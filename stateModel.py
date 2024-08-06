import statsmodels.api as sm

# 为回归模型添加常数项
X_train_selected = sm.add_constant(X_train_selected)

# 建立OLS模型并拟合
model = sm.OLS(y_train, X_train_selected)
results = model.fit()

# 输出模型摘要
print(results.summary())
