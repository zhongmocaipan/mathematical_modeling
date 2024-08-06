from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 建立线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_train_selected, y_train)

# 评估模型性能
y_pred = lin_reg.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差: {mse}')
