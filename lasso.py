from sklearn.linear_model import Lasso

# 使用 Lasso 回归进行特征选择
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# 获取非零系数对应的特征索引
selected_features = np.where(lasso.coef_ != 0)[0]

# 选择重要特征
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]
