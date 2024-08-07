# # # import pandas as pd
# # # import statsmodels.api as sm

# # # # 读取基因数据
# # # file_path = 'X.csv'
# # # data = pd.read_csv(file_path)

# # # # 提取特征和目标变量
# # # X = data.drop(columns=['Unnamed: 0', 'zur_at'])
# # # y = data['zur_at']

# # # # 加入常数项
# # # X = sm.add_constant(X)

# # # # 训练OLS模型
# # # model = sm.OLS(y, X).fit()

# # # # 打印回归结果
# # # summary = model.summary()

# # # # 打印所有 p 值
# # # pvalues = model.pvalues
# # # print("P-values:")
# # # print(pvalues)

# # # # 提取显著基因（p 值小于 0.05）
# # # significant_genes = pvalues[pvalues < 0.05].index

# # # # 删除常数项
# # # significant_genes = significant_genes.drop('const', errors='ignore')

# # # # 将结果输出到 txt 文件
# # # with open('regression_results.txt', 'w') as f:
# # #     f.write(summary.as_text())
# # #     f.write("\nP-values:\n")
# # #     f.write(pvalues.to_string())
# # #     f.write("\nSignificant Genes:\n")
# # #     f.write("\n".join(significant_genes))

# # # # 打印结果存入 txt 文档
# # # print("回归结果已存入 regression_results.txt")
# # # print("Significant Genes:")
# # # print(significant_genes)
# # import pandas as pd
# # import statsmodels.api as sm
# # from statsmodels.stats.outliers_influence import variance_inflation_factor

# # # 读取基因数据
# # file_path = 'X.csv'
# # data = pd.read_csv(file_path)

# # # 提取特征和目标变量
# # X = data.drop(columns=['Unnamed: 0', 'zur_at'])
# # y = data['zur_at']

# # # 加入常数项
# # X = sm.add_constant(X)

# # # 计算 VIF 值
# # vif_data = pd.DataFrame()
# # vif_data["feature"] = X.columns
# # vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# # print(vif_data)

# # # 训练OLS模型
# # model = sm.OLS(y, X).fit()

# # # 打印回归结果
# # summary = model.summary()

# # # 打印所有 p 值
# # pvalues = model.pvalues
# # print("P-values:")
# # print(pvalues)

# # # 提取显著基因（p 值小于 0.05）
# # significant_genes = pvalues[pvalues < 0.05].index

# # # 删除常数项
# # significant_genes = significant_genes.drop('const', errors='ignore')

# # # 将结果输出到 txt 文件
# # with open('regression_results.txt', 'w') as f:
# #     f.write(summary.as_text())
# #     f.write("\nP-values:\n")
# #     f.write(pvalues.to_string())
# #     f.write("\nSignificant Genes:\n")
# #     f.write("\n".join(significant_genes))

# # # 打印结果存入 txt 文档
# # print("回归结果已存入 regression_results.txt")
# # print("Significant Genes:")
# # print(significant_genes)
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 读取基因数据
file_path = 'X.csv'
data = pd.read_csv(file_path)

# 提取特征和目标变量
X = data.drop(columns=['Unnamed: 0', 'zur_at'])
y = data['zur_at']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练Ridge回归模型
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# 预测和计算均方误差
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# 打印和保存结果
coefficients = ridge.coef_
significant_genes = np.array(X.columns)[np.abs(coefficients) > 0.01]

with open('regression_results.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write("Significant Genes:\n")
    for gene in significant_genes:
        f.write(f"{gene}\n")

print("回归结果已存入 regression_results.txt")
print("Significant Genes:")
print(significant_genes)
# import pandas as pd
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # 读取基因数据
# file_path = 'X.csv'
# data = pd.read_csv(file_path)

# # 提取特征和目标变量
# X = data.drop(columns=['Unnamed: 0', 'zur_at'])
# y = data['zur_at']

# # 标准化特征
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # 训练Ridge回归模型
# ridge = Ridge(alpha=1.0)
# ridge.fit(X_train, y_train)

# # 预测和计算均方误差
# y_pred = ridge.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)

# # 打印和保存结果
# coefficients = ridge.coef_
# significant_genes = np.array(X.columns)[np.abs(coefficients) > 0.01]

# with open('regression_results.txt', 'w') as f:
#     f.write(f"Mean Squared Error: {mse}\n")
#     f.write("Significant Genes:\n")
#     for gene in significant_genes:
#         f.write(f"{gene}\n")

# print("回归结果已存入 regression_results.txt")
# print("Significant Genes:")
# print(significant_genes)
