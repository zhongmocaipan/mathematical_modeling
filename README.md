# mathematical_modeling

### 1. 数据准备：

- 加载数据集。
- 将基因表达水平数据（特征）和核黄素产量数据（目标变量）分离。

### 2. 特征选择：

- 由于数据维度较高，可以使用特征选择技术，比如Lasso回归（L1正则化），来减少特征数量。
- Lasso模型通过将不重要的基因系数缩小为零，帮助我们识别出最重要的基因。

### 3. 建模：

- 在选择出的相关基因上，使用线性回归来建模，研究这些基因与核黄素产量之间的关系。
- 也可以根据数据性质，选择Ridge回归或Elastic Net回归。

### 4. 统计显著性检验：

- 对所选变量进行统计检验，验证它们是否对核黄素产量具有显著影响。
  ```

  ```
- 这可以通过检查线性回归模型中系数的p值来实现。

### 数据预处理

```
(new_env) C:\Users\刘芳宜\Desktop\数学建模题目2\mathematical_modeling>python easyModel.py
均方误差: 0.019084997095816344
                            OLS Regression Results
==============================================================================
Dep. Variable:                 zur_at   R-squared:                       0.999
Model:                            OLS   Adj. R-squared:                  0.995
Method:                 Least Squares   F-statistic:                     279.2
Date:                Tue, 06 Aug 2024   Prob (F-statistic):           3.42e-16
Time:                        22:42:41   Log-Likelihood:                 157.68
No. Observations:                  56   AIC:                            -233.4
Df Residuals:                      15   BIC:                            -150.3
Df Model:                          40
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          7.2807      0.005   1524.462      0.000       7.270       7.291
x1             0.0022      0.013      0.168      0.869      -0.026       0.031
x2             0.0218      0.011      2.058      0.057      -0.001       0.044
x3            -0.0185      0.011     -1.739      0.103      -0.041       0.004
x4            -0.0228      0.012     -1.862      0.082      -0.049       0.003
x5             0.0042      0.009      0.473      0.643      -0.015       0.023
x6            -0.0170      0.008     -2.258      0.039      -0.033      -0.001
x7            -0.0050      0.009     -0.533      0.602      -0.025       0.015
x8             0.0187      0.016      1.169      0.261      -0.015       0.053
x9             0.0116      0.009      1.353      0.196      -0.007       0.030
x10            0.0060      0.012      0.511      0.617      -0.019       0.031
x11           -0.0141      0.012     -1.141      0.272      -0.040       0.012
x12           -0.0424      0.011     -3.741      0.002      -0.067      -0.018
x13            0.0052      0.017      0.310      0.761      -0.030       0.041
x14            0.0204      0.014      1.462      0.164      -0.009       0.050
x15           -0.0447      0.013     -3.412      0.004      -0.073      -0.017
x16            0.0141      0.014      1.026      0.321      -0.015       0.043
x17            0.0012      0.013      0.095      0.925      -0.026       0.028
x18            0.0499      0.013      3.816      0.002       0.022       0.078
x19            0.0613      0.018      3.458      0.004       0.024       0.099
x20            0.0107      0.012      0.909      0.378      -0.014       0.036
x21            0.0057      0.014      0.395      0.699      -0.025       0.036
x22            0.0500      0.016      3.128      0.007       0.016       0.084
x23            0.0419      0.017      2.478      0.026       0.006       0.078
x24            0.0107      0.013      0.827      0.421      -0.017       0.038
x25            0.0195      0.022      0.896      0.385      -0.027       0.066
x26            0.0053      0.010      0.556      0.586      -0.015       0.026
x27            0.0275      0.017      1.653      0.119      -0.008       0.063
x28            0.0285      0.016      1.789      0.094      -0.005       0.062
x29            0.0553      0.011      5.168      0.000       0.032       0.078
x30           -0.0132      0.012     -1.134      0.275      -0.038       0.012
x31            0.0029      0.016      0.177      0.862      -0.032       0.038
x32            0.0213      0.016      1.296      0.214      -0.014       0.056
x33            0.0203      0.007      2.951      0.010       0.006       0.035
x34            0.0006      0.011      0.054      0.958      -0.022       0.023
x35            0.0078      0.013      0.584      0.568      -0.021       0.036
x36            0.0121      0.005      2.383      0.031       0.001       0.023
x37           -0.0031      0.006     -0.542      0.596      -0.015       0.009
x38            0.0099      0.008      1.218      0.242      -0.007       0.027
x39           -0.0130      0.006     -2.003      0.064      -0.027       0.001
x40           -0.0077      0.007     -1.139      0.273      -0.022       0.007
==============================================================================
Omnibus:                        3.124   Durbin-Watson:                   2.194
Prob(Omnibus):                  0.210   Jarque-Bera (JB):                2.861
Skew:                           0.548   Prob(JB):                        0.239
Kurtosis:                       2.840   Cond. No.                         36.7
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### 5. 验证：

- 使用交叉验证评估模型性能，确保在不同的数据子集上能够一致识别出关键基因。

以下是实现这一过程的Python代码：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

# 加载数据集
data = pd.read_csv('/mnt/data/X.csv')

# 假设最后一列是核黄素产量，其他列是基因表达水平
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用Lasso回归进行特征选择
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# 获取Lasso模型的系数
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
```

### 代码解释：

1. **数据加载与预处理**：数据集被加载，并使用 `StandardScaler`对特征进行标准化处理。
2. **使用Lasso进行特征选择**：通过Lasso回归识别出对核黄素产量影响最大的基因（非零系数）。
3. **线性回归建模**：在选择的特征上进行线性回归建模，并使用均方误差（MSE）评估模型性能。
4. **统计显著性检验**：通过 `statsmodels`库对选择的基因进行统计显著性检验，输出系数的p值和置信区间。

通过运行这段代码，你可以识别出对核黄素产量有显著影响的基因，并用统计方法验证这些基因的显著性。
