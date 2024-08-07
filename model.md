要识别影响核黄素产量的基因，我们可以采用以下步骤：

1. **数据预处理**：

   - 加载数据集并检查数据的完整性。
   - 标准化数据（如果必要）。
2. **特征选择**：

   - 使用诸如Lasso回归、随机森林重要性分析、或PCA等方法，初步筛选出可能影响核黄素产量的基因。
   - 使用单变量或多变量的统计方法（例如t检验或F检验）进行进一步筛选。
3. **模型构建**：

   - 采用线性回归、岭回归、或Lasso回归等模型，建立核黄素产量与基因表达水平之间的关系。
   - 使用交叉验证或留一法评估模型性能。
4. **显著性检验**：

   - 对所选基因进行统计显著性检验（例如使用p值或调整后的p值）。
   - 如果基因显著性较高，则认为它们可能对核黄素产量有重要影响。

现在我将根据这些步骤进行分析。

数据集包含71个样本和4089个特征，其中包括基因表达水平和核黄素产量。第一列是样本标识符，我们可以忽略它进行后续分析。

接下来，我将执行以下步骤：

1. **提取特征和目标变量**。
2. **特征选择**：使用Lasso回归初步筛选出可能影响核黄素产量的基因。
3. **模型构建与显著性检验**。

我们先进行特征和目标变量的提取。

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 71 entries, 0 to 70
Columns: 4089 entries, Unnamed: 0 to zur_at
dtypes: float64(4088), object(1)
memory usage: 2.2+ MB
结果
(                Unnamed: 0   AADK_at   AAPA_at   ABFA_at     ABH_at   ABNA_at  \
 0        b_Fbat107PT24.CEL  8.492404  8.111451  8.320842  10.287112  8.261279   
 1        b_Fbat107PT30.CEL  7.639380  7.239965  7.289051   9.862287  7.303497   
 2        b_Fbat107PT48.CEL  8.088340  7.855510  7.793395   9.676720  7.098273   
 3        b_Fbat107PT52.CEL  7.886820  7.939513  7.997588   9.680562  7.408494   
 4  knh_102_Fbat289PT24.CEL  6.805762  7.554522  7.609902   8.551953  7.712407   
 
      ABRB_at   ACCA_at   ACCB_at   ACCC_at  ...  YYDF_i_at   YYDG_at  \
 0  10.208273  9.745474  9.818821  9.676227  ...   7.882737  5.440163   
 1   9.500023  9.216008  9.854946  9.650078  ...   7.912135  5.383739   
 2   9.473917  9.580384  9.926077  9.787129  ...   7.354618  5.718795   
 3   9.788725  9.447722  9.852773  9.546915  ...   7.691985  5.561048   
 4   8.490847  8.696249  8.573272  8.589661  ...   7.999088  5.549914   
 
     YYDH_at   YYDI_at   YYDJ_at   YYDK_at   YYXA_at   YYZB_at   YYZE_at  \
 0  4.981684  6.481998  5.363919  7.773494  7.668711  3.762119  7.172823   
 1  4.597438  6.390127  4.763876  7.336261  7.600104  3.761616  7.187113   
 2  4.468228  6.123498  5.400686  7.573388  7.873352  3.741953  7.251285   
 3  4.787030  5.904708  5.139692  7.214269  7.384115  3.887179  7.117375   
 4  5.258139  6.206778  4.910169  6.884474  6.996956  4.034177  7.722534   
 
      zur_at  
 0  7.427105  
 1  7.545287  
 2  7.718321  
 3  7.260654  
 4  6.828164  
 
 [5 rows x 4089 columns],
 None)
```

数据已经被分割为训练集和测试集，并且特征已经标准化。使用Lasso回归进行特征选择，以识别与核黄素产量相关的基因。

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dropping the first column (sample identifiers) and separating features and target
X = data.drop(columns=['Unnamed: 0'])
y = data['zur_at']  # Assuming 'zur_at' is the column representing riboflavin (vitamin B2) yield

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape

```

```
((56, 4088), (15, 4088), (56,), (15,))
```

通过Lasso回归，筛选出了98个与核黄素产量相关的基因。接下来，我们可以使用这些基因来构建线性回归模型，并检验这些基因是否具有显著性。

```
from sklearn.linear_model import LassoCV
import numpy as np

# Using LassoCV for feature selection
lasso = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)

# Selecting features with non-zero coefficients
selected_features = np.where(lasso.coef_ != 0)[0]
selected_genes = X.columns[selected_features]

# Number of selected features and their names
len(selected_genes), selected_genes

```

```
(98,
 Index(['ACOR_at', 'CGEC_at', 'COMEB_at', 'DEGS_at', 'DHBF_at', 'EBRB_at',
        'GABP_at', 'GERAC_at', 'GLNP_at', 'GLPK_at', 'GLTT_at', 'GLVA_at',
        'GLVC_at', 'HBS_at', 'HUTP_at', 'IOLH_at', 'LEUA_at', 'MSMX_at',
        'OPPF_at', 'PHEA_s_at', 'PROA_at', 'SBO_at', 'SECE_r_at', 'SIPW_at',
        'SPOIIIAB_at', 'SPSC_at', 'THYB_at', 'TUAH_at', 'UDK_at', 'UVRC_at',
        'XKDR_at', 'YAAB_at', 'YACB_at', 'YACN_at', 'YACO_at', 'YACP_at',
        'YAZC_at', 'YBBH_at', 'YBFQ_at', 'YCBF_at', 'YCGA_at', 'YCGG_at',
        'YCNF_at', 'YCSE_at', 'YCSI_at', 'YCXD_at', 'YCXE_at', 'YDBF_at',
        'YDIO_at', 'YESR_at', 'YESU_at', 'YFIM_at', 'YFIN_at', 'YFKO_at',
        'YFLA_at', 'YGAO_at', 'YHAQ_at', 'YHCJ_at', 'YJCI_at', 'YJCJ_at',
        'YJOB_at', 'YKCA_at', 'YKKB_at', 'YKOK_at', 'YKOW_at', 'YKRP_at',
        'YKTD_at', 'YKUG_at', 'YKVJ_at', 'YKVN_at', 'YLAJ_at', 'YNBA_at',
        'YNBB_at', 'YNER_at', 'YNGI_at', 'YODC_at', 'YOTL_at', 'YPBD_at',
        'YPGA_at', 'YPKP_at', 'YPOC_at', 'YPUB_at', 'YQGT_at', 'YQIB_at',
        'YQJA_at', 'YQZB_at', 'YRBA_at', 'YRKK_at', 'YRRC_at', 'YRRL_at',
        'YTBE_at', 'YTCG_at', 'YUFN_at', 'YURT_at', 'YUSL_at', 'YVAN_at',
        'YWNE_at', 'zur_at'],
       dtype='object'))
```

我们将进行以下步骤：

1. **构建线性回归模型**：使用这些筛选出的基因。
2. **显著性检验**：通过分析回归系数的p值来检验每个基因的显著性。

现在我将继续执行这些步骤。

```
import statsmodels.api as sm

# Adding a constant for the intercept
X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]
X_train_selected = sm.add_constant(X_train_selected)

# Fitting the OLS (Ordinary Least Squares) model
model = sm.OLS(y_train, X_train_selected).fit()

# Summary of the model
model_summary = model.summary()
model_summary

```
