import seaborn as sns

#缺失值可视化 
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

# 目标变量分布概况  无界约翰逊分布等
import scipy.stats as st
y = Train_data['price']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)

#观察训练集和测试集各个变量的偏度和峰度
Train_data.skew(),Train_data.kurt()
Test_data.skew(),Test_data.kurt()

#观察偏度、峰度的分布情况
# displot()集合了matplotlib的hist()与核函数估计kdeplot的功能，增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新颖用途。
sns.distplot(Train_data.skew(),color='blue',axlabel ='Skewness')
sns.distplot(Train_data.kurt(),color='orange',axlabel ='Kurtness')

#3) 查看预测值的具体频数
plt.hist(Train_data['price'], orientation = 'vertical',histtype = 'bar', color ='red')
plt.show()

#根据字段类型选择变量 也可以改为exclude
category_cols = Train_data.select_dtypes(include=['object','category']).columns.tolist()

# 寻找是否有严重倾斜的特征，统计每个特征频次,
# 目的：判断是否存在某个特征的某个值占比超过99%,一般会删除该特征
def value_counts_sta(df):
    columns = df.columns.tolist()
    lean_cols = []
    data_len = len(df)
    for col in columns:
        df_col = df[col].value_counts().to_frame()
        df_col[col+'_true'] = df_col.index
        df_col = df_col.reset_index(drop=True)
        df_col.rename(columns={col:'counts'},inplace=True)
        top1_value = df_col['counts'].head(1).tolist()[0]
        if top1_value/df_col['counts'].sum() >= 0.99:
            lean_cols.append(col)
    return lean_cols

# 观察类别特征是否有倾斜 'seller', 'offerType'] 删除
value_counts_sta(Train_data)
value_counts_sta(Test_data)

###########----------数字特征观察---------------##############

# 观察训练集和测试集分布是否一致
# 探索特征的贡献
def explore_feature_distibution(X_train,X_test,feature_cols):
    '''
        X_train:训练数据
        X_test:测试数据
        feature_cols:特征字段列表
        kdeplot：核密度估计图，用来估计未知的密度函数
    '''
    for column in feature_cols:
        # 通过核密度估计图可以比较直观的看出数据样本本身的分布特征
        g = sns.kdeplot(X_train[column], color="Red", shade = True)
        g = sns.kdeplot(X_test[column], ax =g, color="Blue", shade= True)
        g.set_xlabel(column)
        g.set_ylabel("Frequency")
        g = g.legend(["train","test"])
        plt.show()

ob_cols = ['name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType',
        'power', 'kilometer', 'regionCode',
        'creatDate', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4',
       'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13',
       'v_14']
# 训练集和测试集数据分布一致
explore_feature_distibution(X_Train,X_Test,ob_cols)

# 观察训练集变量与目标值的相关系数
# 画热力图寻找变量相关性
def fig_heatmap(data):
    plt.figure(figsize=(20, 16))  # 指定绘图对象宽度和高度
    colnm = data.columns.tolist()  # 列表头
    mcorr = data[colnm].corr(method="spearman")  # 相关系数矩阵，即给出了任意两个变量之间的相关系数
    mask = np.zeros_like(mcorr, dtype=np.bool)   # 构造与mcorr同维数矩阵 为bool型
    mask[np.triu_indices_from(mask)] = True      # 角分线右侧为True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 返回matplotlib colormap对象
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')  # 热力图（看两两相似度）
    plt.show()
    
# 找出低相关度的特征 小于等于0.1 ,考虑重构或舍弃
fig_heatmap(Train_data)

## 3) 每个数字特征得分布可视化  先sns.FacetGrid画出轮廓，然后用map填充内容
f = pd.melt(Train_data, value_vars=numeric_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

## 4) 数字特征相互之间（包括目标变量）的关系可视化 pairplot绘制矩阵图
sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()

###########----------类别特征观察---------------##############
## 1) unique分布
for fea in categorical_features:
    print(Train_data[fea].nunique())
    
## 2) 类别特征箱形图可视化 --针对回归类问题 针对分类问题可以画countplot

# 因为 name和 regionCode的类别太稀疏了，这里我们把不稀疏的几类画一下
categorical_features = ['model',
 'brand',
 'bodyType',
 'fuelType',
 'gearbox',
 'notRepairedDamage']
for c in categorical_features:
    Train_data[c] = Train_data[c].astype('category')
    if Train_data[c].isnull().any():
        Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
        Train_data[c] = Train_data[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, "value", "price")

## 3) 类别特征的小提琴图可视化
catg_list = categorical_features
target = 'price'
for catg in catg_list :
    sns.violinplot(x=catg, y=target, data=Train_data)
    plt.show()
    
## 4) 类别特征的柱形图可视化
def bar_plot(x, y, **kwargs):
    sns.barplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(bar_plot, "value", "price")

##  5) 类别特征的每个类别频数可视化(count_plot)
def count_plot(x,  **kwargs):
    sns.countplot(x=x)
    x=plt.xticks(rotation=90)

f = pd.melt(Train_data,  value_vars=categorical_features)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(count_plot, "value")

#看看所有特征的信息统计
df.describe(include = 'all')

#查看数据集的结果是否平衡
sns.countplot(x = 'Admit',order = ['low', 'normal', 'high'],data = df)

#University Rating 分布情况
f, [ax1,ax2] = plt.subplots(2, 1, figsize=(15, 15))
sns.countplot(x = 'University Rating' ,data = df,ax = ax1)
sns.countplot(x = 'University Rating', hue = 'Admit',hue_order = ['low', 'normal', 'high'],data = df , ax = ax2)

# 查看一下GRE Score和Admit的相关性
sns.barplot(x='Admit', y='GRE Score', order=['low', 'normal', 'high'], data=df)
plt.ylim((290, 340))

# 数值类特征之间的关系
# 查看GRE Score, TOEFL Score, CGPA两两之间的相关性
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
sns.regplot(x='GRE Score', y='TOEFL Score', data=df, ax=axes[0])
sns.regplot(x='GRE Score', y='CGPA', data=df, ax=axes[1])
sns.regplot(x='TOEFL Score', y='CGPA', data=df, ax=axes[2])

sns.set(rc={'figure.figsize':(8,6)})
sns.boxplot(x = 'Admit',y = 'GRE Score', data = df, order = ['low', 'normal', 'high'])

#由于职业和雇主的处理非常相似，我们定义函数get_top_amounts()对两个字段分组，组内取top N个值
def get_top_amounts(group,key,n=5):
#传入groupby分组后的对象，返回按照key字段汇总的排序前n的数据
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.sort_values(ascending=False)[:n]
  
grouped = data_vs.groupby('cand_nm')
grouped.apply(get_top_amounts,'contbr_occupation',n=7)

------单变量分析---
# 变量箱线图
colnm = df.columns.tolist()
fig = plt.figure(figsize = (10, 6))

for i in range(12):
    plt.subplot(2,6,i+1)
    sns.boxplot(df[colnm[i]], orient="v", width = 0.5, color = color[0])
    plt.ylabel(colnm[i],fontsize = 12)
#plt.subplots_adjust(left=0.2, wspace=0.8, top=0.9)

plt.tight_layout()
print('\nFigure 1: Univariate Boxplots')

# 变量的频次图
colnm = df.columns.tolist()
plt.figure(figsize = (10, 8))

for i in range(12):
    plt.subplot(4,3,i+1)
    df[colnm[i]].hist(bins = 100, color = color[0])
    plt.xlabel(colnm[i],fontsize = 12)
    plt.ylabel('Frequency')
plt.tight_layout()
print('\nFigure 2: Univariate Histograms')

------双变量分析----
# 观察各变量与目标变量的箱线图
sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.1)

colnm = df.columns.tolist()[:11] + ['total acid']
plt.figure(figsize = (10, 8))
for i in range(12):
    plt.subplot(4,3,i+1)
    sns.boxplot(x ='quality', y = colnm[i], data = df, color = color[1], width = 0.6)    
    plt.ylabel(colnm[i],fontsize = 12)
plt.tight_layout()
print("\nFigure 7: Physicochemical Properties and Wine Quality by Boxplot")

----多变量分析-----
sns.lmplot(x = 'alcohol', y = 'volatile acidity', col='quality', hue = 'quality', 
           data = df,fit_reg = False, size = 3,  aspect = 0.9, col_wrap=3,
           scatter_kws={'s':20})
print("Figure 11-2: Scatter Plots of Alcohol, Volatile Acid and Quality")



