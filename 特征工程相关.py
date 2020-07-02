#时间类特征，做时间差
# 使用时间：data['creatDate'] - data['regDate']，反应汽车使用时间，一般来说价格与使用时间成反比
# 不过要注意，数据里有时间出错的格式，所以我们需要 errors='coerce'
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
                            
#从某些编码特征提取有用信息 类似身份证提取年龄、地区、性别
# 从邮编中提取城市信息，相当于加入了先验知识
data['city'] = data['regionCode'].apply(lambda x : str(x)[:-3])

    
# 日期格式化
def date_proc(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]

#定义日期提取函数
def date_tran(df,fea_col):
    for f in tqdm(fea_col):
        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
    return (df)

#分桶操作 50等距分箱
def cut_group(df,cols,num_bins=50):
    for col in cols:
        df[col+'_bin'] = pd.cut(df[col], num_bins, labels=False)
    return df

### count编码
def count_coding(df,fea_col):
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())
    return(df)
    
#count编码
count_list = ['regDate', 'creatDate', 'model', 'brand', 'regionCode','bodyType','fuelType','name','regDate_year', 'regDate_month', 'regDate_day',
       'regDate_dayofweek' , 'creatDate_month','creatDate_day', 'creatDate_dayofweek','kilometer']
# 类别多的特征统计值的频次作为新的特征      
data = count_coding(data,count_list)

#定义类别特征交叉统计
def cross_cat_num(df,num_col,cat_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
            })
            df = df.merge(feat, on=f1, how='left')
    return(df)

### 类别特征的二阶交叉
from scipy.stats import entropy
def cross_qua_cat_num(df):
    for f_pair in tqdm([
        ['model', 'brand'], ['model', 'regionCode'], ['brand', 'regionCode']
    ]):
        ### 共现次数
        df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)['SaleID'].transform('count')
        ### n unique、熵
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[0], how='left')
        df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
            '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
            '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[1], how='left')
        ### 比例偏好
        df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
        df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']
    return (df)

# 高基类别特征平均数编码
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,KFold
from itertools import product
class MeanEncoder:
    def __init__(self, categorical_features, n_splits=10, target_type='classification', prior_weight_func=None):
        """
        :param categorical_features: list of str, the name of the categorical columns to encode
 
        :param n_splits: the number of splits used in mean encoding
 
        :param target_type: str, 'regression' or 'classification'
 
        :param prior_weight_func:
        a function that takes in the number of observations, and outputs prior weight
        when a dict is passed, the default exponential decay function will be used:
        k: the number of observations needed for the posterior to be weighted equally as the prior
        f: larger f --> smaller slope
        """
 
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.learned_stats = {}
 
        if target_type == 'classification':
            self.target_type = target_type
            self.target_values = []
        else:
            self.target_type = 'regression'
            self.target_values = None
 
        if isinstance(prior_weight_func, dict):
            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
        elif callable(prior_weight_func):
            self.prior_weight_func = prior_weight_func
        else:
            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))
 
    @staticmethod
    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
        X_train = X_train[[variable]].copy()
        X_test = X_test[[variable]].copy()
 
        if target is not None:
            nf_name = '{}_pred_{}'.format(variable, target)
            X_train['pred_temp'] = (y_train == target).astype(int)  # classification
        else:
            nf_name = '{}_pred'.format(variable)
            X_train['pred_temp'] = y_train  # regression
        prior = X_train['pred_temp'].mean()
 
        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)
 
        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values
 
        return nf_train, nf_test, prior, col_avg_y
 
    def fit_transform(self, X, y):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :param y: pandas Series or numpy array, n_samples
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
        if self.target_type == 'classification':
            skf = StratifiedKFold(self.n_splits)
        else:
            skf = KFold(self.n_splits)
 
        if self.target_type == 'classification':
            self.target_values = sorted(set(y))
            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
                                  product(self.categorical_features, self.target_values)}
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        else:
            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new.loc[:, nf_name] = np.nan
                for large_ind, small_ind in skf.split(y, y):
                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
                    X_new.iloc[small_ind, -1] = nf_small
                    self.learned_stats[nf_name].append((prior, col_avg_y))
        return X_new
 
    def transform(self, X):
        """
        :param X: pandas DataFrame, n_samples * n_features
        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
        """
        X_new = X.copy()
 
        if self.target_type == 'classification':
            for variable, target in product(self.categorical_features, self.target_values):
                nf_name = '{}_pred_{}'.format(variable, target)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
        else:
            for variable in self.categorical_features:
                nf_name = '{}_pred'.format(variable)
                X_new[nf_name] = 0
                for prior, col_avg_y in self.learned_stats[nf_name]:
                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
                        nf_name]
                X_new[nf_name] /= self.n_splits
 
        return X_new
        
# 高基类别特征和时间特征进行平均数编码
class_list = ['model','brand','name','regionCode']+date_cols
MeanEnocodeFeature = class_list #声明需要平均数编码的特征
ME = MeanEncoder(MeanEnocodeFeature,target_type='regression') #声明平均数编码的类
X_data = ME.fit_transform(X_data,Y_data)#对训练数据集的X和y进行拟合
#x_train_fav = ME.fit_transform(x_train,y_train_fav)#对训练数据集的X和y进行拟合
X_test = ME.transform(X_test)#对测试集进行编码

from sklearn.model_selection import KFold
### target encoding目标编码，回归场景相对来说做目标编码的选择更多，不仅可以做均值编码，还可以做标准差编码、中位数编码等
enc_cols = []
stats_default_dict = {
    'max': X_data['price'].max(),
    'min': X_data['price'].min(),
    'median': X_data['price'].median(),
    'mean': X_data['price'].mean(),
    'sum': X_data['price'].sum(),
    'std': X_data['price'].std(),
    'skew': X_data['price'].skew(),
    'kurt': X_data['price'].kurt(),
    'mad': X_data['price'].mad()
}
### 暂且选择这三种编码
enc_stats = ['max','min','mean']
skf = KFold(n_splits=10, shuffle=True, random_state=42)
for f in tqdm(['regionCode','brand','regDate_year','creatDate_year','kilometer','model']):
    enc_dict = {}
    for stat in enc_stats:
        enc_dict['{}_target_{}'.format(f, stat)] = stat
        X_data['{}_target_{}'.format(f, stat)] = 0
        X_test['{}_target_{}'.format(f, stat)] = 0
        enc_cols.append('{}_target_{}'.format(f, stat))
    for i, (trn_idx, val_idx) in enumerate(skf.split(X_data, Y_data)):
        trn_x, val_x = X_data.iloc[trn_idx].reset_index(drop=True), X_data.iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['price'].agg(enc_dict)
        val_x = val_x[[f]].merge(enc_df, on=f, how='left')
        test_x = X_test[[f]].merge(enc_df, on=f, how='left')
        for stat in enc_stats:
            val_x['{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            test_x['{}_target_{}'.format(f, stat)] = test_x['{}_target_{}'.format(f, stat)].fillna(stats_default_dict[stat])
            X_data.loc[val_idx, '{}_target_{}'.format(f, stat)] = val_x['{}_target_{}'.format(f, stat)].values 
            X_test['{}_target_{}'.format(f, stat)] += test_x['{}_target_{}'.format(f, stat)].values / skf.n_splits

#偏态处理        
# 左偏 指数处理
all_data['V27'] = all_data['V27'].apply(lambda x:math.exp(x))
all_data['V23'] = all_data['V23'].apply(lambda x:math.exp(x))
all_data['V31'] = all_data['V31'].apply(lambda x:math.exp(x))
all_data['V30'] = all_data['V30'].apply(lambda x:math.exp(x))
all_data['V35'] = all_data['V35'].apply(lambda x:math.exp(x))
all_data['V32'] = all_data['V32'].apply(lambda x:math.exp(x))
all_data['V1'] = all_data['V1'].apply(lambda x:math.exp(x))
all_data['V7'] = all_data['V7'].apply(lambda x:math.exp(x))
all_data['V16'] = all_data['V16'].apply(lambda x:math.exp(x))
all_data['V6'] = all_data['V6'].apply(lambda x:math.exp(x))
all_data['V12'] = all_data['V12'].apply(lambda x:math.exp(x))
all_data["V8"] = np.exp(all_data["V8"])
all_data["V0"] = np.exp(all_data["V0"])
all_data["V18"] = np.exp(all_data["V18"])
all_data["V4"] = np.exp(all_data["V4"])
all_data["V26"] = np.exp(all_data["V26"])
#     all_data["V0_mul_V27"] = np.exp(all_data["V0_mul_V27"])

# 右偏 对数处理
all_data['V25'] = np.log1p(all_data['V25'])



from sklearn.preprocessing import MinMaxScaler
#特征归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(pd.concat([x_train,x_test]).values)
all_data = min_max_scaler.transform(pd.concat([x_train,x_test]).values)        

#特征选择
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#方差
threshold = 0.85
vt = VarianceThreshold().fit(X_train)
# Find feature names
feat_var_threshold = X_train.columns[vt.variances_ > threshold * (1-threshold)]
X_train = X_train[feat_var_threshold]
X_test = X_test[feat_var_threshold]
all_data = pd.concat([X_train, X_test])
print("方差后的shape", all_data.shape)

# 单变量选择
X_scored = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)
feature_scoring = pd.DataFrame({
        'feature': X_train.columns,
        'score': X_scored.scores_
    })
head_feature_num = 18
feat_scored_headnum = feature_scoring.sort_values('score', ascending=False).head(head_feature_num)['feature']
X_train_head = X_train[X_train.columns[X_train.columns.isin(feat_scored_headnum)]]

# 根据特征重要性排序挑选特征，使用null importance feature消除噪声
# 获取树模型的特征重要性
def get_feature_importances(data, shuffle, target, seed=None):
    
    # 特征
    train_features = [f for f in data if f not in [target]]
   
    y = data[target].copy()
    
    # 在造null importance时打乱
    if shuffle:
        # 为了打乱而不影响原始数据，采用了.copy().sample(frac=1.0)这种有点奇怪的做法
        y = data[target].copy().sample(frac=1.0)
    
    # 使用lgb的随机森林模式，据说会比sklearn的随机森林快点
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 30,
        'max_depth': 5,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 4
    }
    
    #训练
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    
    return imp_df

actual_imp = get_feature_importances(data=pd.concat([X_data,Y_data],axis=1),shuffle=False,target='y_label')
actual_imp.sort_values("importance_gain",ascending=False)

# 计算null importance
null_imp_df = pd.DataFrame()
nb_runs = 10
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # 获取当前轮feature impotance
    imp_df = get_feature_importances(data=pd.concat([X_data,Y_data],axis=1), shuffle=True, target='y_label')
    imp_df['run'] = i + 1 
    # 加到合集上去
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # 擦除旧信息
    for l in range(len(dsp)):
        print('b', end='', flush=True)
    # 显示当前轮信息
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)

# 计算原始特征重要性和null importance method的特征重要性的得分
# score = log((1+actual_importance)/(1+null_importance_75))
feature_scores = []
for _f in actual_imp['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp.loc[actual_imp['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp.loc[actual_imp['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    final_score = 0.4*split_score+0.6*gain_score
    feature_scores.append((_f, split_score, gain_score,final_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score','final_score'])
null_importance_score = 'null_importance_score.xlsx'
scores_df.to_excel(null_importance_score,index=False)





