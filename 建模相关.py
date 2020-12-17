#--------------网格调参---------
# 构造模型方法
# 每次调完一个参数，要把 other_params对应的参数更新为最优值。
# xgb
def build_model_xgb(x_train,y_train):
    #  cv 参数 迭代次数
    cv_params = {
#                   'n_estimators':[550, 575, 600, 650, 675,700,750,800,850], 550
#                  'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 9
#                  'min_child_weight': [1, 2, 3, 4, 5, 6], 6
#                 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.6
#                 'subsample': [0.6, 0.7, 0.8, 0.9], 0.7
#                  'colsample_bytree': [0.6, 0.7, 0.8, 0.9],0.6
#                 'reg_alpha': [0.05, 0.1, 1, 2, 3], 
#                  'reg_lambda': [0.05, 0.1, 1, 2, 3],
#                 'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]
                }
    # 初始化参数     
    other_params = {'learning_rate': 0.01,'n_estimators': 500,'max_depth': 3,'min_child_weight': 6,'subsample': 0.8,
                   'colsample_bytree':0.7,'gamma': 0.6,'reg_alpha': 50,'reg_lambda': 100}
    model = xgb.XGBRegressor(**other_params) #, objective ='reg:squarederror'
#     optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
#     optimized_GBM.fit(x_train, y_train)
#     evalute_result = optimized_GBM.scorer_
#     print('每轮迭代运行结果:{0}'.format(evalute_result))
#     print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#     print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    model.fit(x_train,y_train)
    return model

# lgb
def build_model_lgb(x_train,y_train):
        #  cv 参数 迭代次数
#     cv_params = {
# #                     'n_estimators':[550, 575, 600, 650, 675], 650
# #                  'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 4
# #                  'min_child_weight': [1, 2, 3, 4, 5, 6],1
# #                 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],0.1
# #                 'subsample': [0.6, 0.7, 0.8, 0.9], 0.6
# #                  'colsample_bytree': [0.6, 0.7, 0.8, 0.9],0.6
# #                 'reg_alpha': [0.05, 0.1, 1, 2, 3], 1
# #                  'reg_lambda': [0.05, 0.1, 1, 2, 3], 0.05
# #                 'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]
#                 }
    # 初始化参数     
    other_params = {'learning_rate': 0.02,'n_estimators': 650,'max_depth': 2,'min_child_weight': 1,'subsample': 0.8,
                   'colsample_bytree':0.6,'gamma': 0.1,'reg_alpha': 7,'reg_lambda': 5}

    model = lgb.LGBMRegressor(**other_params)
#     optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
#     optimized_GBM.fit(x_train, y_train)
#     evalute_result = optimized_GBM.scorer_
#     print('每轮迭代运行结果:{0}'.format(evalute_result))
#     print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#     print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    model.fit(x_train, y_train)
    return model

# cab
def build_model_cab(x_train,y_train):
            #  cv 参数 迭代次数
    cv_params = {
#                     'n_estimators':[550, 575, 600, 650, 675], 675
#                  'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 6
#                 'subsample': [0.6, 0.7, 0.8, 0.9], 0.6
#                  'reg_lambda': [0.05, 0.1, 1, 2, 3], 3
#                 'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2] 0.07
                }
    other_params = {'learning_rate': 0.02,'n_estimators': 675,'max_depth': 3,'subsample': 0.6,
                  'reg_lambda': 3}
    
    model = cab.CatBoostRegressor(**other_params)
#     optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
#     optimized_GBM.fit(x_train, y_train)
#     evalute_result = optimized_GBM.scorer_
#     print('每轮迭代运行结果:{0}'.format(evalute_result))
#     print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
#     print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    model.fit(x_train,y_train,plot=True)
    return model
    
from sklearn.linear_model import LogisticRegression,ElasticNet

clf = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)

# 线上预测的结果集
onset_predictions = np.zeros(len(X_onset))
# 线下验证的结果集
offset_predictions = np.zeros(len(X_offset))
#  线下cv auc的均值
mean_score = 0
sk = StratifiedKFold(n_splits=5,shuffle=True)
for fold,(train_idx,val_idx) in enumerate(sk.split(X_offset.values,y_offset.values)):
    print('fold {}:'.format(fold+1))
    trn_data = X_offset.iloc[train_idx]
    val_data = X_offset.iloc[val_idx]
    
    clf.fit(trn_data,y_offset.iloc[train_idx])
    #  线下每折的结果集    
    offset_predictions[val_idx] = clf.predict_proba(val_data.values)[:,1]
    mean_score += roc_auc_score(y_offset.iloc[val_idx],offset_predictions[val_idx])/sk.n_splits
    onset_predictions += clf.predict_proba(X_onset.values)[:,1]/sk.n_splits
    
print('lr 线下cv的平均auc:{:<8.5f}'.format(mean_score))
print('lr 线下结果集的auc:{:<8.5f}'.format(roc_auc_score(y_offset,offset_predictions)))
    
# 线上得分
print('lr 线上结果集的auc:{:<8.5f}'.format(roc_auc_score(y_onset,onset_predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,onset_predictions)
print('lr 线上结果集的ks:{:<8.5f}'.format(max(tpr-fpr)))


from sklearn.model_selection import StratifiedKFold
params = {
'boosting_type': 'gbdt',
'objective': 'binary',
'metric':  'auc',
'num_leaves': 30,
'max_depth': -1,
'min_data_in_leaf': 450,
'learning_rate': 0.01,
'feature_fraction': 0.9,
'bagging_fraction': 0.95,
'bagging_freq': 5,
'lambda_l1': 1,
'lambda_l2': 0.001,# 越小l2正则程度越高
'min_gain_to_split': 0.2,
#'device': 'gpu',
'is_unbalance': True
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=99999)
oof = np.zeros(len(X_offset))
predictions = np.zeros(len(X_onset))
mean_score=0
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_offset.values, y_offset.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(X_offset.iloc[trn_idx], label=y_offset.iloc[trn_idx])
    val_data = lgb.Dataset(X_offset.iloc[val_idx], label=y_offset.iloc[val_idx])
    clf = lgb.train(params, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(X_offset.iloc[val_idx], num_iteration=clf.best_iteration)
    mean_score += roc_auc_score(y_offset.iloc[val_idx], oof[val_idx]) / folds.n_splits
    predictions += clf.predict(X_onset, num_iteration=clf.best_iteration) / folds.n_splits

# 线下cv
print("CV score: {:<8.5f}".format(roc_auc_score(y_offset, oof)))
print("mean score: {:<8.5f}".format(mean_score))

# 线上得分
print("lgb online score: {:<8.5f}".format(roc_auc_score(y_onset, predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,predictions)
print('lgb 线上结果集 ks:{:<8.5f}'.format(max(tpr-fpr)))

params={
    'booster':'gbtree',
    'objective ':'binary:logistic',
    'gamma':0.1,                     # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth':6,                   # 构建树的深度 [1:]
    'subsample':0.8,                 # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_by_tree':0.8,          # 构建树时的采样比率 (0:1]
    'eta': 0.01,                     # 如同学习率
    'seed':555,                      # 随机种子
    'silent':1,
    'eval_metric':'auc',
    'n_job':-1,
#     'tree_method':'gpu_hist'
}

# 线上用于预测的矩阵 
test_data = xgb.DMatrix(X_onset,label=y_onset)
# 线上预测的结果集
onset_predictions = np.zeros(len(X_onset))
# 线下验证的结果集
offset_predictions = np.zeros(len(X_offset))
#  线下cv auc的均值
mean_score = 0
sk = StratifiedKFold(n_splits=5,shuffle=True)
for fold,(train_idx,val_idx) in enumerate(sk.split(X_offset.values,y_offset.values)):
    print('fold {}:'.format(fold))
    trn_data = xgb.DMatrix(X_offset.iloc[train_idx],label=y_offset.iloc[train_idx])
    val_data = xgb.DMatrix(X_offset.iloc[val_idx],label=y_offset.iloc[val_idx])
    
    clf = xgb.train(params=params,dtrain=trn_data,num_boost_round=10000,evals=[(trn_data,'train'),(val_data,'val')],
                    early_stopping_rounds=200,verbose_eval=100)
    #  线下每折的结果集    
    offset_predictions[val_idx] = clf.predict(val_data,ntree_limit=clf.best_iteration)
    mean_score += roc_auc_score(y_offset.iloc[val_idx],offset_predictions[val_idx])/sk.n_splits
    onset_predictions += clf.predict(test_data,ntree_limit=clf.best_iteration)/sk.n_splits
    
print('xgb 线下cv的平均auc:{:<8.5f}'.format(mean_score))
print('xgb 线下结果集的auc:{:<8.5f}'.format(roc_auc_score(y_offset,offset_predictions)))
    
# 线上得分
print('xgb 线上结果集的auc:{:<8.5f}'.format(roc_auc_score(y_onset,onset_predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,onset_predictions)
print('xgb 线上结果集的ks:{:<8.5f}'.format(max(tpr-fpr)))

clf = cab.CatBoostClassifier(iterations=100000, 
learning_rate=0.01, 
depth=5, loss_function='Logloss',early_stopping_rounds = 100,eval_metric='AUC')

# 线下预测结果集
offset_predictions = np.zeros(len(X_offset))
# 线上预测结果集
onset_predictions = np.zeros(len(X_onset))
# 线下cv auc平均得分
mean_score = 0
sk = StratifiedKFold(n_splits=5,shuffle=True)

for fold,(train_idx,val_idx) in enumerate(sk.split(X_offset.values,y_offset.values)):
    print('fold {}'.format(fold+1))
    train_data = X_offset.iloc[train_idx]
    val_data = X_offset.iloc[val_idx]
    
    clf.fit(train_data, y_offset.iloc[train_idx], eval_set=(val_data,y_offset.iloc[val_idx]),verbose= 50)
    offset_predictions[val_idx] = clf.predict_proba(val_data)[:,1]
    mean_score += roc_auc_score(y_offset.iloc[val_idx],offset_predictions[val_idx])/sk.n_splits
    onset_predictions += clf.predict_proba(X_onset)[:,1]/sk.n_splits

print('cab 线下cv的平均auc:{:<8.5f}'.format(mean_score))
print('cab 线下结果集的auc:{:<8.5f}'.format(roc_auc_score(y_offset,offset_predictions)))
    
# 线上得分
print('cab 线上结果集的auc:{:<8.5f}'.format(roc_auc_score(y_onset,onset_predictions)))
fpr,tpr,thresholds=roc_curve(y_onset,onset_predictions)
print('cab 线上结果集的ks:{:<8.5f}'.format(max(tpr-fpr)))

#测试git修改文件

    
