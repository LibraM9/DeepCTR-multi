# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3train_station.py
# @time  : 2019/7/26
"""
文件说明：
"""
import numpy as np
np.random.seed(2019)

import pandas as pd
import gc
ori_path = '/home/dev/lm/cm_station/data/fusai/'


# path = "F:/项目相关/1907cm_station/feature/"
path = '/home/dev/lm/cm_station/feature/fusai/' + 'feature/'
out_path = '/home/dev/lm/cm_station/feature/fusai/' + "out/"
# ori_path ="C:\\Users\\gupeng\\Desktop\\yidong\\errorClass\\fusai\\"
# path = ori_path + 'feature\\'
# out_path = ori_path + "out\\"
data_all = pd.read_csv(open(path+'datafusai_all0822_nodrop.csv',encoding="utf8"))
data_label = pd.read_csv(open(path+'data_label0808.csv',encoding="utf8"))
data_label2 = pd.read_csv(open(path+'gupeng0820.csv',encoding="utf8"))
data_near = pd.read_csv(open(path+'near3_warning.csv',encoding="utf8"))
del data_near['time']
del data_near['error']
del data_near['Unnamed: 0']
del data_near['time_last1day']
data_all = data_all.merge(data_label,how='left',on=["id","station","is_train"])
data_all = data_all.merge(data_label2,how='left',on=["id","station","is_train"])
data_all = data_all.merge(data_near,how='left',on=["id","station","is_train"])

# data_gupeng = pd.read_csv(path+"gupengfeaturenew.csv")
# data_all = data_all.merge(data_gupeng,on = ['id','station','is_train'],how = 'left')
del data_all['time'],data_all['station']
train = data_all.loc[data_all.is_train == 1]
train = train.reset_index(drop=True)
###对train筛选，去除概率低的
val = True #使用验证集
n = 23 #分类数量
train_1 = train.loc[train["error"].isin(range(n))]
#####
test = data_all.loc[~(data_all['is_train'] == 1)]
test = test.reset_index(drop=True)
y_train = train_1['error']
y_test = test['error']
del train_1['error'] ,test['error'],train_1['is_train'] ,test['is_train'],train_1["id"],test["id"]
features = ["hour","day","day_of_week"]
features = features +[ i for i in train.columns if "time_all_alert" in i]\
           +[ i for i in train.columns if "time_futureall_alert" in i]\
           +[ i for i in train.columns if "time_last0_alert" in i]\
           +[ i for i in train.columns if "time_last5_alert" in i]\
           +[ i for i in train.columns if "time_future10_alert" in i]\
           +[ i for i in train.columns if "time_last10_alert" in i]\
           + [i for i in train.columns if "time_future5_alert" in i] \
           +[ i for i in train.columns if "time_all_error" in i]\
           +[ i for i in train.columns if "time_futureall_error" in i]\
           +[ i for i in train.columns if "time_last0_error" in i]\
           +[i for i in train.columns if "link_last_" in i]\
           +[i for i in train.columns if "link_future_" in i]

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error,roc_auc_score,accuracy_score
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier

folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=2333)
# folds = KFold(n_splits=5, shuffle=True, random_state=2333)

oof = np.zeros((len(train_1),n))
oof_train = np.zeros((len(train),n))
predictions = np.zeros((len(test),n))
feature_importance_df = pd.DataFrame()

train_x = train_1[features].values
test_x = test[features].values
clf_labels = y_train.values

param = {'objective': 'multiclass',
         'num_class':n,
         'num_leaves': 2**6, #2**5
         # 'min_data_in_leaf': 25,#
         'max_depth': 6,  # 5 2.02949 4 2.02981
         'learning_rate': 0.08, #0.02
         'lambda_l1': 0.13,
         "boosting": "gbdt",
         "feature_fraction": 0.85,
         'bagging_freq': 8,
         "bagging_fraction": 0.9, #0.9
         "metric": 'multi_logloss',
         "verbosity": -1,
         "random_state": 2333,
         "num_threads" : 50}
# lgb
model = "lgb" #0.2778 0.4019
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, clf_labels)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x[trn_idx],label=clf_labels[trn_idx])
    val_data = lgb.Dataset(train_x[val_idx],label=clf_labels[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=50,early_stopping_rounds=100)
    #n*6矩阵
    oof[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    oof_train +=clf.predict(train[features].values, num_iteration=clf.best_iteration) / folds.n_splits
    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits
gc.collect()
sparse_train = clf.predict(train_x, pred_leaf=True, num_iteration=clf.best_iteration)
print("节点维度",sparse_train.shape)
sparse_test = clf.predict(test_x, pred_leaf=True, num_iteration=clf.best_iteration)
node_list = ["node_"+str(i) for i in range(len(sparse_train[0]))]
sparse_train = pd.DataFrame(sparse_train)
sparse_train.columns=node_list
sparse_test = pd.DataFrame(sparse_test)
sparse_test.columns = node_list
del train_x
del test_x
#######################################
import sys
sys.path.append("/home/dev/lm/utils_lm")
sys.path.append("/home/dev/lm/DeepCTR_multi")
import numpy as np

from model_train.a1_preprocessing import NaFilterFeature #删除缺失过多的列
nf = NaFilterFeature(num=0.9)
X_train = nf.fit_transform(train[features])
X_test = nf.transform(test[features])
from model_train.a2_feature_selection import select_primaryvalue_ratio#同值性筛选
features_spr, feature_primaryvalue_ratio = select_primaryvalue_ratio(X_train,ratiolimit=1)
X_train = X_train[features_spr]
X_test = X_test[features_spr]
#用中位数填充
for i in X_train.columns:
    X_train[i] = X_train[i].fillna(X_train[i].median())
    X_test[i] = X_test[i].fillna(X_test[i].median())

#归一化和编码
X_test = X_test.reset_index(drop=True)

sparse_features  = []

dense_features = [i for i in X_train.columns if i not in sparse_features]
target = clf_labels
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr1.models import DeepFM,DCN,xDeepFM
from deepctr1.utils import SingleFeat

#对稠密特征归一化
mms = MinMaxScaler(feature_range=(0, 1))
X_train[dense_features] = mms.fit_transform(X_train[dense_features])
X_test[dense_features] = mms.transform(X_test[dense_features])

#对稀疏特征编码 千维特征速度奇慢
for feat in sparse_features:
    lbe = LabelEncoder()
    X_train[feat] = lbe.fit_transform(X_train[feat])
    X_test[feat] = lbe.transform(X_test[feat])

#加入节点特征
X_train = pd.concat([X_train,sparse_train],axis=1)
X_test = pd.concat([X_test,sparse_test],axis=1)
sparse_features = sparse_features + [i for i in X_train.columns if i[:4]=="node"]
del sparse_train
del sparse_test

sparse_feature_list = [SingleFeat(feat, max(X_train[feat].nunique(),X_train[feat].max()+1))  # since the input is string
                           for feat in sparse_features]
dense_feature_list = [SingleFeat(feat, 0, )
                            for feat in dense_features]

# 5折神经网络
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
# folds = KFold(n_splits=5, shuffle=True, random_state=2333)

oof = np.zeros((len(train_1),n))
predictions = np.zeros((len(test),n))

clf_labels = clf_labels

test_model_input = [X_test[feat.name].values for feat in sparse_feature_list] + \
                   [X_test[feat.name].values for feat in dense_feature_list]

model_name = "dfm"

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, clf_labels)):
    print("fold {}".format(fold_))
    trn_x, trn_y = X_train.loc[trn_idx,:], clf_labels[trn_idx]
    val_x, val_y = X_train.loc[val_idx,:], clf_labels[val_idx]

    train_model_input = [trn_x[feat.name].values for feat in sparse_feature_list] + \
                        [trn_x[feat.name].values for feat in dense_feature_list]
    val_model_input = [val_x[feat.name].values for feat in sparse_feature_list] + \
                        [val_x[feat.name].values for feat in dense_feature_list]

    t_true = pd.get_dummies(pd.Series(trn_y))

    model = DeepFM({"sparse": sparse_feature_list, "dense": dense_feature_list}, task='multi-class', multi_n=n)
    model.compile("adam", "categorical_crossentropy", metrics=['crossentropy'], )  #
    batch_size = 512
    history = model.fit(train_model_input, t_true.values, batch_size=batch_size, epochs=4, verbose=2, validation_split=0, )
    # epochs 5 batch 256 0.9319
    # epochs 4 batch 256 0.9339
    # epochs 4 batch 512 0.9342
    # epochs 3 batch 256 0.9327
    # epochs 2 batch 256 0.9319

    #n*6矩阵
    oof[val_idx] = model.predict(val_model_input, batch_size=batch_size)

    predictions += model.predict(test_model_input, batch_size=batch_size) / folds.n_splits
    gc.collect()
#新的acc函数
import heapq
def acc_new(y_true,oof):
    oof2 = pd.Series(list(oof))
    oof2=oof2.apply(lambda x:list(map(list(x).index, heapq.nlargest(3,x))))
    ans=[]
    for i in range(len(oof2)):
        if y_true[i] in oof2[i]:
            ans.append(1)
        else:
            ans.append(0)
    return sum(ans)/len(ans)

print("CV score: {:<8.5f}".format(log_loss(clf_labels, oof)))

oof_train = pd.DataFrame(oof_train)
if n < 23:
    for i in range(n,23):
        oof_train[i]=0
print("CV score_train: {:<8.5f}".format(log_loss(train["error"].values, oof_train.values)))
feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
ascending=False)
#评价指标： AUC指标 准确率
if val==False:
    oof = oof_train.values #6 0.9342 5 0.9328/0.8969
clf_one_hot = train["error"]
clf_one_hot = pd.get_dummies(clf_one_hot)
auc = roc_auc_score(clf_one_hot, oof,average='weighted')
print("auc:",auc)
#计算acc
acc = acc_new(train["error"].values,oof)
print("acc:",acc)
print("result:",0.5*auc+0.5*acc)
loss = str(0.5*auc+0.5*acc)[2:6]
pd.DataFrame(oof).to_csv(out_path+"oof_{}_{}.csv".format(model_name,loss),index=False)

predictions = pd.DataFrame(predictions)
if n < 23:
    for i in range(n,23):
        predictions[n]=0

column = {'主设备-硬件故障': 0,
 '其他-误告警或自动恢复': 1,
 '动力环境-电力部门供电': 2,
 '主设备-参数配置异常': 3,
 '主设备-设备复位问题': 4,
 '主设备-软件故障': 5,
 '动力环境-开关电源': 6,
 '主设备-设备连线故障': 7,
 '动力环境-动力环境故障': 8,
 '主设备-信源问题': 9,
 '传输系统-光缆故障': 10,
 '传输系统-其他原因': 11,
 '动力环境-高低压设备': 12,
 '动力环境-电源线路故障': 13,
 '动力环境-环境': 14,
 '传输系统-传输设备': 15,
 '动力环境-UPS': 16,
 '动力环境-动环监控系统': 17,
 '主设备-其他': 18,
 '人为操作-告警测试': 19,
 '人为操作-工程施工': 20,
 '人为操作-物业原因': 21,
 '主设备-天馈线故障': 22}
label = list(column.keys())
test_result = np.around(predictions.values,decimals = 3)
pd_test_result=pd.DataFrame(test_result,columns = label)

test_df = pd.read_csv(open(ori_path+"测试故障工单.csv",encoding="gb2312"),parse_dates=["故障发生时间"])
del test_df['故障发生时间'],test_df['涉及告警基站或小区名称'],test_df['故障原因定位（大类）']

sample = pd.read_csv(open(ori_path+"sample.csv",encoding="gb2312"))
del sample["工单号"]

for i in sample.columns:
    test_df[i] = pd_test_result[i]
    test_df[i] = test_df[i].apply(lambda x: str(int(x)) if x == 0 else x)

test_df.to_csv(out_path+'result_{}{}_{}.csv'.format(model_name,n,loss),index = False,encoding='GB2312')
