# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3model_ning.py
# @time  : 2019/7/2
"""
文件说明：
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score,roc_auc_score

oripath = "/home/dev/lm/paipai/ori_data/"
feature_path = '/home/dev/lm/paipai/feature_ning/'

train = pd.read_csv(feature_path+"ning_train.csv")
test = pd.read_csv(feature_path+"ning_test.csv")

y = "early_repay_days"

drop_list= ['age','info_insert_date','taglist','repay_amt','repay_date','early_repay_days','late_repay_days','auditing_date','listing_id','user_id']
features = []
for col in train.columns:
    if col not in drop_list:
        features.append(col)

n = 33

import sys
sys.path.append("/home/dev/lm/utils_lm")
sys.path.append("/home/dev/lm/DeepCTR_multi")

X_train = train[features]
X_test = test[features]
for i in X_train.columns:
    X_train[i] = X_train[i].fillna(X_train[i].median())
    X_test[i] = X_test[i].fillna(X_test[i].median())

X_test = X_test.reset_index(drop=True)

sparse_features  = ["info_insert_date_month","reg_mon_date_month","due_date_day"]
dense_features = [i for i in X_train.columns if i not in sparse_features]
target = y
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr1.models import DeepFM
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

sparse_feature_list = [SingleFeat(feat, X_train[feat].nunique())  # since the input is string
                           for feat in sparse_features]
dense_feature_list = [SingleFeat(feat, 0, )
                            for feat in dense_features]

train_model_input = [X_train[feat.name].values for feat in sparse_feature_list] + \
                    [X_train[feat.name].values for feat in dense_feature_list]
test_model_input = [X_test[feat.name].values for feat in sparse_feature_list] + \
                   [X_test[feat.name].values for feat in dense_feature_list]

t_true = pd.get_dummies(train[target])
model = DeepFM({"sparse": sparse_feature_list,"dense": dense_feature_list}, task='multi-class')
# model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], ) #2.1897
model.compile("adam", "categorical_crossentropy", metrics=['crossentropy'], ) #

history = model.fit(train_model_input, t_true.values,batch_size=512, epochs=10, verbose=2, validation_split=0.2, )
pred_train = model.predict(train_model_input, batch_size=512)
print("train score", log_loss(train[target].values, pred_train))

pred_test = model.predict(test_model_input, batch_size=512)

#####输出结果
# train_prob = pd.DataFrame(pred_train)
# train_dic = {
#     "user_id": train["user_id"].values,
#     "listing_id":train["listing_id"].values,
#     "auditing_date":train["auditing_date"].values,
#     "due_date":train["due_date"].values,
#     "due_amt":train["due_amt"].values,
# }
# for key in train_dic:
#     train_prob[key] = train_dic[key]
# train_prob.to_csv(feature_path + 'sub_ning_dfm.csv', index=None)

test_prob = pd.DataFrame(pred_test)
test_dic = {
    "user_id": test["user_id"].values,
    "listing_id":test["listing_id"].values,
    "auditing_date":test["auditing_date"].values,
    "due_amt":test["due_amt"].values,
}
for key in test_dic:
    test_prob[key] = test_dic[key]
#输出预测概率
# test_prob.to_csv(outpath+'out_dfm368_test.csv',index=None)
for i in range(n-1):
    test_prob[i] = test_prob[i]*test_prob["due_amt"]
#对于训练集评价
def df_rank(df_prob, df_sub):
    for i in range(33):
        print('转换中',i)
        df_tmp = df_prob[['listing_id', i]]
        df_tmp['rank'] = i+1
        df_sub = df_sub.merge(df_tmp,how='left',on=["listing_id",'rank'])
        df_sub.loc[df_sub['rank']==i+1,'repay_amt']=df_sub.loc[df_sub['rank']==i+1,i]
    return df_sub[['listing_id','repay_amt','repay_date']]
#提交
submission = pd.read_csv(open(oripath+"submission.csv",encoding='utf8'),parse_dates=["repay_date"])
submission['rank'] = submission.groupby('listing_id')['repay_date'].rank(ascending=False,method='first')
sub = df_rank(test_prob, submission)
sub.to_csv(feature_path+'sub_ning_dfm.csv',index=None)