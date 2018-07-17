import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import numpy as np
from scipy.sparse import hstack
from util import load_data
import pickle


def xgb_feature_encode(train_data_file, test_data_file, xgb_feature_file):
    all_data = load_data(train_data_file, test_data_file)
    # 训练/测试数据分割
    X_train, X_test, y_train, y_test = train_test_split(all_data['X_train'], all_data['y_train'], test_size=0.1,
                                                        random_state=1)
    # 定义模型
    xgboost = xgb.XGBClassifier(learning_rate=0.05,
                                n_estimators=50,
                                max_depth=3,
                                gamma=0,
                                subsample=0.7,
                                colsample_bytree=0.7)
    # 训练学习
    xgboost.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="mlogloss",
                early_stopping_rounds=10,
                verbose=True)

    # 预测及AUC评测
    y_pred_test = xgboost.predict(X_test)
    num = 0
    for i in range(0, len(y_pred_test)):
        if y_test[i] == y_pred_test[i]:
            num += 1
    print("prediction accuracy is " + str((num) / len(y_pred_test)))

    # xgboost编码原有特征
    X_train_leaves = xgboost.apply(X_train)
    X_test_leaves = xgboost.apply(X_test)
    # 训练样本个数
    train_rows = X_train_leaves.shape[0]
    # 合并编码后的训练数据和测试数据
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)

    (rows, cols) = X_leaves.shape

    # 记录每棵树的编码区间
    cum_count = np.zeros((1, cols), dtype=np.int32)

    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(X_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(X_leaves[:, j])) + cum_count[0][j - 1]

    print('Transform features genenrated by xgboost...')
    # 对所有特征进行ont-hot编码
    for j in range(cols):
        keyMapDict = {}
        if j == 0:
            initial_index = 1
        else:
            initial_index = cum_count[0][j - 1] + 1
        for i in range(rows):
            if X_leaves[i, j] not in keyMapDict:
                keyMapDict[X_leaves[i, j]] = initial_index
                X_leaves[i, j] = initial_index
                initial_index = initial_index + 1
            else:
                X_leaves[i, j] = keyMapDict[X_leaves[i, j]]

    # 基于编码后的特征，将特征处理为libsvm格式且写入文件
    print('Write xgboost learned features to file ...')
    with  open(xgb_feature_file, 'wb') as out_data:
        xgb_feature = [X_leaves, all_data['y_train']]
        pickle.dump(xgb_feature, out_data)


def xgboost_lr_train(train_data_file, test_data_file, xgb_feature_file):
    # load 原始样本数据
    all_data = load_data(train_data_file, test_data_file)
    X_train_origin, X_test_origin, y_train_origin, y_test_origin = train_test_split(all_data['X_train'],
                                                                                    all_data['y_train'],
                                                                                    test_size=0.3,
                                                                                    random_state=42)

    # load xgboost特征编码后的样本数据
    with  open(xgb_feature_file, 'rb') as in_data:
        X_xg_all, y_xg_all = pickle.load(in_data)
    X_train, X_test, y_train, y_test = train_test_split(X_xg_all, y_xg_all, test_size=0.3, random_state=42)

    # lr对原始特征样本模型训练
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr.fit(X_train_origin, y_train_origin)
    joblib.dump(lr, 'model/lr_orgin.m')
    # 预测及AUC评测
    y_pred_test = lr.predict(X_test_origin)

    num = 0
    for i in range(0, len(y_pred_test)):
        if y_test_origin[i] == y_pred_test[i]:
            num += 1
    print("prediction accuracy is " + str((num) / len(y_pred_test)))

    # lr对load xgboost特征编码后的样本模型训练
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'model/lr_xgb.m')
    # 预测及AUC评测
    y_pred_test = lr.predict(X_test)
    num = 0
    for i in range(0, len(y_pred_test)):
        if y_test_origin[i] == y_pred_test[i]:
            num += 1
    print("prediction accuracy is " + str((num) / len(y_pred_test)))

    # 基于原始特征组合xgboost编码后的特征
    X_train_ext = hstack([X_train_origin, X_train])
    del (X_train)
    del (X_train_origin)
    X_test_ext = hstack([X_test_origin, X_test])
    del (X_test)
    del (X_test_origin)

    # lr对组合后的新特征的样本进行模型训练
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    lr.fit(X_train_ext, y_train)
    joblib.dump(lr, 'model/lr_ext.m')
    # 预测及AUC评测
    y_pred_test = lr.predict(X_test_ext)
    num = 0
    for i in range(0, len(y_pred_test)):
        if y_test_origin[i] == y_pred_test[i]:
            num += 1
    print("prediction accuracy is " + str((num) / len(y_pred_test)))


if __name__ == '__main__':
    # 加载数据集
    train_data_file = 'data/new_data/train_set.csv'
    test_data_file = 'data/new_data/train_set.csv'
    xgb_feature_file = 'model/xgb_feature_libsvm'

    xgb_feature_encode(train_data_file, test_data_file, xgb_feature_file)
    xgboost_lr_train(train_data_file, test_data_file, xgb_feature_file)
