import xgboost as xgb
import sys
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost.sklearn import XGBClassifier
csv.field_size_limit(sys.maxsize)


# 读取训练集 按行读取
def read_train(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader][1:]
    X_train = [i[2] for i in data]
    y_train = np.array([int(i[-1]) for i in data])
    print("训练集一共有%d条数据" % (len(data)))
    return X_train, y_train


def read_test(filename):
    print("正在读取测试集数据...")
    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader][1:]
    test_ids = [i[0] for i in data]
    X_test = [i[-1] for i in data]
    print("测试集一共有%d条数据" % (len(data)))
    return test_ids, X_test


def extract_feature(X_train, X_test):
    """
    提取文本特征
    :param data:
    :return:
    """
    print("正在提取特征")
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    X_train_tfidf = tfidf_vec.fit_transform(X_train)
    X_test_tfidf = tfidf_vec.transform(X_test)

    # print(X_train_tfidf.shape,X_test_tfidf.shape)

    return X_train_tfidf, X_test_tfidf
    # return X_train_tfidf.toarray(),X_test_tfidf.toarray()
    # return np.asarray(X_train_tfidf),np.asarray(X_test_tfidf)


def train(X_train, y_train):
    """

    :param X_train:
    :param y_train:
    :return:
    """
    print("开始训练：")
    train_data = xgb.DMatrix(X_train, label=y_train)
    # print(train_data.feature_names)
    param = {
        'max_depth': 8,
        'eta': 0.1,
        'eval_metric': 'merror',
        'silent': 1,
        'objective': 'multi:softmax',
        'num_class': 20
    }
    eval_list = [(train_data, 'train')]
    num_round = 150  # 500
    bst = xgb.train(param, train_data, num_round, eval_list, early_stopping_rounds=20)

    return bst


def train_classify(X_train, y_train):
    """
    使用XGBoostClassifier
    :param X_train:
    :param y_train:
    :return:
    """
    print("正在使用XGBoostClassifier训练")
    model = XGBClassifier(learning_rate=0.1,
                          n_estimators=80,  # 树的个数--1000棵树建立xgboost
                          max_depth=6,  # 树的深度
                          min_child_weight=1,  # 叶子节点最小权重
                          gamma=0.,  # 惩罚项中叶子结点个数前的参数
                          subsample=0.8,  # 随机选择80%样本建立决策树
                          colsample_btree=0.8,  # 随机选择80%特征建立决策树
                          objective='multi:softmax',  # 指定损失函数
                          scale_pos_weight=1,  # 解决样本个数不平衡的问题
                          random_state=27  # 随机数
                          )

    model.fit(X_train,
              y_train,
              eval_set=[(X_train, y_train)],
              eval_metric="mlogloss",
              early_stopping_rounds=10,
              verbose=True)

    return model

def submit_pred(xgb_model, X_test, test_ids):
    print("正在预测结果")
    test_data = xgb.DMatrix(X_test)
    preds = xgb_model.predict(test_data)
    with open('data/results/03_xgb_sub.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(('id', 'class'))
        for test_id, pred in zip(test_ids, preds):
            csv_writer.writerow((test_id, int(pred)))

def submit_pred_classifier(classifier_model, X_test, test_ids):
    print("正在预测classifier结果")
    preds = classifier_model.predict(X_test)
    with open('data/results/03_xgbclassifier_sub.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(('id', 'class'))
        for test_id, pred in zip(test_ids, preds):
            csv_writer.writerow((test_id, int(pred)))

if __name__ == '__main__':
    train_dir = 'data/new_data/train_set.csv'
    test_dir = 'data/new_data/test_set.csv'
    X_train, y_train = read_train(train_dir)
    test_ids, X_test = read_test(test_dir)
    print("读取数据完毕")
    X_train_tfidf, X_test_tfidf = extract_feature(X_train, X_test)
    # print(X_train_tfidf.shape,X_test_tfidf.shape)
    # xgb_model = train(X_train_tfidf, y_train)
    # print("训练完成")
    # submit_pred(xgb_model, X_test_tfidf, test_ids)


    classifier_model = train_classify(X_train_tfidf, y_train)
    submit_pred_classifier(classifier_model, X_test_tfidf, test_ids)
