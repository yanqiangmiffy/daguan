import sys
import pickle
from sklearn.linear_model import LogisticRegression as LR
from feature import FeatureModel
from util import read_train,read_test
import csv
csv.field_size_limit(500*2014)
# 设置tfidf参数
max_feature_cnt = 280000
feature_max_df = 0.9
feature_min_df = 3
ngram_range = (1, 2)


# 设置文件模型保存路径
model_path = 'model/'
tfidf_model_name = model_path + 'tfidf_feature.model'
best_feature_model_name = model_path + 'best_feature.model'
sk_lr_model_name = model_path + 'sklearn.lr.model'

class SKLearnLR(object):
    """
    LR for 文本分类
    """

    def __init__(self, lr_model_name):
        self.lr_model_name = lr_model_name
        self.init_flag = False

    def trainModel(self, train_x, train_y):
        self.clf = LR(C=4, dual=True)
        self.clf.fit(train_x, train_y)
        self.init_flag = True
        pickle.dump(self.clf, open(self.lr_model_name, 'wb'), True)

    def loadModel(self):
        try:
            self.clf = pickle.load(open(self.lr_model_name, 'rb'))
            self.init_flag = True

        except Exception as e:
            print('Load model fail, ' + str(e))
            sys.exit(1)

    def testModel(self, test_x, test_y):
        if not self.init_flag:
            self.loadModel()

        pred_y = self.clf.predict(test_x)

        total = len(test_y)
        correct = 0
        for idx in range(total):
            if pred_y[idx] == test_y[idx]:
                correct += 1

        print('Test LR: ', total, correct, correct * 1.0 / total)

    def predictModel(self, test_x):
        '''
        test_x: darray [samples feature_cnt]
        '''
        if not self.init_flag:
            self.loadModel()

        pred_y = self.clf.predict(test_x)
        return pred_y.tolist()

def train(train_data_file):
    train_x, train_y = read_train(train_data_file)

    feature_transfor = FeatureModel(tfidf_model_name, best_feature_model_name)

    feature_transfor.fit(max_feature_cnt, feature_max_df,
                         feature_min_df, ngram_range, train_x, train_y)

    model_train_x_feature = feature_transfor.transform(train_x)


    # train a single LR model
    print('正在训练单个LR model...')
    lr_clf = SKLearnLR(sk_lr_model_name)
    lr_clf.trainModel(model_train_x_feature, train_y)
    print('训练完成.')


def predict(test_data_file):
    print("正在加载已经训练模型进行预测")
    test_ids, test_x = read_test(test_data_file)
    feature_transfor = FeatureModel(tfidf_model_name, best_feature_model_name)
    feature_transfor.loadModel()
    model_test_x_feature = feature_transfor.transform(test_x)

    lr_clf = SKLearnLR(sk_lr_model_name)
    lr_preds = lr_clf.predictModel(model_test_x_feature)

    with open('data/results/05_lr_chi.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(('id', 'class'))
        for test_id, pred in zip(test_ids, lr_preds):
            csv_writer.writerow((test_id, int(pred)))

    print("预测结果完成了，你可以提交了^_^")

def eval(test_data_file):
    test_x,test_y = read_train(test_data_file)
    feature_transfor = FeatureModel(tfidf_model_name, best_feature_model_name)
    feature_transfor.loadModel()
    model_test_x_feature = feature_transfor.transform(test_x)

    lr_clf = SKLearnLR(sk_lr_model_name)
    lr_clf.testModel(model_test_x_feature, test_y)

if __name__ == '__main__':
    train_data_file='data/new_data/train_set.csv'
    train(train_data_file)

    test_data_file='data/new_data/test_set.csv'
    predict(test_data_file)

    eval(train_data_file)