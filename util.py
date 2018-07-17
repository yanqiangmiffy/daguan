import pandas as pd
import csv
import pickle
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
csv.field_size_limit(500*1000)
def load_data(train_data_file='data/new_data/train_set.csv',
              test_data_file='data/new_data/test_set.csv'):
    print("正在加载数据....")
    if os.path.exists('model/tfidf-feature.pkl'):
        print("加已存在的特征")
        with open('model/tfidf-feature.pkl', 'rb') as in_data:
            data = pickle.load(in_data)
            return data
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    test_id = test[['id']].copy()

    column = "word_seg"

    # ngram_range：词组切分的长度范围
    # max_df：可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。
    # 这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。
    # 如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。如果参数中已经给定了vocabulary，则这个参数无效
    # min_df：类似于max_df，不同之处在于如果某个词的document frequence小于min_df，则这个词不会被当作关键词
    # use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了
    # smooth_idf：idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
    # sublinear_tf：默认为False，如果设为True，则替换tf为1 + log(tf)。

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    train_term_doc = vec.fit_transform(train[column])
    test_term_doc = vec.transform(test[column])

    y = (train['class']).astype(int)

    data=dict()
    data['test_id'] = test_id['id']
    data['X_train'] = train_term_doc
    data['X_test'] = test_term_doc
    data['y_train'] = y

    del test_id['id'],train_term_doc,test_term_doc
    print("正在保存数据")
    with open('model/tfidf-feature.pkl','wb') as out_data:
        pickle.dump(data,out_data,pickle.HIGHEST_PROTOCOL)
    return data




# 读取训练集 按行读取
def read_train(filename):
    print("正在读取训练集数据...")

    if os.path.exists('data/new_data/train.pkl'):
        with open('data/new_data/train.pkl', 'rb') as in_data:
            train=pickle.load(in_data)
            print("测试集一共有%d条数据" % (len(train['X_train'])))
            return train['X_train'],train['y_train']

    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader][1:]
    X_train = [i[2] for i in data]
    y_train = [int(i[-1]) for i in data]
    print("训练集一共有%d条数据" % (len(data)))
    train={'X_train':X_train,'y_train':y_train}
    with open('data/new_data/train.pkl','wb') as out_data:
        pickle.dump(train,out_data,pickle.HIGHEST_PROTOCOL)
    return X_train, y_train


def read_test(filename):
    print("正在读取测试集数据...")

    if os.path.exists('data/new_data/test.pkl'):
        with open('data/new_data/test.pkl', 'rb') as in_data:
            test=pickle.load(in_data)
            print("测试集一共有%d条数据" % (len(test['X_test'])))
            return test['test_ids'],test['X_test']

    with open(filename, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        data = [row for row in csv_reader][1:]
    test_ids = [i[0] for i in data]
    X_test = [i[-1] for i in data]
    test = {'test_ids': test_ids, 'X_test': X_test}
    with open('data/new_data/test.pkl', 'wb') as out_data:
        pickle.dump(test, out_data, pickle.HIGHEST_PROTOCOL)
    return test_ids, X_test

