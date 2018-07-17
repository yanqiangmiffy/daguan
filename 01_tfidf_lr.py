import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time



def load_data():
    print("正在加载数据....")
    train = pd.read_csv('data/new_data/train_set.csv')
    test = pd.read_csv('data/new_data/test_set.csv')
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
    vec = TfidfVectorizer(min_df=3, max_df=0.9)
    # vec = TfidfVectorizer()
    train_term_doc = vec.fit_transform(train[column])
    test_term_doc = vec.transform(test[column])
    y = (train['class']).astype(int)
    data=dict()
    data['id'] = test_id['id']
    data['train'] = train_term_doc
    data['test'] = test_term_doc
    data['y'] = y
    del test_id['id'],train_term_doc,test_term_doc
    return data

# 结果评估
def get_metrics(y_test,y_predicted):
    """
    :param y_test: 真实值
    :param y_predicted: 预测值
    :return:
    """
    # 精确度=真阳性/（真阳性+假阳性）
    precision=precision_score(y_test,y_predicted,pos_label=None,average='weighted')
    # 召回率=真阳性/（真阳性+假阴性）
    recall=recall_score(y_test,y_predicted,pos_label=None,average='weighted')

    # F1
    f1=f1_score(y_test,y_predicted,pos_label=None,average='weighted')
    # 精确率
    accuracy=accuracy_score(y_test,y_predicted)
    return accuracy,precision,recall,f1


def train_tfidf(X_train,X_test,y_train,y_test):
    print("正在训练...")
    clf = LogisticRegression(C=10.0, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_predicted=clf.predict(X_test)
    accuracy, precision, recall, f1=get_metrics(y_test,y_predicted)
    print("accuracy, precision, recall, f1=",accuracy, precision, recall, f1)
    return clf

def predict(clf_model,data):
    preds = clf_model.predict(data['test'])
    # 生成提交结果
    test_pred = pd.DataFrame(preds)
    test_pred.columns = ["class"]
    test_pred["class"] = (test_pred["class"]).astype(int)
    print(test_pred.shape)
    print(data["id"].shape)
    test_pred["id"] = list(data["id"])
    test_pred[["id", "class"]].to_csv('data/results/01_lr_tfidf.csv', index=None)

if __name__ == '__main__':
    t1 = time.time()
    data=load_data()
    X_train,X_test,y_train,y_test=train_test_split(data['train'],data['y'],test_size=0.1,random_state=42)
    clf_model=train_tfidf(X_train,X_test,y_train,y_test)
    predict(clf_model, data)
    t2 = time.time()
    print("time use:", t2 - t1)