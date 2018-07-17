import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
t1=time.time()
train = pd.read_csv('data/new_data/train_set.csv')
test = pd.read_csv('data/new_data/test_set.csv')
test_id = test[["id"]].copy()

column="word_seg"
n = train.shape[0]
# ngram_range：词组切分的长度范围
# max_df：可以设置为范围在[0.0 1.0]的float，也可以设置为没有范围限制的int，默认为1.0。
# 这个参数的作用是作为一个阈值，当构造语料库的关键词集的时候，如果某个词的document frequence大于max_df，这个词不会被当作关键词。
# 如果这个参数是float，则表示词出现的次数与语料库文档数的百分比，如果是int，则表示词出现的次数。如果参数中已经给定了vocabulary，则这个参数无效
# min_df：类似于max_df，不同之处在于如果某个词的document frequence小于min_df，则这个词不会被当作关键词
# use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了
# smooth_idf：idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
# sublinear_tf：默认为False，如果设为True，则替换tf为1 + log(tf)。
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
print(trn_term_doc.shape)
print(trn_term_doc.shape) # 提取特征的维度
test_term_doc = vec.transform(test[column])

y=(train["class"]-1).astype(int)
clf = LogisticRegression(C=4, dual=True)
clf.fit(trn_term_doc, y)
preds=clf.predict_proba(test_term_doc)


#生成提交结果
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('.data/results/00_lr_baseline.csv',index=None)
t2=time.time()
print("time use:",t2-t1)