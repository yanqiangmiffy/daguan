import pandas as pd
import numpy as np
from util import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 加载数据集
train_data_file = 'data/new_data/train_set.csv'
test_data_file = 'data/new_data/test_set.csv'
all_data = load_data(train_data_file, test_data_file)
X_train, X_test, y_train, y_test = train_test_split(all_data['X_train'], all_data['y_train'], test_size=0.1,
                                                    random_state=1)
print(y_train.shape, y_test.shape)

# 训练
svm_clf =clf = LogisticRegression(C=4, dual=True,n_jobs=-1)
svm_clf.fit(X_train, y_train)

# 预测与评估
predicted = svm_clf.predict(X_test)
print("svm prediction accuracy:{:4.4f}".format(np.mean(predicted == y_test)))

# 预测结果
print("正在预测结果")
preds = svm_clf.predict(all_data['X_test'])  # 生成提交结果
test_pred = pd.DataFrame(preds)
test_pred.columns = ["class"]
test_pred["class"] = (test_pred["class"]).astype(int)
test_pred["id"] = list(all_data["test_id"])
test_pred[["id", "class"]].to_csv('data/results/08_sgd.csv', index=None)
