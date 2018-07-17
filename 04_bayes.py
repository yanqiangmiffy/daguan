import pandas as pd
from util import load_data
from sklearn.naive_bayes import MultinomialNB

all_data=load_data()

X_train,y_train = all_data['X_train'],all_data['y_train']


naive_bayes=MultinomialNB()
naive_bayes.fit(X_train,y_train)

predictions = naive_bayes.predict(all_data['X_test'])

test_pred=pd.DataFrame(predictions)
test_pred.columns = ["class"]
test_pred['id']=all_data['test_id']

sub_data = test_pred[['id', 'class']]
sub_data.to_csv('data/new_data/04_sub_bayes.csv', index=False)
print("结果以保存：data/new_data/04_sub_bayes.csv")
