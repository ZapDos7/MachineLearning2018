# -*- coding: utf-8 -*-,
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer#, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


#common for all - loading Datasets
globalDataNum = 100
train_data = pd.read_csv('train_set.csv', sep="\t")
test_data = pd.read_csv('test_set.csv', sep="\t")
train_data = train_data[0:globalDataNum]
test_data = test_data[0:globalDataNum]
categories = train_data.Category
ids = train_data.Id
compons = [2,3,5,7,10,15,20,30,40,50,60,70,80,90,100]
myData = train_data['Content'] + 5 * train_data['Title']
#for question #4 - initialization
metrics_all = [[0 for x in range(5)] for y in range(4)]

#adding English stopwords
eng_stop_words = ENGLISH_STOP_WORDS
myStopWords = {'yes', 'just', "don't", 'didn'}
eng_stop_words = ENGLISH_STOP_WORDS.union(myStopWords)

#
#set(categories)
le = preprocessing.LabelEncoder()
le.fit(categories)
y = le.transform(categories)
set(y)
set(le.inverse_transform(y))
count_vectorizer = CountVectorizer(stop_words=eng_stop_words)
#count_vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,min_df=2,stop_words=eng_stop_words,use_idf=True)
X = count_vectorizer.fit_transform(myData)
"""
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))
X_train_lsa = lsa.fit_transform(X)
X = np.array(X_train_lsa)
"""

"""
#SVM
#playing with parameters
print "Working with Support Vector Machine..."

Cs = [0.1, 1, 10, 100]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'kernel':('linear', 'rbf'), 'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=10)
grid_search.fit(X, y)
params = grid_search.best_params_ #execute it with best params
clf = svm.SVC(params)

clf = svm.SVC(kernel='linear', C='0.1', gamma='0.001')
accuracies = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
count = 0
#LSI time!

for c in compons:
	lsa = TruncatedSVD(n_components = c)
	X_new = lsa.fit_transform(X)
	predicted = cross_val_predict(clf, X, y, cv=10)
	clf.fit(X, y)
	accuracies[count] = metrics.accuracy_score(y, predicted)
	count+=1

maxAcc = np.amax(accuracies)#find max accuracy
ind = accuracies.index(maxAcc)#find its index
maxComps = compons[ind]#get the corresponding component num
#redo the process to keep the best output

lsa = TruncatedSVD(n_components = 30)
X_new = lsa.fit_transform(X)
predicted = cross_val_predict(clf, X, y, cv=10)

clf.fit(X, y)
predicted_categories = le.inverse_transform(predicted)

prec = metrics.precision_score(y, predicted, average='weighted')
rec = metrics.recall_score(y, predicted, average='weighted')
f1 = metrics.f1_score(y, predicted, average='weighted')
met = metrics.accuracy_score(y, predicted)
metrics_all[2][0] = "{0:.4f}%".format(prec * 100)
metrics_all[2][1] = "{0:.4f}%".format(rec * 100)
metrics_all[2][2] = "{0:.4f}%".format(f1 * 100)
metrics_all[2][3] = "{0:.4f}%".format(met * 100)

plt.plot(np.array(cvs),np.array(accuracies))
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.title('SVM')
plt.savefig('SVM.png')
#plt.show()
plt.close()
"""
#Random Forest
print "Working with Random Forest Classifier..."
model = RandomForestClassifier(random_state=0, min_samples_split=10, criterion="entropy", n_estimators=50)
model.fit(X, y)
#predicted = model.predict(X)
#predicted = cross_val_predict(model, X, y, cv=10)
#predicted_categories = le.inverse_transform(predicted)
accuracies = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
count = 0
#LSI time, again!
for c in compons:
	lsa = TruncatedSVD(n_components = c)
	X_new = lsa.fit_transform(X)
	predicted = cross_val_predict(model, X, y, cv=10)
	model.fit(X, y)
	accuracies[count] = metrics.accuracy_score(y, predicted)
	count+=1

maxAcc = np.amax(accuracies)#find max accuracy
ind = accuracies.index(maxAcc)#find its index
maxComps = compons[ind]#get the corresponding component num
#redo the process to keep the best output
lsa = TruncatedSVD(n_components = maxComps)
X_new = lsa.fit_transform(X)
predicted = cross_val_predict(model, X, y, cv=10)
model.fit(X, y)
predicted_categories = le.inverse_transform(predicted)

prec = metrics.precision_score(y, predicted, average='weighted')
rec = metrics.recall_score(y, predicted, average='weighted')
f1 = metrics.f1_score(y, predicted, average='weighted')
met = metrics.accuracy_score(y, predicted)

metrics_all[1][0] = "{0:.4f}%".format(prec * 100)
metrics_all[1][1] = "{0:.4f}%".format(rec * 100)
metrics_all[1][2] = "{0:.4f}%".format(f1 * 100)
metrics_all[1][3] = "{0:.4f}%".format(met * 100)
plt.plot(np.array(accuracies),np.array(accuracies))
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.title('Random Forests')
plt.savefig('RF.png')
#plt.show()


#Multinomial Naive Bayes
#we have to revert from LSI to no preparation of data due to the error discussed in Piazza:
#https://piazza.com/class/jf7ddbdagx94pf?cid=13
"""
set(train_data['Category'])
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
set(y)
set(le.inverse_transform(y))
count_vectorizer = CountVectorizer(stop_words=eng_stop_words)
X = count_vectorizer.fit_transform(myData)
X.shape
X.toarray()
"""
#begin working
print "Working with Multinomial Naive Bayes..."
model = MultinomialNB()
#train model
model.fit(X, y)
#predicted = model.predict(X)
predicted = cross_val_predict(model, X, y, cv=10)
predicted_categories = le.inverse_transform(predicted)
#print classification_report(y, predicted, target_names=list(le.classes_))
prec = metrics.precision_score(y, predicted, average='weighted')
rec = metrics.recall_score(y, predicted, average='weighted')
f1 = metrics.f1_score(y, predicted, average='weighted')
met = metrics.accuracy_score(y, predicted)

metrics_all[0][0] = "{0:.4f}%".format(prec * 100)
metrics_all[0][1] = "{0:.4f}%".format(rec * 100)
metrics_all[0][2] = "{0:.4f}%".format(f1 * 100)
metrics_all[0][3] = "{0:.4f}%".format(met * 100)

#Our implementation -- K nearest neightbours

metrics_all[3][0] = 0
metrics_all[3][1] = 0
metrics_all[3][2] = 0
metrics_all[3][3] = 0







#Our method for question #3 -- beat the benchmark


metrics_all[0][4] = 0
metrics_all[1][4] = 0
metrics_all[2][4] = 0
metrics_all[3][4] = 0



#extracting data to csv -- 4th question
with open('EvaluationMetric_10fold.csv', 'wb') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter='\t',) #tab separated values for my data
	spamwriter.writerow(['Statistic Measure','Naive Bayes','Random Forest','SVM','KNN','My Method']) #titles
	spamwriter.writerow(['Accuracy', [metrics_all[0][3]], [metrics_all[1][3]], [metrics_all[2][3]], [metrics_all[3][3]], [metrics_all[3][4]]])
	spamwriter.writerow(['Precision', [metrics_all[0][0]], [metrics_all[1][0]], [metrics_all[2][0]], [metrics_all[3][0]], [metrics_all[0][4]]])
	spamwriter.writerow(['Recall', [metrics_all[0][1]], [metrics_all[1][1]], [metrics_all[2][1]], [metrics_all[3][1]], [metrics_all[1][4]]])
	spamwriter.writerow(['F-Measure', [metrics_all[0][2]], [metrics_all[1][2]], [metrics_all[2][2]], [metrics_all[3][2]], [metrics_all[2][4]]])

with open('testSet_categories.csv', 'wb') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter='\t',) #tab separated values for my data
	spamwriter.writerow(['ID','Predicted_Category']) #titles
	#add the various text_ids and their predicted category respectively
	for x in range (0,globalDataNum):
		spamwriter.writerow([ids[x],predicted_categories[x]]) #gets the predicted categories from the last method, aka our own.

with open('testSet_categories_for_Kaggle.csv', 'wb') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',',) #tab separated values for my data
	spamwriter.writerow(['Id','Category']) #titles FIXED TO MAKE A FILE ACCEPTABLE BY KAGGLE
	#add the various text_ids and their predicted category respectively
	for x in range (0,globalDataNum):
		spamwriter.writerow([ids[x],predicted_categories[x]]) #gets the predicted categories from the last method, aka our own.