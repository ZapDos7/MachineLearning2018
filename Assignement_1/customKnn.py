from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd
import math
import csv
import random
import operator
import numpy as np

def euclideanDistance(array1,array2,l):
	distance = 0
	for x in range(l):
		distance += pow((array1[x] - array2[x]),2)
	return math.sqrt(distance)

def customKnn():
	train_data = pd.read_csv('train_set.csv', sep="\t")
	test_data = pd.read_csv('train_set.csv', sep="\t")
	
	#Keep only a subset of the data (25 data points) for testing purposes
	globalDataNum = 1000
	train_data = train_data[0:globalDataNum]
	test_data = test_data[0:globalDataNum]
	
	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	
	y = le.transform(train_data["Category"])
	set(y)
	
	#vriskoume to plithos category, tha to xrhsimopoihsoume meta ston algorithmo
	catCount = len(set(y))
	
	count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
	X = count_vectorizer.fit_transform(train_data['Content']+5*train_data['Title'])
	XTest = count_vectorizer.fit_transform(train_data['Content']+5*train_data['Title'])
	
	catNum = le.transform(train_data["Category"])
#	print X.toarray()
	
	k = 6
	#o arithmos samples pou tha valoume sto training, einai k*2*sunoloKathgoriwn, vlepe parakatw
	testSample = k*2
	
	training = []
	xArray = X.toarray()
	xTestArray = XTest.toarray()
#	print XTest.toarray()
	
	for x in range(0,catCount):
		counterK = 0

	#o arithmos samples pou tha valoume sto training set, einai k*2*sunoloKathgoriwn  ^^^
	#se auto to shmeio pairnoume iso arithmo samples gia kathe kathgoria
	#ftiaxnoume to training mas

    ######## Beat the Benchmark ###############
		while(counterK < testSample):
			samples = random.sample(range(1,globalDataNum),1)
	
			if(catNum[samples] == x):
				training.append((xArray[samples],catNum[samples]))
				counterK +=1
	###########################################
	distances = []
	
    #catNum = le.transform(train_data["Category"])
	a = set(train_data['Category'])
	b = test_data['Id']
	print b
	a = list(a)
	a.sort()
#	print a

	for y in range(0,20):
		l = len(xTestArray[y])-1
	#o arithmos samples pou tha valoume sto training, einai k*2*sunoloKathgoriwn ^^^
		for x in range(0,catCount*testSample):
	#upologismos olwn twn distances
			distance = euclideanDistance(xTestArray[y],training[x][0][0],l)
			distances.append((training[x][1][0],distance))
	
	#sort twn votes
		distances.sort(key=operator.itemgetter(1))
		votes = {}
		for x in range (0,k):
			d = distances[x][0]
	
			if d in votes:
				votes[d] += 1
			else:
				votes[d] = 1
	#sort twn votes mazi me distances
		sV = sorted(votes.iteritems(),key=operator.itemgetter(1),reverse=True)
	
		print sV
	#votes
	#	print sV[0][1]
	
	#no1 category
		print ("predicted for data no ",b[y]," category",a[sV[0][0]])
#		print "category %d" %sV[0][0]
#		if(catNum[b[y]] == sV[0][0]):
#			print "predicted"

customKnn()