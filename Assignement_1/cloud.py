# -*- coding: utf-8 -*-,
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd
import csv
import os
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt


train_data = pd.read_csv('train_set.csv', sep="\t")

#train_data = train_data[0:25]

counter = 0

contents = train_data.Content
categories = train_data.Category

bla = set(train_data['Category'])
blo = list(bla)
#print blo

#adding English stopwords
eng_stop_words = ENGLISH_STOP_WORDS
myStopWords = {'yes', 'just', "don't", 'didn'}
eng_stop_words = ENGLISH_STOP_WORDS.union(myStopWords)


for x in blo:
	count = 0
	temp = open("cloud.txt","w+")
	for line in contents:
		querywords = line.split()
		resultwords  = [word for word in querywords if word.lower() not in eng_stop_words]
		result = ' '.join(resultwords)
		if(categories[count] == x):
			temp.write(result)
		count+=1

	temp.close()
	d = path.dirname(__file__)

	# Read the whole text.
	text = open(path.join(d, 'cloud.txt')).read()

	# Generate a word cloud image
	wordcloud = WordCloud().generate(text)

	# Save the generated image with matplotlib:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.savefig('%s.png' % x)
	counter+=1

#	print("removing \n")
	os.remove("cloud.txt")