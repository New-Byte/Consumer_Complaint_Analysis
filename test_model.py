import pickle
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pyttsx3 as spk
import os
from word2number import w2n
import pandas as pd
from sklearn.preprocessing import LabelEncoder

words = pickle.load(open("./data/issues.pkl", 'rb'))
classes = pickle.load(open("./data/classes.pkl", 'rb'))
model = load_model("./data/comment_model.h5")

def clean_up(issue):
	punctuations = ['?','!',',','.',';',':', '@', '#','$', '&', '*', '(', ')']
	sen_words = nltk.word_tokenize(issue)
	sen_words = [lemmatizer.lemmatize(word) for word in sen_words if word not in punctuations]
	return sen_words

def bag_of_words(issue, timelyresponse):
	sen_words = clean_up(issue)
	bag = [0] * len(words)
	for w in sen_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	bag.append(timelyresponse)
	return np.array(bag)

def predict_class(issue, timelyresponse):
	bow = bag_of_words(issue, timelyresponse)
	res = model.predict(np.array([bow]))[0]
	ERROR_THRESHOLD = 0.25
	result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
	result.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in result:
		return_list.append({'result': classes[r[0]], 'probability': str(r[1])})
	return return_list

def categorize(issue, timelyresponse):
	res = predict_class(issue, timelyresponse)
	return res[0]["result"], res[0]["probability"]

lemmatizer = WordNetLemmatizer()

test_data = pd.read_csv("./data/Consumer_Complaints_test_share.csv")

"""Print top 5 records of test dataset"""

print(test_data.head())

le1 = LabelEncoder()
test_data['Timely response?']= le1.fit_transform(test_data['Timely response?'])
print(test_data.head())
i = 0
predictions = []
probability = []
complaints = test_data["Complaint ID"].to_list()
issues = test_data["Issue"].to_list()
timelyresponses = test_data["Timely response?"].to_list()

try:
	while True:
		if i == test_data.shape[0]:
			break
		else:
			res, prob = categorize(issues[i],timelyresponses[i])
			predictions.append(res)
			probability.append(prob)
		i += 1
except Exception as e:
	print('Error: ' + e.message)
	print(i+2, issues[i])

"""Export Predictions to CSV"""
print("Saving the results....")
output = pd.DataFrame({'Complaint Id': complaints,'Customer Disputed': predictions, 'Probability': probability})
output.to_csv('./data/submission.csv', index=False)
print("Your submission was successfully saved!")