import random
import pickle
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print('Loading the datasets...')
"""Load given datasets"""

train_data = pd.read_csv("./data/Consumer_Complaints_train.csv")

"""Print top 5 records of train dataset"""

print(train_data.head())

print('Preprocessing the data....')

percent_missing = train_data.isnull().sum() * 100 / len(train_data)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
print(missing_value_df)


drop_col_train = missing_value_df[missing_value_df['percent_missing']>25]
print(drop_col_train)

drop_col_train= drop_col_train.T

train_data= train_data.drop( columns= drop_col_train.columns)

"""Extract Date, Month, and Year from the "Date Received" Column and create new fields for year, month, and day."""

train_data[["Year","Month", "Day"]] = train_data["Date received"].str.split("-", expand = True)


"""Convert dates from object type to datetime type"""

train_data['Date received'] = pd.to_datetime(train_data['Date received'])
train_data['Date sent to company'] = pd.to_datetime(train_data['Date sent to company'])

"""Calculate the number of days the complaint was with the company

create new field with help given logic
Like, Days held = Date sent to company - Date received
"""

train_data['Number of Days held']= train_data['Date sent to company']- train_data['Date received']

"""Convert "Days Held" to Int(above column)"""
train_data['Days held'] = train_data['Number of Days held'].map(lambda x: np.nan if pd.isnull(x) else x.days)

train_data1= train_data.drop(['Date received','Date sent to company','ZIP code', 'Complaint ID','Number of Days held', 'Company', 'State', 'Days held', "Year", 'Day', 'Month'], axis=1)

#"""Store data of the disputed consumer in the new data frame as "disputed_cons""""

train_data['disputed_cons'] = train_data['Consumer disputed?']
train_data.head()

"""Plot bar graph for the total no of disputes with the help of seaborn"""

# train_data['disputed_cons'].value_counts()

# sns.countplot(x='disputed_cons',data= train_data)
# plt.show()

# """Plot bar graph for the total no of disputes products-wise with help of seaborn"""

# train_data.groupby('Product')['disputed_cons'].value_counts()

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'Product',hue='disputed_cons', data= train_data)
# plt.show()

"""Plot bar graph for the total no of disputes with Top Issues by Highest Disputes , with help of seaborn"""

train_data.groupby('Issue')['disputed_cons'].value_counts().sort_values(ascending=False)

train_data['disputed_cons'].value_counts().iloc[:10].index

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'Issue',hue='disputed_cons', data= train_data ,order=train_data.Issue.value_counts().iloc[:5].index)
# plt.show()

# """Plot bar graph for the total no of disputes by State with Maximum Disputes"""

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'State',hue='disputed_cons', data= train_data ,order=train_data.State.value_counts().iloc[:5].index)
# plt.show()

# """Plot bar graph for the total no of disputes by Submitted Via diffrent source"""

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'Submitted via',hue='disputed_cons', data= train_data ,order=train_data['Submitted via'].value_counts().iloc[:5].index)
# plt.show()

# """Plot bar graph for the total no of disputes wherevCompany's Response to the Complaints"""

# train_data['Company response to consumer'].value_counts()

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'Company response to consumer',hue='disputed_cons', data= train_data ,order=train_data['Company response to consumer'].value_counts().iloc[:5].index)
# plt.show()

# """Plot bar graph for the total no of disputes Whether there are Disputes Instead of Timely Response"""

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'Timely response?',hue='disputed_cons', data= train_data ,order=train_data['Timely response?'].value_counts().iloc[:5].index)
# plt.show()

# """Plot bar graph for the total no of disputes over Year Wise Complaints"""

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'Year',hue='disputed_cons', data= train_data ,order=train_data['Year'].value_counts().iloc[:5].index)
# plt.show()

# """Plot bar graph for the top companies with highest complaints"""

# plt.figure(figsize=(20,5)) 
# sns.countplot(x= 'Company',hue='disputed_cons', data= train_data ,order=train_data['Company'].value_counts().iloc[:5].index)
# plt.show()

"""Change Consumer Disputed Column to 0 and 1(yes to 1, and no to 0)"""

le = LabelEncoder()
train_data1['Consumer disputed?']= le.fit_transform(train_data['Consumer disputed?'])
train_data1['Timely response?']= le.fit_transform(train_data['Timely response?'])
print(train_data1.head()["Consumer disputed?"])
print(train_data1.head()['Timely response?'])

words1 = []
classes = []
documents = []
punctuations = ['?','!',',','.',';',':', '@', '#','$', '&', '*', '(', ')']
i = 0
issues = train_data1['Issue'].tolist()
product = train_data1['Product'].tolist()
timlyresponse = train_data1['Timely response?'].tolist()
disputed = train_data1['Consumer disputed?'].tolist()
lemmatizer = WordNetLemmatizer()
print('Processing the data to build a model....')
while True:
    if i == train_data.shape[0]:
        break
    elif str(disputed[i]) == '0':
        tag = 'No'
    elif str(disputed[i]) == '1':
        tag = 'Yes'
    word_list1 = nltk.word_tokenize(issues[i])
    words1.extend(word_list1)
    documents.append((product[i], word_list1, timlyresponse[i], tag))
    if tag not in classes:
        classes.append(tag)
    i += 1
print(i)
print('Preparing the data for processing....')
words1 = [lemmatizer.lemmatize(word.lower()) for word in words1 if word not in punctuations]

words1 = sorted(set(words1))
classes = sorted(set(classes))

print('Dumping the data....')
pickle.dump(words1, open("./data/issues.pkl", 'wb'))
pickle.dump(classes, open("./data/classes.pkl", 'wb'))

training = []
output = [0] * len(classes)
t_d = []

print('Processing the data...')
for document in documents:
    bag1=[]
    word_patterns1 = document[1]
    word_patterns1 = [lemmatizer.lemmatize(word.lower()) for word in word_patterns1]
    for word in words1:
        bag1.append(1) if word in word_patterns1 else bag1.append(0)
    bag1.append(document[2])
    output_row = list(output)
    output_row[classes.index(document[3])] = 1
    training.append([bag1, output_row])

t_d.append([bag1, output_row])
print('Bag: ', bag1)
print('Output_row: ', output_row)
print('Training: ', t_d)

random.shuffle(training)
training = np.array(training)
t_d = np.array(t_d)
train_x = list(training[:,0])
t_x = list(t_d[:,0])
train_y = list(training[:,1])
t_y = list(t_d[:,1])

print('train_x: ', t_x)
print('train_y: ', t_y)

print('Building the model...')
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

print('Training the model....')
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
'''train_x = np.array(train_x)
train_y = np.array(train_y)'''

hist = model.fit(np.array(train_x), np.array(train_y), epochs=64, batch_size=500, verbose=1)
print('Saving the model....')
model.save("./data/comment_model.h5", hist)

print('Model has graduated and is now ready for the job :)\n')