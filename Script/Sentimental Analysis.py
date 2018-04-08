import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

csv_file = 'Data.csv'
col = ['Text', 'Sentiment']
Dataset = pd.read_csv(csv_file, names=col, encoding='ISO-8859-1')
print (Dataset.head())

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True,strip_accents='ascii', stop_words=stopset)

Y = Dataset.Sentiment

X = vectorizer.fit_transform(Dataset.Text)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y,random_state=42)

clf = naive_bayes.MultinomialNB()
clf.fit(X,Y)



TestData = pd.read_csv('comments_.csv', encoding='ISO-8859-1')
TestData = TestData[['Comment', 'VideoID']]
print (TestData.head())

TDcomments = TestData[['Comment']].values
length_of_comments = len(TDcomments)
x=[]
def make_list(n):
    Senti = [0]*n
    return Senti

def prediction_of_comments(vec):
    vector = vectorizer.transform(vec)
    return clf.predict(vector)
    #print(clf.predict(vector))

def Start_prediction():
    for c in range(0,length_of_comments):
        array = prediction_of_comments(TDcomments[c])
        x.append(array[0])
        len_of_x=len(x)
        percentage = (len_of_x/length_of_comments)*100
        fp="{0:0.01f}".format(percentage)
        print ('Job Completed % ' ,fp)
def main():
    Senti = make_list(length_of_comments)        
    Start_prediction()
    #df_new=pd.DataFrame({'Sentiment':x})
    TestData['Sentiment']=pd.DataFrame({'Sentiment':x})
    print(TestData.head())
    TestData.to_csv('output.csv', encoding='ISO-8859-1')
main()    
