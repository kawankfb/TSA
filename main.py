import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
from nltk.stem import WordNetLemmatizer


def txt_to_csv(file_input,output):

    f=open(file_input,"r")
    lines=f.readlines()
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    for x in lines:
        item=x.split(maxsplit=2)
        if len(item)<3:
            continue
        id = item[0]
        sentiment = item[1]
        text=""
        for i in range(2,len(item)):
            text=text+item[i]
        text=text.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))
        text = text.replace('"', ' ').replace(',', ' ')
        text = lemmatizer.lemmatize(text)
        if text.endswith('\n'):
            text = text.strip('\n')
        with open(output, 'a') as fi:
            print(str(id)+","+sentiment+',"'+text+'"', file=fi)
    f.close()

    def clean_texts(dataframe):
        nltk.download('stopwords')
        all_stopwords = stopwords.words('english')
        text = ""
        text = remove_stopwords(text, stopwords=all_stopwords)
        text = str.lower(text)


#txt_to_csv("twitter-2016train-A.txt","./train.csv")
#txt_to_csv("twitter-2016test-A.txt","./test.csv")
tweets_train=pd.read_csv("train.csv")
tweets_test=pd.read_csv("test.csv")
tweets_train.head()
X_train, X_test = tweets_train.iloc[:, [-1]].values.flatten(), tweets_train.iloc[:, [-2]].values.flatten()
y_train, y_test =  tweets_test.iloc[:, [-1]].values.flatten(), tweets_test.iloc[:, [-2]].values.flatten()

cnt = CountVectorizer(analyzer = 'char',ngram_range=(2,2))

pipeline = Pipeline([
   ('vectorizer',cnt),
   ('model',MultinomialNB())
])

pipeline.fit(X_train,X_test)

y_pred = pipeline.predict(y_train)

nb_acc=metrics.accuracy_score(y_test,y_pred)
print("Accuracy Using Nominal Naiive Bayes : "+str(nb_acc))

nb_acc=metrics.f1_score(y_test,y_pred,average=None)
print("F1 Using Nominal Naiive Bayes : "+str(nb_acc))

