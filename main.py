import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
import imblearn
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

def txt_to_csv(file_input,output):

    f=open(file_input,"r", encoding="UTF-8")
    outputfile = open(output, "w", encoding="UTF-8")
    outputfile.truncate(0)
    outputfile.close()
    lines=f.readlines()
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    all_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for x in lines:
        item=x.split(maxsplit=2)
        if len(item) < 3:
            continue
        id = item[0]
        sentiment = item[1]
        text = item[2]
        text = str.lower(text)
        text = re.sub(r'http\S+', '', text)
        text = remove_stopwords(text, stopwords=all_stopwords)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        word_list = nltk.word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
        if text.endswith('\n'):
            text = text.strip('\n')
        if text != "":
            with open(output, 'a') as fi:
                print(str(id)+","+sentiment+',"'+text+'"', file=fi)
    f.close()


if False:
    txt_to_csv("SemEval2017-task4-dev.subtask-A.english.INPUT.txt", "./train.csv")

tweets_train = pd.read_csv("train.csv", header=None)

X, y = tweets_train.iloc[:, [-1]].values.flatten(), tweets_train.iloc[:, [-2]].values.flatten()


pipeline = Pipeline([
   ('vectorizer', CountVectorizer(analyzer="word",max_features=10000)),
   ('model', MultinomialNB())
])

scores = cross_val_score(pipeline, X, y, cv=7)
print("Accuracy is :%0.3f \n with a standard deviation of : %0.3f" % (scores.mean(), scores.std()))
print()
scores = cross_val_score(pipeline, X, y, cv=7, scoring='f1_macro')
print("F1-Score Average is :%0.3f \n with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
print()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)

ros = imblearn.over_sampling.RandomOverSampler()

X_train=pd.DataFrame(np.stack((X_train, y_train), axis=1),columns=['text','label'])
y_train= X_train.iloc[:, [1]]
tweets_train, tweets_label = ros.fit_resample(X=X_train,y=y_train )
X_train = tweets_train.iloc[:, [0]].values.flatten()
y_train= tweets_train.iloc[:, [1]].values.flatten()

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)


nb_acc=metrics.accuracy_score(y_test, y_pred)
print("Accuracy After Random Oversampling : "+str(nb_acc))
print()

nb_f1=metrics.f1_score(y_test,y_pred,average='macro')
print("Average F1-Score After Random Oversampling : "+str(nb_f1))
print()

nb_f1=metrics.f1_score(y_test,y_pred,average=None)
print("F1-Score of each class After Random Oversampling : "+str(nb_f1))
