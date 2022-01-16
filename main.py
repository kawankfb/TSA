import re
import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import sklearn.metrics as metrics
import imblearn
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

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
        text = ""
        for i in range(2,len(item)):
            if len(item[i]) > 2:
                text = text+item[i]
        text = str.lower(text)
        text = re.sub(r'http\S+', '', text)
        text = remove_stopwords(text, stopwords=all_stopwords)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        word_list = nltk.word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
        # text = text.translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits))
        if text.endswith('\n'):
            text = text.strip('\n')
        if text != "":
            with open(output, 'a') as fi:
                print(str(id)+","+sentiment+',"'+text+'"', file=fi)
    f.close()


# txt_to_csv("twitter-2016train-A.txt", "./train.csv")
#txt_to_csv("SemEval2017-task4-dev.subtask-A.english.INPUT.txt", "./train.csv")
#txt_to_csv("SemEval2017-task4-test.subtask-A.english.txt", "./test.csv")
tweets_train = pd.read_csv("train.csv", header=None)
tweets_test = pd.read_csv("test.csv", header=None)

# rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority')

rus = imblearn.over_sampling.RandomOverSampler()

# tweets_train, y_resampled = rus.fit_resample(X=tweets_train,y=tweets_train.iloc[:, [-2]])

X, y= tweets_train.iloc[:, [-1]].values.flatten(), tweets_train.iloc[:, [-2]].values.flatten()

# X_train, y_train = tweets_test.iloc[:, [-1]].values.flatten(), tweets_test.iloc[:, [-2]].values.flatten()
# X_test, y_test = tweets_test.iloc[:, [-1]].values.flatten(), tweets_test.iloc[:, [-2]].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
plt.hist(y_train)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.show()

plt.hist(y_test)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
plt.show()

cnt = CountVectorizer(analyzer="word")

pipeline = Pipeline([
   ('vectorizer',cnt),
   ('model',MultinomialNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
scores=cross_validate(pipeline,X,y,scoring='accuracy')
print(scores)
nb_acc=metrics.accuracy_score(y_test,y_pred)
print("Accuracy Using Nominal Naiive Bayes : "+str(nb_acc))

nb_acc=metrics.f1_score(y_test,y_pred,average=None)
print("F1 Using Nominal Naiive Bayes : "+str(nb_acc))

#
# text_clf = Pipeline([
#     ('vectorizer', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', RandomForestClassifier()),
# ])
# text_clf.fit(X_train, X_label)
# y_pred = text_clf.predict(y_train)
# nb_acc=metrics.accuracy_score(y_test,y_pred)
# print("Accuracy Using GBC : "+str(nb_acc))
#
# nb_acc=metrics.f1_score(y_test,y_pred,average=None)
# print("F1 Using GBC: "+str(nb_acc))
#