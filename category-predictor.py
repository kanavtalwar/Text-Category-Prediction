from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import pickle

category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos', 
        'rec.sport.hockey': 'Hockey', 'sci.electronics': 'Electronics', 
        'sci.med': 'Medicine'}
stop_words = stopwords.words('english')


training_data = fetch_20newsgroups(subset='train',categories=category_map.keys(), shuffle=True, random_state=5)
data = 'data'
training_data[data] = [w for w in training_data[data] if w not in stop_words]
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data.data)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

classifer = SklearnClassifier(MultinomialNB)
classifier = MultinomialNB().fit(train_tfidf, training_data.target)

testing_data = fetch_20newsgroups(subset='test',categories=category_map.keys(), shuffle=True, random_state=5)
test_tc = count_vectorizer.transform(testing_data.data)
test_tfidf = tfidf.transform(test_tc)
y_pred = classifier.predict(test_tfidf)
print(classification_report(testing_data.target, y_pred))


input_string = raw_input("enter the string ")
input_string1 = [input_string]

input_tc = count_vectorizer.transform(input_string1)

input_tfidf = tfidf.transform(input_tc)
predictions = classifier.predict(input_tfidf)

for category in predictions:
    print(category_map[training_data.target_names[category]])


"""with open('KTcategory-predictor.pkl', 'wb') as file:
    pickle.dump(classifier, file)

with open('KTtfidf.pkl','wb') as file1:
    pickle.dump(tfidf, file1)
    
with open('KTcv.pkl','wb') as file2:
    pickle.dump(train_tc,file2)
 """