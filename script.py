import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("Data/comments_data.csv")
columns = ["Comment", "target"]
comments = df["Comment"].values
labels = []
count_vect = CountVectorizer()
encoded_comments = []
targets = ['ok', 'no', 'maybe']
for i in df["target"].values:
    labels.append(targets.index(i))

for comment in comments:
    encoded_comment = comment.decode('unicode_escape').encode('utf-8')
    encoded_comments.append(encoded_comment)

X_train_counts = count_vect.fit_transform(encoded_comments)
# print X_train_counts.shape

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
# print X_train_tf.shape


clf = MultinomialNB().fit(X_train_tf, labels)

docs_new = ['lunatics','God is love', 'OpenGL on the GPU is fast', 'stupid asshole', 'You are not smart']
X_new_counts = count_vect.transform(docs_new)
# print X_new_counts.shape


tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_new_counts)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# print X_new_tfidf.shape

# print(clf)
print(labels)
predicted = clf.predict(X_new_tfidf)
print predicted
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, targets[category]))