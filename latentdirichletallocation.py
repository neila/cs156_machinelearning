import io
import music21
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_features = 100
n_components = 10

stop_words = set(stopwords.words('english'))
#print(f"stop words: {stop_words}")
file1 = open("kamasutra.txt")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
for r in words:
    if not r in stop_words:
        appendFile = open('filteredtext.txt','a')
        appendFile.write(" "+r)
        appendFile.close()

vect = CountVectorizer(max_features=n_features)
filtertext = open("filteredtext.txt")
X = vect.fit_transform(filtertext)
lda = LatentDirichletAllocation(n_components=n_components, max_iter=10, learning_method='online', learning_offset=50., random_state=0)
document_topics = lda.fit_transform(X)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print("\nTopics in LDA model:")
feature_names = vect.get_feature_names()
print_top_words(lda, feature_names, 10)
