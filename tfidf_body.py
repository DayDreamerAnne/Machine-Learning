from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
import nltk

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def tfidf(tokens_resume,tokens_jobposting):
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=200)
    tf_resume = tfidf.fit_transform(tokens_resume)  # calculate resume tfidf
    tf_jd = tfidf.fit_transform(tokens_jobposting)  # calculate job posting tfidf
    return tf_resume,tf_jd