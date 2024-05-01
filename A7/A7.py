import pandas as pd
import numpy as nu
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text="The Greatest Gold Robbery took place on the night of 15 May 1855, when a shipment of gold to Paris was stolen from the guard's van of the rail service between London and Folkestone. There were four robbers: two employees of the rail company, a former employee and Edward Agar, a career criminal. They took wax impressions of the keys to the train safes and made copies. One of them ensured he was on guard duty when a shipment was taking place, and Agar hid in the guard's van."

from nltk.tokenize import word_tokenize
token= word_tokenize(text)
print("\nWord Tokenised:")
print(token)

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print("\nStop Words")
print(stop_words)

#text=re.sub('[a-zA-Z]',' ', text)
tokens=word_tokenize(text.lower())
filtered_text=[]
for w in tokens:
    if w not in stop_words:
        filtered_text.append(w)
print("\n")
print("Filtered Sentence:")
print(filtered_text)

from nltk.stem import PorterStemmer
ps=PorterStemmer()
#filtered_text1=["wait","waiting","waited","waits"]
for w in token:
    rootWords=ps.stem(w)
print("\n Stemming:")
print(rootWords)

from nltk.stem import WordNetLemmatizer
words=WordNetLemmatizer()
for w in token:
    print("Lemma for {} is {}".format(w,words.lemmatize(w)))

from nltk import pos_tag
pos=pos_tag(token)
print("POS Tagging")
print(pos)

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
                for word, val in idfDict.items():
                    if val > 0: idfDict[word] = math.log(N / float(val))
                    else: idfDict[word] = 0

    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

# Algorithm for Create representation of document by calculating TFIDF
# Step 1: Import the necessary libraries.
from sklearn.feature_extraction.text import TfidfVectorizer
# Step 2: Initialize the Documents.
documentA = 'Jupiter is the largest planet'
documentB = 'Mars is the fourth planet from the Sun'
# Step 3: Create BagofWords (BoW) for Document A and B. word tokenization
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')
# Step 4: Create Collection of Unique words from Document A and B.
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

# Step 5: Create a dictionary of words and their occurrence for each document in the corpus
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1 #How many times each word is repeated
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1
# Step 6: Compute the term frequency for each of our documents.
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
# Step 7: Compute the term Inverse Document Frequency.
print('----------------Term Frequency----------------------')
df = pd.DataFrame([tfA, tfB])
print(df)
# Step 8: Compute the term TF/IDF for all words.
idfs = computeIDF([numOfWordsA, numOfWordsB])
print('----------------Inverse Document Frequency----------------------')
print(idfs)
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
print('------------------- TF-IDF--------------------------------------')
df = pd.DataFrame([tfidfA, tfidfB])
print(df)
