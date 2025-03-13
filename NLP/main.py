import numpy as np
import nltk
import string
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the file
f = open('dt2.txt', 'r', errors='ignore')
dt = f.read()
f.close()

# Convert text to lowercase
dt = dt.lower()

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Tokenize sentences and words
s_token = nltk.sent_tokenize(dt)
w_token = nltk.word_tokenize(dt)

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Function to lemmatize words
def lim_token(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to normalize text (remove punctuation, tokenize, and lemmatize)
remove_punc_dict = str.maketrans('', '', string.punctuation)

def lim_normlz(text):
    return lim_token(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Greeting inputs and responses
greet_in = ('hi', 'hello', 'how are you?')
greet_out = ('Hi!', 'Hey!', 'Hello there!')

def greet(snt):
    for word in snt.split():
        if word.lower() in greet_in:
            return random.choice(greet_out)

# Function to generate bot response
def resp(usr_res):
    global s_token  

    dt1_res = ''
    s_token.append(usr_res)  # Add user input to tokenized sentences

    # Vectorize the text using TF-IDF
    TfidfVec = TfidfVectorizer(tokenizer=lim_normlz, stop_words='english')
    Tfidf = TfidfVec.fit_transform(s_token)

    # Compute cosine similarity
    vals = cosine_similarity(Tfidf[-1], Tfidf[:-1])  # Exclude user input from comparison
    idx = vals.argsort()[0][-1]  # Get the most relevant sentence index

    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]  # Highest similarity score

    if req_tfidf == 0:
        dt1_res = "I am sorry, I am unable to understand!"
    else:
        dt1_res = s_token[idx]  # Return the most relevant response

    s_token.pop()  # Remove user input from sentences
    return dt1_res

# Chatbot interaction loop
flag = True
print("Welcome to my bot!")

while flag:
    usr_res = input().lower()
    if usr_res == 'bye':
        flag = False
        print("Bye!")
    else:
        if greet(usr_res) is not None:
            print("Bot: " + greet(usr_res))
        else:
            print("Bot:", resp(usr_res))
