from bs4 import BeautifulSoup
import re
import string
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stop_words.remove("not")
stop_words.remove("no")
stop_words.remove("nor")
def remove_html(text):
    soup = BeautifulSoup(text,"html.parser")
    return soup.get_text()

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+','',text)

def remove_punctuation(text):
    clean_text = ""

    for ch in text:
        if ch not in string.punctuation:
            clean_text += ch

    return clean_text

def remove_stopwords(text):
    words = text.split()
    
    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)

def preprocess_text(text):
    text = text.lower()
    text = remove_html(text)
    text = remove_url(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)

    return text



