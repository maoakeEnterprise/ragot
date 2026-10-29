"""English stop words, downloaded from NLTK on first import."""
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
