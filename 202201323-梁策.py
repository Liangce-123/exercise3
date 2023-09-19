import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer
import string

from nltk.corpus import gutenberg
moby_dick = gutenberg.raw('melville-moby_dick.txt')

def tokenize_text(text):
    return text.split()

def filter_tokens(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]

def tag_tokens(tokens):
    tagged_tokens = []
    for token in tokens:
        tagged_tokens.append((token, nltk.pos_tag([token])[0][1]))
    return tagged_tokens

def analyze_pos(tagged_tokens):
    pos_freq = FreqDist(tag for word, tag in tagged_tokens)
    return pos_freq.most_common(5)

def lemmatize_tokens(tagged_tokens):
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        pos_dict = {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}
        return pos_dict.get(treebank_tag[0], 'n')

    lemmatized_tokens = []
    for token, pos in tagged_tokens[:20]:
        lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=get_wordnet_pos(pos)))
    return lemmatized_tokens

tokens = tokenize_text(moby_dick)
filtered_tokens = filter_tokens(tokens)
tagged_tokens = tag_tokens(filtered_tokens)
common_pos = analyze_pos(tagged_tokens)
lemmatized_tokens = lemmatize_tokens(tagged_tokens)

print("Most common parts of speech:")
for pos, count in common_pos:
    print(f"{pos}: {count}")

print("Lemmatized tokens:")
print(lemmatized_tokens)