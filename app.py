import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag

# download nltk dependencies if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# preprocessing the text to remove stop words and tokenize the text
def preprocess_text(text):
    # tokenize the text
    tokens = tokenize(text)
    
    # lemmatize the tokens
    lemmatized_text = lemmatize(tokens)
    
    return lemmatized_text

def tokenize(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = []
    
    for token in tokens:
        # remove punctuation and stop words
        if token not in stopwords.words('english') and token.isalnum():
            filtered_tokens.append(token)
    return filtered_tokens

lemmatizer = WordNetLemmatizer()

def lemmatize(tokens):
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=wordnet_pos))
    return " ".join(lemmatized_tokens)  # Return after processing all tokens


def get_wordnet_pos(treebank_tag):
    # Convert treebank tags to wordnet tags
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
print(preprocess_text("The cats are running and mice are hiding"))