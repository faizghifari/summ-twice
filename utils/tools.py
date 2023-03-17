import nltk

nltk.download("punkt")
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

def get_score(text):
    """
    Remove stopwords from text, and compute score based on number of non-stopwords in text
    
    Args:
    text: A string representing the text to be scored.

    Returns:
    A integer value representing the text's score.
    """
    text = ' '.join([word for word in text.split() if word not in stopwords])
    score = len(text.split())
    return score

def get_max_len_query(dataset):
    max_len_q = 3
    for data in dataset:
        for q in data['queries']:
        if len(q.split()) > max_len_q:
            max_len_q = len(q.split())
    return max_len_q

def clean_data(text: str) -> str:
    """
    Cleans the text by removing stopwords, and replacing special markers with whitespace.

    Args:
    text: A string representing the text to be cleaned.

    Returns:
    A string representing the cleaned text.
    """
    text = text.replace('{ disfmarker }', '')
    text = text.replace('[ laughter . ]', ' ')
    text = text.replace('[ inaudible . ] ', '')
    text = ' '.join([w for w in text.split()])
    return text
