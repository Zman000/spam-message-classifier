import re

# Common English stop words
STOPWORDS = {
    "a", "about", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "both", "but", "by", "can", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own",
    "s", "same", "she", "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your"
}

def clean_text(text: str) -> str:
    """Standard text cleaning: lowercase, URL/currency placeholder, strip punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "urladdr", text)
    text = re.sub(r"[$£€]", " moneyprize ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text_standard(text: str) -> str:
    return clean_text(text)

def preprocess_text(text: str, mode: str = "standard") -> str:
    """Preprocess single text for training or inference."""
    if mode == "minimal":
        return str(text).strip()
    
    cleaned = clean_text(text)
    if mode == "with_stopwords":
        words = [w for w in cleaned.split() if w not in STOPWORDS]
        return " ".join(words)
    return cleaned

def preprocess_corpus(corpus, mode: str = "standard"):
    return [preprocess_text(msg, mode=mode) for msg in corpus]

if __name__ == "__main__":
    test_msg = "Congratulations! You won £1000 cash. Visit http://win.com now!"
    print("Standard:", clean_text(test_msg))
    print("Stopwords removed:", preprocess_text(test_msg, mode="with_stopwords"))
