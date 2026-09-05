"""Assignment 2: Data preprocessing pipeline in NLP."""

import html
import re

from nltk.stem import PorterStemmer, WordNetLemmatizer


STOP_WORDS = {
	"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
	"has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
	"to", "was", "were", "will", "with", "this", "these", "those", "or",
}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
	"""Clean text and return tokens, stems, and lemmas.

	The function removes HTML, URLs, punctuation, numbers, and stop words,
	then applies stemming and lemmatization to the remaining tokens.
	"""
	if not isinstance(text, str):
		raise TypeError("text must be a string")

	cleaned_text = html.unescape(text) # this will convert HTML entities to their corresponding characters (eg. &amp; to &)
	cleaned_text = re.sub(r"<[^>]*>", " ", cleaned_text) # remove HTML tags
	cleaned_text = re.sub(r"https?://\S+|www\.\S+", " ", cleaned_text) # remove URLs
	cleaned_text = cleaned_text.lower() # convert to lowercase
	cleaned_text = re.sub(r"[^a-z\s]", " ", cleaned_text) # remove punctuation and numbers
	cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip() # remove extra whitespace

    # Tokenization
	tokens = [token for token in cleaned_text.split() if token not in STOP_WORDS]
    
    # Stemming
	stems = [stemmer.stem(token) for token in tokens]

    # Lemmatization
	try:
		lemmas = [lemmatizer.lemmatize(token) for token in tokens]
	except LookupError:
		# WordNet is an optional NLTK corpus; keep the pipeline usable without it.
		lemmas = tokens.copy()

	return {
		"cleaned_text": cleaned_text,
		"tokens": tokens,
		"stems": stems,
		"lemmas": lemmas,
	}


if __name__ == "__main__":
	sample_text = (
		"<p>Students are learning NLP preprocessing!</p> "
		"Visit https://example.com for more examples."
	)
	result = preprocess_text(sample_text)
	print("Tokens:", result["tokens"])
	print("Stems:", result["stems"])
	print("Lemmas:", result["lemmas"])

