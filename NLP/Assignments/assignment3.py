"""
Using N-Grams to solve word spelling mistakes in a given text.

Spelling Errors:
1. Non-word errors: These are words that do not exist in the dictionary (e.g., "speling" instead of "spelling").
2. Real-word errors: These are words that exist in the dictionary but are used incorrectly in the context (e.g., "A peace of cake" instead of "A piece of cake").
"""

from collections import Counter
import math
import re


WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


class NGramSpellCorrector:
	"""Correct non-word and real-word spelling errors with n-gram scores."""

	def __init__(self, corpus, max_edit_distance=2, smoothing=0.1):
		if not corpus:
			raise ValueError("corpus must contain at least one sentence")
		if max_edit_distance < 1:
			raise ValueError("max_edit_distance must be at least 1")
		if smoothing <= 0:
			raise ValueError("smoothing must be greater than zero")

		self.max_edit_distance = max_edit_distance # this simply means that we will only consider candidate corrections that are within a certain edit distance from the misspelled word-
		self.smoothing = smoothing # this simply means that we add a small constant to the counts of n-grams to avoid zero probabilities for unseen n-grams
		self.unigrams = Counter() # counter for unigrams, which are single words in the training corpus
		self.bigrams = Counter() # counter for bigrams, which are pairs of consecutive words in the training corpus
		self.trigrams = Counter() # counter for trigrams, which are triplets of consecutive words in the training corpus

		for sentence in corpus:
			tokens = self._tokenize(sentence)
			padded_tokens = ["<s>", "<s>"] + tokens + ["</s>"] # we add two start tokens and one end token to the list of tokens to account for the beginning and end of the sentence
			self.unigrams.update(tokens)
			self.bigrams.update(zip(padded_tokens, padded_tokens[1:]))
			# eg. for the sentence "this is a simple sentence", the bigrams would be ("<s>", "<s>"), ("<s>", "this"), ("this", "is"), ("is", "a"), ("a", "simple"), ("simple", "sentence"), ("sentence", "</s>")
			# eg. padded_tokens = ["<s>", "<s>", "this", "is", "a", "simple", "sentence", "</s>"], padded_tokens[1:] = ["<s>", "this", "is", "a", "simple", "sentence", "</s>"], zip(padded_tokens, padded_tokens[1:]) = [("<s>", "<s>"), ("<s>", "this"), ("this", "is"), ("is", "a"), ("a", "simple"), ("simple", "sentence"), ("sentence", "</s>")]
			self.trigrams.update(zip(padded_tokens, padded_tokens[1:], padded_tokens[2:]))

		self.vocabulary = set(self.unigrams)
		self.total_words = sum(self.unigrams.values())

	@staticmethod
	def _tokenize(text):
		return WORD_PATTERN.findall(text.lower())

	@staticmethod
	def _edit_distance(first, second):
		previous_row = list(range(len(second) + 1)) # previous_row = [0, 1, 2, 3, ..., len(second)]
		for first_index, first_character in enumerate(first, start=1): # enumerate(first, start=1) = [(1, first[0]), (2, first[1]), (3, first[2]), ..., (len(first), first[len(first)-1])]
			current_row = [first_index] # current_row = [1], [2], [3], ..., [len(first)]
			for second_index, second_character in enumerate(second, start=1):
				"""
				# Working using raw examples
				Computes the minimum edit distance between the first and second strings.
				For example, if first = "kitten" and second = "sitting", the edit distance is 3 (substitute 'k' with 's', substitute 'e' with 'i', and insert 'g' at the end).
				"""
				insertion = current_row[second_index - 1] + 1
				deletion = previous_row[second_index] + 1
				substitution = previous_row[second_index - 1] + (first_character != second_character)
				current_row.append(min(insertion, deletion, substitution))
			previous_row = current_row
		return previous_row[-1]

	def _candidates(self, word):
		candidates = [
			(candidate, self._edit_distance(word, candidate))
			for candidate in self.vocabulary
			if abs(len(candidate) - len(word)) <= self.max_edit_distance
		]
		return [candidate for candidate, distance in candidates if distance <= self.max_edit_distance]

	def _ngram_score(self, previous_word, word, next_word):
		vocabulary_size = len(self.vocabulary) + 1
		unigram_probability = (self.unigrams[word] + self.smoothing) / (
			self.total_words + self.smoothing * vocabulary_size
		)
		bigram_probability = (self.bigrams[(previous_word, word)] + self.smoothing) / (
			self.unigrams[previous_word] + self.smoothing * vocabulary_size
		)
		trigram_probability = (self.trigrams[(previous_word, word, next_word)] + self.smoothing) / (
			self.bigrams[(previous_word, word)] + self.smoothing * vocabulary_size
		)
		return math.log(unigram_probability) + math.log(bigram_probability) + math.log(trigram_probability)

	def _correct_tokens(self, tokens, only_non_words):
		corrected = tokens.copy()
		padded_tokens = ["<s>"] + corrected + ["</s>"]
		for index, word in enumerate(tokens):
			is_non_word = word not in self.vocabulary
			if only_non_words and not is_non_word:
				continue
			if not only_non_words and is_non_word:
				continue

			candidates = self._candidates(word)
			if not candidates:
				continue
			previous_word = padded_tokens[index]
			next_word = padded_tokens[index + 2]
			best_candidate = max(
				candidates,
				key=lambda candidate: self._ngram_score(previous_word, candidate, next_word),
			)
			if self._ngram_score(previous_word, best_candidate, next_word) > self._ngram_score(
				previous_word, word, next_word
			):
				corrected[index] = best_candidate
				padded_tokens[index + 1] = best_candidate
		return corrected

	def correct_non_word_errors(self, text):
		"""Correct tokens absent from the training vocabulary."""
		tokens = self._tokenize(text)
		return " ".join(self._correct_tokens(tokens, only_non_words=True))

	def correct_real_word_errors(self, text):
		"""Correct known words that have a better context-compatible candidate."""
		tokens = self._tokenize(text)
		return " ".join(self._correct_tokens(tokens, only_non_words=False))

	def correct_text(self, text):
		"""Correct non-word errors first, followed by real-word errors."""
		tokens = self._tokenize(text)
		tokens = self._correct_tokens(tokens, only_non_words=True)
		return " ".join(self._correct_tokens(tokens, only_non_words=False))


if __name__ == "__main__":
	training_corpus = [
		"this is a simple sentence",
		"the student wrote a simple sentence",
		"i had a piece of cake",
		"the piece of cake was delicious",
		"she ate a piece of cake",
	]
	corrector = NGramSpellCorrector(training_corpus)

	print(corrector.correct_non_word_errors("This is a smple sentence"))
	print(corrector.correct_real_word_errors("I had a peace of cake"))
	print(corrector.correct_text("I had a peace of cake and smple tea"))

