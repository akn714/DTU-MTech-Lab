# Text Preprocessing

**Preprocessing is task-dependent.**

### Steps:
1. Text Cleaning
2. Special Character & Punctuation Removal
3. Case Normalization
4. Contraction Expansion
5. Emoji & Emoticon Handling
6. Spelling Correction
7. Tokenization
8. Stop-word Removal
9. Stemming
10. Lemmatization
11. Numbers & Dates Handling
12. POS Tagging & NER

---

**Removing puncutations**<br>
- Q. Why should punctuation not always be removed?
- A. Because punctuation such as ! and ? can contain useful emotional or semantic information, especially in sentiment analysis.

**Case Normalization:**<br>
- Lowercasing reduces vocabulary, but can destroy useful capitalization information for things like Named Entity Recognition (NER).
- Because in NER, Apple could refer to company, while apple could refer to the fruit.

**Expanding Contractions** (don't -> do not, I'm -> I am)<br>
- Contraction expansion converts shortened forms into complete forms.
- It should be done before tokenization.

**Spelling Correction Methods:**<br>
- Edit-distance methods
- SymSpell, pyspellchecker
- Statistical/contextual correction
- Domain-specific dictionaries
- *Note*: Spelling correction reduces noise but can incorrectly modify domain-specific words (like OpenAI, TensorFlow).

**Tokenization**<br>
- Splitting text into smaller units called tokens.
- ***Word Tokenization***: Split into words.
```
    Before: I love NLP
    After: [I, love, NLP]
```
- ***Sentence Tokenization***: Split into sentences.
```
    Before: I love NLP. It is interesting.

    After:
    [
        "I love NLP.",
        "It is interesting."
    ]
```
- ***Subword Tokenization***: Used heavily by modern Transformer models.

**Stop Word Removal (Why remove them?)**<br>
- It can Reduce vocabulary
- It can Reduce dimensionality
- It can Reduce noise
- It can Make search/indexing more efficient
- Stop-word removal is task-dependent because some stop words, especially negations such as “not”, “no”, and “never”, can carry important semantic information.

**Stemming**<br>
- Porter Stemmer
- Snowball Stemmer
- *Note*: Search/indexing where speed is more important than linguistic accuracy.

**Lemmatization**<br>
- Slower than stemming
- Requires accurate POS tagging
- *Note*: Good when linguistic meaning matters

**POS Tagging**<br>
- POS Tagging helps in
    - Lemmatization
    - Information extraction
    - Question answering
    - Search
```
Barack Obama → proper nouns
was → verb
born → verb
Hawaii → proper noun
```

**Named Entity Recognition (NER)**<br>
- Barack Obama → PERSON
- Hawaii → GPE/Location

**POS v/s NER:**<br>
- POS asks: What grammatical role does this word play?
- NER asks: Does this text refer to a real-world entity? If yes, what type?

