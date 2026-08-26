### Levels of Linguistic Analysis
```
Phonology (sound/pronunciation)
   ↓
Morphology (structure of words, unhappy -> un + happy)
   ↓
Syntax (grammer/sentence structure)
   ↓
Semantics (literl meaning)
   ↓
Discourse (language units larger than a single sentence, such as paragraphs, conversations, or entire documents)
   ↓
Pragmatics (the study of how context contributes to the meaning)
```

### NLP Pipeline
```
    Tokenization
          ↓
  Stop-word Removal
          ↓
Stemming/Lemmatization
          ↓
     POS Tagging
          ↓
       Parsing
          ↓
         NER
```

### Approaches to NLP

| Era          | Approach         | Examples                |
| ------------ | ---------------- | ----------------------- |
| 1950s–80s    | Rule-based       | Hand-written rules      |
| 1990s–2000s  | Statistical      | N-grams, HMM, CRF       |
| 2000s–10s    | Machine Learning | Naive Bayes, SVM        |
| 2017–Present | Deep Learning    | RNN, LSTM, Transformers |


### NLP Libraries
- NLTK → learning/prototyping
- spaCy → fast production NLP
- Hugging Face Transformers → pretrained Transformer models
- Gensim → topic modelling/word embeddings
- Stanford CoreNLP → Java NLP toolkit
- TextBlob → simple NLP API

### 2 Minutes Revision
```
NLP:
- AI technique that allows computers to process human language.

Why NLP?
- Huge text + human/machine language gap + manual processing doesn't scale + real-time applications.

Applications:
- Translation, chatbot, search, sentiment, summarization, speech recognition, spam detection.

Main difficulties:
- Ambiguity, context, slang, sarcasm, synonyms, spelling errors, common sense.

Evolution:
- Rule-based → Statistical → ML → Deep Learning/Transformers.

Linguistic levels:
- Phonology → Morphology → Syntax → Semantics → Discourse → Pragmatics.

Pipeline:
- Tokenize → Stopwords → Stem/Lemmatize → POS → Parse → NER.
```