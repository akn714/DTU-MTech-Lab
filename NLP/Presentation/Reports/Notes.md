[source](https://chatgpt.com/c/6ac6853d-e7a0-83e8-b427-6398165e4a50)

#### Embedding Model
- gte-multilingual-base
- This research paper uses this Embedding Model for generating text embedding (as the feature extractor).
```
"I love this phone"
        ↓
gte-multilingual-base
        ↓
[0.21, 0.83, -0.12, ...]
        ↓
Logistic Regression
        ↓
Positive
```

#### 4 Datasets used
- SENTIMENT (3 classes)
    - Positive
    - Neutral
    - Negative   
- TAXI1500 (6 classes)
    - Recommendation
    - Faith
    - Description
    - Sin
    - Grace
    - Violence
- SCENARIO (18 Amazon virtual-assistant domains)
    - shopping
    - weather
    - music
- INTENT (60 very specific virtual-assistant intents)

#### The authors are essentially asking:
```
3 classes
   ↓
6 classes
   ↓
18 classes
   ↓
60 classes
```
> As the number of possible labels increases, do LLMs still work well?

#### 4 Different Approaches for Classification
```
                     SAME TEXT DATA
                           │
             ┌─────────────┼──────────────┐
             │             │              │
             ▼             ▼              ▼
        Zero-shot      Few-shot       Synthetic
           LLM           FastFit          Data
             │             │              │
             │             │              ▼
             │             │        Traditional /
             │             │        LLM classifier
             │             │
             └─────────────┴──────────────┐
                                          │
                                  Full-data models
                                          │
                                          ▼
                                  Compare Accuracy
```

#### Which Models did they use
- GPT-4
- Qwen2.5-7B
- Aya23-8B
- Aya-Expanse-8B

- Zero-shot LLM
- Few-shot FastFit
- Synthetic-data classifier
- Full-data supervised models

#### Approach 1: Zero shot classification
Approach 2: Few shot FastFit