# N-Grams

$$ P(w_n \mid w_{n-1}) =
\frac{\operatorname{Count}(w_{n-1}, w_n)}
{\operatorname{Count}(w_{n-1})}
$$

**In simple words**: How often does the second word appear after the first word?

Example:
- `data drives` occurs 6 times
- `data` occurs 10 times
- Therefore: P(drives∣data)=6/10=0.6

### Applications of N-Grams
- Predict the next word.
- Autocomplete
- Predictive text
- Speech recognition

### Spelling correction using n-grams
```
piece of cake → high probability
peace of cake → low probability


Misspelled word
      ↓
Generate possible words
      ↓
Use N-gram probability
      ↓
Choose best sentence
```

### Problems with n-grams
- **Data Sparsity**:
    - As N increases, possible combinations increase enormously.
    - So many valid combinations will never appear in training data.
- Example: If `"machine learning is powerful"` was never seen during training, a basic model may assign it `P = 0` (Probability = 0)

### Word sense disambugations
- Determines the correct meaning of an ambiguous word from context.