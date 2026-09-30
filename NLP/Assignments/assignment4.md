# Measuring Semantic Similarity with WordNet: Path-Based vs Information Content

How similar are *car* and *automobile*? What about *dog* and *cat*, or *dog* and *car*? Humans answer this instantly, but a computer needs a formal measure. **WordNet**, a lexical database that organizes English words into a hierarchy, lets us compute such a score. In this post we look at two families of WordNet-based measures and how to use them in Python with NLTK.

---

## 1. WordNet in a nutshell

- Words are grouped into **synsets** (sets of synonyms), each representing one *sense*, e.g. `dog.n.01` is "the domestic dog".
- Synsets are linked by **hypernym** ("is-a") relations, forming a tree-like taxonomy: `dog → canine → carnivore → mammal → animal → ... → entity`.
- The **Lowest Common Subsumer (LCS)** of two synsets is their most specific shared ancestor. For *dog* and *cat* it is `carnivore`.

The closer two concepts are in this hierarchy, and the more specific their shared ancestor, the more similar they are.

---

## 2. Setup

```bash
pip install nltk
```

```python
import nltk
from nltk.corpus import wordnet as wn, wordnet_ic

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("wordnet_ic")   # needed only for IC-based measures
```

---

## 3. Category 1: Path-based measures

These use only the **structure of the taxonomy**, meaning distances and depths.

| Measure | Idea |
|---|---|
| **Path Similarity** | `1 / (shortest_path_length + 1)` |
| **Wu-Palmer** | `2 * depth(LCS) / (depth(s1) + depth(s2))` |
| **Leacock-Chodorow** | `-log(path_length / (2 * max_depth))` |

**Wu-Palmer** is a nice choice because it accounts for depth: two nodes that share a *deep* (specific) ancestor are considered more similar than two nodes that only share a very generic one.

```python
dog = wn.synset("dog.n.01")
cat = wn.synset("cat.n.01")

print(dog.wup_similarity(cat))        # ~0.857
print(dog.path_similarity(cat))       # 0.2
print(dog.lowest_common_hypernyms(cat))   # [Synset('carnivore.n.01')]
print(dog.shortest_path_distance(cat))    # 4
```

Scores lie in the range `0 to 1`. A score of `1` means the synsets are identical.

---

## 4. Category 2: Information Content (IC) based measures

Path-based measures assume every edge in the tree represents the same "semantic distance", which is not really true. IC-based measures fix this by adding **corpus statistics**.

### Information Content

```
IC(concept) = -log P(concept)
```

`P(concept)` is the probability of seeing the concept (or any of its descendants) in a corpus. Rare, specific concepts (like `carnivore`) have **high IC**. Generic ones (like `entity`) have **low IC**.

### Common IC measures

| Measure | Formula |
|---|---|
| **Resnik** | `IC(LCS)` |
| **Lin** | `2 * IC(LCS) / (IC(s1) + IC(s2))` |
| **Jiang-Conrath** | `1 / (IC(s1) + IC(s2) - 2 * IC(LCS))` |

- Resnik only looks at the shared ancestor, so scores are unbounded and not normalized.
- **Lin** normalizes by the IC of both synsets, giving a value between `0 and 1`, which is easy to compare with Wu-Palmer.

### Loading IC and computing similarity

NLTK ships pre-computed IC files (built from the Brown corpus and others):

```python
brown_ic = wordnet_ic.ic("ic-brown.dat")

print(dog.lin_similarity(cat, brown_ic))      # ~0.877
print(dog.res_similarity(cat, brown_ic))
print(dog.jcn_similarity(cat, brown_ic))
```

> You can also build your own IC file from any corpus using `wn.ic(corpus_reader, weight_senses_equally=False)`.

---

## 5. Handling words, not synsets

The measures above work on **synsets**, but the input is usually **words**. A word can have many senses, e.g. *car* can be an automobile, a railway carriage or a cable-car. A common strategy is to compare **all sense pairs** and take the **maximum** score:

```python
def best_similarity(word1, word2, measure, pos=wn.NOUN):
    best = (0.0, None, None)
    for s1 in wn.synsets(word1, pos=pos):
        for s2 in wn.synsets(word2, pos=pos):
            score = measure(s1, s2)
            if score is not None and score > best[0]:   # some measures return None
                best = (score, s1, s2)
    return best
```

Then plug in any measure:

```python
wup = lambda a, b: a.wup_similarity(b)
lin = lambda a, b: a.lin_similarity(b, brown_ic)

score, s1, s2 = best_similarity("car", "bicycle", wup)
```

---

## 6. Putting it together

Loop over word pairs and print both scores side by side:

```python
word_pairs = [("car", "automobile"), ("car", "bicycle"),
              ("dog", "cat"), ("dog", "car"), ("gold", "egg")]

for w1, w2 in word_pairs:
    wup_score, _, _ = best_similarity(w1, w2, wup)
    lin_score, _, _ = best_similarity(w1, w2, lin)
    print(f"{w1}-{w2}: Wu-Palmer={wup_score:.3f}  Lin={lin_score:.3f}")
```

### Sample output

| Pair | Wu-Palmer | Lin |
|---|---|---|
| car - automobile | 1.000 | 1.000 |
| coast - shore | 0.909 | 0.963 |
| journey - voyage | 0.952 | 0.778 |
| dog - cat | 0.857 | 0.877 |
| car - bicycle | 0.800 | 0.766 |
| dog - car | 0.667 | 0.381 |
| gold - egg | 0.400 | 0.323 |

---

## 7. Observations

- **Synonyms score 1.0** in both methods since they share the same synset.
- **Related concepts** (dog/cat, car/bicycle) score high, and **unrelated ones** (gold/egg) score low.
- **IC-based scores spread out more.** For *dog - car*, Wu-Palmer gives 0.667 but Lin gives 0.381. The two only share a very generic ancestor, which has low IC, and Lin penalizes that. Path-based measures are more "optimistic" when the tree is shallow.
- **Sense selection matters.** Taking the max over all senses can pick surprising synsets (for *dog - car*, the best match was `pawl.n.01`, a mechanical part, since `dog` also means a catch or latch). For better accuracy, restrict to the most frequent sense (`wn.synsets(word)[0]`) or use word sense disambiguation.

---

## 8. Limitations and tips

- Measures work **within the same part of speech** (nouns with nouns, verbs with verbs). Path and Wu-Palmer need a shared hierarchy, and IC methods are only defined for nouns and verbs.
- Adjectives and adverbs have no hypernym hierarchy, so these measures don't apply to them.
- IC values depend on the corpus used. Different corpora give different scores.
- Pick **Wu-Palmer / Path** when you want a simple, corpus-free baseline. Pick **Lin / Jiang-Conrath** when you want finer discrimination that reflects how specific concepts actually are.

---

## 9. Summary

| Step | What to do |
|---|---|
| 1 | Install NLTK and download `wordnet`, `omw-1.4`, `wordnet_ic` |
| 2 | Load an IC file, e.g. `ic-brown.dat` |
| 3 | For each word, fetch its synsets with `wn.synsets(word, pos=...)` |
| 4 | Score every sense pair with a path-based (`wup_similarity`) and an IC-based (`lin_similarity`) measure |
| 5 | Keep the maximum score per word pair and compare the two methods |
