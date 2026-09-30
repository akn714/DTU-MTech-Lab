"""
Assignment 4: Semantic similarity between word pairs using WordNet.

  1. Path-based method : Wu-Palmer Similarity (uses the taxonomy structure)
  2. IC-based method   : Lin Similarity (uses Information Content from a corpus)
"""

import nltk
from nltk.corpus import wordnet as wn, wordnet_ic

# One-time downloads (skipped automatically if already present)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("wordnet_ic", quiet=True)

# Information Content loaded from the Brown corpus
brown_ic = wordnet_ic.ic("ic-brown.dat")

# Word pairs to compare
word_pairs = [
    ("car", "automobile"),
    ("car", "bicycle"),
    ("dog", "cat"),
    ("dog", "car"),
    ("coast", "shore"),
    ("journey", "voyage"),
    ("gold", "egg"),
]


def best_similarity(word1, word2, measure, pos=wn.NOUN):
    """
    A word has many senses (synsets). We compare every sense of word1
    with every sense of word2 and keep the highest score.
    Returns (score, synset1, synset2).
    """
    best = (0.0, None, None)
    for s1 in wn.synsets(word1, pos=pos):
        for s2 in wn.synsets(word2, pos=pos):
            score = measure(s1, s2)
            # Some measures return None when no path exists
            if score is not None and score > best[0]:
                best = (score, s1, s2)
    return best


def wu_palmer(s1, s2):
    # Path-based: depth of the LCS relative to depths of both synsets
    return s1.wup_similarity(s2)


def lin(s1, s2):
    # IC-based: 2 * IC(LCS) / (IC(s1) + IC(s2))
    return s1.lin_similarity(s2, brown_ic)


print(f"{'Word pair':<22}{'Wu-Palmer':>10}{'Lin':>10}   Best synsets (Lin)")
print("-" * 75)

for w1, w2 in word_pairs:
    wup_score, _, _ = best_similarity(w1, w2, wu_palmer)
    lin_score, a, b = best_similarity(w1, w2, lin)
    synsets = f"{a.name()} / {b.name()}" if a else "-"
    print(f"{w1 + ' - ' + w2:<22}{wup_score:>10.3f}{lin_score:>10.3f}   {synsets}")

# --- Extra: show the shared ancestor (LCS) for one pair to explain the idea ---
s1, s2 = wn.synset("dog.n.01"), wn.synset("cat.n.01")
print("\nLowest common subsumer of dog and cat:", s1.lowest_common_hypernyms(s2))
print("Shortest path length:", s1.shortest_path_distance(s2))
print("Path similarity     :", round(s1.path_similarity(s2), 3))
