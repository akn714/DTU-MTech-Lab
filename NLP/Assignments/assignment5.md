# Word Sense Disambiguation with WordNet and Graph Centrality

The word **bank** means something different in *"He sat on the bank of the river"* and *"The bank approved the loan"*. Picking the right meaning from context is called **Word Sense Disambiguation (WSD)**.

A neat way to do WSD without any training data is to treat **WordNet as a graph** and let **network centrality measures** vote for the most "well-connected" sense of each word. This post walks through the idea and shows how to implement it with NLTK and NetworkX, using these measures:

- Degree centrality
- Betweenness centrality
- Closeness centrality
- PageRank
- Hubs and Authorities (HITS)
- Key Player Problem (KPP)

---

## 1. The core idea

1. A word has several **senses** (WordNet *synsets*), e.g. `bank.n.01` (river bank) and `depository_financial_institution.n.01` (money bank).
2. Words in the same sentence usually talk about the same topic, so **the correct senses of neighbouring words should be semantically close** in WordNet.
3. Build a graph whose nodes are the candidate senses, and connect senses through WordNet's hypernym ("is-a") links.
4. The correct senses end up **densely connected** to each other. Wrong senses stay on the edge of the graph or isolated.
5. Compute a centrality score for each candidate sense and **pick the highest-scoring sense per word**.

---

## 2. Setup

```bash
pip install nltk networkx
```

```python
import itertools
import networkx as nx
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")
nltk.download("omw-1.4")
```

We use nouns only, so all senses live in one hypernym hierarchy and can be connected to each other.

---

## 3. Step 1: Get candidate senses

```python
words = ["bank", "river", "water"]
candidates = {w: wn.synsets(w, pos=wn.NOUN) for w in words}
# 'bank' -> 10 senses, 'river' -> 1, 'water' -> 6
```

---

## 4. Step 2: Build the semantic graph

### Why not link every sense to the root?

The naive approach is to add the full hypernym path (`entity -> ... -> sense`) for every candidate. Do not do this. Every sense then shares generic ancestors such as `entity` or `abstraction`, everything looks connected, and high-level nodes dominate every centrality score.

### Better: connect senses of *different* words through short paths

Following the classic graph-based WSD approach (Navigli and Lapata):

- For every pair of senses belonging to **different words**, find their **Lowest Common Subsumer (LCS)**, the most specific shared ancestor.
- If the two senses are within a maximum distance (say 6 edges), add the two chains `sense -> ... -> LCS` to the graph.
- Senses that have no short connection to any other word's senses stay **isolated**.

```python
MAX_PATH_LEN = 6

def path_up(sense, ancestor):
    """Hypernym chain  sense -> ... -> ancestor."""
    for p in sense.hypernym_paths():
        if ancestor in p:
            return p[p.index(ancestor):][::-1]

G, D = nx.Graph(), nx.DiGraph()        # undirected and directed versions

for s in itertools.chain(*candidates.values()):
    G.add_node(s); D.add_node(s)       # keep every candidate, even if isolated

for w1, w2 in itertools.combinations(words, 2):
    for s1 in candidates[w1]:
        for s2 in candidates[w2]:
            dist = s1.shortest_path_distance(s2)
            if dist is None or dist > MAX_PATH_LEN:
                continue
            for lcs in s1.lowest_common_hypernyms(s2):
                for s in (s1, s2):
                    chain = path_up(s, lcs)
                    for child, parent in zip(chain, chain[1:]):
                        G.add_edge(child, parent)
                        D.add_edge(child, parent)    # direction: sense -> ancestor
```

We keep two graphs because:

- **Undirected `G`** is used for degree, betweenness, closeness, PageRank and KPP.
- **Directed `D`** (sense -> ancestor) is used for HITS, which needs edge direction.

---

## 5. Step 3: Score candidate senses with centrality

Each measure asks a slightly different question about a node's importance.

### 5.1 Degree centrality
Number of neighbours, normalized by `n - 1`. A sense with many links to other words' senses is "popular".

```python
deg = nx.degree_centrality(G)
```

### 5.2 Betweenness centrality
Fraction of shortest paths between all node pairs that pass through the node. High values mean the node is a **bridge** between different parts of the graph.

```python
btw = nx.betweenness_centrality(G)
```

### 5.3 Closeness centrality
Inverse of the average shortest-path distance to all other nodes. High values mean the node is **close to everything**.

```python
clo = nx.closeness_centrality(G)
```

### 5.4 PageRank
A node is important if important nodes link to it. It is computed with a random walk with damping factor (default 0.85).

```python
pr = nx.pagerank(G)
```

### 5.5 Hubs and Authorities (HITS)
On a directed graph each node gets two scores:

- **Authority**: pointed to by many good hubs.
- **Hub**: points to many good authorities.

In our `sense -> ancestor` graph, shared ancestors become authorities, and a candidate sense that points toward well-shared ancestors gets a high **hub** score, so we rank candidates by hub score. `nx.hits(D)` works, but the algorithm is only a few lines of power iteration:

```python
def hits(D, iterations=50):
    hub = {n: 1.0 for n in D}
    for _ in range(iterations):
        auth = {n: sum(hub[u] for u in D.predecessors(n)) for n in D}
        hub  = {n: sum(auth[v] for v in D.successors(n)) for n in D}
        norm = sum(x * x for x in hub.values()) ** 0.5 or 1.0
        hub  = {n: x / norm for n, x in hub.items()}     # normalize each round
    return hub, auth
```

> Note: `nx.hits` fails on a graph with no edges, and its randomized solver can give slightly different floating-point values per run, which matters when you need to break ties. The hand-written version avoids both issues.

### 5.6 Key Player Problem (KPP)

Borgatti's **Key Player Problem** asks: *which nodes are the most important to a network?* It has two flavours:

**KPP-Neg (fragmentation).** Which node, when **removed**, breaks the network into the most pieces?

```
F = 1 - sum( s_k * (s_k - 1) ) / ( n * (n - 1) )      s_k = size of k-th component after removal
```

```python
def kpp_neg(G):
    n, scores = G.number_of_nodes(), {}
    for node in G:
        H = G.copy()
        H.remove_node(node)
        comps = [len(c) for c in nx.connected_components(H)]
        scores[node] = 1 - sum(s * (s - 1) for s in comps) / (n * (n - 1))
    return scores
```

**KPP-Pos (reach).** Which node (or set of nodes) is best placed to **reach all others** quickly?

```
reach(S) = sum over v not in S of  1 / d(S, v)        d(S, v) = distance from nearest member of S
```

```python
def kpp_pos(G, S):
    dist = nx.multi_source_dijkstra_path_length(G, S)
    return sum(1 / d for v, d in dist.items() if v not in S and d > 0)
```

For WSD we score each candidate as a single-node set: `{n: kpp_pos(G, [n]) for n in G}`.

The "real" key player problem looks for the best **set of k nodes**. Since the optimum is expensive to compute, use a **greedy** search: add one node at a time, each time choosing the node that maximizes the set's score.

```python
def greedy_kpp_pos(G, k):
    chosen = []
    for _ in range(k):
        best = max((n for n in G if n not in chosen),
                   key=lambda n: kpp_pos(G, chosen + [n]))
        chosen.append(best)
    return chosen
```

The key players found this way are the concepts that "hold the topic together", for example `body_of_water.n.01` for a river sentence.

---

## 6. Step 4: Choose the sense

For each word, take the candidate with the highest score. Two practical details:

- **Ties.** Many isolated senses score 0. Break ties in favour of the **first listed sense**, since WordNet orders senses by frequency. This is the standard "most frequent sense" fallback.
- **Float noise.** Round scores before comparing (e.g. `round(score, 6)`) so tiny numerical differences don't decide ties.

```python
def disambiguate(words, scores, candidates):
    result = {}
    for w in words:
        best = max(enumerate(candidates[w]),
                   key=lambda t: (round(scores[t[1]], 6), -t[0]))
        result[w] = best[1]
    return result

hubs, _ = hits(D)
all_scores = {"Degree": deg, "Betweenness": btw, "Closeness": clo,
              "PageRank": pr, "HITS (hub)": hubs,
              "KPP-Neg": kpp_neg(G),
              "KPP-Pos": {n: kpp_pos(G, [n]) for n in G}}

for name, scores in all_scores.items():
    print(name, {w: s.name() for w, s in disambiguate(words, scores, candidates).items()})
```

---

## 7. Results

Hand-labelled senses were used to check three short sentences (6 labelled words in total):

| Sentence | Expected |
|---|---|
| He sat on the **bank** of the **river** and watched the water | `bank.n.01` (sloping land), `river.n.01` |
| She moved the **mouse** and typed on the **keyboard** | `mouse.n.04` (device), `keyboard.n.01` |
| The **bank** approved the **loan** and charged interest | `depository_financial_institution.n.01`, `loan.n.01` |

| Measure | Correct |
|---|---|
| Degree | 6 / 6 |
| KPP-Neg | 6 / 6 |
| Closeness | 5 / 6 |
| KPP-Pos | 5 / 6 |
| Betweenness | 4 / 6 |
| PageRank | 4 / 6 |
| HITS (hub) | 4 / 6 |

With only six labelled words this is an illustration, **not a reliable benchmark**. The real lesson is in how the measures behave (next section).

---

## 8. Observations

- **Connected senses win.** In the river sentence, the land sense of *bank* is chosen, because the financial senses have no short path to *river* or *water*.
- **Measures can disagree.** In the finance sentence, closeness and KPP-Pos picked `bank.n.06` ("funds held by a gambling house"). It sits close to *money*-related nodes in the hierarchy, so a distance-based measure likes it, while Degree, PageRank and KPP-Neg picked the depository-institution sense.
- **Global measures favour the centre.** Closeness, KPP-Pos and PageRank reward nodes near the middle of the whole graph. A sense that is central but wrong can beat a correct sense sitting on a small, tightly connected branch.
- **Bridge-based measures are sensitive to graph shape.** Betweenness and KPP-Neg only score high for nodes that connect separate parts of the graph. On very small graphs, where a sense has only one or two paths, the scores are coarse and ties are common.
- **WordNet hierarchy limits accuracy.** Related concepts such as *bank* (institution) and *loan* (financial transaction) sit in different top-level branches of WordNet, so no amount of clever scoring can link them with short hypernym paths. Adding other relations (meronyms, glosses, domain links) usually helps a lot.

---

## 9. Tips and limitations

- **Sentence length matters.** More context words means more evidence. With two words, the graph is tiny and results are fragile.
- **Tune `MAX_PATH_LEN`.** Too small leaves everything disconnected, and too large connects unrelated senses. Values of about 4 to 6 are a common starting point. In practice, going beyond 6 did not improve results.
- **Only nouns here.** Verbs have their own hypernym trees, so noun-verb links need extra relations.
- **Always compare to a baseline.** The "first sense" baseline (`wn.synsets(word)[0]`) is surprisingly hard to beat, so use it to check whether the graph actually helps.
- **Computational cost.** KPP-Neg rebuilds the graph once per node, and greedy KPP-Pos evaluates every node per step. Both are fine for small graphs but slow on large ones.

---

## 10. Summary

| Step | What to do |
|---|---|
| 1 | Pick the content words (nouns) of the sentence |
| 2 | Fetch all candidate synsets for each word |
| 3 | Connect senses of different words via short paths through their LCS, and build an undirected and a directed graph |
| 4 | Score each candidate with degree, betweenness, closeness, PageRank, HITS hub score and KPP-Neg / KPP-Pos |
| 5 | For each word, choose the top-scoring sense, falling back to the first sense on ties |
| 6 | Compare the measures, ideally against labelled data and the most-frequent-sense baseline |
