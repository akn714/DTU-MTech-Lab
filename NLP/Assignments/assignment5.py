"""
Assignment 5: Word Sense Disambiguation (WSD) using WordNet + graph centrality.

Idea:
  1. For every (noun) word in a sentence, take all its candidate senses (synsets).
  2. Build a graph from WordNet: connect senses of DIFFERENT words through
     short hypernym ("is-a") paths via their lowest common ancestor. Senses that
     fit the same topic end up well connected; unrelated senses stay isolated.
  3. Score each candidate sense with a centrality measure.
  4. For each word, pick the sense with the highest score.
"""

import itertools
import networkx as nx
import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ---------------------------------------------------------------- graph ----
MAX_PATH_LEN = 6   # only keep connections that are reasonably short


def path_up(sense, ancestor):
    """Hypernym chain  sense -> ... -> ancestor."""
    for p in sense.hypernym_paths():
        if ancestor in p:
            return p[p.index(ancestor):][::-1]


def build_graph(words):
    """Returns (undirected graph, directed graph, {word: [candidate synsets]})."""
    G, D = nx.Graph(), nx.DiGraph()
    candidates = {w: wn.synsets(w, pos=wn.NOUN) for w in words}

    for s in itertools.chain(*candidates.values()):   # keep every candidate as a node
        G.add_node(s)
        D.add_node(s)

    # connect senses of different words through their lowest common ancestor
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
                            D.add_edge(child, parent)   # direction: sense -> ancestor
    return G, D, candidates


# ----------------------------------------------------------- KPP (Borgatti)
def kpp_neg(G):
    """KPP-Neg: how much does REMOVING a node fragment the graph?
    Fragmentation F = 1 - sum(s_k*(s_k-1)) / (n*(n-1)), s_k = component sizes."""
    n, scores = G.number_of_nodes(), {}
    for node in G:
        H = G.copy()
        H.remove_node(node)
        comps = [len(c) for c in nx.connected_components(H)]
        scores[node] = 1 - sum(s * (s - 1) for s in comps) / (n * (n - 1))
    return scores


def kpp_pos(G, S):
    """KPP-Pos: how well does the node SET S reach everybody else?
    reach = sum over other nodes of 1 / (distance from S to that node)."""
    dist = nx.multi_source_dijkstra_path_length(G, S)
    return sum(1 / d for v, d in dist.items() if v not in S and d > 0)


def greedy_kpp_pos(G, k):
    """Greedily pick the k 'key players' that maximise KPP-Pos."""
    chosen = []
    for _ in range(k):
        best = max((n for n in G if n not in chosen),
                   key=lambda n: kpp_pos(G, chosen + [n]))
        chosen.append(best)
    return chosen


# ------------------------------------------------------------------ HITS --
def hits(D, iterations=50):
    """Hubs & Authorities by power iteration (same idea as nx.hits, but deterministic).
    authority(n) = sum of hub scores of nodes pointing TO n
    hub(n)       = sum of authority scores of nodes n points TO"""
    hub = {n: 1.0 for n in D}
    for _ in range(iterations):
        auth = {n: sum(hub[u] for u in D.predecessors(n)) for n in D}
        hub = {n: sum(auth[v] for v in D.successors(n)) for n in D}
        norm = sum(x * x for x in hub.values()) ** 0.5 or 1.0   # keep numbers small
        hub = {n: x / norm for n, x in hub.items()}
    return hub, auth


# ------------------------------------------------------ centrality scores --
def all_scores(G, D):
    hubs, authorities = hits(D)
    return {
        "Degree":      nx.degree_centrality(G),
        "Betweenness": nx.betweenness_centrality(G),
        "Closeness":   nx.closeness_centrality(G),
        "PageRank":    nx.pagerank(G),
        "HITS (hub)":  hubs,          # a sense "points to" many good ancestors
        "KPP-Neg":     kpp_neg(G),
        "KPP-Pos":     {n: kpp_pos(G, [n]) for n in G},
    }


# ------------------------------------------------------------------- WSD ---
def disambiguate(words, scores, candidates):
    """Pick the highest-scoring sense; on ties the first (most frequent) sense wins.
    Scores are rounded so tiny floating-point noise doesn't break ties at random."""
    result = {}
    for w in words:
        best = max(enumerate(candidates[w]),
                   key=lambda t: (round(scores[t[1]], 6), -t[0]))
        result[w] = best[1]
    return result


def run(sentence, words, gold, tally):
    print("=" * 78)
    print("Sentence:", sentence)
    G, D, candidates = build_graph(words)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print(f"{'Measure':<13}" + "".join(f"{w:<44}" for w in words))
    for name, scores in all_scores(G, D).items():
        result = disambiguate(words, scores, candidates)
        row = f"{name:<13}"
        for w in words:
            mark = ""
            if w in gold:                       # compare with hand-labelled sense
                ok = result[w].name() == gold[w]
                tally[name][0] += ok
                tally[name][1] += 1
                mark = " (ok)" if ok else " (x)"
            row += f"{result[w].name() + mark:<44}"
        print(row)

    print("\nKPP-Pos key players (k=3):", [n.name() for n in greedy_kpp_pos(G, 3)])


if __name__ == "__main__":
    # (sentence, content words, hand-labelled correct senses)
    examples = [
        ("He sat on the bank of the river and watched the water",
         ["bank", "river", "water"],
         {"bank": "bank.n.01", "river": "river.n.01"}),
        ("She moved the mouse and typed on the keyboard",
         ["mouse", "keyboard"],
         {"mouse": "mouse.n.04", "keyboard": "keyboard.n.01"}),
        ("The bank approved the loan and charged interest",
         ["bank", "loan", "interest"],
         {"bank": "depository_financial_institution.n.01", "loan": "loan.n.01"}),
    ]

    tally = {}          # measure -> [correct, total]
    for name in ["Degree", "Betweenness", "Closeness", "PageRank",
                 "HITS (hub)", "KPP-Neg", "KPP-Pos"]:
        tally[name] = [0, 0]

    for sentence, words, gold in examples:
        run(sentence, words, gold, tally)

    print("\n" + "=" * 78)
    print("Accuracy on the hand-labelled words")
    for name, (right, total) in tally.items():
        print(f"  {name:<13}{right}/{total}")
