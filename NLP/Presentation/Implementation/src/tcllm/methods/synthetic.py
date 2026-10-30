"""Synthetic data generation, without an LLM.

The paper generates 1800 synthetic sentiment examples per language by prompting
GPT-4, Qwen2.5-7B and Aya-Expanse-8B (200 per category per model). Those are
multi-GB downloads, so this module fills the same slot in the pipeline with a
**template-and-slot generator**: label-conditioned sentence templates with random
slot substitution, seeded for reproducibility.

The point is to exercise the actual experimental question — *can a classifier
trained only on synthetic data transfer to real held-out tweets?* — with the
volume matched to the paper's (1800 examples, 600 per class). The expected answer
is a visible drop versus training on real data, which is the honest outcome and
consistent with the paper's own finding that synthetic data is competitive but
not equal to real data.

This is a documented deviation. See docs/deviations.md.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..data import Example

# --------------------------------------------------------------------------- #
# slots
# --------------------------------------------------------------------------- #
SUBJECTS = [
    "this movie", "the new update", "this album", "my coffee", "the weather today",
    "this restaurant", "the customer service", "this book", "the concert last night",
    "my new phone", "this app", "the traffic", "this game", "the food here",
    "this show", "the hotel", "my flight", "this podcast", "the exhibition",
    "this laptop", "the delivery", "this series", "my new headphones", "the museum",
    "this cafe", "the bus ride", "this camera", "the workshop", "this playlist",
]
PEOPLE = ["my friend", "the staff", "this guy", "the team", "my boss", "the driver", "the waiter"]

POSITIVE_TEMPLATES = [
    "I absolutely love {s}, it made my whole day",
    "{s} is amazing, I am so happy with it",
    "honestly {s} was fantastic and I would recommend it to anyone",
    "just tried {s} and wow, best decision this week",
    "{s} turned out better than I expected, really pleased",
    "shout out to {p}, they were incredibly helpful and kind",
    "I am so glad I finally got {s}, it works perfectly",
    "{s} is beautiful and works like a charm, love it",
    "ngl {s} exceeded my expectations, super impressive",
    "big thanks to {p} for sorting this out so quickly, brilliant",
    "{s} was a wonderful surprise, I am genuinely delighted",
    "cannot stop smiling about {s}, what a great time",
    "ok so {s} is actually incredible and I am here for it",
    "{s} made me so happy today, exactly what I needed",
    "highly recommend {s}, it is worth every penny",
    "{p} did an outstanding job and I really appreciate it",
    "{s} is easily the best I have had in a long time",
    "feeling grateful for {s}, such a lovely experience",
]

NEGATIVE_TEMPLATES = [
    "I really hate {s}, it completely ruined my day",
    "{s} is awful and I am so disappointed with it",
    "honestly {s} was terrible and I would not recommend it to anyone",
    "just tried {s} and wow, worst decision this week",
    "{s} turned out much worse than I expected, really frustrating",
    "the staff at {p} were rude and completely unhelpful",
    "I regret buying {s}, it is broken and useless",
    "{s} is a mess and barely works, avoid it",
    "ngl {s} was a huge letdown, super disappointing",
    "no thanks to {p} for making this so difficult, awful service",
    "{s} was a horrible surprise, I am genuinely upset",
    "cannot stop thinking about how bad {s} was, what a waste",
    "ok so {s} is actually unusable and I am done with it",
    "{s} made me so angry today, exactly what I did not need",
    "do not bother with {s}, it is a total waste of money",
    "{p} did a terrible job and I am not happy about it",
    "{s} is easily the worst I have had in a long time",
    "so tired of {s}, such a frustrating experience",
]

NEUTRAL_TEMPLATES = [
    "{s} is available now, I might check it out later",
    "has anyone else tried {s} yet, curious what you think",
    "{s} arrived this morning, going to open it soon",
    "reading about {s} in the news today, not sure what to make of it",
    "{s} was released last week according to the announcement",
    "I saw {p} at the station this morning on my way in",
    "{s} is on the schedule for tomorrow, should be interesting",
    "someone mentioned {s} in the meeting earlier today",
    "{s} costs about the same as the older version",
    "there is a new post about {s} on the company blog",
    "{s} is supposed to arrive sometime next week",
    "not sure whether {s} is worth the trip, still deciding",
    "the listing for {s} says it ships in three days",
    "I have {s} saved in my bookmarks for now",
    "{s} came up in conversation at lunch today",
    "planning to look at {s} over the weekend",
    "{s} is one of the options I am considering",
    "just noting that {s} exists in case anyone needs it",
]

ENDINGS = ["", "", "", " #mood", " :)", " :(", " !", " ...", " @user", " :) #blessed"]

TEMPLATES_BY_LABEL = {
    "positive": POSITIVE_TEMPLATES,
    "negative": NEGATIVE_TEMPLATES,
    "neutral": NEUTRAL_TEMPLATES,
}

# The paper's synthetic SENTIMENT volume: 200 per category per LLM x 3 LLMs.
DEFAULT_PER_LABEL = 600


def _fill(template: str, rng: random.Random) -> str:
    text = template
    if "{s}" in text:
        text = text.replace("{s}", rng.choice(SUBJECTS))
    if "{p}" in text:
        text = text.replace("{p}", rng.choice(PEOPLE))
    return text + rng.choice(ENDINGS)


def generate_synthetic(
    labels: Sequence[str],
    per_label: int = DEFAULT_PER_LABEL,
    seed: int = 42,
) -> list[Example]:
    """Generate ``per_label`` synthetic examples for each label.

    Uses no real data at all, so there is nothing here that could leak the test
    split; the demo still runs the leakage guard over the result to prove it.
    """
    rng = random.Random(seed)
    examples: list[Example] = []
    for index, label in enumerate(labels):
        templates = TEMPLATES_BY_LABEL.get(label)
        if not templates:
            raise ValueError(f"no synthetic templates for label {label!r}")
        for _ in range(per_label):
            examples.append(
                Example(text=_fill(rng.choice(templates), rng), label=index, label_name=label)
            )
    rng.shuffle(examples)
    return examples
