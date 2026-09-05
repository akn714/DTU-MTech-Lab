"""
What you mean                   | Regex
--------------------------------|-------------
NOT these characters	        | [^abc]
NOT followed by this pattern	| (?!pattern)
NOT preceded by this pattern	| (?<!pattern)
Start of word	                | \b
"""

import re

PATTERN = re.compile(r"\bh[a-z]+")

text = "hello man, how are you, have you eaten the food, that was crazy, ha ha ha"

print(PATTERN.findall(text))