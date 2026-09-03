# A* Search
**Admissible Heuristics:**<br>
- A heuristic function is said to be admissible if for every node n, `h(n) <= h(n)*`
- Where h(n)* is true/actual cost from node n to goal
- An admissible heuristics never `overestimates` the cost to reach goal.
- Theorem: If h(n) is admissible then A* Search using h(n) as heuristic function is optimal for Tree Search Paradigm.
```
h(sld) is always admissible
h(sld) - straight line distance
```

**Consistent:**<br>
- A heuristic function is said to be consistent if for every node n, for every successor n'
- `h(n) <= cost(n,n') + h(n')`

```
Q. IF f(n), g(n) and h(n) are 3 admissible heusitic functions then check whether the following will be guranteed admissible heuristic functions or not
    - True: min(f(n), g(n), h(n))
    - True: max(f(n), g(n), h(n))
    - True: min(f(n), g(n) + h(n))
    - False: max(f(n) + g(n), h(n))
    - False: f(n) + g(n) + h(n)
```