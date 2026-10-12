import java.util.*;

public class AStarSearch {

    public static void main(String[] args) {

        // Create the graph (edge costs)
        Map<String, Map<String, Integer>> graph = new HashMap<>();

        graph.put("A", Map.of("B", 2, "C", 5));
        graph.put("B", Map.of("A", 2, "C", 2, "D", 4));
        graph.put("C", Map.of("A", 5, "B", 2, "D", 1));
        graph.put("D", Map.of("B", 4, "C", 1));

        // Define start and destination nodes
        String startNode = "A";
        String endNode = "D";

        // Heuristic estimates to the destination D
        Map<String, Integer> heuristic = Map.of(
            "A", 3,
            "B", 2,
            "C", 1,
            "D", 0
        );

        aStar(graph, heuristic, startNode, endNode);
    }

    static void aStar(
            Map<String, Map<String, Integer>> graph,
            Map<String, Integer> heuristic,
            String startNode,
            String endNode) {

        // Actual cost from start to each node
        Map<String, Integer> gScore = new HashMap<>();

        // Store the previous node to reconstruct the path
        Map<String, String> cameFrom = new HashMap<>();

        // Priority queue ordered by f(n) = g(n) + h(n)
        PriorityQueue<String> pq = new PriorityQueue<>(
            Comparator.comparingInt(
                node -> gScore.getOrDefault(node, Integer.MAX_VALUE) + heuristic.getOrDefault(node, 0)
            )
        );

        // Initialize the start node
        gScore.put(startNode, 0);
        pq.offer(startNode);

        while (!pq.isEmpty()) {

            String curr = pq.poll();

            // Destination reached
            if (curr.equals(endNode)) {
                System.out.println("Shortest path: " + reconstructPath(cameFrom, curr));
                System.out.println("Total cost: " + gScore.get(curr));
                return;
            }

            // Explore neighbors
            for (Map.Entry<String, Integer> neighbor: graph.getOrDefault(curr, Collections.emptyMap()).entrySet()) {
                String next = neighbor.getKey(); 
                int edgeCost = neighbor.getValue();

                int tentativeG = gScore.get(curr) + edgeCost;

                // Update if this path is better
                if (tentativeG < gScore.getOrDefault(next, Integer.MAX_VALUE)) {
                    cameFrom.put(next, curr);
                    gScore.put(next, tentativeG);

                    // Reinsert to update its priority
                    pq.remove(next);
                    pq.offer(next);
                }
            }
        }

        System.out.println("No path found from " + startNode + " to " + endNode);
    }

    static List<String> reconstructPath(
            Map<String, String> cameFrom,
            String curr) {

        List<String> path = new ArrayList<>();
        path.add(curr);

        while (cameFrom.containsKey(curr)) {
            curr = cameFrom.get(curr);
            path.add(curr);
        }

        Collections.reverse(path);
        return path;
    }
}