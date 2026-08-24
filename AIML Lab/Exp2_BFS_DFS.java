import java.util.Queue;
import java.util.LinkedList;

class GraphNode {
    int value;
    GraphNode left;
    GraphNode right;

    GraphNode(int val) {
        this.value = val;
        this.left = null;
        this.right = null;
    }
}

public class BFS_DFS {
    public static void main(String[] args) {
        GraphNode root = new GraphNode(0);
        root.left = new GraphNode(1);
        root.right = new GraphNode(2);
        root.left.left = new GraphNode(3);
        root.left.right = new GraphNode(4);
        root.right.right = new GraphNode(5);

        // ========== TRAVERSAL ==========
        System.out.println("== BFS TRAVERSAL ==");
        traverse_bfs(root);

        System.out.println("== DFS TRAVERSAL ==");
        traverse_dfs(root);

        // ========== SHORTEST PATH ==========
        System.out.println("== BFS SHORTEST PATH ==");
        GraphNode g1 = new GraphNode(0);
        g1.left = new GraphNode(1);
        g1.right = new GraphNode(2);
        g1.left.right = new GraphNode(3);
        g1.right.left = new GraphNode(4);
        g1.left.right.right = new GraphNode(5);
        g1.right.left.left = g1.left.right.right;
        shortest_path_bfs(g1, 5);

        System.out.println("== DFS SHORTEST PATH ==");
        GraphNode g2 = new GraphNode(0);
        g2.left = new GraphNode(1);
        g2.right = new GraphNode(2);
        g2.left.left = new GraphNode(3);
        g2.left.right = new GraphNode(4);
        g2.right.right = new GraphNode(5);
        g2.left.right.right = new GraphNode(6);
        g2.right.right.left = g2.left.right.right;
        shortest_path_dfs(g2, 6, 0);
    }

    static void traverse_bfs(GraphNode root) {
        if(root==null) return;
        
        Queue<GraphNode> q = new LinkedList<>();

        q.add(root);

        while(!q.isEmpty()) {
            GraphNode curr = q.poll();
            System.out.println(curr.value);

            if(curr.left!=null) q.add(curr.left);
            if(curr.right!=null) q.add(curr.right);
        }
    }

    static void traverse_dfs(GraphNode root) {
        if(root==null) return;

        System.out.println(root.value);
        
        traverse_dfs(root.left);
        traverse_dfs(root.right);
    }

    static void shortest_path_bfs(GraphNode root, int target) {
        if(root==null) return;
        int edge_count = 0;
        
        Queue<GraphNode> q = new LinkedList<>();

        q.add(root);

        while(!q.isEmpty()) {
            GraphNode curr = q.poll();

            if(curr.value==target) {
                System.out.println("Shortest path length: " + edge_count);
                return;
            }
            edge_count++;

            if(curr.left!=null) q.add(curr.left);
            if(curr.right!=null) q.add(curr.right);
        }
    }

    static boolean shortest_path_dfs(GraphNode root, int target, int edge_count) {
        if(root==null) return false;
        if(root.value==target) {
            System.out.println("Shortest path length: " + edge_count);
            return true;
        }

        if(shortest_path_dfs(root.left, target, edge_count + 1)) {
            return true;
        }
        return shortest_path_dfs(root.right, target, edge_count + 1);
    }
}
