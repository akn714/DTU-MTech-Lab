public class CriticalConnections {
    public static void main(String args[]) {
        List<List<Integer>> connections = new ArrayList<>();
        connections.add(Arrays.asList(0, 1));
        connections.add(Arrays.asList(1, 2));
        connections.add(Arrays.asList(2, 0));
        connections.add(Arrays.asList(1, 3));

        int n = 4;

        List<List<Interger>> crtconn = criticalConnections(n, connections);
    }
    
    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        List<List<Integer>> adj = getAdj(n, connections);
        boolean[] isVisited = new boolean[n];

        int[] arrivaltime = new int[n];
        int[] low = new int[n];
        boolean[] visited = new boolean[n];

        dfs(adj, 0, n, 1, arrivaltime, low, visited);
    }

    public static void dfs(List<List<Integer>> g, int i, int n, int count, int[] at, int[] low, boolean[] visited) {
        if(visited[i]) return low[i];
        at[i] = count;
        low[i] = count;
        for(int node: g.get(i)) {
            dfs(g, node, n, ++count, at, low, visited);n
        }
    }

    public static List<List<Integer>> getAdj(int n, List<List<Integer>> connections) {
        int x, y;
        List<List<Integer>> adj = new ArrayList<>();
        for(List<Integer> e: connections) {
            x = e.get(0);
            y = e.get(1);

            adj.get(x).add(y);
            adj.get(y).add(x);
        }

        return adj;
    }
}



// class Solution {
//     public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
//         List<List<Integer>> adj = getAdj(n, connections);
//         boolean[] isVisited = new boolean[n];

//         int[] arrivaltime = new int[n];
//         int[] low = new int[n];
//         boolean[] visited = new boolean[n];

//         dfs()
//     }

//     public static void dfs(List<List<Integer>> g, int i, int n, int count, int[] at, int[] low, boolean[] visited) {
//         if(visited[i]) return low[i];
//         at[i] = count;
//         low[i] = count;
//         for(List<Integer> 
//         )
//     }

//     public static List<List<Integer>> getAdj(int n, List<List<Integer>> connections) {
//         int x, y;
//         List<List<Integer>> adj = new ArrayList<>();
//         for(List<Integer> e: connections) {
//             x = e.get(0);
//             y = e.get(1);

//             adj.get(x).add(y);
//             adj.get(y).add(x);
//         }

//         return adj;
//     }
// }