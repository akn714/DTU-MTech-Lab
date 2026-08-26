import java.util.Arrays;

public class Experiment2 {
    public static void main(String[] args) {
        int[][] isConnected = {{1,1,0},{1,1,0},{0,0,1}};
        Solution solution = new Solution();
        int result = solution.findCircleNum(isConnected);
        System.out.println(result);
    }
}

class Solution {
    // Problem 1: Number of Provinces
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        int[] visited = new int[n];
        int count = 0;
        int nodes_visited = 0;
        boolean allVisited = false;
        
        for(int i=0;i<n;i++) {
            if(visited[i]==0) {
                visited[i] = 1;
                count++;
                visitNode(isConnected, i, n, visited);
            }
        }

        return count;
    }

    public static void visitNode(int[][] isConnected, int i, int n, int[] visited) {
        if(i>=n) return;

        for(int j=0;j<n;j++) {
            if(i==j) continue;
            if(visited[j]==1) continue;
            else if(isConnected[i][j]==1) {
                visited[j] = 1;
                visitNode(isConnected, j, n, visited);
            }
        }
    }
}