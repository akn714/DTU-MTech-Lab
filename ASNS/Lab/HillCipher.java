import java.util.Arrays;

// Ascii of A-Z is 65-90
// Ascii of a-z is 97-122

public class HillCipher {
    public static void main(String[] args) {
        String PT = "ACT";
        String key = "GYBNQKURP";
        
        String CT = getCipherText(key, PT);

        System.out.println("The Cipher Text is " + CT);
        System.out.println("The Plain Text is " + getPlainText(key, CT));
    }

    static String getCipherText(String key, String PT) {
        int[][] keyMat = getMat(key, key.length()/PT.length(), PT.length());
        int[][] ptMat = getMat(PT, PT.length(), 1);
        String CT = "";

        char[][] mult = matMult(keyMat, ptMat, keyMat.length, ptMat.length, ptMat[0].length);
        
        for(int i=0;i<mult.length;i++) {
            for(int j=0;j<mult[0].length;j++) {
                CT += mult[i][j];
            }
        }

        return CT;
    }

    static int[][] getMat(String str, int x, int y) {
        int[][] mat = new int[x][y];

        int count = 0;
        for(int i=0;i<x;i++) {
            for(int j=0;j<y;j++) {
                mat[i][j] = (int) str.charAt(count) - (int) 'A';
                count++;
            }
        }

        return mat;
    }

    static char[][] matMult(int[][] mat1, int[][] mat2, int n, int m, int p) {
        int[][] res = new int[n][p];
        char[][] charMat = new char[n][p];
        
        for(int i=0;i<n;i++) {
            for(int j=0;j<p;j++) {
                for(int k=0;k<m;k++) {
                    res[i][j] += mat1[i][k] * mat2[k][j];
                }
                charMat[i][j] = (char) (res[i][j] % 26 + (int) 'A');
            }
        }

        return charMat;
    }

    static int[][] getInverse(int[][] mat, int n) {
        int[][] adj = new int[n][n];
        int det = getDeterminant(mat, n);
        int invDet = modInverse(det, 26);

        if(det == 0) {
            System.out.println("Inverse doesn't exist");
            return null;
        }

        adj = getAdjoint(mat, n);

        for(int i=0;i<n;i++) {
            for(int j=0;j<n;j++) {
                adj[i][j] = (adj[i][j] * invDet) % 26;
                if(adj[i][j] < 0) adj[i][j] += 26;
            }
        }

        return adj;
    }

    static int getDeterminant(int[][] mat, int n) {
        int det = 0;
        if(n == 1) return mat[0][0];
        if(n == 2) return (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]);

        for(int i=0;i<n;i++) {
            int[][] temp = new int[n-1][n-1];
            for(int j=1;j<n;j++) {
                for(int k=0;k<n;k++) {
                    if(k < i) temp[j-1][k] = mat[j][k];
                    else if(k > i) temp[j-1][k-1] = mat[j][k];
                }
            }
            det += (i%2==0 ? 1 : -1) * mat[0][i] * getDeterminant(temp, n-1);
        }

        return det;
    }

    static int modInverse(int a, int m) {
        a = a % m;
        for(int x=1;x<m;x++) {
            if((a * x) % m == 1) return x;
        }
        return 1;
    }

    static int[][] getAdjoint(int[][] mat, int n) {
        int[][] adj = new int[n][n];
        if(n == 1) {
            adj[0][0] = 1;
            return adj;
        }

        for(int i=0;i<n;i++) {
            for(int j=0;j<n;j++) {
                int[][] temp = new int[n-1][n-1];
                for(int row=0;row<n;row++) {
                    for(int col=0;col<n;col++) {
                        if(row != i && col != j) {
                            int r = row < i ? row : row - 1;
                            int c = col < j ? col : col - 1;
                            temp[r][c] = mat[row][col];
                        }
                    }
                }
                adj[j][i] = (int) Math.pow(-1, i+j) * getDeterminant(temp, n-1);
            }
        }

        return adj;
    }

    static String getPlainText(String key, String CT) {
        int[][] keyMat = getMat(key, key.length()/CT.length(), CT.length());
        int[][] ctMat = getMat(CT, CT.length(), 1);
        String PT = "";

        int[][] invKeyMat = getInverse(keyMat, keyMat.length);
        char[][] mult = matMult(invKeyMat, ctMat, invKeyMat.length, ctMat.length, ctMat[0].length);
        
        for(int i=0;i<mult.length;i++) {
            for(int j=0;j<mult[0].length;j++) {
                PT += mult[i][j];
            }
        }

        return PT;
    }
}