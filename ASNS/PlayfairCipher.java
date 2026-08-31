import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

public class PlayfairCipher {
    public static void main(String args[]) {
        String[][] mat = {
            {"K","E","Y","W","O"},
            {"R","D","A","B","C"},
            {"F","G","H","I","L"},
            {"M","N","P","Q","S"},
            {"T","U","V","X","Z"}
        };

        String plainText = "ADARSHKUMAR";

        String cipherText = genCipher(plainText, mat);

        System.out.println(cipherText);
    }

    public static String genCipher(String plainText, String[][] mat) {
        List<String> pairs = getPairs(plainText);
        String cipher = "";
        int[][] cords = new int[2][2];
        for(String pair: pairs) {
            cords = getCords(pair, mat);
            // chars in same row
            if(cords[0][1]==cords[1][1]) {
                if(cords[0][0]+1>4) cipher += mat[0][cords[0][1]];
                else cipher += mat[cords[0][0]+1][cords[0][1]];

                if(cords[1][0]+1>4) cipher += mat[0][cords[1][1]];
                else cipher += mat[cords[1][0]+1][cords[1][1]];
            }
            // chars in same column
            else if(cords[0][0]==cords[1][0]){
                if(cords[0][1]+1>4) cipher += mat[cords[0][0]][0];
                else cipher += mat[cords[0][0]][cords[0][1]+1];

                if(cords[1][1]+1>4) cipher += mat[cords[1][0]][0];
                else cipher += mat[cords[1][0]][cords[1][1]+1];
            }
            // chars not in same row or column
            else {
                cipher += mat[cords[0][0]][cords[1][1]];
                cipher += mat[cords[1][0]][cords[0][1]];
            }
        }

        return cipher;
    }

    public static int[][] getCords(String pair, String[][] mat) {
        int[][] cords = new int[2][2];
        for(int i=0;i<mat.length;i++) {
            for(int j=0;j<mat[0].length;j++) {
                if(pair.substring(0,1).equals("J")) {
                    cords[0][0] = 2;
                    cords[0][1] = 3;
                }
                else if(mat[i][j].equals(pair.substring(0,1))) {
                    cords[0][0] = i;
                    cords[0][1] = j;
                }
                if(pair.substring(1,2).equals("J")) {
                    cords[1][0] = 2;
                    cords[1][1] = 3;
                }
                else if(mat[i][j].equals(pair.substring(1,2))) {
                    cords[1][0] = i;
                    cords[1][1] = j;
                }
            }
        }

        return cords;
    }

    public static List<String> getPairs(String plainText) {
        List<String> pairs = new ArrayList<>();
        int i = 0;
        String padding = "X";
        String s = "";
        while(i<plainText.length()) {
            s = "";
            if(i==plainText.length()-1) {
                s += plainText.charAt(i);
                s += padding;
                i += 2;
            }
            else if(i+1<plainText.length() && plainText.charAt(i)==plainText.charAt(i+1)) {
                s += plainText.charAt(i);
                s += padding;
                i++;
            }
            else {
                s += plainText.charAt(i);
                s += plainText.charAt(i+1);
                i += 2;
            }
            pairs.add(s);
        }

        return pairs;
    }
}