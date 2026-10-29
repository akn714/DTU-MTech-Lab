import java.util.Scanner;

public class SDESKeyGeneration {

    static int[] P10 = {3, 5, 2, 7, 4, 10, 1, 9, 8, 6};
    static int[] P8 = {6, 3, 7, 4, 8, 5, 10, 9};

    static String permute(String key, int[] table) {
        StringBuilder result = new StringBuilder();

        for (int position : table) {
            result.append(key.charAt(position - 1));
        }

        return result.toString();
    }

    static String leftShift(String bits, int count) {
        return bits.substring(count) + bits.substring(0, count);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter 10-bit key: ");
        String key = sc.nextLine();

        if (!key.matches("[01]{10}")) {
            System.out.println("Invalid key. Enter exactly 10 binary bits.");
            sc.close();
            return;
        }

        String permutedKey = permute(key, P10);

        String left = permutedKey.substring(0, 5);
        String right = permutedKey.substring(5);

        left = leftShift(left, 1);
        right = leftShift(right, 1);

        String k1 = permute(left + right, P8);

        left = leftShift(left, 2);
        right = leftShift(right, 2);

        String k2 = permute(left + right, P8);

        System.out.println("Subkey K1: " + k1);
        System.out.println("Subkey K2: " + k2);

        sc.close();
    }
}