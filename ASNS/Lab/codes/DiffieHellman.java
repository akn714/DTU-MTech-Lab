import java.util.Scanner;

public class DiffieHellman {
    public static void main(String args[]) {
        int p = get("Enter prime number p: ");
        int g = get("Enter generator g: ");

        int a = get("Enter private key of alice: ");
        int b = get("Enter private key of bob: ");
        
        // Generating Public Keys
        int A = getMod(g, a, p);
        int B = getMod(g, b, p);

        System.out.println("Public Key A: " + A);
        System.out.println("Public Key B: " + B);

        // Generating Secret Key
        int secretKey1 = getMod(B, a, p); // Alice generating secret key using Bob's public key
        int secretKey2 = getMod(A, b, p); // Bob generating secret key using Alice's public key

        System.out.println("Alice has Secret key 1: " + secretKey1);
        System.out.println("Bob has Secret key 2: " + secretKey2);
        System.out.println(secretKey1==secretKey2);
    }

    static int get(String s) {
        Scanner sc = new Scanner(System.in);
        
        System.out.print(s);
        int x = sc.nextInt();
        return x;
    }

    static int getMod(int base, int exp, int MOD) {
        long result = 1;
        long b = base % MOD;

        while (exp > 0) {
            if ((exp & 1) == 1) {
                result = (result * b) % MOD;
            }

            b = (b * b) % MOD;
            exp >>= 1;
        }

        return (int) result;
    }

}