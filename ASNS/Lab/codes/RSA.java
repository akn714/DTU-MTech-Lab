import java.util.Arrays;
import java.util.Scanner;

public class RSA {
    public static void main(String args[]) {
        Scanner sc = new Scanner(System.in);

        System.out.println("==== KEY GENERATION ====");
        int p = get("Enter P: ", sc);
        int q = get("Enter Q: ", sc);

        int N = p*q;
        int phi_n = (p-1) * (q-1);
        
        int[] keys = keyGeneration(phi_n);
        System.out.println("Public Key: " + keys[0]);
        System.out.println("Private Key: " + keys[1]);


        System.out.println();
        System.out.println("==== Encryption ====");
        System.out.print("Enter the number to be encrypted (0 ≤ message < " + N + "): ");
        int message = sc.nextInt();
        int encrypted_message = (int) (Math.pow(message, keys[0]) % N);
        System.out.println("Encrypted message: " + encrypted_message);

        System.out.println();
        System.out.println("==== Decryption ====");
        int decrypted_message = (int) (Math.pow(encrypted_message, keys[1]) % N);
        System.out.println("Decrypted message: " + decrypted_message);
    }

    static int get(String s, Scanner sc) {
        System.out.print(s);
        int x = sc.nextInt();
        return x;
    }

    // generating public key and private key
    static int[] keyGeneration(int phi_n) {
        int public_key = getCoprime(phi_n);
        int private_key = modInverse(public_key, phi_n);
        return new int[] {public_key, private_key};
    }
    
    static int modInverse(int a, int m) {
        a = a % m;
        for(int x=1;x<m;x++) {
            if((a * x) % m == 1) return x;
        }
        return 1;
    }

    // Euclidean Algorithm to get gcd
    public static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // Returns true if the numbers are coprime, false otherwise
    public static int getCoprime(int phi_n) {
        for(int i=2;i<phi_n;i++) {
            if(gcd(i, phi_n) == 1) return i;
        }

        return 1;
    }
}