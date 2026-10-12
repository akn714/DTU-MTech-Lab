// Digital Signature Standard (DSS) using RSA based signature schema

import java.util.Scanner;

public class DSSUsingRSA {
    public static void main(String args[]) {
        Scanner sc = new Scanner(System.in);

        System.out.println("==== KEY GENERATION ====");
        int p = get("Enter P: ", sc);
        int q = get("Enter Q: ", sc);

        int N = p * q;
        int phi_n = (p - 1) * (q - 1);

        int[] keys = keyGeneration(phi_n);
        int public_key = keys[0];  // e - published, used by anyone to verify
        int private_key = keys[1]; // d - kept secret, used by the signer only
        System.out.println("Public Key  (e, N): (" + public_key + ", " + N + ")");
        System.out.println("Private Key (d, N): (" + private_key + ", " + N + ")");

        System.out.println();
        System.out.println("==== SIGNING (Sender) ====");
        System.out.print("Enter the message to be signed: ");
        sc.nextLine(); // consume the newline left behind by nextInt()
        String message = sc.nextLine();

        // The message itself is never signed, only its hash
        int hash = hash(message, N);
        int signature = modPow(hash, private_key, N); // S = H^d mod N
        System.out.println("Message: " + message);
        System.out.println("Hash (H): " + hash);
        System.out.println("Signature (S = H^d mod N): " + signature);
        System.out.println("Sender transmits the pair (message, signature)");

        System.out.println();
        System.out.println("==== VERIFICATION (Receiver) ====");
        int recovered_hash = modPow(signature, public_key, N); // H' = S^e mod N
        int receiver_hash = hash(message, N);
        System.out.println("Hash of received message (H): " + receiver_hash);
        System.out.println("Hash recovered from signature (H' = S^e mod N): " + recovered_hash);
        System.out.println("Signature is " + (recovered_hash == receiver_hash
                ? "VALID - message is authentic"
                : "INVALID - message or signature was altered"));

        System.out.println();
        System.out.println("==== VERIFICATION OF A TAMPERED MESSAGE ====");
        String tampered_message = "Hacked: " + message;
        System.out.println("Tampered message: " + tampered_message);
        System.out.println("Signature is " + (modPow(signature, public_key, N) == hash(tampered_message, N)
                ? "VALID - message is authentic"
                : "INVALID - message or signature was altered"));

        sc.close();
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

    // Extended Euclidean Algorithm to find d such that (a * d) % m == 1
    static int modInverse(int a, int m) {
        long old_r = a, r = m;
        long old_s = 1, s = 0;

        while (r != 0) {
            long quotient = old_r / r;

            long temp = old_r - quotient * r;
            old_r = r;
            r = temp;

            temp = old_s - quotient * s;
            old_s = s;
            s = temp;
        }

        if (old_r != 1) return 1; // a and m are not coprime, inverse does not exist
        return (int) ((old_s % m + m) % m);
    }

    // Euclidean Algorithm to get gcd
    public static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // Returns the smallest number coprime to phi_n, used as the public exponent
    public static int getCoprime(int phi_n) {
        for(int i=2;i<phi_n;i++) {
            if(gcd(i, phi_n) == 1) return i;
        }

        return 1;
    }

    // Hash of the message, reduced modulo N so that it fits the RSA modulus
    static int hash(String message, int N) {
        int h = 7; // non zero seed so that even an empty message hashes to a valid value
        for (int i = 0; i < message.length(); i++) {
            h = (h * 31 + message.charAt(i)) % N;
        }
        return h;
    }

    // Fast modular exponentiation, (base ^ exponent) % mod
    static int modPow(int base, int exponent, int mod) {
        long result = 1;
        long b = base % mod;

        while (exponent > 0) {
            if (exponent % 2 == 1) result = (result * b) % mod;
            b = (b * b) % mod;
            exponent /= 2;
        }

        return (int) result;
    }
}
