import java.util.Scanner;
import java.security.SecureRandom;

public class ElGamal {

    static long modPow(long base, long exp, long mod) {
        long result = 1;
        base %= mod;

        while (exp > 0) {
            if ((exp & 1) == 1) {
                result = result * base % mod;
            }

            base = base * base % mod;
            exp >>= 1;
        }

        return result;
    }

    static long modInverse(long a, long p) {
        long t = 0, newT = 1;
        long r = p, newR = a % p;

        while (newR != 0) {
            long q = r / newR;

            long temp = t - q * newT;
            t = newT;
            newT = temp;

            temp = r - q * newR;
            r = newR;
            newR = temp;
        }

        if (r != 1) {
            throw new ArithmeticException("Inverse does not exist");
        }

        return (t % p + p) % p;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        SecureRandom random = new SecureRandom();

        System.out.print("Enter prime number p: ");
        long p = sc.nextLong();

        System.out.print("Enter generator g: ");
        long g = sc.nextLong();

        System.out.print("Enter private key x: ");
        long x = sc.nextLong();

        if (p <= 2 || g <= 1 || g >= p || x < 1 || x > p - 2) {
            System.out.println("Invalid parameters.");
            sc.close();
            return;
        }

        long y = modPow(g, x, p);

        System.out.println("Public Key (p, g, y): ("
                + p + ", " + g + ", " + y + ")");
        System.out.println("Private Key x: " + x);

        System.out.print("Enter message M (0 <= M < p): ");
        long m = sc.nextLong();

        if (m < 0 || m >= p) {
            System.out.println("Invalid message.");
            sc.close();
            return;
        }

        long k = 1 + random.nextInt((int) (p - 2));

        long c1 = modPow(g, k, p);
        long c2 = (m * modPow(y, k, p)) % p;

        System.out.println("Ciphertext C1: " + c1);
        System.out.println("Ciphertext C2: " + c2);

        long sharedSecret = modPow(c1, x, p);
        long inverse = modInverse(sharedSecret, p);

        long decrypted = (c2 * inverse) % p;

        System.out.println("Decrypted message: " + decrypted);

        sc.close();
    }
}