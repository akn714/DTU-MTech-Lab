import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Scanner;

public class SHA512 {
    public static void main(String[] args)
            throws NoSuchAlgorithmException {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter the message: ");
        String message = sc.nextLine();

        MessageDigest md = MessageDigest.getInstance("SHA-512");
        byte[] hash = md.digest(message.getBytes());

        StringBuilder hex = new StringBuilder();

        for (byte b : hash) {
            hex.append(String.format("%02x", b & 0xff));
        }

        System.out.println("SHA-512 Hash: " + hex);

        sc.close();
    }
}