public class CaeserCipher {
    public static void main(String args[]) {
        String plaintext = "Hello";
        int key = 10;

        String ciphertext = encrypt(plaintext, key);

        System.out.println(ciphertext);
        System.out.println(decrypt(ciphertext, key));
    }

    static String encrypt(String pt, int key) {
        key = key%26;
        String ct = "";
        for(int i=0;i<pt.length();i++) {
            char ch = pt.charAt(i);
            if (ch >= 'A' && ch <= 'Z') {
                ct += (char) ('A' + (ch - 'A' + key) % 26);
            } else if (ch >= 'a' && ch <= 'z') {
                ct += (char) ('a' + (ch - 'a' + key) % 26);
            }
        }

        return ct;
    }

    static String decrypt(String ct, int key) {
        key = key%26;
        String pt = "";
        for(int i=0;i<ct.length();i++) {
            char ch = ct.charAt(i);
            if (ch >= 'A' && ch <= 'Z') {
                pt += (char) ('A' + (ch - 'A' - key + 26) % 26);
            } else if (ch >= 'a' && ch <= 'z') {
                pt += (char) ('a' + (ch - 'a' - key + 26) % 26);
            }
        }

        return pt;
    }
}