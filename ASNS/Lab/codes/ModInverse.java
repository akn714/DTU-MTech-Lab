public class ModInverse {
    public static void main(String args[]) {
        int a = 3;
        int m = 26;
        System.out.println("Modular Inverse of " + a + " under modulo " + m + " is " + modInverse(a, m));
    }
    
    static int modInverse(int a, int m) {
        a = a % m;
        for(int x=1;x<m;x++) {
            if((a * x) % m == 1) return x;
        }
        return 1;
    }
}