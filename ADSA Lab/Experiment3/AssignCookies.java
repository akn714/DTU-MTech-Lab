import java.util.*;

public class AssignCookies {
    public static int findContentChildren(int[] greed, int[] cookies) {
        Arrays.sort(greed);
        Arrays.sort(cookies);

        int child = 0;
        int cookie = 0;

        while (child < greed.length && cookie < cookies.length) {
            if (cookies[cookie] >= greed[child]) child++;
            cookie++;
        }

        return child;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter number of children: ");
        int n = sc.nextInt();
        int[] greed = new int[n];

        System.out.println("Enter greed factor of each child:");
        for (int i = 0; i < n; i++) {
            greed[i] = sc.nextInt();
        }

        System.out.print("Enter number of cookies: ");
        int m = sc.nextInt();
        int[] cookies = new int[m];

        System.out.println("Enter cookie sizes:");
        for (int i = 0; i < m; i++) {
            cookies[i] = sc.nextInt();
        }

        int result = findContentChildren(greed, cookies);
        System.out.println("Maximum number of content children: " + result);

        sc.close();
    }
}