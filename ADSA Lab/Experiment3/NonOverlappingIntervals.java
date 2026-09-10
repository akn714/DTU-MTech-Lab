import java.util.*;

public class NonOverlappingIntervals {
    public static int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0) {
            return 0;
        }

        // Sort by ending time
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[1], b[1]));

        int selected = 1;
        int lastEnd = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {
            // Non-overlapping interval
            if (intervals[i][0] >= lastEnd) {
                selected++;
                lastEnd = intervals[i][1];
            }
        }

        // Number of intervals that must be removed
        return intervals.length - selected;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter number of intervals: ");
        int n = sc.nextInt();
        int[][] intervals = new int[n][2];

        System.out.println("Enter start and end time of each interval:");
        for (int i = 0; i < n; i++) {
            System.out.print("Interval " + (i + 1) + ": ");
            intervals[i][0] = sc.nextInt();
            intervals[i][1] = sc.nextInt();
        }

        int result = eraseOverlapIntervals(intervals);
        System.out.println("Minimum number of intervals to remove: " + result);

        sc.close();
    }
}