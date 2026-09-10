import java.util.*;

public class MaximumNumberOfEvents {

    public static int maxEvents(int[][] events) {

        // Sort events by starting day
        Arrays.sort(events, (a, b) -> Integer.compare(a[0], b[0]));

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        int n = events.length;
        int i = 0;
        int day = 0;
        int attended = 0;

        while (i < n || !minHeap.isEmpty()) {

            // If no active events, jump to the next event's start day
            if (minHeap.isEmpty()) {
                day = Math.max(day, events[i][0]);
            }

            // Add all events that have started
            while (i < n && events[i][0] <= day) {
                minHeap.add(events[i][1]);
                i++;
            }

            // Remove events that have already expired
            while (!minHeap.isEmpty() && minHeap.peek() < day) {
                minHeap.poll();
            }

            // Attend the event that ends earliest
            if (!minHeap.isEmpty()) {
                minHeap.poll();
                attended++;
                day++;
            }
        }

        return attended;
    }

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of events: ");
        int n = sc.nextInt();

        int[][] events = new int[n][2];

        System.out.println("Enter start and end day of each event:");

        for (int i = 0; i < n; i++) {
            System.out.print("Event " + (i + 1) + ": ");
            events[i][0] = sc.nextInt();
            events[i][1] = sc.nextInt();
        }

        int result = maxEvents(events);

        System.out.println("Maximum number of events that can be attended: " + result);

        sc.close();
    }
}