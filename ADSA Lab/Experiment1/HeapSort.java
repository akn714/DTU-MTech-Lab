import java.util.Arrays;

public class HeapSort {
    public static void main(String[] args) {
        int[] arr = {6,2,1,7,5,4};
        
        // Displaying the unsorted array
        System.out.println("== UNSORTED ARRAY: " + Arrays.toString(arr) + " ==");
        
        // Sorting the array using merge sort
        heapSort(arr);

        // Displaying the sorted array
        System.out.println("== FINAL SORTED ARRAY IN DESCENDING ORDER: " + Arrays.toString(arr) + " ==");
    }

    static void heapSort(int[] arr) {
        int n = arr.length;

        for(int i=n/2-1;i>=0;i--) {
            heapify(arr, n, i);
        }

        for(int i=n-1;i>0;i--) {
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;

            heapify(arr, i, 0);
        }
    }

    static void heapify(int[] arr, int n, int i) {
        int smallest = i;
        int left = 2*i + 1;
        int right = 2*i + 2;

        if(left<n && arr[left]<arr[smallest]) smallest = left;
        if(right<n && arr[right]<arr[smallest]) smallest = right;

        if(smallest!=i) {
            int temp = arr[i];
            arr[i] = arr[smallest];
            arr[smallest] = temp;

            heapify(arr, n, smallest);
        }
    }
}