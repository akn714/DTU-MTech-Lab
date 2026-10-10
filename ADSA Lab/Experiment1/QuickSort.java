import java.util.Arrays;

public class QuickSort {
    public static void main(String[] args) {
        int[] arr = {23,4,7,8,2,3,5,7,2,3,8,9,2,3,4};

        // Displaying the unsorted array
        System.out.println("== UNSORTED ARRAY: " + Arrays.toString(arr) + " ==");
        
        // Sorting the array using merge sort
        quickSort(arr, arr.length-1, 0, arr.length-2);

        // Displaying the sorted array
        System.out.println("== FINAL SORTED ARRAY: " + Arrays.toString(arr) + " ==");
    }

    static void quickSort(int[] arr, int pivot, int start, int end) {
        if(end<start) return;

        int i=start-1, j=start;
        int temp;

        while(j<pivot) {
            if(arr[j]<arr[pivot]) {
                i++;
                temp = arr[j];
                arr[j] = arr[i];
                arr[i] = temp;
            }
            j++;
        }
        i++;
        temp = arr[pivot];
        arr[pivot] = arr[i];
        arr[i] = temp;

        quickSort(arr, pivot, i+1, end);
        quickSort(arr, i-1, start, i-2);
    }
}