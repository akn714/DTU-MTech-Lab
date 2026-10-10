import java.util.ArrayList;
import java.util.Arrays;

public class MergeSort {
    public static void main(String[] args) {
        int[] arr = {6,2,1,7,5,4};

        // Displaying the unsorted array
        System.out.println("== UNSORTED ARRAY: " + Arrays.toString(arr) + " ==");
        
        // Sorting the array using merge sort
        mergeSort(arr, 0, arr.length-1);

        // Displaying the sorted array
        System.out.println("== FINAL SORTED ARRAY IN DESCENDING ORDER: " + Arrays.toString(arr) + " ==");
    }

    static void mergeSort(int[] arr, int start, int end) {
        // return if there is only one element left
        if(start==end) return;

        // Dividing array in 2
        int mid = start + (end - start) / 2;
        mergeSort(arr, start, mid);
        mergeSort(arr, mid + 1, end);

        // Merging the 2 arrays
        mergeArray(arr, start, mid, mid + 1, end);
    }

    // Function to merge the array in sorted order
    static void mergeArray(int[] arr, int start1, int end1, int start2, int end2) {
        int[] sorted = new int[end2-start1+1];
        int start = start1, end = end2, i = 0;

        // System.out.println("----------------------------------");
        // System.out.println("Array before merging: " + Arrays.toString(arr));
        // System.out.println("-- Unsorted subarray: " + Arrays.toString(sorted));

        while(start1<=end1 || start2<=end2) {
            if(start1>end1) {
                sorted[i++] = arr[start2++];
            }
            else if (start2>end2) {
                sorted[i++] = arr[start1++];
            }
            else {
                if(arr[start1]>arr[start2]) sorted[i++] = arr[start1++];
                else sorted[i++] = arr[start2++];
            }
        }

        int k = 0;
        for(int j=start;j<=end;j++) {
            arr[j] = sorted[k++];
        }
        
        // System.out.println("-- Sorted subarray: " + Arrays.toString(sorted));
        // System.out.println("Array after merging: " + Arrays.toString(arr));
        // System.out.println("----------------------------------");
    }
}