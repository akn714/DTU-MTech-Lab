/*
LAB 1: Linear and Binary Search

Problem 1: Upper and Lower Bound
Problem 2: First and Last Occurrence
*/



public class Search {
    public static void main(String[] args) {
        int[] arr = {1,2,4,5,6,6,6,6,8,10,15,16};
        int target = 7;

        System.out.println("Linear Search: " + linearSearch(arr, target));
        System.out.println("Binary Search: " + binarySearch(arr, target));

        System.out.println("First Occurance: " + firstOccurance(arr, 6));
        System.out.println("Last Occurance: " + lastOccurance(arr, 6));
    }

    static int linearSearch(int[] arr, int target) {
        for(int i=0;i<arr.length;i++) {
            if(arr[i]==target) return i;
        }
        return -1;
    }

    static int binarySearch(int[] arr, int target) {
        int start = 0;
        int end = arr.length - 1;
        int mid;

        while(start<=end) {
            mid = start + (end - start) / 2;

            if(arr[mid]==target) return mid;
            else if(arr[mid]>target) end = mid - 1;
            else start = mid + 1;
        }

        return -1;
    }

    static int firstOccurance(int[] arr, int target) {
        int start = 0;
        int end = arr.length - 1;
        int mid;

        while(start<end) {
            mid = start + (end - start) / 2;

            if(arr[mid]<target) {
                start = mid + 1;
            }
            else {
                end = mid;
            }
        }

        return start;
    }

    static int lastOccurance(int[] arr, int target) {
        int start = 0;
        int end = arr.length - 1;
        int mid;

        while(start<end) {
            mid = start + (end - start) / 2;

            if(arr[mid]>target) {
                end = mid - 1;
            }
            else {
                start = mid + 1;
            }
        }

        return end;
    }
}