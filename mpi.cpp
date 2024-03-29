#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SIZE 1000000 // Maximum size of the array

// Function to swap two elements
void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function for Quicksort
int partition(int arr[], int low, int high)
{
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// Quicksort function
void quicksort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

// Parallel Quicksort using MPI
void parallel_quicksort(int arr[], int n, int rank, int size)
{
    int chunk_size = n / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? n : start + chunk_size;

    quicksort(arr, start, end - 1);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        int *temp = (int *)malloc(n * sizeof(int));
        int *recvcounts = (int *)malloc(size * sizeof(int));
        int *displs = (int *)malloc(size * sizeof(int));
        int offset = 0;

        for (int i = 0; i < size; i++)
        {
            int chunk_start = i * chunk_size;
            int chunk_end = (i == size - 1) ? n : chunk_start + chunk_size;
            recvcounts[i] = chunk_end - chunk_start;
            displs[i] = offset;
            offset += recvcounts[i];
        }

        MPI_Gatherv(arr, end - start, MPI_INT, temp, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        quicksort(temp, 0, n - 1);

        for (int i = 0; i < n; i++)
        {
            arr[i] = temp[i];
        }

        free(temp);
        free(recvcounts);
        free(displs);
    }
    else
    {
        MPI_Gatherv(arr, end - start, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Test case
    int arr[MAX_SIZE];
    int n = MAX_SIZE;
    srand(time(NULL));
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 1000000; // Generate random numbers between 0 and 999999
    }

    double start_time = MPI_Wtime();
    parallel_quicksort(arr, n, rank, size);
    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        printf("Execution time (MPI): %f seconds\n", end_time - start_time);
        // Uncomment the following line to print the sorted array
        // for (int i = 0; i < n; i++) printf("%d ", arr[i]);
    }

    MPI_Finalize();
    return 0;
}