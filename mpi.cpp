#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>

using namespace std;
using namespace std::chrono;

void exchange(int &x, int &y) {
    int temp = x;
    x = y;
    y = temp;
}

int divide(vector<int> &arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            exchange(arr[i], arr[j]);
        }
    }
    exchange(arr[i + 1], arr[high]);
    return i + 1;
}

void sort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int pivot_index = divide(arr, low, high);
        sort(arr, low, pivot_index - 1);
        sort(arr, pivot_index + 1, high);
    }
}

int main(int argc, char *argv[]) {
    const size_t arraySize = 1000000;
    vector<int> array(arraySize);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Fill the vector with random numbers
    srand(time(0) + rank);
    for (size_t i = 0; i < arraySize; i++) {
        array[i] = rand();
    }

    // Scatter the array among processes
    int chunkSize = arraySize / size;
    vector<int> localArray(chunkSize);
    MPI_Scatter(&array[0], chunkSize, MPI_INT, &localArray[0], chunkSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort the local array
    auto startTime = high_resolution_clock::now();
    sort(localArray, 0, localArray.size() - 1);
    auto endTime = high_resolution_clock::now();

    // Gather the sorted local arrays into the root process
    vector<int> sortedArray(arraySize);
    MPI_Gather(&localArray[0], chunkSize, MPI_INT, &sortedArray[0], chunkSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Display total time taken for sorting
    if (rank == 0) {
        auto elapsedTime = duration_cast<microseconds>(endTime - startTime);
        cout << "Parallel Sorting took " << elapsedTime.count() << " microseconds." << endl;
    }

    MPI_Finalize();
    return 0;
}
