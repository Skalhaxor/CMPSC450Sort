#include <iostream>
#include <omp.h>

using namespace std;

#define ARRAY_LENGTH 50

void sort(float *data, int length);

int main()
{
	/*float* sortedArray = new float[ARRAY_LENGTH];
	for (int i = 0; i < ARRAY_LENGTH; ++i) {
		sortedArray[i] = i;
	}

	float* backwards = new float[ARRAY_LENGTH];
	for (int i = 0; i < ARRAY_LENGTH; ++i) {
		backwards[i] = ARRAY_LENGTH-i;
	}*/

	float* random = new float[ARRAY_LENGTH];
	for (int i = 0; i < ARRAY_LENGTH; ++i) {
		random[i] = (float)rand();
	}

	double time = omp_get_wtime();
	sort(random, ARRAY_LENGTH);
	time = omp_get_wtime() - time;

	cin.get();
}

void sort(float *data, int length) {
    // Initialize //////////////////////////////////////////////////////
	int  numThreads       = omp_get_max_threads();
	int* dataStartIndexes = new int[numThreads];
	int* dataLengths      = new int[numThreads];
	int  dataLength       = length / numThreads;

    // Partition Data //////////////////////////////////////////////////
	for (int i = 0; i < numThreads - 1; ++i) {
		dataLengths[i]      = dataLength;
		dataStartIndexes[i] = dataLength*i;
	}

	// make the last thread handle the left over data
    // (will be >= dataLength and < 2*dataLength)
	dataStartIndexes[numThreads - 1] = dataLength * (numThreads - 1);
	dataLengths[numThreads - 1]      = length - dataStartIndexes[numThreads - 1];
	
    // Startup Threads /////////////////////////////////////////////////
	// If a SHARED variable in a parallel region is read by the threads executing the region,
	// but not written to by any of the threads, then specify that variable to be FIRSTPRIVATE instead of SHARED.
	// This avoids accessing the variable by dereferencing a pointer, and avoids cache conflicts. 
#pragma omp parallel firstprivate(dataStartIndexes, dataLengths)
	{
        // this is just a test
		int threadId = omp_get_thread_num();
#pragma omp critical 
		cout << "I'm Thread " << threadId << " and handle data from " << dataStartIndexes[threadId] <<
			" to " << dataStartIndexes[threadId] + dataLengths[threadId] - 1 << endl;

        // Quicksort ///////////////////////////////////////////////////


        // Choose Sample Data //////////////////////////////////////////
        // choose numThreads amount from each thread's data; has to be non-random

#pragma omp single
        {
            // Multimerge Samples //////////////////////////////////////


            // Choose Pivots ///////////////////////////////////////////
            // choose p - 1 pivots from the merged samples, non-randomly
        }


        // Partion by Pivots ///////////////////////////////////////////
        // each thread partitions it's data into classes based on the pivots


        // Multimerge Classes //////////////////////////////////////////
        // the i-th thread multimerges all the i-th classes of each thread's partitioned data


        // Consolidate /////////////////////////////////////////////////
	}
}
