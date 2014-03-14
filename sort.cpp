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

	int  numThreads       = omp_get_max_threads();
	int* dataStartIndexes = new int[numThreads];
	int* dataLengths      = new int[numThreads];
	int  dataLength       = length / numThreads;
	for (int i = 0; i < numThreads - 1; ++i) {
		dataLengths[i]      = dataLength;
		dataStartIndexes[i] = dataLength*i;
	}

	// make the last thread handle the rest of the data
	// (could be a few more if the data length isn't divisible by
	// the number of threads
	dataStartIndexes[numThreads - 1] = dataLength * (numThreads - 1);
	dataLengths[numThreads - 1]      = length - dataStartIndexes[numThreads - 1];
	
	// partition data
	// If a SHARED variable in a parallel region is read by the threads executing the region,
	// but not written to by any of the threads, then specify that variable to be FIRSTPRIVATE instead of SHARED.
	// This avoids accessing the variable by dereferencing a pointer, and avoids cache conflicts. 
#pragma omp parallel firstprivate(dataStartIndexes, dataLengths)
	{
		int threadId = omp_get_thread_num();
#pragma omp critical 
		cout << "I'm Thread " << threadId << " and handle data from " << dataStartIndexes[threadId] <<
			" to " << dataStartIndexes[threadId] + dataLengths[threadId] - 1 << endl;



	}
}