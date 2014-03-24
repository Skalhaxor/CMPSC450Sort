// Created by: Alexander Anderson, Jason Killian
// CMPSC 450 Homework 2
// Date: March 23, 2014

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

// define this to get lots of trace output from the parallel sort algorithm
//#define DEBUG_TRACE

// define this to have the program run std::sort and the parallel sort and make sure they
//produce the right output
//#define CHECK_ALG

#define NUM_ITER 10           // number of times to run the algorithm for testing
#define NUM_ARG  4            // number of arguments being passed on command line

// structure to put in the multimerge priority queue
struct PriorityInfo
{
    float value;              // value to prioritize
    int   arrayIndex;         // index to arrays that contain info on arrays that are being multimerged

    // constructor
    PriorityInfo(float val, int index) :
        value(val),
        arrayIndex(index)
    {}

    // overload the < operator so the priority queue will put minimum value on top
    bool operator<(const PriorityInfo& rhs) const {
        return value > rhs.value;
    }
};

struct WriteBuffer
{
    int    index;
    int*   segmentLengths;
    float* data;

    WriteBuffer() {
        index    = 0;
        writeNum = 0;
    }

    void init(int length, int numThreads) {
        data           = new float[length];
        segmentLengths = new int[numThreads];
    }

    void finalizeWrite(int length) {
        segmentLengths[writeNum] = length;
        index += length;
        ++writeNum;
    }

    ~WriteBuffer() {
        delete[] data;
        delete[] segmentLengths;
    }

private:
    int writeNum;
};

int    gen_input(float *A, int n, int input_type);
float* sort(float* data, int length);
float* standardSort(float* data, int length);
void   multimerge(float* arrays[], const int lengths[], const int numArrays, float newArray[],
                                                                         const int newLength);

int main(int argc, char* argv[])
{
    if (argc != NUM_ARG) {
        fprintf(stderr, "%s <n> <input_type> <alg_type>\n", argv[0]);
        fprintf(stderr, "input_type 0: uniform random\n");
        fprintf(stderr, "           1: already sorted\n");
        fprintf(stderr, "           2: almost sorted\n");
        fprintf(stderr, "           3: single unique value\n");
        fprintf(stderr, "           4: sorted in reverse\n");
        fprintf(stderr, "alg_type   0: use C++ std::sort\n");
        fprintf(stderr, "           1: use parallel sort\n");
        exit(1);
    }

    int n;
    n = atoi(argv[1]);
    assert(n > 0);
    assert(n <= 1000000000);

    float* A;
    A = new float[n];
    assert(A != 0);

    int input_type = atoi(argv[2]);
    assert(input_type >= 0);
    assert(input_type <= 4);

    int alg_type = atoi(argv[3]);
    assert((alg_type == 0) || (alg_type == 1));

    std::cout << "Number of Elements: " << n << "\n";
    switch (input_type) {
    case 0: std::cout << "Input Type: Uniform Random\n"; break;
    case 1: std::cout << "Input Type: Sorted\n"; break;
    case 2: std::cout << "Input Type: Almost Sorted\n"; break;
    case 3: std::cout << "Input Type: Single Value\n"; break;
    case 4: std::cout << "Input Type: Reversed\n"; break;
    }
    switch (alg_type) {
    case 0: std::cout << "Alg Type: std::sort\n"; break;
    case 1: std::cout << "Alg Type: Parallel Sort on " << omp_get_max_threads() << " threads\n"; break;
    }
    
    for (int i = 0; i < NUM_ITER; ++i) {
        gen_input(A, n, input_type);

#ifdef CHECK_ALG
        float* r1 = standardSort(A, n);
        float* r2 = sort(A, n);
        for (int j = 0; j<n; ++j) {
            assert(r1[j] == r2[j]);
        }
        delete[] r1;
        delete[] r2;
#endif //CHECK_ALG
  
        float* result;
        int    printDigs = n > 5 ? 5 : n;

        double time = omp_get_wtime();
        if (alg_type == 0) {
            result = standardSort(A, n);
        }
        else /*if (alg_type == 1)*/ {
            result = sort(A, n);
        }
        time = omp_get_wtime() - time;
        
        printf("Iteration %d: %9.3lfs  First numbers: (", i + 1, time);
        for (int j = 0; j < printDigs; ++j) {
            printf(" %.3f", result[j]);
        }
        printf(")\n");

        delete[] result;
    }

    return 0;
}

float* sort(float *data, int length) {
    // Initialize //////////////////////////////////////////////////////
    int          numThreads       = omp_get_max_threads();
    int          numThreadsSq     = numThreads * numThreads;
    int*         dataStartIndexes = new int[numThreads];
    int*         dataLengths      = new int[numThreads];
    int          segmentLength    = length / numThreads;
    float*       samples          = new float[numThreadsSq];
    float*       pivots           = new float[numThreads + 1];
    WriteBuffer* dataInBuffs      = new WriteBuffer[numThreads];
    float**      sortedDatas      = new float*[numThreads];
    float*       finalData        = new float[length];

    for (int i = 0; i < numThreads; ++i) {
        dataInBuffs[i].init(length, numThreads);
    }

    // Partition Data //////////////////////////////////////////////////
    for (int i = 0; i < numThreads - 1; ++i) {
        dataLengths[i]      = segmentLength;
        dataStartIndexes[i] = segmentLength*i;
    }

    // make the last thread handle the left over data (will be >= dataLength and < 2*dataLength)
    dataStartIndexes[numThreads - 1] = segmentLength * (numThreads - 1);
    dataLengths[numThreads - 1]      = length - dataStartIndexes[numThreads - 1];
    
    // Startup Threads /////////////////////////////////////////////////
    // If a SHARED variable in a parallel region is read by the threads executing the region,
    // but not written to by any of the threads, then specify that variable to be FIRSTPRIVATE instead
    // of SHARED. This avoids accessing the variable by dereferencing a pointer, and avoids cache
    // conflicts. 
#pragma omp parallel firstprivate(dataStartIndexes, dataLengths, numThreads, numThreadsSq)
    {
        int threadId   = omp_get_thread_num();
        int dataStart  = dataStartIndexes[threadId];
        int dataLength = dataLengths[threadId];


#ifdef DEBUG_TRACE
#pragma omp critical 
        std::cout << "I'm Thread " << threadId << " and handle data from " << dataStartIndexes[threadId] <<
            " to " << dataStartIndexes[threadId] + dataLengths[threadId] - 1 << std::endl;
#endif // DEBUG_TRACE


        // Quicksort ///////////////////////////////////////////////////
        // should each thread operate on the main array but only in its own secgment?
        // or should each thread copy its data segment to its own local private array?
        // either way, we should just find a library function to do this prob
        std::vector<float> dataVec(data + dataStart, data + dataStart + dataLength);
        std::sort(dataVec.begin(), dataVec.end());


#ifdef DEBUG_TRACE
#pragma omp critical 
        {
            std::cout << "Thread " << threadId << " sort:\n";
            for (int i = 0; i < dataLength; ++i)
                std::cout << dataVec[i] << std::endl;
        }
#endif // DEBUG_TRACE

        
        // Choose Sample Data //////////////////////////////////////////
        // choose numThreads amount from each thread's data; has to be non-random
        int sampleStart = numThreads * threadId;
        int interval    = dataLength / numThreads;
        for (int i = 0; i < numThreads; ++i) {
            float sample = dataVec[i*interval];
            samples[sampleStart + i] = sample;
        }

        // wait until all threads have written samples before continuing
#pragma omp barrier

#pragma omp single
        {

#ifdef DEBUG_TRACE
            std::cout << "The samples: " << std::endl;
            for (int i = 0; i < numThreadsSq; ++i)
                std::cout << samples[i] << std::endl;
#endif // DEBUG_TRACE


            // Multimerge Samples //////////////////////////////////////
            int*    lengths       = new int[numThreads];
            float** arrays        = new float*[numThreads];
            float*  sortedSamples = new float[numThreadsSq];
            for (int i = 0; i < numThreads; ++i) {
                lengths[i] = numThreads;
                arrays[i]  = samples + i * numThreads;
            }
            multimerge(arrays, lengths, numThreads, sortedSamples, numThreadsSq);
            delete[] arrays;
            delete[] lengths;

#ifdef DEBUG_TRACE
            std::cout << "Sorted samples: " << std::endl;
            for (int i = 0; i < numThreadsSq; ++i)
                std::cout << sortedSamples[i] << std::endl;
#endif // DEBUG_TRACE


            // Choose Pivots ///////////////////////////////////////////
            // choose p - 1 pivots from the merged samples, non-randomly
            interval = numThreads;
            for (int i = 0; i < numThreads; ++i) {
                float pivot = sortedSamples[numThreads * i];
                pivots[i] = pivot;
            }
            // largest value of samples may not be largest max value in data
            // so make the endpoint of the last partition the largest possible float value
            pivots[numThreads] = FLT_MAX;


#ifdef DEBUG_TRACE
            std::cout << "The pivots: " << std::endl;
            for (int i = 0; i <= numThreads; ++i)
                std::cout << pivots[i] << std::endl;
#endif // DEBUG_TRACE


        } // implicit barrier


        // Partion by Pivots ///////////////////////////////////////////
        // each thread partitions it's data into classes based on the pivots
        // Our pivots look like this:
        //   | |    |     |      |
        //   xxxxxxxxxxxxxxxxxxxxx
        std::vector<int> partitionStartIndices(numThreads);
        std::vector<int> partitionLengths(numThreads);
        int  currPartition         = 0;

        partitionStartIndices[0] = 0;
        partitionLengths[0]      = 0;

        for (int i = 0; i < dataLength; ++i) {
            float dataPoint = dataVec[i];
 
            if (pivots[currPartition] <= dataPoint && dataPoint < pivots[currPartition+1] ) {
                partitionLengths[currPartition]++;
            }
            else {
                currPartition++;
                partitionStartIndices[currPartition] = i;
                partitionLengths[currPartition] = 0;
                // the next element might not be in the next partition, so reclassify it
                i--;
            }
        }


#ifdef DEBUG_TRACE
#pragma omp critical
        {
            std::cout << "For thread " << threadId << ":\n";
            for (int i = 0; i < numThreads; ++i) {
                std::cout << " Partition " << i << " starts at index " << partitionStartIndices[i]
                    << " and is " << partitionLengths[i] << " long." << std::endl;
            }
        }
#endif // DEBUG_TRACE


#pragma omp barrier

        // Multimerge Classes //////////////////////////////// //////////
        // the i-th thread multimerges all the i-th classes of each thread's partitioned data
#pragma omp critical
        {
            //write data from each partition to corresponding thread
            for (int i = 0; i < numThreads; ++i) {
                WriteBuffer* buf = dataInBuffs + i;
                int dataToWriteLength = partitionLengths[i];
                int dataStartIndex = partitionStartIndices[i];
                for (int j = 0; j < dataToWriteLength; ++j) {
                    buf->data[j + buf->index] = dataVec[j + dataStartIndex];
                }
                buf->finalizeWrite(dataToWriteLength);
            }
        }

#pragma omp barrier


#ifdef DEBUG_TRACE
#pragma omp critical
        {
            std::cout << "For thread " << threadId << ":\n";
            for (int i = 0; i < dataInBuffs[threadId].index; ++i) {
                std::cout << dataInBuffs[threadId].data[i] << std::endl;
            }
        }
#endif // DEBUG_TRACE


        // each thread sorts the data the other threads passed to it
        float* dataPointer = dataInBuffs[threadId].data;
        float** arrays = new float*[numThreads];
        float*  sortedData = new float[dataInBuffs[threadId].index];
        for (int i = 0; i < numThreads; ++i) {
            arrays[i] = dataPointer;
            dataPointer += dataInBuffs[threadId].segmentLengths[i];
        }
        multimerge(arrays, dataInBuffs[threadId].segmentLengths, numThreads, sortedData, dataInBuffs[threadId].index);
        sortedDatas[threadId] = sortedData;
        delete[] arrays;

#ifdef DEBUG_TRACE
#pragma omp critical
        {
            std::cout << "For thread " << threadId << ":\n";
            for (int i = 0; i < dataInBuffs[threadId].index; ++i) {
                std::cout << sortedData[i] << std::endl;
            }
        }
#endif // DEBUG_TRACE


#pragma omp barrier

        // Consolidate /////////////////////////////////////////////////
#pragma omp single
        {
            int pos = 0;
            for (int i = 0; i < numThreads; ++i) {
                for (int j = 0; j < dataInBuffs[i].index; ++j) {
                    finalData[pos + j] = sortedDatas[i][j];
                }
                pos += dataInBuffs[i].index;
            }
        }
    } // end omp parallel

    // clean up allocated memory
    delete[] dataStartIndexes;
    delete[] dataLengths;
    delete[] samples;
    delete[] pivots;
    delete[] dataInBuffs;
    for (int i = 0; i < numThreads; ++i) {
        delete[] sortedDatas[i];
    }
    delete[] sortedDatas;

    return finalData;
}

// takes in multiple arrays of already sorted data and sorts them together into one new array
//
// arrays    [in]  - array of arrays to sort; they must all be already sorted
// lengths   [in]  - lengths of each array to sort; parallel with arrays
// numArrays [in]  - the number of arrays being sorted (length of arrays)
// newArray  [out] - the array containing all data sorted together; will be returned
// newLength [in]  - length of newArray
void multimerge(float* arrays[], const int lengths[], const int numArrays, float newArray[],
                                                                         const int newLength) {
    int* indexes       = new int[numArrays];          // current index into each array in arrays;
                                                      //   parallel with arrays
    int  newArrayIndex = 0;                           // current index into newArray
    std::priority_queue<PriorityInfo> curPriorities;  // priority queue of current smallest values from
                                                      //   each array in arrays

    // initialize indexes and insert into curPriorities the first value from each array in arrays
    for (int i = 0; i < numArrays; ++i) {
        if (lengths[i] > 0) {
            indexes[i] = 1;
            curPriorities.push( PriorityInfo(arrays[i][0], i) );
        }
    }

    // pop the top of the priority queue and add that value to newArray; then add to curPriorities the
    //   next value in the array that the popped value came from
    while (!curPriorities.empty() && newArrayIndex < newLength) {
        PriorityInfo cur = curPriorities.top();     // save the element at the top of curPriorities

        curPriorities.pop();

        // add the popped value to newArray
        newArray[newArrayIndex] = cur.value;
        ++newArrayIndex;

        // as long as there are still values in the array that the popped value came from, add the next
        //   element to curPriorities
        if (indexes[cur.arrayIndex] < lengths[cur.arrayIndex]) {
            curPriorities.push( PriorityInfo(arrays[cur.arrayIndex][ indexes[cur.arrayIndex] ],
                                                                                 cur.arrayIndex) );
            ++indexes[cur.arrayIndex];
        }
    }

    delete[] indexes;
}

float* standardSort(float* data, int length) {
    std::vector<float> dataVec(data, data + length);
    std::sort(dataVec.begin(), dataVec.end());

    float* result = new float[dataVec.size()];
    std::copy(dataVec.begin(), dataVec.end(), result);

    return result;
}

/* generate different inputs for testing sort */
int gen_input(float *A, int n, int input_type) {
    int i;

    if (input_type == 0) {                      /* uniform random values */
        srand(123);

        for (i = 0; i<n; i++) {
            A[i] = ((float)rand()) / ((float)RAND_MAX) * 10000;
        }
    }
    else if (input_type == 1) {                 /* sorted values */
        for (i = 0; i<n; i++) {
            A[i] = (float)i;
        }
    }
    else if (input_type == 2) {                 /* almost sorted */
        for (i = 0; i<n; i++) {
            A[i] = (float)i;
        }

        /* do a few shuffles */
        int num_shuffles = (n / 100) + 1;
        srand(1234);
        for (i = 0; i<num_shuffles; i++) {
            int j = (rand() % n);
            int k = (rand() % n);

            /* swap A[j] and A[k] */
            float tmpval = A[j];
            A[j] = A[k];
            A[k] = tmpval;
        }
    }
    else if (input_type == 3) {                 /* array with single unique value */
        for (i = 0; i<n; i++) {
            A[i] = 1.0;
        }
    }
    else {                                      /* sorted in reverse */
        for (i = 0; i<n; i++) {
            A[i] = (float)(n + 1.0 - i);
        }
    }

    return 0;

}
