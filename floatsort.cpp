#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>

#define ARRAY_LENGTH 50       // length of array to be sorted

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
    bool operator<(const PriorityInfo& rhs) const
    {
        return value > rhs.value;
    }
};

struct WriteBuffer
{
    int index;
    int* segmentLengths;
    float* data;
    WriteBuffer() {
        index = 0;
        writeNum = 0;
    }

    void init(int length, int numThreads) {
        data = new float[length];
        segmentLengths = new int[numThreads];
    }

    void finalizeWrite(int length) {
        segmentLengths[writeNum] = length;
        index += length;
        writeNum++;
    }

private:
    int writeNum;
};

float* sort(float *data, int length);
void multimerge(float* arrays[], const int lengths[], const int numArrays, float newArray[],
                                                                         const int newLength);

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
    float* sortedData = sort(random, ARRAY_LENGTH);
    time = omp_get_wtime() - time;

    std::cout << std::endl << "Sorted Data: " << std::endl;
    for (int i = 0; i < ARRAY_LENGTH; ++i) {
        std::cout << sortedData[i] << std::endl;
    }

    std::cout << "Press any key to continue...\n";

    float* test = new float[5];
    test[0] = 23; // start of first partition
    test[1] = 55;
    test[2] = 29;   // start of second partition
    test[3] = 102;
    test[4] = 20; // start of third and fourth partition. Third partition has no length

    int lengths[] = { 2, 2, 0, 1 };
    float sorted[5];
    float* arrays[] = { test, test + 2, test + 4, test + 4 };
    multimerge(arrays, lengths, 4, sorted, 5);
    for (int i = 0; i < 5; ++i)
        std::cout << sorted[i] << ", ";

    std::cin.get();
}

float* sort(float *data, int length) {
    // Initialize //////////////////////////////////////////////////////
    int          numThreads       = omp_get_max_threads();
    int          numThreadsSq     = numThreads * numThreads;
    int*         dataStartIndexes = new int[numThreads];
    int*         dataLengths      = new int[numThreads];
    int          segmentLength    = length / numThreads;
    float*       samples          = new float[numThreads];
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

        // this is just a test for debugging
#pragma omp critical 
        std::cout << "I'm Thread " << threadId << " and handle data from " << dataStartIndexes[threadId] <<
            " to " << dataStartIndexes[threadId] + dataLengths[threadId] - 1 << std::endl;

        // Quicksort ///////////////////////////////////////////////////
        // should each thread operate on the main array but only in its own secgment?
        // or should each thread copy its data segment to its own local private array?
        // either way, we should just find a library function to do this prob
        std::vector<float> dataVec(data + dataStart, data + dataStart + dataLength);
        std::sort(dataVec.begin(), dataVec.end());
#pragma omp critical 
        {
            std::cout << "Thread " << threadId << " sort:\n";
            for (int i = 0; i < dataLength; ++i)
                std::cout << dataVec[i] << std::endl;
        }
        
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
            std::cout << "The samples: " << std::endl;
            for (int i = 0; i < numThreadsSq; ++i)
                std::cout << samples[i] << std::endl;
            // Multimerge Samples //////////////////////////////////////
            int*    lengths       = new int[numThreads];
            float** arrays        = new float*[numThreads];
            float*  sortedSamples = new float[numThreadsSq];
            for (int i = 0; i < numThreads; ++i) {
                lengths[i] = numThreads;
                arrays[i]  = samples + i * numThreads;
            }
            multimerge(arrays, lengths, numThreads, sortedSamples, numThreadsSq);

            std::cout << "Sorted samples: " << std::endl;
            for (int i = 0; i < numThreadsSq; ++i)
                std::cout << sortedSamples[i] << std::endl;

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

            std::cout << "The pivots: " << std::endl;
            for (int i = 0; i <= numThreads; ++i)
                std::cout << pivots[i] << std::endl;
        }


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

#pragma omp critical
        {
            std::cout << "For thread " << threadId << ":\n";
            for (int i = 0; i < numThreads; ++i) {
                std::cout << " Partition " << i << " starts at index " << partitionStartIndices[i]
                    << " and is " << partitionLengths[i] << " long." << std::endl;
            }
        }
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

#pragma omp critical
        {
            std::cout << "For thread " << threadId << ":\n";
            for (int i = 0; i < dataInBuffs[threadId].index; ++i) {
                std::cout << dataInBuffs[threadId].data[i] << std::endl;
            }
        }

        float* dataPointer = dataInBuffs[threadId].data;
        float** arrays = new float*[numThreads];
        float*  sortedData = new float[dataInBuffs[threadId].index];
        for (int i = 0; i < numThreads; ++i) {
            arrays[i] = dataPointer;
            dataPointer += dataInBuffs[threadId].segmentLengths[i];
        }
        multimerge(arrays, dataInBuffs[threadId].segmentLengths, numThreads, sortedData, dataInBuffs[threadId].index);
        sortedDatas[threadId] = sortedData;

#pragma omp critical
        {
            std::cout << "For thread " << threadId << ":\n";
            for (int i = 0; i < dataInBuffs[threadId].index; ++i) {
                std::cout << sortedData[i] << std::endl;
            }
        }

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
    int* indexes       = new int[numArrays];       // current index into each array in arrays; parallel
                                                   //   with arrays
    int  newArrayIndex = 0;                        // current index into newArray
    std::priority_queue<PriorityInfo> curPriorities;    // priority queue of current smallest values from
                                                   //   each array in arrays

    // initialize indexes and insert into curPriorities the first value from each array in arrays
    for (int i = 0; i < numArrays; ++i) {
        indexes[i] = 1;
        curPriorities.push( PriorityInfo(arrays[i][0], i) );
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
