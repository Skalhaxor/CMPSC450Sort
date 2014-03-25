// Created by: Alexander Anderson, Jason Killian
// CMPSC 450 Homework 2
// Date: March 23, 2014

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <map>
#include <algorithm>
#include <omp.h>
#include <vector>

// define this to get lots of trace output from the parallel sort algorithm
//#define DEBUG_TRACE

#define NUM_ITER      10           // number of times to run the algorithm for testing
#define NUM_ARG       4            // number of arguments being passed on command line

int find_uniq_stl_map(const char **str_array, const int num_strings);
int find_uniq_stl_sort(const char **str_array, const int num_strings); 
int parallel_sort(std::string* data, int length);

int main(int argc, char* argv[])
{
    if (argc != NUM_ARG) {
        fprintf(stderr, "%s <input file> <n> <alg_type>\n", argv[0]);
        fprintf(stderr, "alg_type 0: use STL sort, then find unique strings\n");
        fprintf(stderr, "         1: use STL map\n");
        fprintf(stderr, "         2: use parallel sort\n");

        exit(1);
    }

    char* inputFile = argv[1];
    std::ifstream stream;
    stream.open(inputFile);

    int n;
    n = atoi(argv[2]);
    assert(n > 0);
    assert(n <= 1000000000);

    int alg_type = atoi(argv[3]);
    assert((alg_type >= 0) || (alg_type <= 2));

    switch (alg_type) {
    case 0: std::cout << "Alg Type: std::sort\n"; break;
    case 1: std::cout << "Alg Type: std::map\n"; break;
    case 2: std::cout << "Alg Type: parallel unique on " << omp_get_max_threads() << " threads\n"; break;
    }

    std::string* stringArray = new std::string[n];
    const char** cStrArray   = new const char*[n];

    for (int i = 0; i < n && !stream.eof(); ++i) {
        getline(stream, stringArray[i]);
        cStrArray[i] = stringArray[i].c_str();

        if (stream.fail()) {
            std::cerr << "Error while reading from file" << std::endl;
            exit(1);
        }
    }


#ifdef DEBUG_TRACE
    std::cout << "The initial string array" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << cStrArray[i] << std::endl;
    }
#endif // DEBUG_TRACE


    for (int i = 0; i < NUM_ITER; ++i) {
        int numUniqueStrings;

        double time = omp_get_wtime();
        if (alg_type == 0) {
            numUniqueStrings = find_uniq_stl_sort(cStrArray, n);
        }
        else if (alg_type == 1) {
            numUniqueStrings = find_uniq_stl_map(cStrArray, n);
        }
        else {
            numUniqueStrings = parallel_sort(stringArray, n);
        }
        time = omp_get_wtime() - time;

        printf("Iteration %d: %9.3lfs;  Num Unique Strings: %d\n", i, time, numUniqueStrings);
    }

    delete[] stringArray;
    delete[] cStrArray;
}

int parallel_sort(std::string* data, int length) {
    int numThreads   = omp_get_max_threads();
    int numThreadsSq = numThreads * numThreads;
    std::vector<std::string> samples(numThreadsSq);
    std::vector<std::string> dividers(numThreads - 1);

    srand(5353);

    int* myNumUnique = new int[numThreads];    // # of unique strings in each thread's bucket
    int  numUnique   = 0;                      // total # of unique strings
    std::vector<std::vector<int> > myOccurenceCount(numThreads);  // the # of occurences of each unique
                                                                 //   string for each thread's bucket
    std::vector<int> occurenceCount;           // the # of occurences of each string

    // get random samples
    for (int i = 0; i < numThreadsSq; ++i) {
        std::string selected = data[rand() % length];
        samples[i] = selected;
    }

    // sort samples
    std::sort(samples.begin(), samples.end());
      
    // choose numThreads-1 samples as bucket dividers
    for (int i = 0; i < numThreads - 1; ++i) {
        dividers[i] = samples[numThreads * (i + 1)];
    }
    
    // data of each thread to sort
    std::vector<std::vector<std::string> > datas(numThreads);

    // the thread with id #1 found a string that belongs to thread #3 and adds it
    // buckets[1][3].push_back(string)
    std::vector<std::vector<std::vector<std::string> > > buckets(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        buckets[i] = std::vector<std::vector<std::string> >(numThreads);
        for (int j = 0; j < numThreads; ++j) {
            buckets[i][j] = std::vector<std::string>();
        }
    }

#pragma omp parallel firstprivate(numThreads, numThreadsSq)
    {
        int threadId = omp_get_thread_num();
        // [parallel] go through section of data over each string and insert into threads vector if string falls in threads range
#pragma omp for
        for (int i = 0; i < length; ++i)
        {
            int threadToGive = 0;
            std::string str = data[i];
            while (threadToGive < numThreads - 1 && str.compare(dividers[threadToGive]) >= 0) {  // not sure if this logic is right!
                threadToGive++;
            }
            buckets[threadId][threadToGive].push_back(str);
        }


        std::vector<std::string> myData = datas[threadId];
        for (int i = 0; i < numThreads; ++i) {
            myData.insert(myData.end(), buckets[i][threadId].begin(), buckets[i][threadId].end());
        }

#ifdef DEBUG_TRACE
#pragma omp critical
        {
            std::cout << "Thread " << threadId << " elements:\n";
            for (unsigned int i = 0; i < myData.size(); ++i) {
                std::cout << myData[i] << std::endl;
            }
        }
#pragma omp barrier
#endif

        // each thread has a vector[numThreads]
        // as the thread scans through its portion of the data, insert data into the proper vector[index]
        // barrier
        // each thread now reads its segment of the data from the other threads and inserts into its own data vector

        // each thread sorts the strings in its bucket
        std::sort(myData.begin(), myData.end());

        // each thread finds the # of uniques in its bucket and the # of occurences for those uniques
        int numOccurences = 1;

        if (myData.empty()) {
            myNumUnique[threadId] = 0;
        }
        else {
            myNumUnique[threadId] = 1;
        }
        
        for (unsigned int i = 1; i < myData.size(); ++i) {
            if (myData[i].compare(myData[i-1]) == 0) {
                ++numOccurences;
            }
            else {
                ++myNumUnique[threadId];
                myOccurenceCount[threadId].push_back(numOccurences);
                numOccurences = 1;
            }
        }
        myOccurenceCount[threadId].push_back(numOccurences);
    }

    // consolidate the unique counts and occurences
    for (int i = 0; i < numThreads; ++i) {
        numUnique += myNumUnique[i];
        occurenceCount.insert(occurenceCount.end(), myOccurenceCount[i].begin(), myOccurenceCount[i].end());
    }


#ifdef DEBUG_TRACE
    std::cout << "There are " << numUnique << " strings" << std::endl;
    std::cout << "The # of occurences of each unique string in sorted order:" << std::endl;
    for (int i = 0; i < occurenceCount.size(); ++i) {
        std::cout << occurenceCount[i] << std::endl;
    }
#endif // DEBUG_TRACE


    delete[] myNumUnique;

    return numUnique;
}


// Serial methods--------------------------------------------------------------------------------------
int find_uniq_stl_map(const char** str_array, const int num_strings) {

    char** B;
    B = (char**) malloc(num_strings * sizeof(char*));
    assert(B != NULL);

    int* counts;
    counts = (int*) malloc(num_strings * sizeof(int));
    
    memcpy(B, str_array, num_strings * sizeof(char*));

    int i;
    for (i = 0; i < num_strings; ++i) {
        counts[i] = 0;
    }

    std::map<std::string, int> str_map;

    for (i = 0; i < num_strings; ++i) {
        std::string curr_str(B[i]);
        //curr_str.assign(B[i], strlen(B[i]));
        ++str_map[curr_str];
    }

    free(B);
    free(counts);

    return str_map.size();
}


/* comparison routine for STL sort */
class compare_str_cmpf {
public:
    bool operator() (char *u, char *v) {

        int cmpval = strcmp(u, v);

        if (cmpval < 0)
            return true;
        else
            return false;
    }
};

int find_uniq_stl_sort(const char** str_array, const int num_strings) {
    char** B;
    B = (char**) malloc(num_strings * sizeof(char*));
    assert(B != NULL);

    int* counts;
    counts = (int*) malloc(num_strings * sizeof(int));

    memcpy(B, str_array, num_strings * sizeof(char*));

    int i;
    for (i = 0; i < num_strings; ++i) {
        counts[i] = 0;
    }

    compare_str_cmpf cmpf;
    std::sort(B, B+num_strings, cmpf);

    /* determine number of unique strings 
        and count of each string */
    int num_uniq_strings        = 1;
    int string_occurrence_count = 1;
    for (i = 1; i < num_strings; ++i) {
        if (strcmp(B[i], B[i-1]) != 0) {
            ++num_uniq_strings;
            counts[i-1] = string_occurrence_count;
            string_occurrence_count = 1;
        } else {
            ++string_occurrence_count;
        }
    }
    counts[num_strings-1] = string_occurrence_count;
    
    free(B);
    free(counts);

    return num_uniq_strings;
}
// End Serial methods----------------------------------------------------------------------------------
