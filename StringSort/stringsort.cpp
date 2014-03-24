// Created by: Alexander Anderson, Jason Killian
// CMPSC 450 Homework 2
// Date: March 23, 2014

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <map>
#include <algorithm>
#include <omp.h>
#include <vector>

#define PTR(x) *((T**)(&(x)))

// Pool allocator for STL library, source: http://forums.codeguru.com/showthread.php?406108-A-faster-std-set&p=2049091#post2049091
template <typename T>
class bestAlloc {
public:
    typedef T value_type;

    typedef value_type * pointer;
    typedef const value_type * const_pointer;
    typedef value_type & reference;
    typedef const value_type & const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template <typename U>
    struct rebind {
        typedef bestAlloc<U> other;
    };

    class Block {
    public:
        T* ptr;
        size_type size;

        Block(){};
        Block(const Block &b) {
            ptr = b.ptr;
            size = b.size;
        }

        Block& operator=(const Block &b) {
            ptr = b.ptr;
            size = b.size;

            return *this;
        }

        void Initialize() {
            for (size_type i = 0; i<size - 1; i++) {
                PTR(ptr[i]) = &ptr[i + 1];
            }

            PTR(ptr[size - 1]) = NULL;
        };
    };

    std::vector<Block> m_vecBlocks;

    T* firstFree;

    size_type dSize;

    size_type numAllocated;

    void AddBlock() {
        Block b;
        b.size = dSize;

        if (dSize < 128 * 1024)dSize *= 4;

        assert(sizeof(T) >= sizeof(T*));

        b.ptr = (T*)(new char[b.size * sizeof(T)]);

        if (b.ptr == NULL)
            throw std::bad_alloc();

        b.Initialize(); // initialize list of pointers

        PTR(b.ptr[b.size - 1]) = firstFree;
        firstFree = b.ptr;

        m_vecBlocks.push_back(b);
    }

    T* malloc_(size_type n) {
        assert(n == 1); // only single element malloc supported

        if (firstFree == NULL) {
            AddBlock();
        }

        T* ret = firstFree;
        firstFree = PTR(*firstFree);
        numAllocated++;
        return ret;
    }

    void free_(void * const ptr, const size_type n) {
        assert(n == 1);

        if (ptr == NULL)return;

        T* p = (T*)ptr;

        PTR(*p) = firstFree;
        firstFree = p;

        numAllocated--;

        if (numAllocated == 0)releaseMemory();
    }

    void releaseMemory() {
        for (size_type i = 0; i<m_vecBlocks.size(); i++) {
            delete[]((char*)m_vecBlocks[i].ptr);
        }

        m_vecBlocks.clear();

        firstFree = NULL;
        numAllocated = 0;
        dSize = 8;
    }

public:
    bestAlloc() {
        dSize = 8;
        numAllocated = 0;
        firstFree = NULL;
        m_vecBlocks.clear();
    }

    bestAlloc(const bestAlloc<T> &a) {
        dSize = 8;
        numAllocated = 0;
        firstFree = NULL;
        m_vecBlocks.clear();
    }
private:
    bestAlloc& operator=(const bestAlloc<T> &a) {
        dSize = 8;
        numAllocated = 0;
        firstFree = NULL;
        m_vecBlocks.clear();

        return *this;
    }
public:
    // not explicit, mimicking std::allocator [20.4.1]
    template <typename U>
    bestAlloc(const bestAlloc<U> &a) {
        dSize = 8;
        numAllocated = 0;
        firstFree = NULL;
        m_vecBlocks.clear();
    }

    ~bestAlloc() {
        releaseMemory();
    }

    static pointer address(reference r) {
        return &r;
    }
    static const_pointer address(const_reference s) {
        return &s;
    }
    static size_type max_size() {
        return (std::numeric_limits<size_type>::max)();
    }
    void construct(const pointer ptr, const value_type & t) {
        new (ptr)T(t);
    }
    void destroy(const pointer ptr) {
        ptr->~T();
        (void)ptr; // avoid unused variable warning
    }

    // always different
    bool operator==(const bestAlloc &) const {
        return false;
    }
    bool operator!=(const bestAlloc &) const {
        return true;
    }

    pointer allocate(const size_type n) {
        const pointer ret = malloc_(n);

        if (ret == 0)
            throw std::bad_alloc();

        return ret;
    }

    pointer allocate(const size_type n, const void * const) {
        return allocate(n);
    }

    pointer allocate() {
        const pointer ret = malloc_(1);

        if (ret == 0)
            throw std::bad_alloc();

        return ret;
    }

    void deallocate(const pointer ptr, const size_type n) {
        free_(ptr, n);
    }

    void deallocate(const pointer ptr) {
        free_(ptr, 1);
    }
};

#undef PTR

// define this to get lots of trace output from the parallel sort algorithm
//#define DEBUG_TRACE

#define NUM_ITER      10           // number of times to run the algorithm for testing
#define NUM_ARG       4            // number of arguments being passed on command line

int find_uniq_stl_map(const char **str_array, const int num_strings);
int find_uniq_stl_sort(const char **str_array, const int num_strings); 
int parallel_sort();

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

    std::string* stringArray = new std::string[n];
    const char** cStrArray   = new const char*[n];

    // NOTE: consider using mmap to get access to the whole file and read lines parallely
    for (int i = 0; i < n && !stream.eof(); ++i) {
        getline(stream, stringArray[i]);
        cStrArray[i] = stringArray[i].c_str();
        if (stream.fail()) {
            std::cerr << "Error while reading from file" << std::endl;
            exit(1);
        }
    }

#ifdef DEBUG_TRACE
    for (int i = 0; i < n; ++i) {
        std::cout << stringArray[i] << std::endl;
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
            numUniqueStrings = parallel_sort();
        }
        time = omp_get_wtime() - time;

        printf("Iteration %d: %9.3lfms;  Num Unique Strings: %d\n", i, time, numUniqueStrings);
    }

    delete[] stringArray;
}

int parallel_sort() {
    return 0;
}







// Serial methods
int find_uniq_stl_map(const char **str_array, const int num_strings) {

    char **B;
    B = (char **) malloc(num_strings * sizeof(char *));
    assert(B != NULL);

    int *counts;
    counts = (int *) malloc(num_strings * sizeof(int));
        
    int i;

    memcpy(B, str_array, num_strings * sizeof(char *));

    for (i=0; i<num_strings; i++) {
        counts[i] = 0;
    }

    std::map<std::string, int, std::less<std::string>, bestAlloc<std::string>> str_map;

    for (i=0; i<num_strings; i++) {
        std::string curr_str(B[i]);
        //curr_str.assign(B[i], strlen(B[i]));
        str_map[curr_str]++;
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

int find_uniq_stl_sort(const char **str_array, const int num_strings) {

    char **B;
    B = (char **) malloc(num_strings * sizeof(char *));
    assert(B != NULL);

    int *counts;
    counts = (int *) malloc(num_strings * sizeof(int));
        
    int i;

    memcpy(B, str_array, num_strings * sizeof(char *));

    for (i=0; i<num_strings; i++) {
        counts[i] = 0;
    }

    compare_str_cmpf cmpf;
    std::sort(B, B+num_strings, cmpf);

    /* determine number of unique strings 
        and count of each string */
    int num_uniq_strings = 1;
    int string_occurrence_count = 1;
    for (i=1; i<num_strings; i++) {
        if (strcmp(B[i], B[i-1]) != 0) {
            num_uniq_strings++;
            counts[i-1] = string_occurrence_count;
            string_occurrence_count = 1;
        } else {
            string_occurrence_count++;
        }
    }
    counts[num_strings-1] = string_occurrence_count;
    
    free(B);
    free(counts);


    return num_uniq_strings;

}