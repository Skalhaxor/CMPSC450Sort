// Created by: Alexander Anderson, Jason Killian
// CMPSC 450 Homework 2
// Date: March 23, 2014

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define NUM_ITER 10           // number of times to run the algorithm for testing
#define NUM_ARG  4            // number of arguments being passed on command line

int main(int argc, char* argv[])
{
    if (argc != NUM_ARG) {
        fprintf(stderr, "%s <input file> <n> <alg_type>\n", argv[0]);
        fprintf(stderr, "alg_type 0: use C qsort, then find unique strings\n");
        fprintf(stderr, "         1: use inline qsort, then find unique strings\n");
        fprintf(stderr, "         2: use STL sort, then find unique strings\n");
        fprintf(stderr, "         3: use STL map\n");
        fprintf(stderr, "         4: use parallel sort\n");

        exit(1);
    }

    char* inputFile = argv[1];

    int n;
    n = atoi(argv[2]);
    assert(n > 0);
    assert(n <= 1000000000);

    int alg_type = atoi(argv[3]);
    assert((alg_type >= 0) || (alg_type <= 4));
}