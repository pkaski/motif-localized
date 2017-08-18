
/* 
 * This file is part of an experimental software implementation of
 * vertex-localized graph motif search for GPUs utilizing the constrained 
 * multilinear sieving framework. 
 * 
 * The source code is subject to the following license.
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2017 P. Kaski, S. Thejaswi
 * Copyright (c) 2014 A. Björklund, P. Kaski, Ł. Kowalik, J. Lauri
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<time.h>
#include<sys/utsname.h>
#include<string.h>
#include<stdarg.h>
#include<assert.h>
#include<ctype.h>
#include<sys/time.h>
#include<cuda.h>

/**************************************************** Configuration & types. */

#define THREADS_IN_WARP 32

typedef long int index_t;  // default to 64-bit indexing

#include"gf.h"
#include"ffprng.h"

#define MAX_K 32
#define MAX_SHADES 32

#define BUILD_PARALLEL     // do a parallel CPU build
#ifdef BUILD_PARALLEL
#define MAX_THREADS 128
#include<omp.h>
#endif

typedef unsigned int shade_map_t;

/******************************************************************** Flags. */

index_t flag_bin_input    = 0; // default to ASCII input

/************************************************************ Common macros. */

/* Linked list navigation macros. */

#define pnlinknext(to,el) { (el)->next = (to)->next; (el)->prev = (to); (to)->next->prev = (el); (to)->next = (el); }
#define pnlinkprev(to,el) { (el)->prev = (to)->prev; (el)->next = (to); (to)->prev->next = (el); (to)->prev = (el); }
#define pnunlink(el) { (el)->next->prev = (el)->prev; (el)->prev->next = (el)->next; }
#define pnrelink(el) { (el)->next->prev = (el); (el)->prev->next = (el); }


/********************************************************** Error reporting. */

#define ERROR(...) error(__FILE__,__LINE__,__func__,__VA_ARGS__);

static void error(const char *fn, int line, const char *func, 
                  const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, 
            "ERROR [file = %s, line = %d]\n"
            "%s: ",
            fn,
            line,
            func);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();    
}

/******************************************************** Get the host name. */

#define MAX_HOSTNAME 256

const char *sysdep_hostname(void)
{
    static char hn[MAX_HOSTNAME];

    struct utsname undata;
    uname(&undata);
    strcpy(hn, undata.nodename);
    return hn;
}

/******************************************************** Available threads. */

index_t num_threads(void)
{
#ifdef BUILD_PARALLEL
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/********************************************* Memory allocation & tracking. */

#define MALLOC(x) malloc_wrapper(x)
#define FREE(x) free_wrapper(x)

index_t malloc_balance = 0;

struct malloc_track_struct
{
    void *p;
    size_t size;
    struct malloc_track_struct *prev;
    struct malloc_track_struct *next;
};

typedef struct malloc_track_struct malloc_track_t;

malloc_track_t malloc_track_root;
size_t malloc_total = 0;

#define MEMTRACK_STACK_CAPACITY 256
size_t memtrack_stack[MEMTRACK_STACK_CAPACITY];
index_t memtrack_stack_top = -1;

void *malloc_wrapper(size_t size)
{
    if(malloc_balance == 0) {
        malloc_track_root.prev = &malloc_track_root;
        malloc_track_root.next = &malloc_track_root;
    }
    void *p = malloc(size);
    if(p == NULL)
        ERROR("malloc fails");
    malloc_balance++;

    malloc_track_t *t = (malloc_track_t *) malloc(sizeof(malloc_track_t));
    t->p = p;
    t->size = size;
    pnlinkprev(&malloc_track_root, t);
    malloc_total += size;
    for(index_t i = 0; i <= memtrack_stack_top; i++)
        if(memtrack_stack[i] < malloc_total)
            memtrack_stack[i] = malloc_total;    
    return p;
}

void free_wrapper(void *p)
{
    malloc_track_t *t = malloc_track_root.next;
    for(;
        t != &malloc_track_root;
        t = t->next) {
        if(t->p == p)
            break;
    }
    if(t == &malloc_track_root)
        ERROR("FREE issued on a non-tracked pointer %p", p);
    malloc_total -= t->size;
    pnunlink(t);
    free(t);
    
    free(p);
    malloc_balance--;
}

index_t *alloc_idxtab(index_t n)
{
    index_t *t = (index_t *) MALLOC(sizeof(index_t)*n);
    return t;
}

void push_memtrack(void) 
{
    assert(memtrack_stack_top + 1 < MEMTRACK_STACK_CAPACITY);
    memtrack_stack[++memtrack_stack_top] = malloc_total;
}

size_t pop_memtrack(void)
{
    assert(memtrack_stack_top >= 0);
    return memtrack_stack[memtrack_stack_top--];    
}

size_t current_mem(void)
{
    return malloc_total;
}

double inGiB(size_t s) 
{
    return (double) s / (1 << 30);
}

void print_current_mem(void)
{
    fprintf(stdout, "{curr: %.2lfGiB}", inGiB(current_mem()));
    fflush(stdout);
}

void print_pop_memtrack(void)
{
    fprintf(stdout, "{peak: %.2lfGiB}", inGiB(pop_memtrack()));
    fflush(stdout);
}

/******************************************************* Timing subroutines. */

#define TIME_STACK_CAPACITY 256
double start_stack[TIME_STACK_CAPACITY];
index_t start_stack_top = -1;

void push_time(void) 
{
    assert(start_stack_top + 1 < TIME_STACK_CAPACITY);
#ifdef BUILD_PARALLEL
    start_stack[++start_stack_top] = omp_get_wtime();
#else
    start_stack[++start_stack_top] = (double) clock()/CLOCKS_PER_SEC;
#endif
}

double pop_time(void)
{
#ifdef BUILD_PARALLEL
    double wstop = omp_get_wtime();
#else
    double wstop = (double) clock()/CLOCKS_PER_SEC;
#endif
    assert(start_stack_top >= 0);
    double wstart = start_stack[start_stack_top--];
    return (double) (1000.0*(wstop-wstart));
}

/****************************************************************** Sorting. */

void shellsort(index_t n, index_t *a)
{
    index_t h = 1;
    index_t i;
    for(i = n/3; h < i; h = 3*h+1)
        ;
    do {
        for(i = h; i < n; i++) {
            index_t v = a[i];
            index_t j = i;
            do {
                index_t t = a[j-h];
                if(t <= v)
                    break;
                a[j] = t;
                j -= h;
            } while(j >= h);
            a[j] = v;
        }
        h /= 3;
    } while(h > 0);
}

#define LEFT(x)      (x<<1)
#define RIGHT(x)     ((x<<1)+1)
#define PARENT(x)    (x>>1)

void heapsort_indext(index_t n, index_t *a)
{
    /* Shift index origin from 0 to 1 for convenience. */
    a--; 
    /* Build heap */
    for(index_t i = 2; i <= n; i++) {
        index_t x = i;
        while(x > 1) {
            index_t y = PARENT(x);
            if(a[x] <= a[y]) {
                /* heap property ok */
                break;              
            }
            /* Exchange a[x] and a[y] to enforce heap property */
            index_t t = a[x];
            a[x] = a[y];
            a[y] = t;
            x = y;
        }
    }

    /* Repeat delete max and insert */
    for(index_t i = n; i > 1; i--) {
        index_t t = a[i];
        /* Delete max */
        a[i] = a[1];
        /* Insert t */
        index_t x = 1;
        index_t y, z;
        while((y = LEFT(x)) < i) {
            z = RIGHT(x);
            if(z < i && a[y] < a[z]) {
                index_t s = z;
                z = y;
                y = s;
            }
            /* Invariant: a[y] >= a[z] */
            if(t >= a[y]) {
                /* ok to insert here without violating heap property */
                break;
            }
            /* Move a[y] up the heap */
            a[x] = a[y];
            x = y;
        }
        /* Insert here */
        a[x] = t; 
    }
}

/****************************************************** Bitmap manipulation. */

void bitset(index_t *map, index_t j, index_t value)
{
    assert((value & (~1UL)) == 0);
    map[j/64] = (map[j/64] & ~(1UL << (j%64))) | ((value&1) << (j%64));  
}

index_t bitget(index_t *map, index_t j)
{
    return (map[j/64]>>(j%64))&1UL;
}

/************************************************** Random numbers and such. */

index_t irand(void)
{
    return (((index_t) rand())<<31)^((index_t) rand());
}

/**************************************************** (Parallel) prefix sum. */

index_t prefixsum(index_t n, index_t *a, index_t k)
{

#ifdef BUILD_PARALLEL
    index_t s[MAX_THREADS];
    index_t nt = num_threads();
    assert(nt < MAX_THREADS);

    index_t length = n;
    index_t block_size = length/nt;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t tsum = (stop-start+1)*k;
        for(index_t u = start; u <= stop; u++)
            tsum += a[u];
        s[t] = tsum;
    }

    index_t run = 0;
    for(index_t t = 1; t <= nt; t++) {
        index_t v = s[t-1];
        s[t-1] = run;
        run += v;
    }
    s[nt] = run;

#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? length-1 : (start+block_size-1);
        index_t trun = s[t];
        for(index_t u = start; u <= stop; u++) {
            index_t tv = a[u];
            a[u] = trun;
            trun += tv + k;
        }
        assert(trun == s[t+1]);    
    }

#else

    index_t run = 0;
    for(index_t u = 0; u < n; u++) {
        index_t tv = a[u];
        a[u] = run;
        run += tv + k;
    }

#endif

    return run; 
}


/*********************** Search for an interval of values in a sorted array. */

inline index_t get_interval(index_t n, index_t *a, 
                            index_t lo_val, index_t hi_val,
                            index_t *iv_start, index_t *iv_end)
{
    assert(n >= 0);
    if(n == 0) {
        *iv_start = 0; 
        return 0;
    }
    assert(lo_val <= hi_val);
    // find first element in interval (if any) with binary search
    index_t lo = 0;
    index_t hi = n-1;
    // at or above lo, and at or below hi (if any)
    while(lo < hi) {
        index_t mid = (lo+hi)/2; // lo <= mid < hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi (if any)
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo (if any), lo <= hi
            else
                hi = mid;   // at or below hi (exists) 
        }
        // 0 <= lo <= n-1
    }
    if(a[lo] < lo_val || a[lo] > hi_val) {
        // array contains no values in interval
        if(a[lo] < lo_val) {
            lo++;
            assert(lo == n || a[lo+1] > hi_val);
        } else {
            assert(lo == 0 || a[lo-1] < lo_val);
        }
        *iv_start = lo; 
        *iv_end   = hi;
        return 0; 
    }
    assert(lo_val <= a[lo] && a[lo] <= hi_val);
    *iv_start = lo;
    // find interval end (last index in interval) with binary search
    lo = 0;
    hi = n-1;
    // last index (if any) is at or above lo, and at or below hi
    while(lo < hi) {
        index_t mid = (lo+hi+1)/2; // lo < mid <= hi
        index_t v = a[mid];
        if(hi_val < v) {
            hi = mid-1;     // at or below hi, lo <= hi
        } else {
            if(v < lo_val)
                lo = mid+1; // at or above lo
            else
                lo = mid;   // at or above lo, lo <= hi
        }
    }
    assert(lo == hi);
    *iv_end = lo; // lo == hi
    return 1+*iv_end-*iv_start; // return cut size
}


/********************************** Initialize an array with random scalars. */

void randinits_scalar(scalar_t *a, index_t s, ffprng_scalar_t seed)
{
    ffprng_t base;
    FFPRNG_INIT(base, seed);
    index_t nt = num_threads();
    index_t block_size = s/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? s-1 : (start+block_size-1);
        FFPRNG_FWD(gen, start, base);
        for(index_t i = start; i <= stop; i++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, gen);
            scalar_t rs = (scalar_t) rnd;           
            a[i] = rs;
        }
    }
}



/********************************************************************* CUDA. */

/************************ CUDA error wrapper (adapted from CUDA By Example). */

#define CUDA_WRAP(err) (error_wrap(err,__FILE__,__LINE__))

static void error_wrap(cudaError_t err,
                       const char *fn,
                       int line) {
    if(err != cudaSuccess) {
        fprintf(stderr,
                "error [%s, line %d]: %s\n",
                fn,
                line,
                cudaGetErrorString(err));
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}

/***************************************************** Line-sum for the GPU. */

/*
 * The following kernel adapted [in particular, sans the commentary!] 
 * from Mark Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA
 *
 * http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *
 */

template <index_t block_size>
__device__ void device_line_sum_finish(volatile line_t *s, index_t a)
{
    // Remarks:
    //
    // 1)
    // Observe the volatile decl above to instruct the compiler
    // *not* to reorder the share mem transactions below
    //
    // 2)
    // What is below takes simultaneously place for a = 0,1,...,31
    // __in parallel__, all data now in s[0],s[1],...,s[63]
    //

    if(block_size >= 64)
        LINE_ADD(s[a],s[a],s[a + 32]); // ... now in s[0],s[1],...,s[31]
    if(block_size >= 32)
        LINE_ADD(s[a],s[a],s[a + 16]); // ... now in s[0],s[1],...,s[15]
    if(block_size >= 16)
        LINE_ADD(s[a],s[a],s[a +  8]); // ... now in s[0],s[1],...,s[7]
    if(block_size >=  8)
        LINE_ADD(s[a],s[a],s[a +  4]); // ... now in s[0],s[1],s[2],s[3]
    if(block_size >=  4)
        LINE_ADD(s[a],s[a],s[a +  2]); // ... now in s[0],s[1]
    if(block_size >=  2)
        LINE_ADD(s[a],s[a],s[a +  1]); // ... now in s[0]
}

template <index_t block_size>
__global__ void device_line_sum_block(index_t      dg,
                                      index_t      q,
                                      index_t      seg,
                                      line_array_t *d_in,
                                      line_array_t *d_out)
{
    // Many a thread hereby commence their labours in this block ...
    index_t a        = threadIdx.x;         // my index inside my block
    index_t span     = 2*block_size;        // one block spans *twice* the data
    index_t major    = (index_t) blockIdx.x+blockIdx.y*gridDim.x;
    index_t i        = major*span + a;      // accumulate from here ...
    index_t i_end    = i + q;               // ... to here (exclusive)
    index_t stride   = span*dg;             // ... with a stride that isolates
                                            //     us from whatever the
                                            //     __other__ blocks are doing,
                                            //     asynchronously

    extern __shared__ line_t s[]; // cells for me and my mates
                                  // (in my block); my cell is s[a],
                                  // I shall write to no other cells
                                  // (except at init)

    // Start my work, my brave mates working in parallel with me ...

    line_t sum;
    LINE_SET_ZERO(sum);
    while(i < i_end) {
        line_t t1, t2;
        LINE_LOAD(t1, d_in, seg, i);
        LINE_LOAD(t2, d_in, seg, i + block_size); // span twice the data
        LINE_ADD(t1, t1, t2);
        LINE_ADD(sum, sum, t1);
        i += stride;          // ... stride past all the other blocks
    }
    LINE_MOV(s[a], sum);
    LINE_SET_ZERO(s[a+block_size]); // small inputs may refer here, so zero it
    __syncthreads();   // sync with my mates

    // All data now in s[0],s[1],...,s[min(511,block_size)]
    if(block_size >= 512) { if(a < 256) { LINE_ADD(s[a],s[a],s[a + 256]); } __syncthreads(); }
    // All data now in s[0],s[1],...,s[min(255,block_size)]
    if(block_size >= 256) { if(a < 128) { LINE_ADD(s[a],s[a],s[a + 128]); } __syncthreads(); }
    // All data now in s[0],s[1],...,s[min(127,block_size)]
    if(block_size >= 128) { if(a <  64) { LINE_ADD(s[a],s[a],s[a +  64]); } __syncthreads(); }
    // All data now in s[0],s[1],...,s[min(63,block_size)]

    if(a < 32) {
        // Most of my mates are done, but I remain in the wrap-up detail ...
        device_line_sum_finish<block_size>(s, a);
    }
    if(a == 0) {
        // Ha! I get to output all the efforts due to me and my mates ...
        LINE_STORE(d_out, seg, major, s[0]);
    }
}

__global__ void device_last_line(index_t      p,
                                 index_t      seg,
                                 line_array_t *d_in,
                                 scalar_t     *d_sum_out)
{
    index_t v = blockDim.x*((index_t) blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    if(v < p) {
        line_t l;
        LINE_LOAD(l, d_in, seg, v);
        LINE_SUM(d_sum_out[v], l);
    }
}

void driver_line_sum(index_t       p,
                     index_t       l,
                     index_t       seg,
                     line_array_t  *d_s0,
                     line_array_t  *d_s1,
                     scalar_t      *h_sum)
{

    index_t n = l;              // number of lines to sum up
    index_t pref_threads = 512; // preferred threads per block 
                                // (must be a power of 2)

    while(n > 1) {
        index_t dg, db;
        size_t sm;

        if(n >= 2*pref_threads) {
            db = pref_threads;
            dg = n/(2*db); // one block spans _twice_ the data
        } else {
            db = n/2; // one block spans _twice_ the data
            dg = 1;
        }
        sm = sizeof(line_t)*db*2; // enough share mem to span twice the threads
        index_t pdg = p*dg;

        // Create a 2D grid to satisfy GPU hardware
        index_t pdgx = pdg >= (1 << 16) ? (1 << 15) : pdg;
        index_t pdgy = pdg >= (1 << 16) ? pdg / (1 << 15) : 1;
        dim3 pdg2(pdgx,pdgy);
        dim3 db2(db,1);

        switch(db) {
        case 1024:
            device_line_sum_block<1024><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 512:
            device_line_sum_block<512><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 256:
            device_line_sum_block<256><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 128:
            device_line_sum_block<128><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 64:
            device_line_sum_block< 64><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 32:
            device_line_sum_block< 32><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 16:
            device_line_sum_block< 16><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 8:
            device_line_sum_block<  8><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 4:
            device_line_sum_block<  4><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 2:
            device_line_sum_block<  2><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        case 1:
            device_line_sum_block<  1><<<pdg2,db2,sm>>>(dg,n,seg,d_s0,d_s1); 
            break;
        default:
            assert(0); 
            break;
        }
        cudaDeviceSynchronize();
        CUDA_WRAP(cudaGetLastError());
        n = dg;

        /* Restore invariant. */
        line_array_t *save = d_s0;
        d_s0               = d_s1;
        d_s1               = save;

    }
    /* Invariant: Input (= Output) is in d_s0. */

    /* Sum up the last lines to a scalar. */
    index_t dbl = 1;
    index_t dgl = 1;
    while(dbl < p)
        dbl *= 2;
    if(dbl > pref_threads) {
        dgl = dbl / pref_threads;
        dbl = pref_threads;
    }
    index_t dglx = dgl >= (1 << 16) ? (1 << 15) : dgl;
    index_t dgly = dgl >= (1 << 16) ? dgl / (1 << 15) : 1;
    dim3 dgl2(dglx,dgly);
    dim3 dbl2(dbl,1);
    device_last_line<<<dgl2,dbl2>>>(p, seg, d_s0, (scalar_t *) d_s1);
    cudaDeviceSynchronize();
    CUDA_WRAP(cudaGetLastError());

    CUDA_WRAP(cudaMemcpy(h_sum, d_s1,
                         p*sizeof(scalar_t), cudaMemcpyDeviceToHost));
}

/************************** Init shade variables for sieve (host-side init). */

void init_shades(index_t         n,
                 index_t         n0,
                 index_t         k,
                 index_t         num_shades,
                 shade_map_t     *h_s,
                 ffprng_scalar_t seed,
                 scalar_t        *h_z)
{
    assert(num_shades <= MAX_SHADES);
    scalar_t wdj[k*k];
    ffprng_t base;
    FFPRNG_INIT(base, seed);

    for(index_t i = 0; i < k; i++) {
        for(index_t j = 0; j < k; j++) {
            ffprng_scalar_t rnd;
            FFPRNG_RAND(rnd, base);
            wdj[i*k+j] = (scalar_t) rnd;
        }
    }

    index_t nt = num_threads();
    index_t block_size = n/nt;
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t t = 0; t < nt; t++) {
        ffprng_t gen;
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        FFPRNG_FWD(gen, k*start, base);
        for(index_t i = start; i <= stop; i++) {
            if(i < n0) {
                scalar_t vi[k];
                shade_map_t shades_u = h_s[i];            
                for(index_t j = 0; j < k; j++) {
                    ffprng_scalar_t rnd;
                    FFPRNG_RAND(rnd, gen);
                    scalar_t rs = (scalar_t) rnd;                   
                    rs = rs & (-((scalar_t)((shades_u >> j)&(j < num_shades))));
                    vi[j] = rs;
                }
                for(index_t j = 0; j < k; j++) {
                    scalar_t uj = 0;
                    for(index_t d = 0; d < k; d++) {
                        scalar_t ln = 0;
                        REF_SCALAR_MUL(ln, wdj[j*k+d], vi[d]); 
                                                      // SMUL [host]: n0*k*k
                        REF_SCALAR_ADD(uj, uj, ln);
                    }
                    h_z[i*k+j] = uj;                  // SW [host]: n0*k
                }
            } else {
                for(index_t j = 0; j < k; j++)
                    h_z[i*k+j] = 0;                   // SW [host]: (n-n0)*k
            }
        }
    }

    // total SW:   n*k
    // total SMUL: n0*k*k
}

/****************************************************************** Sieving. */

#define LINE_IDX(n, gl, l, u, a) ((((l)-1)*(n)*(gl))+((u)*(gl))+(a))

__global__
void device_constrained_sieve_pre(index_t       n,
                                  index_t       k,
                                  index_t       gl,
                                  index_t       seg,
                                  index_t       pfx,
                                  scalar_t      *d_z,
                                  line_array_t  *d_s)
{
    index_t job = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    index_t u = job/gl;
    index_t a = job%gl;
    index_t aa = pfx + a*SCALARS_IN_LINE;
    line_t ln;
    LINE_SET_ZERO(ln);
    for(index_t j = 0; j < SCALARS_IN_LINE; j++) {
        index_t aaj = aa+j;
        scalar_t xuaaj;
        SCALAR_SET_ZERO(xuaaj);
        for(index_t l = 0; l < k; l++) {
            scalar_t z_ul = d_z[u*k+l];     // SR [warp, cached]: n*k
            z_ul = z_ul & (-(((aaj) >> l)&1));
            SCALAR_ADD(xuaaj, xuaaj, z_ul);
        }
        LINE_STORE_SCALAR(ln, j, xuaaj);
    }
    index_t l1ua = LINE_IDX(n, gl, 1, u, a);
    LINE_STORE(d_s, seg, l1ua, ln);                       // LW: n*gl

    // total SR: n*k
    // total LW: n*gl
}

void driver_constrained_sieve_pre(index_t          n,
                                  index_t          k,
                                  index_t          gl,
                                  index_t          seg,
                                  index_t          pfx,
                                  index_t          dg,
                                  index_t          db,
                                  scalar_t         *d_z,
                                  line_array_t     *d_s)
{

    // Create a 2D grid to satisfy GPU hardware
    index_t dgx = dg >= (1 << 16) ? (1 << 15) : dg;
    index_t dgy = dg >= (1 << 16) ? dg / (1 << 15) : 1;
    dim3 dg2(dgx,dgy);
    dim3 db2(db,1);
    device_constrained_sieve_pre<<<dg2,db2>>>(n,k,gl,seg,pfx,d_z,d_s);
    cudaDeviceSynchronize();
    CUDA_WRAP(cudaGetLastError());
}

/********************************** Generating function for k-arborescences. */

#ifndef VERTICES_PER_GENF_THREAD
#define VERTICES_PER_GENF_THREAD 1   // must be a power of 2 (max 16 for k=10)
#endif

__global__
void device_karb_genf_round(index_t        n,   
                            index_t        l,
                            index_t        k,
                            index_t        gl,  
                            index_t        b,
                            index_t        seg,
                            index_t        *d_pos,
                            index_t        *d_adj,
                            scalar_t       *d_y,
                            line_array_t   *d_s
#ifdef GF_LOG_EXP_LOOKUP
                          , scalar_t       *d_lookup_log, 
                            scalar_t       *d_lookup_exp
#endif
)
{
    index_t job = blockDim.x*(blockIdx.x+blockIdx.y*gridDim.x)+threadIdx.x;
    index_t a   = job % gl;
    index_t u_start = (job / gl) * VERTICES_PER_GENF_THREAD;
    index_t u_end = u_start + VERTICES_PER_GENF_THREAD - 1;

    #pragma unroll 1
    for(index_t u = u_start; u <= u_end; u++) {
        index_t p   = d_pos[u];                  // IR [warp]: (k-1)*n
        index_t deg = d_adj[p];                  // IR [warp]: (k-1)*n
        line_t p_lu;
        LINE_SET_ZERO(p_lu);
    
        #pragma unroll 1
        for(index_t j = 1; j <= deg; j++) { 
            index_t v = d_adj[p+j];              // IR [warp, cached]: (k-1)*m
            line_t p_luv;
            LINE_SET_ZERO(p_luv);
            #pragma unroll 1           
            for(index_t l1 = 1; l1 < l; l1++) {
                // \sum_{l=2}^k \sum_{l1=1}^{l-1} 1 
                //    = \sum_{l=2}^k (l-1) 
                //    = k(k-1)/2
                index_t l2 = l-l1;
                index_t l1u = LINE_IDX(n, gl, l1, u, a);
                line_t p_l1u;
                LINE_LOAD(p_l1u, d_s, seg, l1u);         // LR: m*gl*k(k-1)/2
                index_t l2v = LINE_IDX(n, gl, l2, v, a);
                line_t p_l2v;
                LINE_LOAD(p_l2v, d_s, seg, l2v);         // LR: m*gl*k(k-1)/2
                line_t p_l1u_l2v;
                LINE_MUL(p_l1u_l2v, p_l1u, p_l2v);       // LMUL: m*gl*k(k-1)/2
                LINE_ADD(p_luv, p_luv, p_l1u_l2v);
            }
            scalar_t y_luv = d_y[(l-1)*b+p+j];   // SR [warp, cached]: (k-1)*m
            line_t res;
            LINE_MUL_SCALAR(res, p_luv, y_luv);          // LMUL: m*gl*(k-1)
            LINE_ADD(p_lu, p_lu, res);
        }    
        index_t lu = LINE_IDX(n, gl, l, u, a);
        LINE_STORE(d_s, seg, lu, p_lu);                  // LW: n*gl*(k-1)
    }

    // total IR:    2*(k-1)*n+(k-1)*m
    // total SR:    (k-1)*m
    // total LR+LW: m*gl*k(k-1) + n*gl*(k-1)
    // total LMUL:  m*gl*k(k-1)/2 + m*gl*(k-1) 
}

line_array_t *driver_karb_genf(index_t        n,
                               index_t        k,
                               index_t        gl,
                               index_t        b,
                               index_t        seg,
                               index_t        dg,
                               index_t        db,
                               index_t        *d_pos,
                               index_t        *d_adj,
                               scalar_t       *d_y,
                               line_array_t   *d_s
#ifdef GF_LOG_EXP_LOOKUP
                             , scalar_t       *d_lookup_log, 
                               scalar_t       *d_lookup_exp
#endif
)
{
    // Create a 2D grid to satisfy GPU hardware
    index_t dgx = dg >= (1 << 16) ? (1 << 15) : dg;
    index_t dgy = dg >= (1 << 16) ? dg / (1 << 15) : 1;

    dim3 dg2(dgx,dgy);
    dim3 db2(db,1);

    assert(k >= 1);
    if(k >= 2) {
        for(index_t l = 2; l <= k; l++) {
            device_karb_genf_round<<<dg2,db2>>>(n,
                                                l,
                                                k,
                                                gl,
                                                b,
                                                seg,
                                                d_pos,
                                                d_adj,
                                                d_y,
                                                d_s
#ifdef GF_LOG_EXP_LOOKUP
                                              , d_lookup_log, 
                                                d_lookup_exp
#endif
                                               );
            cudaDeviceSynchronize();
            CUDA_WRAP(cudaGetLastError());
        }
    } 

    return d_s;
}

/*********************************************** Stub to warm up GPU device. */

void lightup_stub(void)
{
    fprintf(stdout, "lightup: ");
    push_time();

    index_t n    = 1024;
    index_t seed = 123456789;

    push_time();
    /* Allocate space in host memory. */
    scalar_t *h_x = (scalar_t *) MALLOC(n*sizeof(scalar_t));
    randinits_scalar(h_x, n, seed);

    scalar_t *d_x;

    /* Now light up the hardware. */
    fprintf(stdout, " {malloc:");
    push_time();
    push_time();
    /* Set up space in device memory. */
    CUDA_WRAP(cudaMalloc(&d_x, n*sizeof(scalar_t)));
    double time1 = pop_time();

    push_time();
    /* Upload input to device. */
    CUDA_WRAP(cudaMemcpy(d_x, h_x, n*sizeof(scalar_t), cudaMemcpyHostToDevice));
    double time2 = pop_time();
    double time0 = pop_time();
    fprintf(stdout, " %.2lfms %.2lfms %.2lfms}",
                    time1, time2, time0);
    
    
    fprintf(stdout, " {free:");
    /* Free device memory. */
    push_time();
    CUDA_WRAP(cudaFree(d_x));
    time0 = pop_time();
    fprintf(stdout, " %.2lfms}", time0);

    /* Free working space in host memory. */
    FREE(h_x);
    time0 = pop_time();
    fprintf(stdout, " [%.2lfms]\n", time0);
    fflush(stdout);
}

/******************************************************* The k-motif oracle. */

index_t oracle(index_t        n0,
               index_t        k,
               index_t        *h_pos,
               index_t        *h_adj,
               index_t        num_shades,
               shade_map_t    *h_s,
               index_t        seed,
               scalar_t       *master_vsum) 
{
       
    assert(k < 31);
    assert(n0 > 0);

    index_t m0 = h_pos[n0-1]+h_adj[h_pos[n0-1]]+1-n0;
    index_t b0 = n0+m0;

    index_t n = 1;
    while(n < n0)
        n = n*2;
    index_t m = m0;
    index_t b = n+m;

    /* Invariant: n must be a power of two. */

    index_t sum_size = 1 << k;
    assert(SCALARS_IN_LINE <= sum_size);

    index_t g = sum_size; // g scalars of work

    while(LINE_ARRAY_SIZE((size_t) k*n*g) > 
#if defined(GPU_M2090)
          (size_t) 1 << 32       //  4 GiB peak allocation for the M2090
#elif defined(GPU_K40) || defined(GPU_K80)
          10*((size_t) 1 << 30)  // 10 GiB peak allocation for the K40
#elif defined(GPU_P100)
          16*((size_t) 1 << 30)  // 16 GiB peak allocation for the P100
#else
#error "choose one of GPU_M2090 or GPU_K40 or GPU_K80 or GPU_P100"
#endif
         )
        g /= 2;
    assert(g >= SCALARS_IN_LINE);

    index_t outer = sum_size / g;      // number of iterations for outer loop
    index_t gl = g / SCALARS_IN_LINE;  // gl scalar-lines of work

    index_t num_processors = 16;       // should be a power of 2
    index_t max_block      = 32;       // should be a power of 2

    index_t work = n*gl;
    index_t work_per_processor = work / num_processors;
    index_t dg, db;
    if(work_per_processor < THREADS_IN_WARP) {
        dg = work / THREADS_IN_WARP;
        db = THREADS_IN_WARP;
    } else {
        db = work / num_processors;
        if(db > max_block)
            db = max_block;
        dg = work / db;
    }
    assert(dg >= 1);               // must have enough work
    assert(db >= THREADS_IN_WARP);

    /* Invariant:  n*gl == work == dg*db */

    assert(dg % VERTICES_PER_GENF_THREAD == 0);

    /* Light up the device to avoid cold start. */
    lightup_stub();

    /* Start timing. */
    float time;
    cudaEvent_t start, stop;
    CUDA_WRAP(cudaEventCreate(&start));
    CUDA_WRAP(cudaEventCreate(&stop));
    CUDA_WRAP(cudaEventRecord(start, 0));

    /* Allocate vertex sum buffer. */
    scalar_t *h_vs = (scalar_t *) MALLOC(n*sizeof(scalar_t));

    /* Allocate working space in host memory. */
    scalar_t *h_y = (scalar_t *) MALLOC(b*k*sizeof(scalar_t));
    scalar_t *h_z = (scalar_t *) MALLOC(n*k*sizeof(scalar_t));
    index_t *h_pospad = (index_t *) MALLOC((n-n0)*sizeof(index_t));
    index_t *h_adjpad = (index_t *) MALLOC((b-b0)*sizeof(index_t));

    /* Init & set up padding. */
    init_shades(n, n0, k, num_shades, h_s, seed, h_z); 
    randinits_scalar(h_y, b*k, seed); 

#ifdef BUILD_PARALLEL
#pragma omp parallel for 
#endif
   for(index_t i = 0; i < n-n0; i++)
        h_pospad[i] = b0+i;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < b-b0; i++)
        h_adjpad[i] = 0;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t i = 0; i < n0; i++)
        master_vsum[i] = 0;

    /* Set up inputs and scratch space in device memory. */
    index_t *d_pos;      CUDA_WRAP(cudaMalloc(&d_pos, n*sizeof(index_t)));
    index_t *d_adj;      CUDA_WRAP(cudaMalloc(&d_adj, b*sizeof(index_t)));
    scalar_t *d_y;       CUDA_WRAP(cudaMalloc(&d_y,   b*k*sizeof(scalar_t)));
    scalar_t *d_z;       CUDA_WRAP(cudaMalloc(&d_z,   n*k*sizeof(scalar_t)));
    scalar_t *d_sum_out; CUDA_WRAP(cudaMalloc(&d_sum_out, sizeof(scalar_t)));
    index_t seg = LINE_SEGMENT_SIZE(k*n*g);
    line_array_t *d_s;   CUDA_WRAP(cudaMalloc(&d_s, LINE_ARRAY_SIZE(k*n*g)));

    /* Upload input to device. */
    CUDA_WRAP(cudaMemcpy(d_pos, h_pos, n0*sizeof(index_t),    cudaMemcpyHostToDevice));
    CUDA_WRAP(cudaMemcpy(d_adj, h_adj, b0*sizeof(index_t),    cudaMemcpyHostToDevice));
    CUDA_WRAP(cudaMemcpy(d_y,   h_y,   b*k*sizeof(scalar_t),  cudaMemcpyHostToDevice));
    CUDA_WRAP(cudaMemcpy(d_z,   h_z,   n*k*sizeof(scalar_t),  cudaMemcpyHostToDevice));
    CUDA_WRAP(cudaMemcpy(d_pos + n0, h_pospad, (n-n0)*sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_WRAP(cudaMemcpy(d_adj + b0, h_adjpad, (b-b0)*sizeof(index_t), cudaMemcpyHostToDevice));

#ifdef GF_LOG_EXP_LOOKUP
    gf_precompute_exp_log();
    scalar_t *d_lookup_log; CUDA_WRAP(cudaMalloc(&d_lookup_log, GF_LOG_LOOKUP_SIZE));
    scalar_t *d_lookup_exp; CUDA_WRAP(cudaMalloc(&d_lookup_exp, GF_EXP_LOOKUP_SIZE)); 
    CUDA_WRAP(cudaMemcpy(d_lookup_log, h_lookup_log, GF_LOG_LOOKUP_SIZE, cudaMemcpyHostToDevice));
    CUDA_WRAP(cudaMemcpy(d_lookup_exp, h_lookup_exp, GF_EXP_LOOKUP_SIZE, cudaMemcpyHostToDevice));
#endif

    /* Free working space in host memory. */
    FREE(h_y);
    FREE(h_z);
    FREE(h_pospad);
    FREE(h_adjpad);

    /* Now run the work. */

    scalar_t master_sum;
    SCALAR_SET_ZERO(master_sum);

    for(index_t out = 0; out < outer; out++) {
        driver_constrained_sieve_pre(n, k, gl, seg, 
                                     g*out, dg, db, d_z, d_s);
        line_array_t *d_g = driver_karb_genf(n, 
                                             k, 
                                             gl, 
                                             b, 
                                             seg,
                                             dg/VERTICES_PER_GENF_THREAD, 
                                             db, 
                                             d_pos, 
                                             d_adj, 
                                             d_y, 
                                             d_s
#ifdef GF_LOG_EXP_LOOKUP
                                           , d_lookup_log, 
                                             d_lookup_exp
#endif
);
        driver_line_sum(n,
                        gl, 
                        seg, 
                        d_g + (k-1)*n*gl,
                        d_g,
                        h_vs);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
        for(index_t i = 0; i < n0; i++)
            REF_SCALAR_ADD(master_vsum[i],  master_vsum[i], h_vs[i]);

    }

    for(index_t i = 0; i < n0; i++) // could do a host-side parallel sum here
        REF_SCALAR_ADD(master_sum, master_sum, master_vsum[i]);

    /* Stop timing. */

    CUDA_WRAP(cudaEventRecord(stop, 0));
    CUDA_WRAP(cudaEventSynchronize(stop));
    CUDA_WRAP(cudaEventElapsedTime(&time, start, stop));

    /* All done, now print out some statistics. */



    // total IR:    2*(k-1)*n+(k-1)*m             (genf)

    // total SW:    n*k                           (host init)
    // total SR:    n*k                           (pre)
    // total SR:    (k-1)*m                       (genf)

    // total LW:    n*gl                          (pre)
    // total LR+LW: m*gl*k(k-1) + n*gl*(k-1)      (genf)
    // total LR:    n*gl                          (sum)

    // total SMUL:  n0*k*k                        (host init)
    // total LMUL:  m*gl*k(k-1)/2 + m*gl*(k-1)    (genf)

    double line_rw_inner    = (double) m*gl*k*(k-1) + n*gl*(k-1) + 2*n*gl;
    double line_mul_inner   = (double) m*gl*k*(k-1)/2 + m*gl*(k-1);
    double line_rw_total    = line_rw_inner*outer;
    double line_mul_total   = line_mul_inner*outer;
    double bytes_rw_total   = EFFECTIVE_BYTES_IN_LINE*line_rw_total;
    double scalar_mul_total = line_mul_total*SCALARS_IN_LINE;
    
    double rw_rate            = bytes_rw_total / (time/1000.0);
    double mul_rate           = scalar_mul_total / time;
    double total_instr_in_mul = LINE_MUL_INSTR*scalar_mul_total/SCALARS_IN_LINE;
    double instr_in_mul_rate  = total_instr_in_mul / time;

    fprintf(stdout,
            "oracle: "
            SCALAR_FORMAT_STRING
            " %10.2fms [%6.3lfGiB %7.2lfGiB/s %7.2lfGHz %7.2fGHz] %ld %d",
            (scalar_t) master_sum,
            time,
            inGiB(LINE_ARRAY_SIZE(k*n*g)+
                  n*sizeof(index_t)+
                  b*sizeof(index_t)+
                  b*k*sizeof(scalar_t)+
                  n*k*sizeof(scalar_t)+
                  sizeof(scalar_t)),
            rw_rate/((double)(1<<30)),
            mul_rate/((double)1e6),
            instr_in_mul_rate/((double) 1e6),
            gl,
            master_sum != 0);
    fflush(stdout);

    /* Free device memory. */
    CUDA_WRAP(cudaFree(d_pos));
    CUDA_WRAP(cudaFree(d_adj));
    CUDA_WRAP(cudaFree(d_y));
    CUDA_WRAP(cudaFree(d_z));
    CUDA_WRAP(cudaFree(d_s));
    CUDA_WRAP(cudaFree(d_sum_out));
#ifdef PRECOMPUTE_GF_2_8
    CUDA_WRAP(cudaFree(d_lookup_log));
    CUDA_WRAP(cudaFree(d_lookup_exp));  
#endif

    /* Free sum buffer. */
    FREE(h_vs);


    return master_sum != 0;
}

/***************************************************************** End CUDA. */




/************************************************ Rudimentary graph builder. */

typedef struct 
{
    index_t num_vertices;
    index_t num_edges;
    index_t edge_capacity;
    index_t *edges;
    index_t *colors;
} graph_t;

static index_t *enlarge(index_t m, index_t m_was, index_t *was)
{
    assert(m >= 0 && m_was >= 0);

    index_t *a = (index_t *) MALLOC(sizeof(index_t)*m);
    index_t i;
    if(was != (void *) 0) {
        for(i = 0; i < m_was; i++) {
            a[i] = was[i];
        }
        FREE(was);
    }
    return a;
}

graph_t *graph_alloc(index_t n)
{
    assert(n >= 0);

    index_t i;
    graph_t *g = (graph_t *) MALLOC(sizeof(graph_t));
    g->num_vertices = n;
    g->num_edges = 0;
    g->edge_capacity = 100;
    g->edges = enlarge(2*g->edge_capacity, 0, (index_t *) 0);
    g->colors = (index_t *) MALLOC(sizeof(index_t)*n);
    for(i = 0; i < n; i++)
        g->colors[i] = -1;
    return g;
}

void graph_free(graph_t *g)
{
    FREE(g->edges);
    FREE(g->colors);
    FREE(g);
}

void graph_add_edge(graph_t *g, index_t u, index_t v)
{
    assert(u >= 0 && 
           v >= 0 && 
           u < g->num_vertices &&
           v < g->num_vertices);

    if(g->num_edges == g->edge_capacity) {
        g->edges = enlarge(4*g->edge_capacity, 2*g->edge_capacity, g->edges);
        g->edge_capacity *= 2;
    }

    assert(g->num_edges < g->edge_capacity);

    index_t *e = g->edges + 2*g->num_edges;
    g->num_edges++;
    e[0] = u;
    e[1] = v;
}

index_t *graph_edgebuf(graph_t *g, index_t cap)
{
    g->edges = enlarge(2*g->edge_capacity+2*cap, 2*g->edge_capacity, g->edges);
    index_t *e = g->edges + 2*g->num_edges;
    g->edge_capacity += cap;
    g->num_edges += cap;
    return e;
}

void graph_set_color(graph_t *g, index_t u, index_t c)
{
    assert(u >= 0 && u < g->num_vertices && c >= 0);
    g->colors[u] = c;
}

/************************************ Basic motif query processing routines. */

struct motifq_struct
{
    index_t     is_stub;
    index_t     n;
    index_t     k;
    index_t     *pos;
    index_t     *adj;
    index_t     nl;
    index_t     *l;  
    index_t     ns;
    shade_map_t *shade;
    scalar_t    *vsum;
};

typedef struct motifq_struct motifq_t;

void adjsort(index_t n, index_t *pos, index_t *adj)
{
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        heapsort_indext(deg, adj + pu + 1);
    }
}

void motifq_free(motifq_t *q)
{
    if(!q->is_stub) {
        FREE(q->pos);
        FREE(q->adj);
        FREE(q->l);
        FREE(q->shade);
        FREE(q->vsum);
    }
    FREE(q);
}

index_t motifq_execute(motifq_t *q)
{
    if(q->is_stub)
        return 0;
    return oracle(q->n, q->k, q->pos, q->adj, q->ns, q->shade, irand(), q->vsum);
}

/************** Project a query by cutting out a given interval of vertices. */

index_t get_poscut(index_t n, index_t *pos, index_t *adj, 
                   index_t lo_v, index_t hi_v,
                   index_t *poscut)
{
    // Note: assumes the adjacency lists are sorted
    assert(lo_v <= hi_v);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < lo_v; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t cs, ce;
        index_t l = get_interval(deg, adj + pu + 1,
                                 lo_v, hi_v,
                                 &cs, &ce);
        poscut[u] = deg - l;
    }

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = hi_v+1; u < n; u++) {
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t cs, ce;
        index_t l = get_interval(deg, adj + pu + 1,
                                 lo_v, hi_v,
                                 &cs, &ce);
        poscut[u-hi_v-1+lo_v] = deg - l;
    }

    index_t ncut = n - (hi_v-lo_v+1);
    index_t run = prefixsum(ncut, poscut, 1);
    return run;
}

motifq_t *motifq_cut(motifq_t *q, index_t lo_v, index_t hi_v)
{
    // Note: assumes the adjacency lists are sorted

    index_t n = q->n;
    index_t *pos = q->pos;
    index_t *adj = q->adj;    
    assert(0 <= lo_v && lo_v <= hi_v && hi_v < n);

    // Fast-forward a stub NO when the interval 
    // [lo_v,hi_v] contains an element in q->l
    for(index_t i = 0; i < q->nl; i++) {
        if(q->l[i] >= lo_v && q->l[i] <= hi_v) {
            motifq_t *qs = (motifq_t *) MALLOC(sizeof(motifq_t));
            qs->is_stub = 1;
            return qs;
        }
    }

    index_t ncut = n - (hi_v-lo_v+1);
    index_t *poscut = alloc_idxtab(ncut);
    index_t bcut = get_poscut(n, pos, adj, lo_v, hi_v, poscut);
    index_t *adjcut = alloc_idxtab(bcut);
    index_t gap = hi_v-lo_v+1;

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < ncut; v++) {
        index_t u = v;
        if(u >= lo_v)
            u += gap;
        index_t pu = pos[u];
        index_t degu = adj[pu];
        index_t cs, ce;
        index_t l = get_interval(degu, adj + pu + 1,
                                 lo_v, hi_v,
                                 &cs, &ce);
        index_t pv = poscut[v];
        index_t degv = degu - l;
        adjcut[pv] = degv;
        // could parallelize this too
        for(index_t i = 0; i < cs; i++)
            adjcut[pv + 1 + i] = adj[pu + 1 + i];
        // could parallelize this too
        for(index_t i = cs; i < degv; i++)
            adjcut[pv + 1 + i] = adj[pu + 1 + i + l] - gap;
    }

    motifq_t *qq = (motifq_t *) MALLOC(sizeof(motifq_t));
    qq->is_stub = 0;
    qq->n = ncut;
    qq->k = q->k;
    qq->pos = poscut;
    qq->adj = adjcut;
    qq->nl = q->nl;
    qq->l = (index_t *) MALLOC(sizeof(index_t)*qq->nl);
    for(index_t i = 0; i < qq->nl; i++) {
        index_t u = q->l[i];
        assert(u < lo_v || u > hi_v);
        if(u > hi_v)
            u -= gap;
        qq->l[i] = u;
    }
    qq->ns = q->ns;
    qq->shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*ncut);
    for(index_t v = 0; v < ncut; v++) {
        index_t u = v;
        if(u >= lo_v)
            u += gap;
        qq->shade[v] = q->shade[u];
    }
    qq->vsum = (scalar_t *) MALLOC(sizeof(scalar_t)*qq->n);

    return qq;
}

/***************** Project a query with given projection & embedding arrays. */

#define PROJ_UNDEF 0xFFFFFFFFFFFFFFFFUL

index_t get_posproj(index_t n, index_t *pos, index_t *adj, 
                    index_t nproj, index_t *proj, index_t *embed,
                    index_t *posproj)
{

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++) {
        index_t u = embed[v];
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t degproj = 0;
        for(index_t i = 0; i < deg; i++) {
            index_t w = proj[adj[pu + 1 + i]];
            if(w != PROJ_UNDEF)
                degproj++;
        }
        posproj[v] = degproj;
    }

    index_t run = prefixsum(nproj, posproj, 1);
    return run;
}

motifq_t *motifq_project(motifq_t *q, 
                         index_t nproj, index_t *proj, index_t *embed,
                         index_t nl, index_t *l)
{
    index_t n = q->n;
    index_t *pos = q->pos;
    index_t *adj = q->adj;    
 
    index_t *posproj = alloc_idxtab(nproj);
    index_t bproj = get_posproj(n, pos, adj, nproj, proj, embed, posproj);
    index_t *adjproj = alloc_idxtab(bproj);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++) {
        index_t pv = posproj[v];
        index_t u = embed[v];
        index_t pu = pos[u];
        index_t deg = adj[pu];
        index_t degproj = 0;
        for(index_t i = 0; i < deg; i++) {
            index_t w = proj[adj[pu + 1 + i]];
            if(w != PROJ_UNDEF)
                adjproj[pv + 1 + degproj++] = w;
        }
        adjproj[pv] = degproj;
    }

    motifq_t *qq = (motifq_t *) MALLOC(sizeof(motifq_t));
    qq->is_stub = 0;
    qq->n = nproj;
    qq->k = q->k;
    qq->pos = posproj;
    qq->adj = adjproj;

    // Now project the l array

    assert(q->nl == 0); // l array comes from lister    
    qq->nl = nl;
    qq->l = (index_t *) MALLOC(sizeof(index_t)*nl);
    for(index_t i = 0; i < nl; i++) {
        index_t u = proj[l[i]];
        assert(u != PROJ_UNDEF); // query is a trivial NO !
        qq->l[i] = u;
    }

    // Next set up the projected shades

    qq->ns = q->ns;
    qq->shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*nproj);

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        index_t v = proj[u];
        if(v != PROJ_UNDEF)
            qq->shade[v] = q->shade[u];
    }

    // Reserve a unique shade to every vertex in l
    // while keeping the remaining shades available

    // Reserve shades first ... 
    index_t *l_shade = (index_t *) MALLOC(sizeof(index_t)*nl);
    shade_map_t reserved_shades = 0;
    for(index_t i = 0; i < nl; i++) {
        index_t v = qq->l[i];
        index_t j = 0;
        for(; j < qq->ns; j++)
            if(((qq->shade[v] >> j)&1) == 1 && 
               ((reserved_shades >> j)&1) == 0)
                break;
        assert(j < qq->ns);
        reserved_shades |= 1UL << j;
        l_shade[i] = j;
    }
    // ... then clear all reserved shades in one pass

#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t v = 0; v < nproj; v++)
        qq->shade[v] &= ~reserved_shades;

    // ... and finally set reserved shades
    for(index_t i = 0; i < nl; i++) {
        index_t v = qq->l[i];
        qq->shade[v] = 1UL << l_shade[i];
    }
    FREE(l_shade);

    qq->vsum = (scalar_t *) MALLOC(sizeof(scalar_t)*qq->n);

    return qq;
}

/*************************************************** The interval extractor. */

struct ivlist_struct
{
    index_t start;
    index_t end;
    struct ivlist_struct *prev;
    struct ivlist_struct *next;
};

typedef struct ivlist_struct ivlist_t;

typedef struct ivext_struct 
{
    index_t     n;
    index_t     k;
    ivlist_t    *queue;
    ivlist_t    *active_queue_head;
    ivlist_t    *spare_queue_head;
    ivlist_t    *embed_list;
} ivext_t;

void ivext_enqueue_spare(ivext_t *e, ivlist_t *iv)
{
    pnlinknext(e->spare_queue_head,iv);
}

void ivext_enqueue_active(ivext_t *e, ivlist_t *iv)
{
    pnlinkprev(e->active_queue_head,iv);
}

ivlist_t *ivext_dequeue_first_nonsingleton(ivext_t *e)
{
    ivlist_t *iv = e->active_queue_head->next;  
    for(; 
        iv != e->active_queue_head; 
        iv = iv->next)
        if(iv->end - iv->start + 1 > 1)
            break;
    assert(iv != e->active_queue_head);
    pnunlink(iv);
    return iv;
}

ivlist_t *ivext_get_spare(ivext_t *e)
{
    assert(e->spare_queue_head->next != e->spare_queue_head);
    ivlist_t *iv = e->spare_queue_head->next;
    pnunlink(iv);
    return iv;
}

void ivext_reset(ivext_t *e)
{
    e->active_queue_head = e->queue + 0;
    e->spare_queue_head  = e->queue + 1;
    e->active_queue_head->next = e->active_queue_head;
    e->active_queue_head->prev = e->active_queue_head;
    e->spare_queue_head->prev  = e->spare_queue_head;
    e->spare_queue_head->next  = e->spare_queue_head;  
    e->embed_list = (ivlist_t *) 0;

    for(index_t i = 0; i < e->k + 2; i++)
        ivext_enqueue_spare(e, e->queue + 2 + i); // rot-safe
    ivlist_t *iv = ivext_get_spare(e);
    iv->start = 0;
    iv->end = e->n-1;
    ivext_enqueue_active(e, iv);
}

ivext_t *ivext_alloc(index_t n, index_t k)
{
    ivext_t *e = (ivext_t *) MALLOC(sizeof(ivext_t));
    e->n = n;
    e->k = k;
    e->queue = (ivlist_t *) MALLOC(sizeof(ivlist_t)*(k+4)); // rot-safe
    ivext_reset(e);
    return e;
}

void ivext_free(ivext_t *e)
{
    ivlist_t *el = e->embed_list;
    while(el != (ivlist_t *) 0) {
        ivlist_t *temp = el;
        el = el->next;
        FREE(temp);
    }
    FREE(e->queue);
    FREE(e);
}

void ivext_project(ivext_t *e, ivlist_t *iv)
{
    for(ivlist_t *z = e->active_queue_head->next; 
        z != e->active_queue_head; 
        z = z->next) {
        assert(z->end < iv->start ||
               z->start > iv->end);
        if(z->start > iv->end) {
            z->start -= iv->end-iv->start+1;
            z->end   -= iv->end-iv->start+1;
        }
    }

    ivlist_t *em = (ivlist_t *) MALLOC(sizeof(ivlist_t));
    em->start    = iv->start;
    em->end      = iv->end;
    em->next     = e->embed_list;
    e->embed_list = em;
}

index_t ivext_embed(ivext_t *e, index_t u)
{
    ivlist_t *el = e->embed_list;
    while(el != (ivlist_t *) 0) {
        if(u >= el->start)
            u += el->end - el->start + 1;
        el = el->next;
    }
    return u;
}

ivlist_t *ivext_halve(ivext_t *e, ivlist_t *iv)
{
    assert(iv->end - iv->start + 1 >= 2);
    index_t mid = (iv->start + iv->end)/2;  // mid < iv->end    
    ivlist_t *h = ivext_get_spare(e);
    h->start = iv->start;
    h->end = mid;
    iv->start = mid+1;
    return h;
}
    
index_t ivext_queue_size(ivext_t *e)
{
    index_t s = 0;
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next)
        s += iv->end-iv->start+1;
    return s;
}

index_t ivext_num_active_intervals(ivext_t *e)
{
    index_t s = 0;
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next)
        s++;
    return s;
}

void ivext_queue_print(FILE *out, ivext_t *e, index_t rot)
{
    index_t j = 0;
    char x[16384];
    char y[16384];
    y[0] = '\0';
    sprintf(x, "%c%12ld [", 
            rot == 0 ? ' ' : 'R',
            ivext_queue_size(e));
    strcat(y, x);
    for(ivlist_t *iv = e->active_queue_head->next; 
        iv != e->active_queue_head; 
        iv = iv->next) {
        assert(iv->start <= iv->end);
        if(iv->start < iv->end)
            sprintf(x, 
                    "%s[%ld:%ld]", 
                    j++ == 0 ? "" : ",",
                    ivext_embed(e, iv->start),
                    ivext_embed(e, iv->end));
        else
            sprintf(x, 
                    "%s[%ld]", 
                    j++ == 0 ? "[" : ",",
                    ivext_embed(e, iv->start));
        strcat(y, x);
    }   
    strcat(y, "] ");
    fprintf(out, "%-120s", y);
    fflush(out);
}

index_t extract_match(index_t is_root, motifq_t *query, index_t *match)
{
    // Assumes adjancency lists of query are sorted.

    fprintf(stdout, "extract: %ld %ld %ld\n", query->n, query->k, query->nl);
    push_time();
    assert(query->k <= query->n);
    ivext_t *e = ivext_alloc(query->n, query->k);
    ivext_queue_print(stdout, e, 0);
    if(!motifq_execute(query)) {
        fprintf(stdout, " -- false\n");
        ivext_free(e);
        if(!is_root)
            motifq_free(query);
        double time = pop_time();
        fprintf(stdout, "extract done [%.2lf ms]\n", time);
        return 0;
    }
    fprintf(stdout, " -- true\n");
           
    while(ivext_queue_size(e) > e->k) {
        ivlist_t *iv = ivext_dequeue_first_nonsingleton(e);
        ivlist_t *h = ivext_halve(e, iv);
        ivext_enqueue_active(e, iv);
        motifq_t *qq = motifq_cut(query, h->start, h->end);
        ivext_queue_print(stdout, e, 0);
        if(motifq_execute(qq)) {
            fprintf(stdout, " -- true\n");
            if(!is_root)
                motifq_free(query);
            query = qq;
            is_root = 0;
            ivext_project(e, h);
            ivext_enqueue_spare(e, h);
        } else {
            fprintf(stdout, " -- false\n");
            motifq_free(qq);
            pnunlink(iv);
            ivext_enqueue_active(e, h);
            qq = motifq_cut(query, iv->start, iv->end);
            ivext_queue_print(stdout, e, 0);
            if(motifq_execute(qq)) {
                fprintf(stdout, " -- true\n");
                if(!is_root)
                    motifq_free(query);
                query = qq;
                is_root = 0;
                ivext_project(e, iv);
                ivext_enqueue_spare(e, iv);
            } else {
                fprintf(stdout, " -- false\n");
                motifq_free(qq);
                ivext_enqueue_active(e, iv);
                while(ivext_num_active_intervals(e) > e->k) {
                    // Rotate queue until outlier is out ...
                    ivlist_t *iv = e->active_queue_head->next;  
                    pnunlink(iv);
                    qq = motifq_cut(query, iv->start, iv->end);
                    ivext_queue_print(stdout, e, 1);
                    if(motifq_execute(qq)) {
                        fprintf(stdout, " -- true\n");
                        if(!is_root)
                            motifq_free(query);
                        query = qq;
                        is_root = 0;
                        ivext_project(e, iv);
                        ivext_enqueue_spare(e, iv);
                    } else {
                        fprintf(stdout, " -- false\n");
                        motifq_free(qq);
                        ivext_enqueue_active(e, iv);
                    }
                }
            }
        }
    }
    for(index_t i = 0; i < query->k; i++)
        match[i] = ivext_embed(e, i);
    ivext_free(e);
    if(!is_root)
        motifq_free(query);
    double time = pop_time();
    fprintf(stdout, "extract done [%.2lf ms]\n", time);
    return 1;
}

/*************************************************************** The lister. */

#define M_QUERY       0
#define M_OPEN        1
#define M_CLOSE       2
#define M_REWIND_U    3
#define M_REWIND_L    4

index_t command_mnemonic(index_t command) 
{
    return command >> 60;   
}

index_t command_index(index_t command)
{
    return command & (~(0xFFUL<<60));
}

index_t to_command_idx(index_t mnemonic, index_t idx)
{
    assert(idx < (1UL << 60));
    return (mnemonic << 60)|idx;
}

index_t to_command(index_t mnemonic)
{
    return to_command_idx(mnemonic, 0UL);
}

typedef struct 
{
    index_t n;              // number of elements in universe
    index_t k;              // size of the sets to be listed
    index_t *u;             // upper bound as a bitmap
    index_t u_size;         // size of upper bound
    index_t *l;             // lower bound 
    index_t l_size;         // size of lower bound
    index_t *stack;         // a stack for maintaining state
    index_t stack_capacity; // ... the capacity of the stack    
    index_t top;            // index of stack top
    motifq_t *root;         // the root query
} lister_t;

void lister_push(lister_t *t, index_t word)
{
    assert(t->top + 1 < t->stack_capacity);
    t->stack[++t->top] = word;
}

index_t lister_pop(lister_t *t)
{
    return t->stack[t->top--];
}

index_t lister_have_work(lister_t *t)
{
    return t->top >= 0;
}

index_t lister_in_l(lister_t *t, index_t j)
{
    for(index_t i = 0; i < t->l_size; i++)
        if(t->l[i] == j)
            return 1;
    return 0;
}

void lister_push_l(lister_t *t, index_t j)
{
    assert(!lister_in_l(t, j) && t->l_size < t->k);
    t->l[t->l_size++] = j;
}

void lister_pop_l(lister_t *t)
{
    assert(t->l_size > 0);
    t->l_size--;
}

void lister_reset(lister_t *t)
{
    t->l_size = 0;
    t->top = -1;
    lister_push(t, to_command(M_QUERY));
    for(index_t i = 0; i < t->n; i++)
        bitset(t->u, i, 1);
    t->u_size = t->n;
}

lister_t *lister_alloc(index_t n, index_t k, motifq_t *root)
{
    assert(n >= 1 && n < (1UL << 60) && k >= 1 && k <= n);
    lister_t *t = (lister_t *) MALLOC(sizeof(lister_t));
    t->n = n;
    t->k = k;
    t->u = alloc_idxtab((n+63)/64);
    t->l = alloc_idxtab(k);
    t->stack_capacity = n + k*(k+1+2*k) + 1;
    t->stack = alloc_idxtab(t->stack_capacity);
    lister_reset(t);
    t->root = root;
    if(t->root != (motifq_t *) 0) {
        assert(t->root->n == t->n);
        assert(t->root->k == t->k);
        assert(t->root->nl == 0);
    }
    return t;
}

void lister_free(lister_t *t)
{
    if(t->root != (motifq_t *) 0)
        motifq_free(t->root);
    FREE(t->u);
    FREE(t->l);
    FREE(t->stack);
    FREE(t);
}

void lister_get_proj_embed(lister_t *t, 
                           index_t  **proj_out, 
                           index_t  **embed_out)
{
    index_t n = t->n;
    index_t usize = t->u_size;

    index_t *embed = (index_t *) MALLOC(sizeof(index_t)*usize);
    index_t *proj  = (index_t *) MALLOC(sizeof(index_t)*n);

    // could parallelize this (needs parallel prefix sum)
    index_t run = 0;
    for(index_t i = 0; i < n; i++) {
        if(bitget(t->u, i)) {
            proj[i]    = run;
            embed[run] = i;
            run++;
        } else {
            proj[i] = PROJ_UNDEF;
        }
    }
    assert(run == usize);

    *proj_out  = proj;
    *embed_out = embed;
}

void lister_query_setup(lister_t *t, motifq_t **q_out, index_t **embed_out)
{
    index_t *proj;
    index_t *embed;

    // set up the projection with u and l
    lister_get_proj_embed(t, &proj, &embed);
    motifq_t *qq = motifq_project(t->root, 
                                  t->u_size, proj, embed, 
                                  t->l_size, t->l);
    FREE(proj);

    *q_out     = qq;
    *embed_out = embed;
}

index_t lister_extract(lister_t *t, index_t *s)
{
    // assumes t->u contains all elements of t->l
    // (otherwise query is trivial no)

    assert(t->root != (motifq_t *) 0);
    
    if(t->u_size == t->n) {
        // rush the root query without setting up a copy
        return extract_match(1, t->root, s);
    } else {
        // a first order of business is to set up the query 
        // based on the current t->l and t->u; this includes
        // also setting up the embedding back to the root,
        // in case we are lucky and actually discover a match
        motifq_t *qq; // will be released by extractor
        index_t *embed;
        lister_query_setup(t, &qq, &embed);
        
        // now execute the interval extractor ...
        index_t got_match = extract_match(0, qq, s);
        
        // ... and embed the match (if any) 
        if(got_match) {
            for(index_t i = 0; i < t->k; i++)
                s[i] = embed[s[i]];
        }
        FREE(embed);
        return got_match;
    }
}

index_t lister_run(lister_t *t, index_t *s)
{
    while(lister_have_work(t)) {
        index_t cmd = lister_pop(t);
        index_t mnem = command_mnemonic(cmd);
        index_t idx = command_index(cmd);
        switch(mnem) {
        case M_QUERY:
            if(t->k <= t->u_size && lister_extract(t, s)) {
                // we have discovered a match, which we need to
                // put on the stack to continue work when the user
                // requests this
                for(index_t i = 0; i < t->k; i++)
                    lister_push(t, s[i]);
                lister_push(t, to_command_idx(M_OPEN, t->k-1));
                // now report our discovery to user
                return 1;
            }
            break;
        case M_OPEN:
            {
                index_t *x = t->stack + t->top - t->k + 1;
                index_t k = 0;
                for(; k < idx; k++)
                    if(!lister_in_l(t, x[k]))
                        break;
                if(k == idx) {
                    // opening on last element of x not in l
                    // so we can dispense with x as long as we remember to 
                    // insert x[idx] back to u when rewinding
                    for(index_t j = 0; j < t->k; j++)
                        lister_pop(t); // axe x from stack
                    if(!lister_in_l(t, x[idx])) {
                        bitset(t->u, x[idx], 0); // remove x[idx] from u
                        t->u_size--;
                        lister_push(t, to_command_idx(M_REWIND_U, x[idx]));
                        lister_push(t, to_command(M_QUERY));
                    }
                } else {
                    // have still other elements of x that we need to
                    // open on, so must keep x in stack 
                    // --
                    // invariant that controls stack size:
                    // each open increases l by at least one
                    lister_push(t, to_command_idx(M_CLOSE, idx));
                    if(!lister_in_l(t, x[idx])) {
                        bitset(t->u, x[idx], 0); // remove x[idx] from u
                        t->u_size--;
                        lister_push(t, to_command_idx(M_REWIND_U, x[idx]));
                        // force x[0],x[1],...,x[idx-1] to l
                        index_t j = 0;
                        for(; j < idx; j++) {
                            if(!lister_in_l(t, x[j])) {
                                if(t->l_size >= t->k)
                                    break;
                                lister_push_l(t, x[j]);
                                lister_push(t, 
                                            to_command_idx(M_REWIND_L, x[j]));
                            }
                        }
                        if(j == idx)
                            lister_push(t, to_command(M_QUERY));
                    }
                }
            }
            break;
        case M_CLOSE:
            assert(idx > 0);
            lister_push(t, to_command_idx(M_OPEN, idx-1));
            break;
        case M_REWIND_U:
            bitset(t->u, idx, 1);
            t->u_size++;
            break;
        case M_REWIND_L:
            lister_pop_l(t);
            break;
        }
    }
    lister_push(t, to_command(M_QUERY));
    return 0;
}

/******************************************************* Root query builder. */

motifq_t *root_build(graph_t *g, index_t k, index_t *kk)
{
    push_memtrack();

    index_t n = g->num_vertices;
    index_t m = 2*g->num_edges;
    index_t *pos = alloc_idxtab(n);
    index_t *adj = alloc_idxtab(n+m);
    index_t ns = k;
    shade_map_t *shade = (shade_map_t *) MALLOC(sizeof(shade_map_t)*n);

    motifq_t *root = (motifq_t *) MALLOC(sizeof(motifq_t));
    root->is_stub = 0;
    root->n       = g->num_vertices;
    root->k       = k;
    root->pos     = pos;
    root->adj     = adj;
    root->nl      = 0;
    root->l       = (index_t *) MALLOC(sizeof(index_t)*root->nl);
    root->ns      = ns;
    root->shade   = shade;
    root->vsum    = (scalar_t *) MALLOC(sizeof(scalar_t)*root->n);

    push_time();
    fprintf(stdout, "root build ... ");
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        pos[u] = 0;
    double time = pop_time();
    fprintf(stdout, "[zero: %.2lf ms] ", time);
    fflush(stdout);
    
    push_time();
    index_t *e = g->edges;
#ifdef BUILD_PARALLEL
   // Parallel occurrence count
   // -- each thread is responsible for a group of bins, 
   //    all threads scan the entire list of edges
    index_t nt = num_threads();
    index_t block_size = n/nt;
#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < m; j++) {
            index_t u = e[j];
            if(start <= u && u <= stop)                
                pos[u]++; // I am responsible for u, record adjacency to u
        }
    }
#else
    for(index_t j = 0; j < m; j++)
        pos[e[j]]++;
#endif

    index_t run = prefixsum(n, pos, 1);
    assert(run == n+m);
    time = pop_time();
    fprintf(stdout, "[pos: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++)
        adj[pos[u]] = 0;

    e = g->edges;
#ifdef BUILD_PARALLEL
    // Parallel aggregation to bins 
    // -- each thread is responsible for a group of bins, 
    //    all threads scan the entire list of edges
    nt = num_threads();
    block_size = n/nt;
#pragma omp parallel for
    for(index_t t = 0; t < nt; t++) {
        index_t start = t*block_size;
        index_t stop = (t == nt-1) ? n-1 : (start+block_size-1);
        for(index_t j = 0; j < m; j+=2) {
            index_t u0 = e[j+0];
            index_t u1 = e[j+1];
            if(start <= u0 && u0 <= stop) {
                // I am responsible for u0, record adjacency to u1
                index_t pu0 = pos[u0];
                adj[pu0 + 1 + adj[pu0]++] = u1;
            }
            if(start <= u1 && u1 <= stop) {
                // I am responsible for u1, record adjacency to u0
                index_t pu1 = pos[u1];
                adj[pu1 + 1 + adj[pu1]++] = u0;
            }
        }
    }
#else
    for(index_t j = 0; j < m; j+=2) {
        index_t u0 = e[j+0];
        index_t u1 = e[j+1];
        index_t p0 = pos[u0];
        index_t p1 = pos[u1];       
        adj[p1 + 1 + adj[p1]++] = u0;
        adj[p0 + 1 + adj[p0]++] = u1;
    }
#endif
    time = pop_time();
    fprintf(stdout, "[adj: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
    adjsort(n, pos, adj);
    time = pop_time();
    fprintf(stdout, "[adjsort: %.2lf ms] ", time);
    fflush(stdout);

    push_time();
#ifdef BUILD_PARALLEL
#pragma omp parallel for
#endif
    for(index_t u = 0; u < n; u++) {
        shade_map_t s = 0;
        for(index_t j = 0; j < k; j++)
            if(g->colors[u] == kk[j])
                s |= 1UL << j;
        shade[u] = s;
    }
    time = pop_time();
    fprintf(stdout, "[shade: %.2lf ms] ", time);
    fflush(stdout);

    time = pop_time();
    fprintf(stdout, "done. [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    fflush(stdout);

    return root;
}

/***************************************************** Input reader (ASCII). */

void skipws(FILE *in)
{
    int c;
    do {
        c = fgetc(in);
        if(c == '#') {
            do {
                c = fgetc(in);
            } while(c != EOF && c != '\n');
        }
    } while(c != EOF && isspace(c));
    if(c != EOF)
        ungetc(c, in);
}

#define CMD_NOP          0
#define CMD_TEST_UNIQUE  1
#define CMD_TEST_COUNT   2
#define CMD_RUN_ORACLE   3
#define CMD_LIST_FIRST   4
#define CMD_LIST_ALL     5

const char *cmd_legend[] = { "no operation", "test unique", "test count", "run oracle", "list first", "list all" };

void reader_ascii(FILE *in, 
                  graph_t **g_out, index_t *k_out, index_t **kk_out, 
                  index_t *cmd_out, index_t **cmd_args_out)
{
    push_time();
    push_memtrack();
    
    index_t n = 0;
    index_t m = 0;
    graph_t *g = (graph_t *) 0;
    index_t i, j, d, k;
    index_t *kk = (index_t *) 0;
    index_t cmd = CMD_NOP;
    index_t *cmd_args = (index_t *) 0;
    skipws(in);
    while(!feof(in)) {
        skipws(in);
        int c = fgetc(in);
        switch(c) {
        case 'p':
            if(g != (graph_t *) 0)
                ERROR("duplicate parameter line");
            skipws(in);
            if(fscanf(in, "motif %ld %ld", &n, &m) != 2)
                ERROR("invalid parameter line");
            if(n <= 0 || m < 0) 
                ERROR("invalid input parameters (n = %ld, m = %ld)", n, m);
            g = graph_alloc(n);
            break;
        case 'e':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before edges");
            skipws(in);
            if(fscanf(in, "%ld %ld", &i, &j) != 2)
                ERROR("invalid edge line");
            if(i < 1 || i > n || j < 1 || j > n)
                ERROR("invalid edge (i = %ld, j = %ld with n = %ld)", 
                      i, j, n);
            graph_add_edge(g, i-1, j-1);
            break;
        case 'n':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before vertex colors");
            skipws(in);
            if(fscanf(in, "%ld %ld", &i, &d) != 2)
                ERROR("invalid color line");
            if(i < 1 || i > n || d < 1)
                ERROR("invalid color line (i = %ld, d = %ld with n = %ld)", 
                      i, d, n);
            graph_set_color(g, i-1, d-1);
            break;
        case 'k':
            if(g == (graph_t *) 0)
                ERROR("parameter line must be given before motif");
            skipws(in);
            if(fscanf(in, "%ld", &k) != 1)
                ERROR("invalid motif line");
            if(k < 1 || k > n)
                ERROR("invalid motif line (k = %ld with n = %d)", k, n);
            kk = alloc_idxtab(k);
            for(index_t u = 0; u < k; u++) {
                skipws(in);
                if(fscanf(in, "%ld", &i) != 1)
                    ERROR("error parsing motif line");
                if(i < 1)
                    ERROR("invalid color on motif line (i = %ld)", i);
                kk[u] = i-1;
            }
            break;
        case 't':
            if(g == (graph_t *) 0 || kk == (index_t *) 0)
                ERROR("parameter and motif lines must be given before test");
            skipws(in);
            {
                char cmdstr[128];
                if(fscanf(in, "%100s", cmdstr) != 1)
                    ERROR("invalid test command");
                if(!strcmp(cmdstr, "unique")) {
                    cmd_args = alloc_idxtab(k);
                    for(index_t u = 0; u < k; u++) {
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 1 || i > n)
                            ERROR("invalid test line entry (i = %ld)", i);
                        cmd_args[u] = i-1;
                    }
                    heapsort_indext(k, cmd_args);
                    for(index_t u = 1; u < k; u++)
                        if(cmd_args[u-1] >= cmd_args[u])
                            ERROR("test line contains duplicate entries");
                    cmd = CMD_TEST_UNIQUE;
                } else {
                    if(!strcmp(cmdstr, "count")) {
                        cmd_args = alloc_idxtab(1);
                        skipws(in);
                        if(fscanf(in, "%ld", &i) != 1)
                            ERROR("error parsing test line");
                        if(i < 0)
                            ERROR("count on test line cannot be negative");
                        cmd = CMD_TEST_COUNT;
                        cmd_args[0] = i;
                    } else {
                        ERROR("unrecognized test command \"%s\"", cmdstr);
                    }
                }
            }
            break;
        case EOF:
            break;
        default:
            ERROR("parse error");
        }
    }

    if(g == (graph_t *) 0)
        ERROR("no graph given in input");
    if(kk == (index_t *) 0)
        ERROR("no motif given in input");

    for(index_t i = 0; i < n; i++) {
        if(g->colors[i] == -1)
            ERROR("no color assigned to vertex i = %ld", i);
    }
    double time = pop_time();
    fprintf(stdout, 
            "input: n = %ld, m = %ld, k = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            k,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");    
    
    *g_out = g;
    *k_out = k;
    *kk_out = kk;
    *cmd_out = cmd;
    *cmd_args_out = cmd_args;
}

/**************************************************** Input reader (binary). */

#define BIN_MAGIC 0x1234567890ABCDEFUL

void reader_bin(FILE *in, 
                graph_t **g_out, index_t *k_out, index_t **kk_out, 
                index_t *cmd_out, index_t **cmd_args_out)
{
    push_time();
    push_memtrack();
    
    index_t magic = 0;
    index_t n = 0;
    index_t m = 0;
    graph_t *g = (graph_t *) 0;
    index_t k = 0;
    index_t has_target = 0;
    index_t *kk = (index_t *) 0;
    index_t cmd = CMD_NOP;
    index_t *cmd_args = (index_t *) 0;
    
    if(fread(&magic, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    if(magic != BIN_MAGIC)
        ERROR("error reading input");
    if(fread(&n, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    if(fread(&m, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    assert(n >= 0 && m >= 0 && m%2 == 0);
    g = graph_alloc(n);
    index_t *e = graph_edgebuf(g, m/2);
    if(fread(e, sizeof(index_t), m, in) != m)
        ERROR("error reading input");
    if(fread(g->colors, sizeof(index_t), n, in) != n)
        ERROR("error reading input");
    if(fread(&has_target, sizeof(index_t), 1UL, in) != 1UL)
        ERROR("error reading input");
    assert(has_target == 0 || has_target == 1);
    if(has_target) {
        if(fread(&k, sizeof(index_t), 1UL, in) != 1UL)
            ERROR("error reading input");
        assert(k >= 0);
        kk = alloc_idxtab(k);
        if(fread(kk, sizeof(index_t), k, in) != k)
            ERROR("error reading input");         
        if(fread(&cmd, sizeof(index_t), 1UL, in) != 1UL)
            ERROR("error reading input");         
        switch(cmd) {
        case CMD_NOP:
            break;
        case CMD_TEST_UNIQUE:
            cmd_args = alloc_idxtab(k);
            if(fread(cmd_args, sizeof(index_t), k, in) != k)
                ERROR("error reading input");         
            shellsort(k, cmd_args);
            break;          
        case CMD_TEST_COUNT:
            cmd_args = alloc_idxtab(1);
            if(fread(cmd_args, sizeof(index_t), 1UL, in) != 1UL)
                ERROR("error reading input");                         
            break;          
        default:
            ERROR("invalid command in binary input stream");
            break;          
        }
    }

    double time = pop_time();
    fprintf(stdout, 
            "input: n = %ld, m = %ld, k = %ld [%.2lf ms] ", 
            g->num_vertices,
            g->num_edges,
            k,
            time);
    print_pop_memtrack();
    fprintf(stdout, " ");
    print_current_mem();
    fprintf(stdout, "\n");
    
    *g_out = g;
    *k_out = k;
    *kk_out = kk;
    *cmd_out = cmd;
    *cmd_args_out = cmd_args;
}


/****************************************************** Program entry point. */

int main(int argc, char **argv)
{

    push_time();
    push_memtrack();
    
    index_t arg_cmd = CMD_NOP;
    index_t have_seed = 0;
    index_t seed = 123456789;
    for(index_t f = 1; f < argc; f++) {
        if(argv[f][0] == '-') {
            if(!strcmp(argv[f], "-bin")) {
                flag_bin_input = 1;
            }
            if(!strcmp(argv[f], "-ascii")) {
                flag_bin_input = 0;
            }
            if(!strcmp(argv[f], "-oracle")) {
                arg_cmd = CMD_RUN_ORACLE;
            }
            if(!strcmp(argv[f], "-first")) {
                arg_cmd = CMD_LIST_FIRST;
            }
            if(!strcmp(argv[f], "-all")) {
                arg_cmd = CMD_LIST_ALL;
            }
            if(!strcmp(argv[f], "-seed")) {
                if(f == argc - 1)
                    ERROR("random seed missing from command line");
                seed = atol(argv[++f]);
                have_seed = 1;
            }
        }
    }
    fprintf(stdout, "invoked as:");
    for(index_t f = 0; f < argc; f++)
        fprintf(stdout, " %s", argv[f]);
    fprintf(stdout, "\n");

    if(have_seed == 0) {
        fprintf(stdout, 
                "no random seed given, defaulting to %ld\n", seed);
    }
    fprintf(stdout, "random seed = %ld\n", seed);
    
    srand(seed); 

    graph_t *g;
    index_t k;
    index_t *kk;
    index_t input_cmd;
    index_t *cmd_args;
    if(flag_bin_input) {
        reader_bin(stdin, &g, &k, &kk, &input_cmd, &cmd_args);
    } else {
        reader_ascii(stdin, &g, &k, &kk, &input_cmd, &cmd_args);
    }
    index_t cmd = input_cmd;  // by default execute command in input stream
    if(arg_cmd != CMD_NOP)
        cmd = arg_cmd;        // override command in input stream

    motifq_t *root = root_build(g, k, kk);
    graph_free(g);
    FREE(kk);

    fprintf(stdout, "command: %s\n", cmd_legend[cmd]);
    fflush(stdout);

    push_time();
    switch(cmd) {
    case CMD_NOP:
        motifq_free(root);
        break;
    case CMD_TEST_UNIQUE:
        {
            index_t n = root->n;
            index_t k = root->k;
            lister_t *t = lister_alloc(n, k, root);
            index_t *get = alloc_idxtab(k);
            index_t ct = 0;
            while(lister_run(t, get)) {
                assert(ct == 0);
                fprintf(stdout, "found %ld: ", ct);
                for(index_t i = 0; i < k; i++)
                    fprintf(stdout, "%ld%s", get[i], i == k-1 ? "\n" : " ");
                for(index_t l = 0; l < k; l++)
                    assert(get[l] == cmd_args[l]);
                ct++;
            }
            assert(ct == 1);
            FREE(get);
            lister_free(t);
        }
        break;
    case CMD_LIST_FIRST:
    case CMD_LIST_ALL:
    case CMD_TEST_COUNT:
        {
            index_t n = root->n;
            index_t k = root->k;
            lister_t *t = lister_alloc(n, k, root);
            index_t *get = alloc_idxtab(k);
            index_t ct = 0;
            while(lister_run(t, get)) {
                fprintf(stdout, "found %ld: ", ct);
                for(index_t i = 0; i < k; i++)
                    fprintf(stdout, "%ld%s", get[i], i == k-1 ? "\n" : " ");
                ct++;
                if(cmd == CMD_LIST_FIRST)
                    break;
            }
            if(cmd == CMD_TEST_COUNT) {
                fprintf(stdout, "count = %ld, target = %ld\n", ct, cmd_args[0]);
                assert(ct == cmd_args[0]);
            }
            FREE(get);
            lister_free(t);
        }
        break;
    case CMD_RUN_ORACLE:
        if(motifq_execute(root)) {
            index_t support_size = 0;
            assert(!root->is_stub);
            scalar_t *master_vsum = root->vsum;
            for(index_t i = 0; i < root->n; i++) {
                if(master_vsum[i] != 0) {
                    support_size++;
                }
            }
            fprintf(stdout, " -- true [%ld]\n", support_size);
        } else {
            fprintf(stdout, " -- false [0]\n");
        }
        motifq_free(root);
        break;
    default:
        assert(0);
        break;
    }
    double time = pop_time();
    fprintf(stdout, "command done [%.2lf ms]\n", time);
    if(input_cmd != CMD_NOP)
        FREE(cmd_args);

    time = pop_time();
    fprintf(stdout, "grand total [%.2lf ms] ", time);
    print_pop_memtrack();
    fprintf(stdout, "\n");
    fprintf(stdout, "host: %s\n", sysdep_hostname());
    fprintf(stdout, 
            "build: %s\n",
        LINE_TYPE);
    fprintf(stdout, 
            "compiler: gcc %d.%d.%d\n",
            __GNUC__,
            __GNUC_MINOR__,
            __GNUC_PATCHLEVEL__);
    fflush(stdout);
    assert(malloc_balance == 0);
    assert(memtrack_stack_top < 0);

    return 0;
}
