
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

#ifndef GF_H
#define GF_H

/********************************** Basic types for finite field arithmetic. */

// Terminology
// -----------
//
//   Scalar:      one field element (e.g. from GF(2^32))
//
//   Line:        the unit of data that one thread processes, 
//                usually about one or two cache lines; 
//                each line consists of a fixed number scalars 
//                (with varying encodings); intuition
//                "a _line_ is orthogonal to a SIMD _lane_",
//                "as broad a hardware-supported word as possible"
//
//   Line array:  an array consisting of lines
//                (warning: a line may reside in _non-contiguous_ locations
//                          in a line array)
//
//   Word:        the basic unit of data from which scalars and lines are built
//                (e.g. a 32-bit word)
//
//   Segment:     an internal parameter used to address a non-contiguous 
//                line array; ideally would want to dispense with this
//                since it breaches an otherwise clean interface
//

/* 
 * Remarks: 
 *
 * 1)
 * Assumes an index data type "index_t" is defined. 
 *
 * 2)
 * Variables of type "int" are assumed to be large enough
 * to hold quantities for purposes of shifting a variable. 
 *
 * 3) 
 * The primitive polynomials used in field construction:
 *
 * degree  8:  [x^8 +] x^4 + x^3 + x^2 + 1            ~ 0x1D = 29
 * degree 16: [x^16 +] x^5 + x^3 + x^2 + 1            ~ 0x2D = 45
 * degree 32: [x^32 +] x^7 + x^5 + x^3 + x^2 + x + 1  ~ 0xAF = 175
 * degree 64: [x^64 +] x^4 + x^3 + x + 1              ~ 0x1B = 27
 *
 * http://www.ams.org/journals/mcom/1962-16-079/S0025-5718-1962-0148256-1/S0025-5718-1962-0148256-1.pdf
 *
 */

#define GF2_8_MODULUS   0x01D
#define GF2_16_MODULUS  0x02D
#define GF2_32_MODULUS  0x0AF
#define GF2_64_MODULUS  0x01BL

typedef unsigned int word_t;    // GPU defaults to 32-bit words
typedef uint4 word4_t;          // uint4 is a type supported by 
                                // the NVIDIA compiler nvcc

// Pick one definition from what is below.

// GF(2^32)

//#define LINE_1_GF2_32
//#define LINE_4_GF2_32
//#define LINE_8_GF2_32

// GF(2^8)

//#define LINE_1_GF2_8   
//#define LINE_4_GF2_8
//#define LINE_16_GF2_8
//#define LINE_32_GF2_8

//#define LINE_32_GF2_8_BITSLICE

//#define LINE_1_GF2_8_EXPLOG
//#define LINE_4_GF2_8_EXPLOG
//#define LINE_16_GF2_8_EXPLOG
//#define LINE_32_GF2_8_EXPLOG

/************************ One-word line representing an element of GF(2^32). */

#ifdef LINE_1_GF2_32
#define LINE_TYPE "1 x GF(2^{32}) with one 32-bit word"

#define GF2_B       32
#define GF2_MODULUS GF2_32_MODULUS

typedef word_t scalar_t;
typedef word_t line_t;
typedef word_t line_array_t;

#define SCALARS_IN_LINE 1

#define LINE_ARRAY_SIZE(b) (sizeof(word_t)*(size_t)(b))
#define LINE_SEGMENT_SIZE(b) 0 // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ((word_t *) ptr)[idx+0*seg];\

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((word_t *) ptr)[idx+0*seg] = (source);\

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[idx+0*seg];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx+0*seg]   = source;\

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
    target = left^right;\

#define LINE_MUL(target, left, right)\
    GF2_32_MUL(target, left, right);\

#define LINE_MUL_SCALAR(target, source, scalar)\
    LINE_MUL(target, source, scalar);\

#define LINE_SUM(target, source)\
    target = source;\

#define LINE_SET_ZERO(target)\
{\
    target = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
    target = source+0*idx;\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_32_MUL(target, left, right);\

#define LINE_MUL_INSTR (195)

#define WORD_TO_SCALAR(x) ((x)&0xFFFFFFFF)

#endif 

/************************** 4-word line representing 4 elements of GF(2^32). */

#ifdef LINE_4_GF2_32
#define LINE_TYPE "4 x GF(2^{32}) with four 32-bit words"

#define GF2_B       32
#define GF2_MODULUS GF2_32_MODULUS

typedef word_t scalar_t;
typedef word4_t line_t;
typedef word4_t line_array_t;

#define SCALARS_IN_LINE 4

#define LINE_ARRAY_SIZE(b) (sizeof(word4_t)*((size_t)b)/4)
#define LINE_SEGMENT_SIZE(b) 0 // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ((word_t *) ptr)[idx+0*seg];\

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((word_t *) ptr)[idx+0*seg] = (source);\

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[idx+0*seg];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx+0*seg]   = source;\

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
{\
    target.x = left.x^right.x;\
    target.y = left.y^right.y;\
    target.z = left.z^right.z;\
    target.w = left.w^right.w;\
}\

#define LINE_MUL(target, left, right)\
    GF2_32_MUL_QUAD(target.x, target.y, target.z, target.w,\
                      left.x,   left.y,   left.z,   left.w,\
                     right.x,  right.y,  right.z,  right.w);\

#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    line_t temp;\
    temp.x = scalar;\
    temp.y = temp.x;\
    temp.z = temp.x;\
    temp.w = temp.x;\
    LINE_MUL(target, source, temp);\
}\

#define LINE_SUM(target, source)\
{\
    scalar_t temp;\
    temp = source.x^source.y^source.z^source.w;\
    target = temp;\
}\

#define LINE_SET_ZERO(target)\
{\
    target.x = 0;\
    target.y = 0;\
    target.z = 0;\
    target.w = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
    target.x = (idx == 0) ? source : target.x;\
    target.y = (idx == 1) ? source : target.y;\
    target.z = (idx == 2) ? source : target.z;\
    target.w = (idx == 3) ? source : target.w;\

#define SCALAR_SET_ZERO(target)\
    target = 0;\
    
#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_32_MUL(target, left, right);\

#define LINE_MUL_INSTR (4*195)

#define WORD_TO_SCALAR(x) ((x)&0xFFFFFFFF)

#endif 

/************************** 8-word line representing 8 elements of GF(2^32). */

#ifdef LINE_8_GF2_32
#define LINE_TYPE "8 x GF(2^{32}) with eight 32-bit words"

/* Note: we have some register pressure here -- each line eats up _8_ regs. */

/* Caveat: scalar loads and stores are somewhat subtle because
 *         the line consists of two contiguous blocks of words. */

#define GF2_B       32
#define GF2_MODULUS GF2_32_MODULUS

typedef word_t scalar_t;
typedef word4_t line_t[2];
  // typedef word4_t[2] line_t;    
  // -- this is how it really should read, but the syntax of C is what it is...
typedef word4_t line_array_t;

#define SCALARS_IN_LINE 8

#define LINE_ARRAY_SIZE(b) (sizeof(word4_t)*((size_t)b)/4)
#define LINE_SEGMENT_SIZE(b) (b/2/4)

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ((word_t *) ptr)[((idx)>>1)+4*((idx)&1)*seg];\

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((word_t *) ptr)[((idx)>>1)+4*((idx)&1)*seg] = (source);\

#define LINE_LOAD(target,ptr,seg,idx)\
    target[0] = ptr[idx];\
    target[1] = ptr[seg+idx];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx]   = source[0];\
    ptr[seg+idx] = source[1];\

#define LINE_MOV(target, source)\
    target[0] = source[0];\
    target[1] = source[1];\

#define LINE_ADD(target, left, right)\
{\
    target[0].x = left[0].x^right[0].x;\
    target[0].y = left[0].y^right[0].y;\
    target[0].z = left[0].z^right[0].z;\
    target[0].w = left[0].w^right[0].w;\
    target[1].x = left[1].x^right[1].x;\
    target[1].y = left[1].y^right[1].y;\
    target[1].z = left[1].z^right[1].z;\
    target[1].w = left[1].w^right[1].w;\
}\

#define LINE_MUL(target, left, right)\
    GF2_32_MUL_QUAD(target[0].x, target[0].y, target[0].z, target[0].w,\
                      left[0].x,   left[0].y,   left[0].z,   left[0].w,\
                     right[0].x,  right[0].y,  right[0].z,  right[0].w);\
    GF2_32_MUL_QUAD(target[1].x, target[1].y, target[1].z, target[1].w,\
                      left[1].x,   left[1].y,   left[1].z,   left[1].w,\
                     right[1].x,  right[1].y,  right[1].z,  right[1].w);\

#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    line_t temp;\
    temp[0].x = scalar;\
    temp[0].y = temp[0].x;\
    temp[0].z = temp[0].x;\
    temp[0].w = temp[0].x;\
    temp[1].x = temp[0].x;\
    temp[1].y = temp[0].x;\
    temp[1].z = temp[0].x;\
    temp[1].w = temp[0].x;\
    LINE_MUL(target, source, temp);\
}\

#define LINE_SUM(target, source)\
{\
    scalar_t temp;\
    temp = source[0].x^source[0].y^source[0].z^source[0].w\
          ^source[1].x^source[1].y^source[1].z^source[1].w;\
    target = temp;\
}\

#define LINE_SET_ZERO(target)\
{\
    target[0].x = 0;\
    target[0].y = 0;\
    target[0].z = 0;\
    target[0].w = 0;\
    target[1].x = 0;\
    target[1].y = 0;\
    target[1].z = 0;\
    target[1].w = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
    target[0].x = (idx == 0) ? source : target[0].x;\
    target[0].y = (idx == 1) ? source : target[0].y;\
    target[0].z = (idx == 2) ? source : target[0].z;\
    target[0].w = (idx == 3) ? source : target[0].w;\
    target[1].x = (idx == 4) ? source : target[1].x;\
    target[1].y = (idx == 5) ? source : target[1].y;\
    target[1].z = (idx == 6) ? source : target[1].z;\
    target[1].w = (idx == 7) ? source : target[1].w;\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_32_MUL(target, left, right);\

#define LINE_MUL_INSTR (2*4*195)  // 1560 instructions in one batch

#define WORD_TO_SCALAR(x) ((x)&0xFFFFFFFF)

#endif 


/************************** One-word line that packs one element of GF(2^8). */

#ifdef LINE_1_GF2_8
#define LINE_TYPE "1 x GF(2^8) with one 32-bit word"

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word_t line_t;
typedef word_t line_array_t;

#define SCALARS_IN_LINE 1
#define EFFECTIVE_BYTES_IN_LINE ((size_t) 1)

#define LINE_ARRAY_SIZE(b) (sizeof(word_t)*(size_t)(b))
#define LINE_SEGMENT_SIZE(b) 0 // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ptr[idx+0*seg]\

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ptr[idx+0*seg] = (source);\

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[idx+0*seg];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx+0*seg]   = source;\

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
    target = left^right;\

#define LINE_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, left, right);\

#define LINE_MUL_SCALAR(target, source, scalar)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_SUM(target, source)\
{\
    target = source;\
}\

#define LINE_SET_ZERO(target)\
{\
    target = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
    target = WORD_TO_SCALAR(source) + 0*idx;    \

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_MUL_INSTR (128)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif

/*************************** One-word line that packs 4 elements of GF(2^8). */

#ifdef LINE_4_GF2_8
#define LINE_TYPE "4 x GF(2^8) with one 32-bit word"

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

typedef word_t scalar_t;
typedef word_t line_t;
typedef word_t line_array_t;

#define SCALARS_IN_LINE 4

#define LINE_ARRAY_SIZE(b) (sizeof(word_t)*(size_t)(b)/4)
#define LINE_SEGMENT_SIZE(b) 0 // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = (word_t) ((unsigned char *) ptr)[idx+0*seg];   \

/* CAVEAT: Threads can interfere with each other. */

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((unsigned char *) ptr)[(idx)+0*(seg)] = (unsigned char) (source);  \

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[(idx)+0*(seg)];     \

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[(idx)+0*(seg)]   = source;    \

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
    target = left^right;\

#define LINE_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, left, right);      \

#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    line_t temp;\
    word_t temp_s = WORD_TO_SCALAR(scalar);\
    temp = (temp_s<<24)|(temp_s<<16)|(temp_s<<8)|(temp_s);\
    GF2_8_MUL_QUAD(target, source, temp);\
}\

#define LINE_SUM(target, source)\
{\
    scalar_t temp = (((source)>>24)^((source)>>16)^((source)>>8)^(source))&0xFF;\
    target = temp;\
}\

#define LINE_SET_ZERO(target)\
{\
    target = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
    int amount = 8*(idx);                   \
    scalar_t mask = 0xFF << amount;             \
    target = ((target)&~mask)|(WORD_TO_SCALAR(source)<<amount);   \

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, left, right);\

#define LINE_MUL_INSTR (128)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif




/************************** A 4-word line that packs 16 elements of GF(2^8). */

#ifdef LINE_16_GF2_8
#define LINE_TYPE "16 x GF(2^8) with four 32-bit words"

/* A line consists of one word4_t, 
 * this word4_t (4 word_t's in total, 4*32 bits) 
 * represents 16 elements of GF(2^8), four elements packed to each word. */

/* Caveat: scalar loads and stores are somewhat subtle because
 *         the line consists of two contiguous blocks of words. */

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word4_t line_t;
typedef word4_t line_array_t;

#define SCALARS_IN_LINE 16

#define LINE_ARRAY_SIZE(b) (sizeof(unsigned char)*(b))
#define LINE_SEGMENT_SIZE(b) 0  // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ((unsigned char *) ptr)[(idx)+0*(seg)];    \

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((unsigned char *) ptr)[(idx)+0*(seg)] = (source);  \

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[idx];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx]   = source;\

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
{\
    target.x = left.x^right.x;\
    target.y = left.y^right.y;\
    target.z = left.z^right.z;\
    target.w = left.w^right.w;\
}\

#define LINE_SET_ZERO(target)\
{\
    target.x = 0;\
    target.y = 0;\
    target.z = 0;\
    target.w = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
{\
    int amount = 8*((idx)&0x03);\
    scalar_t mask = 0xFF << amount;\
    switch((idx)>>2) {\
    case 0:\
        target.x = ((target.x)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 1:\
        target.y = ((target.y)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 2:\
        target.z = ((target.z)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 3:\
        target.w = ((target.w)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    }\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define LINE_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target.x, left.x, right.x);      \
    GF2_8_MUL_QUAD(target.y, left.y, right.y);      \
    GF2_8_MUL_QUAD(target.z, left.z, right.z);      \
    GF2_8_MUL_QUAD(target.w, left.w, right.w);      \

#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    word_t temp_s = WORD_TO_SCALAR(scalar);\
    word_t temp = (temp_s<<24)|(temp_s<<16)|(temp_s<<8)|(temp_s);\
    GF2_8_MUL_QUAD(target.x, source.x, temp);      \
    GF2_8_MUL_QUAD(target.y, source.y, temp);      \
    GF2_8_MUL_QUAD(target.z, source.z, temp);      \
    GF2_8_MUL_QUAD(target.w, source.w, temp);      \
}\

#define LINE_SUM(target, source)\
{\
    word_t temp = source.x^source.y^source.z^source.w;\
    scalar_t temp2 = ((temp>>24)^(temp>>16)^(temp>>8)^temp)&0xFF;\
    target = temp2;\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_MUL_INSTR (4*128)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif



/************************* An 8-word line that packs 32 elements of GF(2^8). */

#ifdef LINE_32_GF2_8
#define LINE_TYPE "32 x GF(2^8) with eight 32-bit words"

/* A line consists of two word4_t's, 
 * these two word4_t's (8 word_t's in total, 8*32 bits) 
 * represent 32 elements of GF(2^8), four elements packed to each word. */

/* Caveat: scalar loads and stores are somewhat subtle because
 *         the line consists of two contiguous blocks of words. */

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word4_t line_t[2];
typedef word4_t line_array_t;

#define SCALARS_IN_LINE 32

#define LINE_ARRAY_SIZE(b) (sizeof(unsigned char)*(b))
#define LINE_SEGMENT_SIZE(b) (b/2/4/4)

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ((unsigned char *) ptr)[((idx)>>1)+4*4*((idx)&1)*seg];\

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((unsigned char *) ptr)[((idx)>>1)+4*4*((idx)&1)*seg] = (source);\

#define LINE_LOAD(target,ptr,seg,idx)\
    target[0] = ptr[idx];\
    target[1] = ptr[seg+idx];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx]   = source[0];\
    ptr[seg+idx] = source[1];\

#define LINE_MOV(target, source)\
    target[0] = source[0];\
    target[1] = source[1];\

#define LINE_ADD(target, left, right)\
{\
    target[0].x = left[0].x^right[0].x;\
    target[0].y = left[0].y^right[0].y;\
    target[0].z = left[0].z^right[0].z;\
    target[0].w = left[0].w^right[0].w;\
    target[1].x = left[1].x^right[1].x;\
    target[1].y = left[1].y^right[1].y;\
    target[1].z = left[1].z^right[1].z;\
    target[1].w = left[1].w^right[1].w;\
}\

#define LINE_SET_ZERO(target)\
{\
    target[0].x = 0;\
    target[0].y = 0;\
    target[0].z = 0;\
    target[0].w = 0;\
    target[1].x = 0;\
    target[1].y = 0;\
    target[1].z = 0;\
    target[1].w = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
{\
    int amount = 8*((idx)&0x03);\
    scalar_t mask = 0xFF << amount;\
    switch((idx)>>2) {\
    case 0:\
        target[0].x = ((target[0].x)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 1:\
        target[0].y = ((target[0].y)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 2:\
        target[0].z = ((target[0].z)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 3:\
        target[0].w = ((target[0].w)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 4:\
        target[1].x = ((target[1].x)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 5:\
        target[1].y = ((target[1].y)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 6:\
        target[1].z = ((target[1].z)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 7:\
        target[1].w = ((target[1].w)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    }\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define LINE_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target[0].x, left[0].x, right[0].x);      \
    GF2_8_MUL_QUAD(target[0].y, left[0].y, right[0].y);      \
    GF2_8_MUL_QUAD(target[0].z, left[0].z, right[0].z);      \
    GF2_8_MUL_QUAD(target[0].w, left[0].w, right[0].w);      \
    GF2_8_MUL_QUAD(target[1].x, left[1].x, right[1].x);      \
    GF2_8_MUL_QUAD(target[1].y, left[1].y, right[1].y);      \
    GF2_8_MUL_QUAD(target[1].z, left[1].z, right[1].z);      \
    GF2_8_MUL_QUAD(target[1].w, left[1].w, right[1].w);      \

#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    word_t temp_s = WORD_TO_SCALAR(scalar);\
    word_t temp = (temp_s<<24)|(temp_s<<16)|(temp_s<<8)|(temp_s);\
    GF2_8_MUL_QUAD(target[0].x, source[0].x, temp);      \
    GF2_8_MUL_QUAD(target[0].y, source[0].y, temp);      \
    GF2_8_MUL_QUAD(target[0].z, source[0].z, temp);      \
    GF2_8_MUL_QUAD(target[0].w, source[0].w, temp);      \
    GF2_8_MUL_QUAD(target[1].x, source[1].x, temp);      \
    GF2_8_MUL_QUAD(target[1].y, source[1].y, temp);      \
    GF2_8_MUL_QUAD(target[1].z, source[1].z, temp);      \
    GF2_8_MUL_QUAD(target[1].w, source[1].w, temp);      \
}\

#define LINE_SUM(target, source)\
{\
    word_t temp = source[0].x^source[0].y^source[0].z^source[0].w^\
                    source[1].x^source[1].y^source[1].z^source[1].w;\
    scalar_t temp2 = ((temp>>24)^(temp>>16)^(temp>>8)^temp)&0xFF;\
    target = temp2;\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_MUL_INSTR (8*128)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif


/************** An 8-word line that packs 32 bit-sliced elements of GF(2^8). */

#ifdef LINE_32_GF2_8_BITSLICE
#define LINE_TYPE "32 x GF(2^8) with eight 32-bit words, bit sliced"

/* A line consists of two word4_t's, 
 * these two word4_t's (8 word_t's in total, 8*32 bits) 
 * represent 32 elements of GF(2^8), each word represents _one bit_
 * in each element of GF(2^8). */

/* Caveat: scalar loads and stores are somewhat subtle because
 *         the line consists of two contiguous blocks of words. */

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word4_t line_t[2];
typedef word4_t line_array_t;

#define SCALARS_IN_LINE 32

#define LINE_ARRAY_SIZE(b) (sizeof(unsigned char)*(b))
#define LINE_SEGMENT_SIZE(b) (b/2/4/4)

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
{\
    index_t j = idx;\
    word4_t lo = ptr[j/32];\
    word4_t hi = ptr[seg+j/32];\
    target = (((lo.x >> (j%32))&1)<<0)|\
             (((lo.y >> (j%32))&1)<<1)|\
             (((lo.z >> (j%32))&1)<<2)|\
             (((lo.w >> (j%32))&1)<<3)|\
             (((hi.x >> (j%32))&1)<<4)|\
             (((hi.y >> (j%32))&1)<<5)|\
             (((hi.z >> (j%32))&1)<<6)|\
             (((hi.w >> (j%32))&1)<<7);\
}\

/* CAVEAT: Threads can interfere with each other. */

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
{\
    index_t j = idx;\
    scalar_t byte = source;\
    ptr[j/32].x     = (ptr[j/32].x     & (~(1<<(j%32))))|(((byte>>0)&1)<<(j%32));\
    ptr[j/32].y     = (ptr[j/32].y     & (~(1<<(j%32))))|(((byte>>1)&1)<<(j%32));\
    ptr[j/32].z     = (ptr[j/32].z     & (~(1<<(j%32))))|(((byte>>2)&1)<<(j%32));\
    ptr[j/32].w     = (ptr[j/32].w     & (~(1<<(j%32))))|(((byte>>3)&1)<<(j%32));\
    ptr[seg+j/32].x = (ptr[seg+j/32].x & (~(1<<(j%32))))|(((byte>>4)&1)<<(j%32));\
    ptr[seg+j/32].y = (ptr[seg+j/32].y & (~(1<<(j%32))))|(((byte>>5)&1)<<(j%32));\
    ptr[seg+j/32].z = (ptr[seg+j/32].z & (~(1<<(j%32))))|(((byte>>6)&1)<<(j%32));\
    ptr[seg+j/32].w = (ptr[seg+j/32].w & (~(1<<(j%32))))|(((byte>>7)&1)<<(j%32));\
}\
    

#define LINE_LOAD(target,ptr,seg,idx)\
    target[0] = ptr[idx];\
    target[1] = ptr[seg+idx];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx]     = source[0];\
    ptr[seg+idx] = source[1];\

#define LINE_MOV(target, source)\
    target[0] = source[0];\
    target[1] = source[1];\

#define LINE_ADD(target, left, right)\
{\
    target[0].x = left[0].x^right[0].x;\
    target[0].y = left[0].y^right[0].y;\
    target[0].z = left[0].z^right[0].z;\
    target[0].w = left[0].w^right[0].w;\
    target[1].x = left[1].x^right[1].x;\
    target[1].y = left[1].y^right[1].y;\
    target[1].z = left[1].z^right[1].z;\
    target[1].w = left[1].w^right[1].w;\
}\

#define LINE_MUL(target, left, right)\
    GF2_8_MUL_BITSLICE(target[0].x, target[0].y, target[0].z, target[0].w,\
                       target[1].x, target[1].y, target[1].z, target[1].w, \
                       left[0].x, left[0].y, left[0].z, left[0].w,      \
                       left[1].x, left[1].y, left[1].z, left[1].w,      \
                       right[0].x, right[0].y, right[0].z, right[0].w,  \
                       right[1].x, right[1].y, right[1].z, right[1].w); \

#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    line_t temp;\
    temp[0].x = -(((scalar)>>0)&1);\
    temp[0].y = -(((scalar)>>1)&1);\
    temp[0].z = -(((scalar)>>2)&1);\
    temp[0].w = -(((scalar)>>3)&1);\
    temp[1].x = -(((scalar)>>4)&1);\
    temp[1].y = -(((scalar)>>5)&1);\
    temp[1].z = -(((scalar)>>6)&1);\
    temp[1].w = -(((scalar)>>7)&1);\
    LINE_MUL(target, source, temp);\
}\

#define LINE_SUM(target, source)\
{\
    scalar_t temp = 0;\
    for(int i = 0; i < 32; i++) {\
      temp ^= ((source[0].x>>i)&1)<<0;\
      temp ^= ((source[0].y>>i)&1)<<1;\
      temp ^= ((source[0].z>>i)&1)<<2;\
      temp ^= ((source[0].w>>i)&1)<<3;\
      temp ^= ((source[1].x>>i)&1)<<4;\
      temp ^= ((source[1].y>>i)&1)<<5;\
      temp ^= ((source[1].z>>i)&1)<<6;\
      temp ^= ((source[1].w>>i)&1)<<7;\
    }\
    target = temp;\
}\

#define LINE_SET_ZERO(target)\
{\
    target[0].x = 0;\
    target[0].y = 0;\
    target[0].z = 0;\
    target[0].w = 0;\
    target[1].x = 0;\
    target[1].y = 0;\
    target[1].z = 0;\
    target[1].w = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
{\
    target[0].x = (target[0].x & (~(1<<(idx))))|((((source)>>0)&1)<<(idx));\
    target[0].y = (target[0].y & (~(1<<(idx))))|((((source)>>1)&1)<<(idx));\
    target[0].z = (target[0].z & (~(1<<(idx))))|((((source)>>2)&1)<<(idx));\
    target[0].w = (target[0].w & (~(1<<(idx))))|((((source)>>3)&1)<<(idx));\
    target[1].x = (target[1].x & (~(1<<(idx))))|((((source)>>4)&1)<<(idx));\
    target[1].y = (target[1].y & (~(1<<(idx))))|((((source)>>5)&1)<<(idx));\
    target[1].z = (target[1].z & (~(1<<(idx))))|((((source)>>6)&1)<<(idx));\
    target[1].w = (target[1].w & (~(1<<(idx))))|((((source)>>7)&1)<<(idx));\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_MUL_INSTR (141)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif



/******** One-word line that packs one element of GF(2^8), exp/log multiply. */

#ifdef LINE_1_GF2_8_EXPLOG
#define LINE_TYPE "1 x GF(2^8) with one 32-bit word [exp/log multiply]"

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

#define GF_LOG_EXP_LOOKUP

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word_t line_t;
typedef word_t line_array_t;

#define SCALARS_IN_LINE 1
#define EFFECTIVE_BYTES_IN_LINE ((size_t) 1)

#define LINE_ARRAY_SIZE(b) (sizeof(word_t)*(size_t)(b))
#define LINE_SEGMENT_SIZE(b) 0 // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ptr[idx+0*seg]\

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ptr[idx+0*seg] = (source);\

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[idx+0*seg];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx+0*seg] = source;\

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
    target = left^right;\

#define LINE_MUL(target, left, right) {\
    scalar_t lll = (left)&0xFF;\
    scalar_t rrr = (right)&0xFF;\
    scalar_t ttt;\
    if(lll == 0 || rrr == 0) {\
        ttt = 0;\
    } else {\
        ttt = d_lookup_exp[d_lookup_log[lll]+\
                           d_lookup_log[rrr]];\
    }\
    target = ttt;\
}

#define LINE_MUL_SCALAR(target, source, scalar)\
    LINE_MUL(target,source,scalar);\

#define LINE_SUM(target, source)\
{\
    target = source;\
}\

#define LINE_SET_ZERO(target)\
{\
    target = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
    target = WORD_TO_SCALAR(source) + 0*idx;    \

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_MUL_INSTR (0)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif


/****** One-word line that packs four elements of GF(2^8), exp/log multiply. */

#ifdef LINE_4_GF2_8_EXPLOG
#define LINE_TYPE "4 x GF(2^8) with one 32-bit word [exp/log multiply]"

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

#define GF_LOG_EXP_LOOKUP

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word_t line_t;
typedef word_t line_array_t;

#define SCALARS_IN_LINE 4

#define LINE_ARRAY_SIZE(b) (sizeof(word_t)*(size_t)(b)/4)
#define LINE_SEGMENT_SIZE(b) 0 // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = (word_t) ((unsigned char *) ptr)[idx+0*seg];   \

/* CAVEAT: Threads can interfere with each other. */

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((unsigned char *) ptr)[(idx)+0*(seg)] = (unsigned char) (source);  \

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[(idx)+0*(seg)];     \

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[(idx)+0*(seg)]   = source;    \

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
    target = left^right;\

#define LOOKUP_MUL(target, left, right) {\
    scalar_t lll = (left)&0xFF;\
    scalar_t rrr = (right)&0xFF;\
    scalar_t ttt;\
    if(lll == 0 || rrr == 0) {\
        ttt = 0;\
    } else {\
        ttt = d_lookup_exp[d_lookup_log[lll]+\
                           d_lookup_log[rrr]];\
    }\
    target = ttt;\
}

#define LINE_MUL(target, left, right) {\
    word_t llll = left;\
    word_t rrrr = right;\
    scalar_t pppp;\
    word_t tttt;\
    LOOKUP_MUL(tttt, llll, rrrr);\
    LOOKUP_MUL(pppp, llll>>8, rrrr>>8);\
    tttt |= (pppp<<8);\
    LOOKUP_MUL(pppp, llll>>16, rrrr>>16);\
    tttt |= (pppp<<16);\
    LOOKUP_MUL(pppp, llll>>24, rrrr>>24);\
    tttt |= (pppp<<24);\
    target = tttt;\
}
    
#define LINE_MUL_SCALAR(target, source, scalar) {\
    word_t temp = (scalar)&0xFF;\
    temp = (temp<<24)|(temp<<16)|(temp<<8)|temp;\
    LINE_MUL(target,source,temp);\
}

#define LINE_SUM(target, source)\
{\
    scalar_t temp = (((source)>>24)^((source)>>16)^((source)>>8)^(source))&0xFF;\
    target = temp;\
}\

#define LINE_SET_ZERO(target)\
{\
    target = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
    int amount = 8*(idx);                   \
    scalar_t mask = 0xFF << amount;             \
    target = ((target)&~mask)|(WORD_TO_SCALAR(source)<<amount);   \

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, left, right);\

#define LINE_MUL_INSTR (0)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif


/******** A 4-word line that packs 16 elements of GF(2^8), exp/log multiply. */

#ifdef LINE_16_GF2_8_EXPLOG
#define LINE_TYPE "16 x GF(2^8) with four 32-bit words [exp/log multiply]"

/* A line consists of one word4_t, 
 * this word4_t (4 word_t's in total, 4*32 bits) 
 * represents 16 elements of GF(2^8), four elements packed to each word. */

/* Caveat: scalar loads and stores are somewhat subtle because
 *         the line consists of two contiguous blocks of words. */

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

#define GF_LOG_EXP_LOOKUP

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word4_t line_t;
typedef word4_t line_array_t;

#define SCALARS_IN_LINE 16

#define LINE_ARRAY_SIZE(b) (sizeof(unsigned char)*(b))
#define LINE_SEGMENT_SIZE(b) 0  // not used

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ((unsigned char *) ptr)[(idx)+0*(seg)];    \

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((unsigned char *) ptr)[(idx)+0*(seg)] = (source);  \

#define LINE_LOAD(target,ptr,seg,idx)\
    target = ptr[idx];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx]   = source;\

#define LINE_MOV(target, source)\
    target = source;\

#define LINE_ADD(target, left, right)\
{\
    target.x = left.x^right.x;\
    target.y = left.y^right.y;\
    target.z = left.z^right.z;\
    target.w = left.w^right.w;\
}\

#define LINE_SET_ZERO(target)\
{\
    target.x = 0;\
    target.y = 0;\
    target.z = 0;\
    target.w = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
{\
    int amount = 8*((idx)&0x03);\
    scalar_t mask = 0xFF << amount;\
    switch((idx)>>2) {\
    case 0:\
        target.x = ((target.x)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 1:\
        target.y = ((target.y)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 2:\
        target.z = ((target.z)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 3:\
        target.w = ((target.w)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    }\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define LOOKUP_MUL(target, left, right) {\
    scalar_t lll = (left)&0xFF;\
    scalar_t rrr = (right)&0xFF;\
    scalar_t ttt;\
    if(lll == 0 || rrr == 0) {\
        ttt = 0;\
    } else {\
        ttt = d_lookup_exp[d_lookup_log[lll]+\
                           d_lookup_log[rrr]];\
    }\
    target = ttt;\
}

#define WORD_MUL(target, left, right) {\
    word_t llll = left;\
    word_t rrrr = right;\
    scalar_t pppp;\
    word_t tttt;\
    LOOKUP_MUL(tttt, llll, rrrr);\
    LOOKUP_MUL(pppp, llll>>8, rrrr>>8);\
    tttt |= (pppp<<8);\
    LOOKUP_MUL(pppp, llll>>16, rrrr>>16);\
    tttt |= (pppp<<16);\
    LOOKUP_MUL(pppp, llll>>24, rrrr>>24);\
    tttt |= (pppp<<24);\
    target = tttt;\
}

#define LINE_MUL(target, left, right)\
    WORD_MUL(target.x, left.x, right.x);      \
    WORD_MUL(target.y, left.y, right.y);      \
    WORD_MUL(target.z, left.z, right.z);      \
    WORD_MUL(target.w, left.w, right.w);      \

    
#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    word_t temp_s = WORD_TO_SCALAR(scalar);\
    word_t temp = (temp_s<<24)|(temp_s<<16)|(temp_s<<8)|(temp_s);\
    WORD_MUL(target.x, source.x, temp);      \
    WORD_MUL(target.y, source.y, temp);      \
    WORD_MUL(target.z, source.z, temp);      \
    WORD_MUL(target.w, source.w, temp);      \
}\

#define LINE_SUM(target, source)\
{\
    word_t temp = source.x^source.y^source.z^source.w;\
    scalar_t temp2 = ((temp>>24)^(temp>>16)^(temp>>8)^temp)&0xFF;\
    target = temp2;\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_MUL_INSTR (0)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif



/******* An 8-word line that packs 32 elements of GF(2^8), exp/log multiply. */

#ifdef LINE_32_GF2_8_EXPLOG
#define LINE_TYPE "32 x GF(2^8) with eight 32-bit words [exp/log multiply]"

/* A line consists of two word4_t's, 
 * these two word4_t's (8 word_t's in total, 8*32 bits) 
 * represent 32 elements of GF(2^8), four elements packed to each word. */

/* Caveat: scalar loads and stores are somewhat subtle because
 *         the line consists of two contiguous blocks of words. */

#define GF2_B       8
#define GF2_MODULUS GF2_8_MODULUS

#define GF_LOG_EXP_LOOKUP

typedef word_t scalar_t; // use the 8 least significant bits of a word
typedef word4_t line_t[2];
typedef word4_t line_array_t;

#define SCALARS_IN_LINE 32

#define LINE_ARRAY_SIZE(b) (sizeof(unsigned char)*(b))
#define LINE_SEGMENT_SIZE(b) (b/2/4/4)

#define LINE_ARRAY_LOAD_SCALAR(target,ptr,seg,idx)\
    target = ((unsigned char *) ptr)[((idx)>>1)+4*4*((idx)&1)*seg];\

#define LINE_ARRAY_STORE_SCALAR(ptr,seg,idx,source)\
    ((unsigned char *) ptr)[((idx)>>1)+4*4*((idx)&1)*seg] = (source);\

#define LINE_LOAD(target,ptr,seg,idx)\
    target[0] = ptr[idx];\
    target[1] = ptr[seg+idx];\

#define LINE_STORE(ptr,seg,idx,source)\
    ptr[idx]   = source[0];\
    ptr[seg+idx] = source[1];\

#define LINE_MOV(target, source)\
    target[0] = source[0];\
    target[1] = source[1];\

#define LINE_ADD(target, left, right)\
{\
    target[0].x = left[0].x^right[0].x;\
    target[0].y = left[0].y^right[0].y;\
    target[0].z = left[0].z^right[0].z;\
    target[0].w = left[0].w^right[0].w;\
    target[1].x = left[1].x^right[1].x;\
    target[1].y = left[1].y^right[1].y;\
    target[1].z = left[1].z^right[1].z;\
    target[1].w = left[1].w^right[1].w;\
}\

#define LINE_SET_ZERO(target)\
{\
    target[0].x = 0;\
    target[0].y = 0;\
    target[0].z = 0;\
    target[0].w = 0;\
    target[1].x = 0;\
    target[1].y = 0;\
    target[1].z = 0;\
    target[1].w = 0;\
}\

#define LINE_STORE_SCALAR(target,idx,source)\
{\
    int amount = 8*((idx)&0x03);\
    scalar_t mask = 0xFF << amount;\
    switch((idx)>>2) {\
    case 0:\
        target[0].x = ((target[0].x)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 1:\
        target[0].y = ((target[0].y)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 2:\
        target[0].z = ((target[0].z)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 3:\
        target[0].w = ((target[0].w)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 4:\
        target[1].x = ((target[1].x)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 5:\
        target[1].y = ((target[1].y)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 6:\
        target[1].z = ((target[1].z)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    case 7:\
        target[1].w = ((target[1].w)&~mask)|(WORD_TO_SCALAR(source)<<amount);\
        break;\
    }\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\


#define LOOKUP_MUL(target, left, right) {\
    scalar_t lll = (left)&0xFF;\
    scalar_t rrr = (right)&0xFF;\
    scalar_t ttt;\
    if(lll == 0 || rrr == 0) {\
        ttt = 0;\
    } else {\
        ttt = d_lookup_exp[d_lookup_log[lll]+\
                           d_lookup_log[rrr]];\
    }\
    target = ttt;\
}

#define WORD_MUL(target, left, right) {\
    word_t llll = left;\
    word_t rrrr = right;\
    scalar_t pppp;\
    word_t tttt;\
    LOOKUP_MUL(tttt, llll, rrrr);\
    LOOKUP_MUL(pppp, llll>>8, rrrr>>8);\
    tttt |= (pppp<<8);\
    LOOKUP_MUL(pppp, llll>>16, rrrr>>16);\
    tttt |= (pppp<<16);\
    LOOKUP_MUL(pppp, llll>>24, rrrr>>24);\
    tttt |= (pppp<<24);\
    target = tttt;\
}

#define LINE_MUL(target, left, right)\
    WORD_MUL(target[0].x, left[0].x, right[0].x);\
    WORD_MUL(target[0].y, left[0].y, right[0].y);\
    WORD_MUL(target[0].z, left[0].z, right[0].z);\
    WORD_MUL(target[0].w, left[0].w, right[0].w);\
    WORD_MUL(target[1].x, left[1].x, right[1].x);\
    WORD_MUL(target[1].y, left[1].y, right[1].y);\
    WORD_MUL(target[1].z, left[1].z, right[1].z);\
    WORD_MUL(target[1].w, left[1].w, right[1].w);\

#define LINE_MUL_SCALAR(target, source, scalar)\
{\
    word_t temp_s = WORD_TO_SCALAR(scalar);\
    word_t temp = (temp_s<<24)|(temp_s<<16)|(temp_s<<8)|(temp_s);\
    WORD_MUL(target[0].x, source[0].x, temp);\
    WORD_MUL(target[0].y, source[0].y, temp);\
    WORD_MUL(target[0].z, source[0].z, temp);\
    WORD_MUL(target[0].w, source[0].w, temp);\
    WORD_MUL(target[1].x, source[1].x, temp);\
    WORD_MUL(target[1].y, source[1].y, temp);\
    WORD_MUL(target[1].z, source[1].z, temp);\
    WORD_MUL(target[1].w, source[1].w, temp);\
}\

#define LINE_SUM(target, source)\
{\
    word_t temp = source[0].x^source[0].y^source[0].z^source[0].w^\
                    source[1].x^source[1].y^source[1].z^source[1].w;\
    scalar_t temp2 = ((temp>>24)^(temp>>16)^(temp>>8)^temp)&0xFF;\
    target = temp2;\
}\

#define SCALAR_SET_ZERO(target)\
    target = 0;\

#define SCALAR_ADD(target, left, right)\
    target = left^right;\

#define SCALAR_MUL(target, left, right)\
    GF2_8_MUL_QUAD(target, source, scalar);\

#define LINE_MUL_INSTR (0)

#define WORD_TO_SCALAR(x) ((x)&0x0FF)

#endif



#ifndef EFFECTIVE_BYTES_IN_LINE
#define EFFECTIVE_BYTES_IN_LINE sizeof(line_t)
#endif


/*********************************************** Multiplication subroutines. */

#define REPEAT_32_TIMES(x) REP32(x)
#define REPEAT_8_TIMES(x) REP8(x)
#define REP32(x) REP16(x) REP16(x)
#define REP16(x) REP8(x) REP8(x)
#define REP8(x) REP4(x) REP4(x)
#define REP4(x) x x x x

// 3 + 6*32 = 3 + 192 = 195 instructions for GF(2^32) multiply
//
// Modulus: [x^32 +] x^7 + x^5 + x^3 + x^2 + x + 1  ~ 0xAF = 175 (primitive)

#define GF2_32_MUL(tt,ll,rr)\
    asm("{\n\t"\
        "     .reg .u32     rx;\n\t"\
        "     .reg .u32     ry;\n\t"\
        "     .reg .pred    p1;\n\t"\
        "     .reg .pred    p2;\n\t"\
        "     mov.b32       rx, %1;\n\t"\
        "     brev.b32      ry, %2;\n\t"\
        "     xor.b32       %0, %0, %0;\n\t"\
        REPEAT_32_TIMES(\
         "    setp.lt.s32  p1, ry, 0;\n\t"\
         "    setp.lt.s32  p2, rx, 0;\n\t"\
         "    add.u32      ry, ry, ry;\n\t"\
         "@p1 xor.b32      %0, %0, rx;\n\t"\
         "    add.u32      rx, rx, rx;\n\t"\
         "@p2 xor.b32      rx, rx, 175;\n\t"\
        )\
        "}"\
        : "=r"(tt)\
        : "r"(ll), "r"(rr));\

// Quad-multiply version of the above
//
// Modulus: [x^32 +] x^7 + x^5 + x^3 + x^2 + x + 1  ~ 0xAF = 175 (primitive)

#define GF2_32_MUL_QUAD(tt0,tt1,tt2,tt3,\
                        ll0,ll1,ll2,ll3,\
                        rr0,rr1,rr2,rr3)\
    asm("{\n\t"\
        "       .reg .u32     q0rx;\n\t"\
        "       .reg .u32     q0ry;\n\t"\
        "       .reg .pred    q0p1;\n\t"\
        "       .reg .pred    q0p2;\n\t"\
        "       .reg .u32     q1rx;\n\t"\
        "       .reg .u32     q1ry;\n\t"\
        "       .reg .pred    q1p1;\n\t"\
        "       .reg .pred    q1p2;\n\t"\
        "       .reg .u32     q2rx;\n\t"\
        "       .reg .u32     q2ry;\n\t"\
        "       .reg .pred    q2p1;\n\t"\
        "       .reg .pred    q2p2;\n\t"\
        "       .reg .u32     q3rx;\n\t"\
        "       .reg .u32     q3ry;\n\t"\
        "       .reg .pred    q3p1;\n\t"\
        "       .reg .pred    q3p2;\n\t"\
        "       mov.b32       q0rx, %4;\n\t"\
        "       brev.b32      q0ry, %8;\n\t"\
        "       mov.b32       q1rx, %5;\n\t"\
        "       brev.b32      q1ry, %9;\n\t"\
        "       mov.b32       q2rx, %6;\n\t"\
        "       brev.b32      q2ry, %10;\n\t"\
        "       mov.b32       q3rx, %7;\n\t"\
        "       brev.b32      q3ry, %11;\n\t"\
        "       xor.b32       %0, %0, %0;\n\t"\
        "       xor.b32       %1, %1, %1;\n\t"\
        "       xor.b32       %2, %2, %2;\n\t"\
        "       xor.b32       %3, %3, %3;\n\t"\
        REPEAT_32_TIMES(\
         "      setp.lt.s32  q0p1, q0ry, 0;\n\t"\
         "      setp.lt.s32  q1p1, q1ry, 0;\n\t"\
         "      setp.lt.s32  q2p1, q2ry, 0;\n\t"\
         "      setp.lt.s32  q3p1, q3ry, 0;\n\t"\
         "      setp.lt.s32  q0p2, q0rx, 0;\n\t"\
         "      setp.lt.s32  q1p2, q1rx, 0;\n\t"\
         "      setp.lt.s32  q2p2, q2rx, 0;\n\t"\
         "      setp.lt.s32  q3p2, q3rx, 0;\n\t"\
         "      add.u32      q0ry, q0ry, q0ry;\n\t"\
         "      add.u32      q1ry, q1ry, q1ry;\n\t"\
         "      add.u32      q2ry, q2ry, q2ry;\n\t"\
         "      add.u32      q3ry, q3ry, q3ry;\n\t"\
         "@q0p1 xor.b32      %0, %0, q0rx;\n\t"\
         "@q1p1 xor.b32      %1, %1, q1rx;\n\t"\
         "@q2p1 xor.b32      %2, %2, q2rx;\n\t"\
         "@q3p1 xor.b32      %3, %3, q3rx;\n\t"\
         "      add.u32      q0rx, q0rx, q0rx;\n\t"\
         "      add.u32      q1rx, q1rx, q1rx;\n\t"\
         "      add.u32      q2rx, q2rx, q2rx;\n\t"\
         "      add.u32      q3rx, q3rx, q3rx;\n\t"\
         "@q0p2 xor.b32      q0rx, q0rx, 175;\n\t"\
         "@q1p2 xor.b32      q1rx, q1rx, 175;\n\t"\
         "@q2p2 xor.b32      q2rx, q2rx, 175;\n\t"\
         "@q3p2 xor.b32      q3rx, q3rx, 175;\n\t"\
        )\
        "}"\
        : "=r"(tt0), "=r"(tt1), "=r"(tt2), "=r"(tt3)\
        : "r"(ll0),   "r"(ll1),  "r"(ll2),  "r"(ll3),\
          "r"(rr0),   "r"(rr1),  "r"(rr2),  "r"(rr3)\
        );

// 128 = 8 + 8*15 instructions for 
// 4 x GF(2^8) multiply -- 4 multiplications executed in parallel
//                         on one 32-bit register 
//
// Caveat: e.g. shifts take 2 cycles on an NVIDIA M2090/Fermi T20A
//
// Modulus:  [x^8 +] x^4 + x^3 + x^2 + 1            ~ 0x1D = 29 (primitive)

#define GF2_8_MUL_QUAD(tt,ll,rr)\
    asm("{\n\t"\
        "     .reg .u32     rx;\n\t"\
        "     .reg .u32     ry;\n\t"\
        "     .reg .u32     l;\n\t"\
        "     .reg .u32     nl;\n\t"\
        "     .reg .u32     h;\n\t"\
        "     .reg .u32     nh;\n\t"\
        "     .reg .u32     m;\n\t"\
        "     .reg .u32     t1;\n\t"\
        "     .reg .u32     t2;\n\t"\
        "     .reg .u32     t3;\n\t"\
        "     mov.b32       rx, %1;\n\t"\
        "     mov.b32       ry, %2;\n\t"\
        "     mov.u32       l,  0x01010101U;\n\t"\
        "     mov.u32       h,  0x80808080U;\n\t"\
        "     not.b32       nl, l;\n\t"\
        "     not.b32       nh, h;\n\t"\
        "     mov.u32       m,  0x1D1D1D1D;\n\t"\
        "     xor.b32       %0, %0, %0;\n\t"\
        REPEAT_8_TIMES(\
          "   and.b32       t1, rx, h;\n\t"\
          "   and.b32       t2, ry, l;\n\t"\
          "   shr.u32       t1, t1, 7;\n\t"\
          "   and.b32       t3, ry, nl;\n\t"\
          "   add.u32       t2, t2, nh;\n\t"\
          "   shr.u32       ry, t3, 1;\n\t"\
          "   add.u32       t1, t1, nh;\n\t"\
          "   xor.b32       t3, t2, nh;\n\t"\
          "   and.b32       t2, rx, nh;\n\t"\
          "   xor.b32       t1, t1, nh;\n\t"\
          "   and.b32       t3, rx, t3;\n\t"\
          "   add.u32       t2, t2, t2;\n\t"\
          "   and.b32       t1, t1, m;\n\t"\
          "   xor.b32       %0, %0, t3;\n\t"\
          "   xor.b32       rx, t1, t2;\n\t"\
        )\
        "}"\
        : "=r"(tt)\
        : "r"(ll), "r"(rr));\


/*
 * A simplified bit-sliced Mastrovito multiplier for GF(2^8)
 * (141 gates)
 *
 * Modulus:  [x^8 +] x^4 + x^3 + x^2 + 1            ~ 0x1D = 29 (primitive)
 *
 * First few powers of the generator & degrees of nonzero monomials
 *
 *  0: 0x01 0
 *  1: 0x02 1
 *  2: 0x04 2
 *  3: 0x08 3
 *  4: 0x10 4
 *  5: 0x20 5
 *  6: 0x40 6
 *  7: 0x80 7
 * -------------------
 *  8: 0x1D 0 2 3 4
 *  9: 0x3A 1 3 4 5
 * 10: 0x74 2 4 5 6
 * 11: 0xE8 3 5 6 7
 * 12: 0xCD 0 2 3 6 7
 * 13: 0x87 0 1 2 7
 * 14: 0x13 0 1 4
 * 
 */

#define GF2_8_MUL_BITSLICE(z0,z1,z2,z3,z4,z5,z6,z7,x0,x1,x2,x3,x4,x5,x6,x7,y0,y1,y2,y3,y4,y5,y6,y7)\
    asm("{\n\t"\
        "    .reg .u32    a;\n\t"\
        "    .reg .u32    t8;\n\t"\
        "    .reg .u32    t9;\n\t"\
        "    .reg .u32    t10;\n\t"\
        "    .reg .u32    t11;\n\t"\
        "    .reg .u32    t12;\n\t"\
        "    .reg .u32    t13;\n\t"\
        "    .reg .u32    t14;\n\t"\
        "    and.b32      %0, %8, %16;\n\t"\
        "    and.b32      %1, %8, %17;\n\t"\
        "    and.b32      a, %9, %16;\n\t"\
        "    xor.b32      %1, %1, a;\n\t"\
        "    and.b32      %2, %8, %18;\n\t"\
        "    and.b32      a, %9, %17;\n\t"\
        "    xor.b32      %2, %2, a;\n\t"\
        "    and.b32      a, %10, %16;\n\t"\
        "    xor.b32      %2, %2, a;\n\t"\
        "    and.b32      %3, %8, %19;\n\t"\
        "    and.b32      a, %9, %18;\n\t"\
        "    xor.b32      %3, %3, a;\n\t"\
        "    and.b32      a, %10, %17;\n\t"\
        "    xor.b32      %3, %3, a;\n\t"\
        "    and.b32      a, %11, %16;\n\t"\
        "    xor.b32      %3, %3, a;\n\t"\
        "    and.b32      %4, %8, %20;\n\t"\
        "    and.b32      a, %9, %19;\n\t"\
        "    xor.b32      %4, %4, a;\n\t"\
        "    and.b32      a, %10, %18;\n\t"\
        "    xor.b32      %4, %4, a;\n\t"\
        "    and.b32      a, %11, %17;\n\t"\
        "    xor.b32      %4, %4, a;\n\t"\
        "    and.b32      a, %12, %16;\n\t"\
        "    xor.b32      %4, %4, a;\n\t"\
        "    and.b32      %5, %8, %21;\n\t"\
        "    and.b32      a, %9, %20;\n\t"\
        "    xor.b32      %5, %5, a;\n\t"\
        "    and.b32      a, %10, %19;\n\t"\
        "    xor.b32      %5, %5, a;\n\t"\
        "    and.b32      a, %11, %18;\n\t"\
        "    xor.b32      %5, %5, a;\n\t"\
        "    and.b32      a, %12, %17;\n\t"\
        "    xor.b32      %5, %5, a;\n\t"\
        "    and.b32      a, %13, %16;\n\t"\
        "    xor.b32      %5, %5, a;\n\t"\
        "    and.b32      %6, %8, %22;\n\t"\
        "    and.b32      a, %9, %21;\n\t"\
        "    xor.b32      %6, %6, a;\n\t"\
        "    and.b32      a, %10, %20;\n\t"\
        "    xor.b32      %6, %6, a;\n\t"\
        "    and.b32      a, %11, %19;\n\t"\
        "    xor.b32      %6, %6, a;\n\t"\
        "    and.b32      a, %12, %18;\n\t"\
        "    xor.b32      %6, %6, a;\n\t"\
        "    and.b32      a, %13, %17;\n\t"\
        "    xor.b32      %6, %6, a;\n\t"\
        "    and.b32      a, %14, %16;\n\t"\
        "    xor.b32      %6, %6, a;\n\t"\
        "    and.b32      %7, %8, %23;\n\t"\
        "    and.b32      a, %9, %22;\n\t"\
        "    xor.b32      %7, %7, a;\n\t"\
        "    and.b32      a, %10, %21;\n\t"\
        "    xor.b32      %7, %7, a;\n\t"\
        "    and.b32      a, %11, %20;\n\t"\
        "    xor.b32      %7, %7, a;\n\t"\
        "    and.b32      a, %12, %19;\n\t"\
        "    xor.b32      %7, %7, a;\n\t"\
        "    and.b32      a, %13, %18;\n\t"\
        "    xor.b32      %7, %7, a;\n\t"\
        "    and.b32      a, %14, %17;\n\t"\
        "    xor.b32      %7, %7, a;\n\t"\
        "    and.b32      a, %15, %16;\n\t"\
        "    xor.b32      %7, %7, a;\n\t"\
        "    and.b32      t8, %9, %23;\n\t"\
        "    and.b32      a, %10, %22;\n\t"\
        "    xor.b32      t8, t8, a;\n\t"\
        "    and.b32      a, %11, %21;\n\t"\
        "    xor.b32      t8, t8, a;\n\t"\
        "    and.b32      a, %12, %20;\n\t"\
        "    xor.b32      t8, t8, a;\n\t"\
        "    and.b32      a, %13, %19;\n\t"\
        "    xor.b32      t8, t8, a;\n\t"\
        "    and.b32      a, %14, %18;\n\t"\
        "    xor.b32      t8, t8, a;\n\t"\
        "    and.b32      a, %15, %17;\n\t"\
        "    xor.b32      t8, t8, a;\n\t"\
        "    and.b32      t9, %10, %23;\n\t"\
        "    and.b32      a, %11, %22;\n\t"\
        "    xor.b32      t9, t9, a;\n\t"\
        "    and.b32      a, %12, %21;\n\t"\
        "    xor.b32      t9, t9, a;\n\t"\
        "    and.b32      a, %13, %20;\n\t"\
        "    xor.b32      t9, t9, a;\n\t"\
        "    and.b32      a, %14, %19;\n\t"\
        "    xor.b32      t9, t9, a;\n\t"\
        "    and.b32      a, %15, %18;\n\t"\
        "    xor.b32      t9, t9, a;\n\t"\
        "    and.b32      t10, %11, %23;\n\t"\
        "    and.b32      a, %12, %22;\n\t"\
        "    xor.b32      t10, t10, a;\n\t"\
        "    and.b32      a, %13, %21;\n\t"\
        "    xor.b32      t10, t10, a;\n\t"\
        "    and.b32      a, %14, %20;\n\t"\
        "    xor.b32      t10, t10, a;\n\t"\
        "    and.b32      a, %15, %19;\n\t"\
        "    xor.b32      t10, t10, a;\n\t"\
        "    and.b32      t11, %12, %23;\n\t"\
        "    and.b32      a, %13, %22;\n\t"\
        "    xor.b32      t11, t11, a;\n\t"\
        "    and.b32      a, %14, %21;\n\t"\
        "    xor.b32      t11, t11, a;\n\t"\
        "    and.b32      a, %15, %20;\n\t"\
        "    xor.b32      t11, t11, a;\n\t"\
        "    and.b32      t12, %13, %23;\n\t"\
        "    and.b32      a, %14, %22;\n\t"\
        "    xor.b32      t12, t12, a;\n\t"\
        "    and.b32      a, %15, %21;\n\t"\
        "    xor.b32      t12, t12, a;\n\t"\
        "    and.b32      t13, %14, %23;\n\t"\
        "    and.b32      a, %15, %22;\n\t"\
        "    xor.b32      t13, t13, a;\n\t"\
        "    and.b32      t14, %15, %23;\n\t"\
        "    xor.b32      %0, %0, t8;\n\t"\
        "    xor.b32      %2, %2, t8;\n\t"\
        "    xor.b32      %3, %3, t8;\n\t"\
        "    xor.b32      %4, %4, t8;\n\t"\
        "    xor.b32      %1, %1, t9;\n\t"\
        "    xor.b32      %3, %3, t9;\n\t"\
        "    xor.b32      %4, %4, t9;\n\t"\
        "    xor.b32      %5, %5, t9;\n\t"\
        "    xor.b32      %2, %2, t10;\n\t"\
        "    xor.b32      %4, %4, t10;\n\t"\
        "    xor.b32      %5, %5, t10;\n\t"\
        "    xor.b32      %6, %6, t10;\n\t"\
        "    xor.b32      %3, %3, t11;\n\t"\
        "    xor.b32      %5, %5, t11;\n\t"\
        "    xor.b32      %6, %6, t11;\n\t"\
        "    xor.b32      %7, %7, t11;\n\t"\
        "    xor.b32      %0, %0, t12;\n\t"\
        "    xor.b32      %2, %2, t12;\n\t"\
        "    xor.b32      %3, %3, t12;\n\t"\
        "    xor.b32      %6, %6, t12;\n\t"\
        "    xor.b32      %7, %7, t12;\n\t"\
        "    xor.b32      %0, %0, t13;\n\t"\
        "    xor.b32      %1, %1, t13;\n\t"\
        "    xor.b32      %2, %2, t13;\n\t"\
        "    xor.b32      %7, %7, t13;\n\t"\
        "    xor.b32      %0, %0, t14;\n\t"\
        "    xor.b32      %1, %1, t14;\n\t"\
        "    xor.b32      %4, %4, t14;\n\t"\
        "}"\
        :\
        "=r"(z0),\
        "=r"(z1),\
        "=r"(z2),\
        "=r"(z3),\
        "=r"(z4),\
        "=r"(z5),\
        "=r"(z6),\
        "=r"(z7)\
        :\
        "r"(x0),\
        "r"(x1),\
        "r"(x2),\
        "r"(x3),\
        "r"(x4),\
        "r"(x5),\
        "r"(x6),\
        "r"(x7),\
        "r"(y0),\
        "r"(y1),\
        "r"(y2),\
        "r"(y3),\
        "r"(y4),\
        "r"(y5),\
        "r"(y6),\
        "r"(y7)\
        );

#endif

/************************************* Reference subroutines (testing only). */

#define REF_SCALAR_ADD(target, left, right) { target = gf2_add_ref(left, right); }
#if GF2_B == 8
#define REF_SCALAR_MUL(target, left, right) { target = gf2_8_mul_ref(left, right); }
#define SCALAR_FORMAT_STRING "0x%02X"
#endif
#if GF2_B == 32
#define REF_SCALAR_MUL(target, left, right) { target = gf2_32_mul_ref(left, right); }
#define SCALAR_FORMAT_STRING "0x%08X"
#endif
#if GF2_B == 64
#define REF_SCALAR_MUL(target, left, right) { target = gf2_64_mul_ref(left, right); }
#define SCALAR_FORMAT_STRING "0x%016lX"
#endif

inline scalar_t gf2_add_ref(scalar_t x, scalar_t y)
{   
    return x^y;
}

// Modulus:  [x^8 +] x^4 + x^3 + x^2 + 1            ~ 0x1D = 29 (primitive)

inline scalar_t gf2_8_mul_ref(scalar_t x, scalar_t y)
{   
    word_t z = 0;
    for(int i = 0; i < 8; i++) {
        word_t f = (scalar_t) (x & 0x080);
        if(y & 1)
            z ^= x;
        y = y >> 1;
        x = (x&0x07F) << 1;
        if(f)
            x ^= (scalar_t) GF2_8_MODULUS;
    }
    return z;
}

// Modulus: [x^32 +] x^7 + x^5 + x^3 + x^2 + x + 1   ~ 0xAF = 175 (primitive)

inline scalar_t gf2_32_mul_ref(scalar_t x, scalar_t y)
{   
    scalar_t z = 0;
    for(int i = 0; i < 32; i++) {
        scalar_t f = (scalar_t) (x & 0x80000000);
        if(y & 1)
            z ^= x;
        y >>= 1;
        x <<= 1;
        if(f)
            x ^= (scalar_t) GF2_32_MODULUS;
    }
    return z;
}

// Modulus: [x^64 +] x^4 + x^3 + x + 1               ~ 0x1B = 27 (primitive)

inline scalar_t gf2_64_mul_ref(scalar_t x, scalar_t y)
{   
    scalar_t z = 0;
    for(int i = 0; i < 64; i++) {
        scalar_t f = (scalar_t) (x & 0x8000000000000000L);
        if(y & 1)
            z ^= x;
        y >>= 1;
        x <<= 1;
        if(f)
            x ^= (scalar_t) GF2_64_MODULUS;
    }
    return z;
}

/************************************* Log/exp look-up table initialization. */

#ifdef GF_LOG_EXP_LOOKUP

#define GF_LOG_LOOKUP_SIZE (sizeof(scalar_t)*256)
#define GF_EXP_LOOKUP_SIZE (sizeof(scalar_t)*512)

scalar_t h_lookup_log[256];
scalar_t h_lookup_exp[512];

void gf_precompute_exp_log(void)
{   
    scalar_t v, g;  
    v = 0x01;   
    g = 0x02;
    // modulus is primitive so 0x02 == x generates the mult group   
    for(index_t i = 0; i < 511; i++) {      
        h_lookup_exp[i] = v;        
        if(i < 256)
            h_lookup_log[v] = i;        
        REF_SCALAR_MUL(v, v, g);        
    }   
}

#endif

#ifndef LINE_TYPE
#error "No line type selected"
#endif


