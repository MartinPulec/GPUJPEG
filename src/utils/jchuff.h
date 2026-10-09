/*
 * jchuff.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying .c file.
 *
 * This file contains declarations for Huffman entropy encoding routines
 * that are shared between the sequential encoder (jchuff.c) and the
 * progressive encoder (jcphuff.c).  No other modules need to see these.
 */

#ifndef UTILS_TABLE_GEN_OPTIMAL_H
#define UTILS_TABLE_GEN_OPTIMAL_H

#ifdef __cplusplus
extern "C" {
#endif

struct gpujpeg_table_huffman_encoder;

void
jpeg_gen_optimal_table(struct gpujpeg_table_huffman_encoder* htbl, int freq[]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ! defined UTILS_TABLE_GEN_OPTIMAL_H
