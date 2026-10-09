/*
 * jchuff.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009-2011, 2014-2016, 2018-2024, D. R. Commander.
 * Copyright (C) 2015, Matthieu Darbois.
 * Copyright (C) 2018, Matthias Räncker.
 * Copyright (C) 2020, Arm Limited.
 * Copyright (C) 2022, Felix Hanau.
 *
 * The authors make NO WARRANTY or representation, either express or implied,
 * with respect to this software, its quality, accuracy, merchantability, or
 * fitness for a particular purpose.  This software is provided "AS IS", and you,
 * its user, assume the entire risk as to its quality and accuracy.
 *
 * This software is copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding.
 * All Rights Reserved except as specified below.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * software (or portions thereof) for any purpose, without fee, subject to these
 * conditions:
 * (1) If any part of the source code for this software is distributed, then this
 * README file must be included, with this copyright and no-warranty notice
 * unaltered; and any additions, deletions, or changes to the original files
 * must be clearly indicated in accompanying documentation.
 * (2) If only executable code is distributed, then the accompanying
 * documentation must state that "this software is based in part on the work of
 * the Independent JPEG Group".
 * (3) Permission for use of this software is granted only if the user accepts
 * full responsibility for any undesirable consequences; the authors accept
 * NO LIABILITY for damages of any kind.
 *
 * These conditions apply to any software derived from or based on the IJG code,
 * not just to the unmodified library.  If you use our work, you ought to
 * acknowledge us.
 *
 * Permission is NOT granted for the use of any IJG author's name or company name
 * in advertising or publicity relating to this software or products derived from
 * it.  This software may be referred to only as "the Independent JPEG Group's
 * software".
 *
 * We specifically permit and encourage the use of this software as the basis of
 * commercial products, provided that all warranty or liability claims are
 * assumed by the product vendor.
 *
 *
 * This file contains Huffman entropy encoding routines.
 *
 * Much of the complexity here has to do with supporting output suspension.
 * If the data destination module demands suspension, we want to be able to
 * back up to the start of the current MCU.  To do this, we copy state
 * variables into local working storage, and update them back to the
 * permanent JPEG objects only upon successful completion of an MCU.
 *
 * NOTE: All referenced figures are from
 * Recommendation ITU-T T.81 (1992) | ISO/IEC 10918-1:1994.
 */

#include "jchuff.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../gpujpeg_table.h"

/*
 * Generate the best Huffman code table for the given counts, fill htbl.
 * Note this is also used by jcphuff.c and jclhuff.c.
 *
 * The JPEG standard requires that no symbol be assigned a codeword of all
 * one bits (so that padding bits added at the end of a compressed segment
 * can't look like a valid code).  Because of the canonical ordering of
 * codewords, this just means that there must be an unused slot in the
 * longest codeword length category.  Annex K (Clause K.2) of
 * Rec. ITU-T T.81 (1992) | ISO/IEC 10918-1:1994 suggests reserving such a slot
 * by pretending that symbol 256 is a valid symbol with count 1.  In theory
 * that's not optimal; giving it count zero but including it in the symbol set
 * anyway should give a better Huffman code.  But the theoretically better code
 * actually seems to come out worse in practice, because it produces more
 * all-ones bytes (which incur stuffed zero bytes in the final file).  In any
 * case the difference is tiny.
 *
 * The JPEG standard requires Huffman codes to be no more than 16 bits long.
 * If some symbols have a very small but nonzero probability, the Huffman tree
 * must be adjusted to meet the code length restriction.  We currently use
 * the adjustment method suggested in JPEG section K.2.  This method is *not*
 * optimal; it may not choose the best possible limited-length code.  But
 * typically only very-low-frequency symbols will be given less-than-optimal
 * lengths, so the code is almost optimal.  Experimental comparisons against
 * an optimal limited-length-code algorithm indicate that the difference is
 * microscopic --- usually less than a hundredth of a percent of total size.
 * So the extra complexity of an optimal algorithm doesn't seem worthwhile.
 */
void
jpeg_gen_optimal_table(struct gpujpeg_table_huffman_encoder*htbl, int freq[])
{
#define MAX_CLEN  32            /* assumed maximum initial code length */
  uint8_t bits[MAX_CLEN + 1];     /* bits[k] = # of symbols with code length k */
  int bit_pos[MAX_CLEN + 1];    /* # of symbols with smaller code length */
  int codesize[257];            /* codesize[k] = code length of symbol k */
  int nz_index[257];            /* index of nonzero symbol in the original freq
                                   array */
  int others[257];              /* next symbol in current branch of tree */
  int c1, c2;
  int p, i, j;
  int num_nz_symbols;
  long v, v2;

  /* This algorithm is explained in section K.2 of the JPEG standard */

  memset(bits, 0, sizeof(bits));
  memset(codesize, 0, sizeof(codesize));
  for (i = 0; i < 257; i++)
    others[i] = -1;             /* init links to empty */

  freq[256] = 1;                /* make sure 256 has a nonzero count */
  /* Including the pseudo-symbol 256 in the Huffman procedure guarantees
   * that no real symbol is given code-value of all ones, because 256
   * will be placed last in the largest codeword category.
   */

  /* Group nonzero frequencies together so we can more easily find the
   * smallest.
   */
  num_nz_symbols = 0;
  for (i = 0; i < 257; i++) {
    if (freq[i]) {
      nz_index[num_nz_symbols] = i;
      freq[num_nz_symbols] = freq[i];
      num_nz_symbols++;
    }
  }

  /* Huffman's basic algorithm to assign optimal code lengths to symbols */

  for (;;) {
    /* Find the two smallest nonzero frequencies; set c1, c2 = their symbols */
    /* In case of ties, take the larger symbol number.  Since we have grouped
     * the nonzero symbols together, checking for zero symbols is not
     * necessary.
     */
    c1 = -1;
    c2 = -1;
    v = 1000000000L;
    v2 = 1000000000L;
    for (i = 0; i < num_nz_symbols; i++) {
      if (freq[i] <= v2) {
        if (freq[i] <= v) {
          c2 = c1;
          v2 = v;
          v = freq[i];
          c1 = i;
        } else {
          v2 = freq[i];
          c2 = i;
        }
      }
    }

    /* Done if we've merged everything into one frequency */
    if (c2 < 0)
      break;

    /* Else merge the two counts/trees */
    freq[c1] += freq[c2];
    /* Set the frequency to a very high value instead of zero, so we don't have
     * to check for zero values.
     */
    freq[c2] = 1000000001L;

    /* Increment the codesize of everything in c1's tree branch */
    codesize[c1]++;
    while (others[c1] >= 0) {
      c1 = others[c1];
      codesize[c1]++;
    }

    others[c1] = c2;            /* chain c2 onto c1's tree branch */

    /* Increment the codesize of everything in c2's tree branch */
    codesize[c2]++;
    while (others[c2] >= 0) {
      c2 = others[c2];
      codesize[c2]++;
    }
  }

  /* Now count the number of symbols of each code length */
  for (i = 0; i < num_nz_symbols; i++) {
    /* The JPEG standard seems to think that this can't happen, */
    /* but I'm paranoid... */
    if (codesize[i] > MAX_CLEN) {
        puts("OVERFLOW");
        abort();
      // ERREXIT(cinfo, JERR_HUFF_CLEN_OVERFLOW);
    }

    bits[codesize[i]]++;
  }

  /* Count the number of symbols with a length smaller than i bits, so we can
   * construct the symbol table more efficiently.  Note that this includes the
   * pseudo-symbol 256, but since it is the last symbol, it will not affect the
   * table.
   */
  p = 0;
  for (i = 1; i <= MAX_CLEN; i++) {
    bit_pos[i] = p;
    p += bits[i];
  }

  /* JPEG doesn't allow symbols with code lengths over 16 bits, so if the pure
   * Huffman procedure assigned any such lengths, we must adjust the coding.
   * Here is what Rec. ITU-T T.81 | ISO/IEC 10918-1 says about how this next
   * bit works: Since symbols are paired for the longest Huffman code, the
   * symbols are removed from this length category two at a time.  The prefix
   * for the pair (which is one bit shorter) is allocated to one of the pair;
   * then, skipping the BITS entry for that prefix length, a code word from the
   * next shortest nonzero BITS entry is converted into a prefix for two code
   * words one bit longer.
   */

  for (i = MAX_CLEN; i > 16; i--) {
    while (bits[i] > 0) {
      j = i - 2;                /* find length of new prefix to be used */
      while (bits[j] == 0)
        j--;

      bits[i] -= 2;             /* remove two symbols */
      bits[i - 1]++;            /* one goes in this length */
      bits[j + 1] += 2;         /* two new symbols in this length */
      bits[j]--;                /* symbol of this length is now a prefix */
    }
  }

  /* Remove the count for the pseudo-symbol 256 from the largest codelength */
  while (bits[i] == 0)          /* find largest codelength still in use */
    i--;
  bits[i]--;

  /* Return final symbol counts (only for lengths 0..16) */
  memcpy(htbl->bits, bits, sizeof(htbl->bits));

  /* Return a list of the symbols sorted by code length */
  /* It's not real clear to me why we don't need to consider the codelength
   * changes made above, but Rec. ITU-T T.81 | ISO/IEC 10918-1 seems to think
   * this works.
   */
  for (i = 0; i < num_nz_symbols - 1; i++) {
    htbl->huffval[bit_pos[codesize[i]]] = (uint8_t)nz_index[i];
    bit_pos[codesize[i]]++;
  }

  /* Set sent_table FALSE so updated table will be written to JPEG file. */
  // htbl->sent_table = FALSE;
}
