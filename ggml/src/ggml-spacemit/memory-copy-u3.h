// SPDX-License-Identifier: MIT
// C009: three-load copy, verified by D011; all transfer stays timed.
#pragma once
#include "memory-copy.h"

// All bytes stay compressed. A100 VLEN=128 bytes, so m8 holds 1024 bytes.
// Three independent loads precede all their stores. The original VLA
// helper handles the exact remainder without reading or writing beyond bytes.
inline void tg_copy_weights_u3(uint8_t * dst,const uint8_t * src,size_t bytes) {
    constexpr size_t step=3072;
    size_t count=bytes/step;
    const size_t copied=count*step;
    if(count) {
        auto * current_src=src;
        auto * current_dst=dst;
        asm volatile(
            "li t0, %[STEP]\n\t"
            "vsetvli t1, x0, e8, m8, ta, ma\n\t"
            "1:\n\t"
            "vle8.v v0, (%[SRC])\n\t"
            "addi t1, %[SRC], 1024\n\t"
            "vle8.v v8, (t1)\n\t"
            "addi t1, t1, 1024\n\t"
            "vle8.v v16, (t1)\n\t"
            "vse8.v v0, (%[DST])\n\t"
            "addi t1, %[DST], 1024\n\t"
            "vse8.v v8, (t1)\n\t"
            "addi t1, t1, 1024\n\t"
            "vse8.v v16, (t1)\n\t"
            "add %[SRC], %[SRC], t0\n\t"
            "add %[DST], %[DST], t0\n\t"
            "addi %[CNT], %[CNT], -1\n\t"
            "bnez %[CNT], 1b\n\t"
            :[SRC] "+&r"(current_src),[DST] "+&r"(current_dst),[CNT] "+&r"(count)
            :[STEP] "i"(step)
            :"memory","t0","t1",
             "v0","v1","v2","v3","v4","v5","v6","v7",
             "v8","v9","v10","v11","v12","v13","v14","v15",
             "v16","v17","v18","v19","v20","v21","v22","v23");
    }
    tg_copy_weights(dst+copied,src+copied,bytes-copied);
}
