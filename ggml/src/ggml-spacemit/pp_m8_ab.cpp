// SPDX-License-Identifier: MIT
// Full-M8 i8 tile with A and B prepacked into the IME2 dot layout.
#include "pp_q4k.hpp"

namespace q4k_pp {

void gemm_m8_q8_ab_prepacked(const PackedABlockM8Ime * packed_a, const PackedBlockQ8N32Ime * packed_b,
                            float * output, std::size_t k_blocks, std::size_t ldc) {
    const std::uint8_t * a_data = reinterpret_cast<const std::uint8_t *>(packed_a);
    const std::uint8_t * b_data = reinterpret_cast<const std::uint8_t *>(packed_b);
    const std::size_t ldc_bytes = ldc * sizeof(float);

    __asm__ volatile(
        "vsetvli        t0, x0, e32, m1     \n\t"
        "vxor.vv        v24, v24, v24       \n\t"
        "vxor.vv        v25, v25, v25       \n\t"
        "vxor.vv        v26, v26, v26       \n\t"
        "vxor.vv        v27, v27, v27       \n\t"
        "vxor.vv        v28, v28, v28       \n\t"
        "vxor.vv        v29, v29, v29       \n\t"
        "vxor.vv        v30, v30, v30       \n\t"
        "vxor.vv        v31, v31, v31       \n\t"
        "mv             t4, %[BK]           \n\t"

        ".align 4                           \n\t"
        "BLK_LOOP%=:                        \n\t"
        // Two prearranged A-scale vectors directly match raw dot lanes.
        "vl2r.v         v0, (%[A])          \n\t"
        "addi           %[A], %[A], 256   \n\t"

        "vsetvli        t0, x0, e16, mf2    \n\t"
        "vle16.v        v12, (%[B])         \n\t"
        "addi           %[B], %[B], 64      \n\t"
        "vfwcvt.f.f.v   v14, v12            \n\t"

        "vsetvli        t0, x0, e8, m1      \n\t"
        "vl2r.v         v2, (%[A])         \n\t"
        "addi           %[A], %[A], 256   \n\t"
        "vl4r.v         v4, (%[B])          \n\t"
        "addi           %[B], %[B], 512     \n\t"
        "vl4r.v         v8, (%[B])          \n\t"
        "addi           %[B], %[B], 512     \n\t"

        "vsetvli        t0, x0, e32, m1     \n\t"

        "vxor.vv        v16, v16, v16       \n\t"
        "vxor.vv        v17, v17, v17       \n\t"
        "vxor.vv        v18, v18, v18       \n\t"
        "vxor.vv        v19, v19, v19       \n\t"
        "vxor.vv        v20, v20, v20       \n\t"
        "vxor.vv        v21, v21, v21       \n\t"
        "vxor.vv        v22, v22, v22       \n\t"
        "vxor.vv        v23, v23, v23       \n\t"
        "vmadot         v16, v2, v4, i8     \n\t"
        "vmadot         v18, v2, v6, i8     \n\t"
        "vmadot         v20, v2, v8, i8     \n\t"
        "vmadot         v22, v2, v10, i8    \n\t"
        "vmadot         v16, v3, v5, i8     \n\t"
        "vmadot         v18, v3, v7, i8     \n\t"
        "vmadot         v20, v3, v9, i8     \n\t"
        "vmadot         v22, v3, v11, i8    \n\t"

        // Broadcast each N8 quarter's B scales to match the raw dot lanes.
        "vpack.vv       v4, v14, v14, 2     \n\t"
        "vpack.vv       v6, v4, v4, 3       \n\t"
        "vpack.vv       v8, v5, v5, 3       \n\t"

        "vfcvt.f.x.v    v16, v16            \n\t"
        "vfcvt.f.x.v    v17, v17            \n\t"
        "vfcvt.f.x.v    v18, v18            \n\t"
        "vfcvt.f.x.v    v19, v19            \n\t"
        "vfcvt.f.x.v    v20, v20            \n\t"
        "vfcvt.f.x.v    v21, v21            \n\t"
        "vfcvt.f.x.v    v22, v22            \n\t"
        "vfcvt.f.x.v    v23, v23            \n\t"
        "vfmul.vv       v16, v16, v6        \n\t"
        "vfmul.vv       v17, v17, v6        \n\t"
        "vfmul.vv       v18, v18, v7        \n\t"
        "vfmul.vv       v19, v19, v7        \n\t"
        "vfmul.vv       v20, v20, v8        \n\t"
        "vfmul.vv       v21, v21, v8        \n\t"
        "vfmul.vv       v22, v22, v9        \n\t"
        "vfmul.vv       v23, v23, v9        \n\t"
        "vfmacc.vv      v24, v0, v16        \n\t"
        "vfmacc.vv      v25, v1, v17        \n\t"
        "vfmacc.vv      v26, v0, v18        \n\t"
        "vfmacc.vv      v27, v1, v19        \n\t"
        "vfmacc.vv      v28, v0, v20        \n\t"
        "vfmacc.vv      v29, v1, v21        \n\t"
        "vfmacc.vv      v30, v0, v22        \n\t"
        "vfmacc.vv      v31, v1, v23        \n\t"
        "addi           t4, t4, -1          \n\t"
        "bgtz           t4, BLK_LOOP%=      \n\t"

        // Undo the dot result layout once, after the entire K reduction.
        "vpack.vv       v0, v24, v26, 2     \n\t"
        "vpack.vv       v2, v25, v27, 2     \n\t"
        "vpack.vv       v4, v28, v30, 2     \n\t"
        "vpack.vv       v6, v29, v31, 2     \n\t"
        "vpack.vv       v16, v0, v4, 3      \n\t"
        "vpack.vv       v18, v1, v5, 3      \n\t"
        "vpack.vv       v20, v2, v6, 3      \n\t"
        "vpack.vv       v22, v3, v7, 3      \n\t"
        "mv             t2, %[DST]          \n\t"
        "vse32.v        v16, (t2)           \n\t"
        "add            t2, t2, %[LDC]     \n\t"
        "vse32.v        v17, (t2)           \n\t"
        "add            t2, t2, %[LDC]     \n\t"
        "vse32.v        v18, (t2)           \n\t"
        "add            t2, t2, %[LDC]     \n\t"
        "vse32.v        v19, (t2)           \n\t"
        "add            t2, t2, %[LDC]     \n\t"
        "vse32.v        v20, (t2)           \n\t"
        "add            t2, t2, %[LDC]     \n\t"
        "vse32.v        v21, (t2)           \n\t"
        "add            t2, t2, %[LDC]     \n\t"
        "vse32.v        v22, (t2)           \n\t"
        "add            t2, t2, %[LDC]     \n\t"
        "vse32.v        v23, (t2)           \n\t"
        : [A] "+r"(a_data), [B] "+r"(b_data)
        : [DST] "r"(output), [LDC] "r"(ldc_bytes), [BK] "r"(k_blocks)
        : "cc", "memory", "t0", "t2", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
          "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7");
}

}  // namespace q4k_pp
