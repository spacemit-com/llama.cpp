#pragma once
// Vendored from spine-kernel (MIT, (c) 2025-2026 SpacemiT) at commit 29955e5
// ("feat(pack-cast): add IME2 Q41Block to FP16 conversion").  Only the pieces
// the q4_0_f16 path uses live here: the asm macro headers, the ime2 pack_cast
// and mmt4d inner kernels, the rvv reducesum inner kernel, and the three
// dispatch functions.  Keep this file verbatim-synced with the source tree;
// the upstream tree stays the reference for changes.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>

#ifdef _MSC_VER
#    define SPINE_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__CLANG__)
#    define SPINE_FORCE_INLINE inline __attribute__((always_inline))
#else
#    define SPINE_FORCE_INLINE inline
#endif

#ifndef NDEBUG
#    define SPE_ASSERT(expr) assert(expr)
#else
#    define SPE_ASSERT(expr) ;
#endif

#include <functional>

#include "repack.h"  // Q41Block / _spe_fp16_ / uint4x2_t (vendored data types)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#else
#    include <riscv_vector.h>
#endif

#ifndef SPE_KERNELS_PRELUDE_SHIM
#define SPE_KERNELS_PRELUDE_SHIM
#endif

// ---- build/_deps/spine-arch-info-src/inc/spine_arch_info.h: arch descriptor
// unions and the IME2 arch bit (verbatim; the toString/GetSpineCoreInfo
// helpers stay in the spine-runtime side and are not vendored). ----
constexpr uint64_t SpineInvalidCoreID = 0xFFFFFFFF;

enum class SpineCoreArchID : uint16_t {
    CoreArchNone = 0,
    CoreArchX60  = 0x503C,
    CoreArchX100 = 0x5064,
    CoreArchX200 = 0x50C8,
    CoreArchA60  = 0xA03C,
    CoreArchA100 = 0xA064,
    CoreArchA200 = 0xA0C8,
};

union SpineArchExtensionInfoDef {
    uint32_t value;

    struct {
        uint8_t rvv;
        uint8_t ime;
        uint8_t ame;
        uint8_t vme;
    } fields;
};

union SpineArchInfoDef {
    uint64_t value;

    struct {
        SpineCoreArchID           core_arch_id;
        uint8_t                   matrix_info;
        uint8_t                   vector_info;
        SpineArchExtensionInfoDef core_ext_info;
    } fields;
};

// ---- src/spe_kernel_common_inner.h:230 (verbatim) ----
#define SPE_ARCH_IME2 ((uint8_t) (1 << 1))

// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef PACK_ASM_IME2
#define PACK_ASM_IME2

#define PACK_X8_M32N16_VLD_M8_0               \
    "vle8.v         v0, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v1, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v2, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v3, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v4, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v5, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v6, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v7, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t"

#define PACK_X8_M32N16_VLD_M8_1               \
    "vle8.v         v8, (s2)            \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v9, (s3)            \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v10, (s2)           \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v11, (s3)           \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v12, (s2)           \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v13, (s3)           \n\t" \
    "add            s3, s3, s1          \n\t" \
    "vle8.v         v14, (s2)           \n\t" \
    "add            s2, s2, s1          \n\t" \
    "vle8.v         v15, (s3)           \n\t" \
    "add            s3, s3, s1          \n\t"

#define PACK_X8_M32N16_N128_0                 \
    "vpack.vv       v16, v0, v1, 1      \n\t" \
    "vpack.vv       v18, v2, v3, 1      \n\t" \
    "vpack.vv       v20, v4, v5, 1      \n\t" \
    "vpack.vv       v22, v6, v7, 1      \n\t" \
    "vpack.vv       v0, v16, v18, 2     \n\t" \
    "vpack.vv       v2, v17, v19, 2     \n\t" \
    "vpack.vv       v4, v20, v22, 2     \n\t" \
    "vpack.vv       v6, v21, v23, 2     \n\t" \
    "vpack.vv       v16, v0, v4, 3      \n\t" \
    "vpack.vv       v18, v1, v5, 3      \n\t" \
    "vpack.vv       v20, v2, v6, 3      \n\t" \
    "vpack.vv       v22, v3, v7, 3      \n\t"

#define PACK_X8_M32N16_N128_1                 \
    "vpack.vv       v24, v8, v9, 1      \n\t" \
    "vpack.vv       v26, v10, v11, 1    \n\t" \
    "vpack.vv       v28, v12, v13, 1    \n\t" \
    "vpack.vv       v30, v14, v15, 1    \n\t" \
    "vpack.vv       v8, v24, v26, 2     \n\t" \
    "vpack.vv       v10, v25, v27, 2    \n\t" \
    "vpack.vv       v12, v28, v30, 2    \n\t" \
    "vpack.vv       v14, v29, v31, 2    \n\t" \
    "vpack.vv       v24, v8, v12, 3     \n\t" \
    "vpack.vv       v26, v9, v13, 3     \n\t" \
    "vpack.vv       v28, v10, v14, 3    \n\t" \
    "vpack.vv       v30, v11, v15, 3    \n\t"

#define PACK_X8_M32N16_VST_M16N64_0           \
    "vse8.v         v16, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v24, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v17, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v25, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v18, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v26, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v19, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v27, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t"

#define PACK_X8_M32N16_VST_M16N64_1           \
    "vse8.v         v20, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v28, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v21, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v29, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v22, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v30, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t" \
    "vse8.v         v23, (s4)           \n\t" \
    "addi           s4, s4, 512         \n\t" \
    "vse8.v         v31, (s5)           \n\t" \
    "addi           s5, s5, 512         \n\t"

#define PACK_X16_M16N8                                          \
    "vpack.vv               v16, v0, v1, 1                \n\t" \
    "vpack.vv               v18, v2, v3, 1                \n\t" \
    "vpack.vv               v20, v4, v5, 1                \n\t" \
    "vpack.vv               v22, v6, v7, 1                \n\t" \
                                                                \
    "vpack.vv               v0, v16, v18, 2               \n\t" \
    "vpack.vv               v2, v17, v19, 2               \n\t" \
    "vpack.vv               v4, v20, v22, 2               \n\t" \
    "vpack.vv               v6, v21, v23, 2               \n\t" \
                                                                \
    "vpack.vv               v16, v0, v4, 3                \n\t" \
    "vpack.vv               v18, v1, v5, 3                \n\t" \
    "vpack.vv               v20, v2, v6, 3                \n\t" \
    "vpack.vv               v22, v3, v7, 3                \n\t" \
                                                                \
    "vpack.vv               v24, v8, v9, 1                \n\t" \
    "vpack.vv               v26, v10, v11, 1              \n\t" \
    "vpack.vv               v28, v12, v13, 1              \n\t" \
    "vpack.vv               v30, v14, v15, 1              \n\t" \
                                                                \
    "vpack.vv               v8, v24, v26, 2               \n\t" \
    "vpack.vv               v10, v25, v27, 2              \n\t" \
    "vpack.vv               v12, v28, v30, 2              \n\t" \
    "vpack.vv               v14, v29, v31, 2              \n\t" \
                                                                \
    "vpack.vv               v24, v8, v12, 3               \n\t" \
    "vpack.vv               v26, v9, v13, 3               \n\t" \
    "vpack.vv               v28, v10, v14, 3              \n\t" \
    "vpack.vv               v30, v11, v15, 3              \n\t"
#endif

// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

// Flat instruction sequences for VLEN1024; each sequence documents its vtype.
// These macros contain only load, conversion, packing or store instructions.
// vtype setup, loop counters, branches and operand bindings live in the inner kernel.
// All strides are in bytes. No macro expands another macro.

// clang-format off

// Four independent row addresses: t0/t1/t5/t6 enter at rows 0/1/2/3.
// Read rows 0..3, advance each pointer by four rows, then read rows 4..7.
// Output: FP32 rows in v16..v31 (EMUL=2); pointers remain at rows 4/5/6/7.
#define PACK_CAST_LOAD_F32_M8K64                                               \
    "vle32.v        v16, (t0)                                            \n\t" \
    "vle32.v        v18, (t1)                                            \n\t" \
    "vle32.v        v20, (t5)                                            \n\t" \
    "vle32.v        v22, (t6)                                            \n\t" \
    "sh2add         t0, %[stride], t0                                    \n\t" \
    "sh2add         t1, %[stride], t1                                    \n\t" \
    "sh2add         t5, %[stride], t5                                    \n\t" \
    "sh2add         t6, %[stride], t6                                    \n\t" \
    "vle32.v        v24, (t0)                                            \n\t" \
    "vle32.v        v26, (t1)                                            \n\t" \
    "vle32.v        v28, (t5)                                            \n\t" \
    "vle32.v        v30, (t6)                                            \n\t"

// Narrow eight FP32 rows at v16/v18/.../v30 to v0..v7; VL is 32 or 64.
#define PACK_CAST_NARROW_F32_F16_M8_LO                                         \
    "vfncvt.f.f.w   v0, v16                                              \n\t" \
    "vfncvt.f.f.w   v1, v18                                              \n\t" \
    "vfncvt.f.f.w   v2, v20                                              \n\t" \
    "vfncvt.f.f.w   v3, v22                                              \n\t" \
    "vfncvt.f.f.w   v4, v24                                              \n\t" \
    "vfncvt.f.f.w   v5, v26                                              \n\t" \
    "vfncvt.f.f.w   v6, v28                                              \n\t" \
    "vfncvt.f.f.w   v7, v30                                              \n\t"

// Narrow eight FP32 rows at v16/v18/.../v30 to v8..v15; VL is 32 or 64.
#define PACK_CAST_NARROW_F32_F16_M8_HI                                         \
    "vfncvt.f.f.w   v8, v16                                              \n\t" \
    "vfncvt.f.f.w   v9, v18                                              \n\t" \
    "vfncvt.f.f.w   v10, v20                                             \n\t" \
    "vfncvt.f.f.w   v11, v22                                             \n\t" \
    "vfncvt.f.f.w   v12, v24                                             \n\t" \
    "vfncvt.f.f.w   v13, v26                                             \n\t" \
    "vfncvt.f.f.w   v14, v28                                             \n\t" \
    "vfncvt.f.f.w   v15, v30                                             \n\t"

// Reorder eight rows v0..v7 into eight K8 tiles v8..v15; preserve v16..v31.
#define PACK_CAST_PACK_F16_M8K64                                               \
    "vpack.vv       v8,  v0,  v1,  1                                     \n\t" \
    "vpack.vv       v10, v2,  v3,  1                                     \n\t" \
    "vpack.vv       v12, v4,  v5,  1                                     \n\t" \
    "vpack.vv       v14, v6,  v7,  1                                     \n\t" \
    "vpack.vv       v0,  v8,  v10, 2                                     \n\t" \
    "vpack.vv       v2,  v9,  v11, 2                                     \n\t" \
    "vpack.vv       v4,  v12, v14, 2                                     \n\t" \
    "vpack.vv       v6,  v13, v15, 2                                     \n\t" \
    "vpack.vv       v8,  v0,  v4,  3                                     \n\t" \
    "vpack.vv       v10, v1,  v5,  3                                     \n\t" \
    "vpack.vv       v12, v2,  v6,  3                                     \n\t" \
    "vpack.vv       v14, v3,  v7,  3                                     \n\t"

// Store the eight 8x8 tiles at MB16/MB32 tile_stride intervals. t2 ends at tile 7.
#define PACK_CAST_STORE_F16_M8K64                                              \
    "vse16.v        v8, (t2)                                             \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v9, (t2)                                             \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v10, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v11, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v12, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v13, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v14, (t2)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "vse16.v        v15, (t2)                                            \n\t"

// Store eight 16x8 tiles from PACK_X16_M16N8: v16..v23 are top halves, v24..v31 bottom halves.
#define PACK_CAST_STORE_F16_M16K64                                             \
    "vse16.v        v16, (t0)                                            \n\t" \
    "vse16.v        v24, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v17, (t0)                                            \n\t" \
    "vse16.v        v25, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v18, (t0)                                            \n\t" \
    "vse16.v        v26, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v19, (t0)                                            \n\t" \
    "vse16.v        v27, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v20, (t0)                                            \n\t" \
    "vse16.v        v28, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v21, (t0)                                            \n\t" \
    "vse16.v        v29, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v22, (t0)                                            \n\t" \
    "vse16.v        v30, (t1)                                            \n\t" \
    "addi           t0, t0, %[tile_stride]                               \n\t" \
    "addi           t1, t1, %[tile_stride]                               \n\t" \
    "vse16.v        v23, (t0)                                            \n\t" \
    "vse16.v        v31, (t1)                                            \n\t"

// e16/m8, VL=512: expand all 32 rows from packed bytes in v4..v7.
// v24..v31 hold interleaved U4 bytes; v8..v23 then hold the two I16/FP16 halves.
// Keep v0 (the row mask) intact. Every integer difference fits exactly in FP16.
#define PACK_CAST_UNPACK_U4_F16_M32K32                                         \
    "vzext.vf2      v8, v4                                               \n\t" \
    "vsrl.vi        v16, v8, 4                                           \n\t" \
    "vand.vi        v8, v8, 15                                           \n\t" \
    "vsll.vi        v16, v16, 8                                          \n\t" \
    "vor.vv         v24, v8, v16                                         \n\t" \
    "vzext.vf2      v8, v24                                              \n\t" \
    "vzext.vf2      v16, v28                                             \n\t" \
    "vsub.vx        v8, v8, %[zero_point]                                \n\t" \
    "vsub.vx        v16, v16, %[zero_point]                              \n\t" \
    "vfcvt.f.x.v    v8, v8                                               \n\t" \
    "vfcvt.f.x.v    v16, v16                                             \n\t"

// Load FP16 scales for rows 0..15; t1 is the current block scale base.
#define PACK_CAST_LOAD_SCALE_F16_M16_LO                                        \
    "flh            ft0, 0(t1)                                           \n\t" \
    "flh            ft1, 2(t1)                                           \n\t" \
    "flh            ft2, 4(t1)                                           \n\t" \
    "flh            ft3, 6(t1)                                           \n\t" \
    "flh            ft4, 8(t1)                                           \n\t" \
    "flh            ft5, 10(t1)                                          \n\t" \
    "flh            ft6, 12(t1)                                          \n\t" \
    "flh            ft7, 14(t1)                                          \n\t" \
    "flh            fa0, 16(t1)                                          \n\t" \
    "flh            fa1, 18(t1)                                          \n\t" \
    "flh            fa2, 20(t1)                                          \n\t" \
    "flh            fa3, 22(t1)                                          \n\t" \
    "flh            fa4, 24(t1)                                          \n\t" \
    "flh            fa5, 26(t1)                                          \n\t" \
    "flh            fa6, 28(t1)                                          \n\t" \
    "flh            fa7, 30(t1)                                          \n\t"

// Load FP16 scales for rows 16..31; t1 is the current block scale base.
#define PACK_CAST_LOAD_SCALE_F16_M16_HI                                        \
    "flh            ft0, 32(t1)                                          \n\t" \
    "flh            ft1, 34(t1)                                          \n\t" \
    "flh            ft2, 36(t1)                                          \n\t" \
    "flh            ft3, 38(t1)                                          \n\t" \
    "flh            ft4, 40(t1)                                          \n\t" \
    "flh            ft5, 42(t1)                                          \n\t" \
    "flh            ft6, 44(t1)                                          \n\t" \
    "flh            ft7, 46(t1)                                          \n\t" \
    "flh            fa0, 48(t1)                                          \n\t" \
    "flh            fa1, 50(t1)                                          \n\t" \
    "flh            fa2, 52(t1)                                          \n\t" \
    "flh            fa3, 54(t1)                                          \n\t" \
    "flh            fa4, 56(t1)                                          \n\t" \
    "flh            fa5, 58(t1)                                          \n\t" \
    "flh            fa6, 60(t1)                                          \n\t" \
    "flh            fa7, 62(t1)                                          \n\t"

// e16/m1, VL=64: scale rows 0..15 in v8..v15.
// v0 selects each register's high 32 lanes; v1/v2 alternate the paired row scales.
#define PACK_CAST_SCALE_F16_M16K32_LO                                          \
    "vfmv.v.f       v1, ft0                                              \n\t" \
    "vfmv.v.f       v2, ft2                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft3, v0                                      \n\t" \
    "vfmul.vv       v8, v8, v1                                           \n\t" \
    "vfmul.vv       v9, v9, v2                                           \n\t" \
    "vfmv.v.f       v1, ft4                                              \n\t" \
    "vfmv.v.f       v2, ft6                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft7, v0                                      \n\t" \
    "vfmul.vv       v10, v10, v1                                         \n\t" \
    "vfmul.vv       v11, v11, v2                                         \n\t" \
    "vfmv.v.f       v1, fa0                                              \n\t" \
    "vfmv.v.f       v2, fa2                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa3, v0                                      \n\t" \
    "vfmul.vv       v12, v12, v1                                         \n\t" \
    "vfmul.vv       v13, v13, v2                                         \n\t" \
    "vfmv.v.f       v1, fa4                                              \n\t" \
    "vfmv.v.f       v2, fa6                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa7, v0                                      \n\t" \
    "vfmul.vv       v14, v14, v1                                         \n\t" \
    "vfmul.vv       v15, v15, v2                                         \n\t"

// e16/m1, VL=64: scale rows 16..31 in v16..v23.
// v0 selects each register's high 32 lanes; v1/v2 alternate the paired row scales.
#define PACK_CAST_SCALE_F16_M16K32_HI                                          \
    "vfmv.v.f       v1, ft0                                              \n\t" \
    "vfmv.v.f       v2, ft2                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft3, v0                                      \n\t" \
    "vfmul.vv       v16, v16, v1                                         \n\t" \
    "vfmul.vv       v17, v17, v2                                         \n\t" \
    "vfmv.v.f       v1, ft4                                              \n\t" \
    "vfmv.v.f       v2, ft6                                              \n\t" \
    "vfmerge.vfm    v1, v1, ft5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, ft7, v0                                      \n\t" \
    "vfmul.vv       v18, v18, v1                                         \n\t" \
    "vfmul.vv       v19, v19, v2                                         \n\t" \
    "vfmv.v.f       v1, fa0                                              \n\t" \
    "vfmv.v.f       v2, fa2                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa1, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa3, v0                                      \n\t" \
    "vfmul.vv       v20, v20, v1                                         \n\t" \
    "vfmul.vv       v21, v21, v2                                         \n\t" \
    "vfmv.v.f       v1, fa4                                              \n\t" \
    "vfmv.v.f       v2, fa6                                              \n\t" \
    "vfmerge.vfm    v1, v1, fa5, v0                                      \n\t" \
    "vfmerge.vfm    v2, v2, fa7, v0                                      \n\t" \
    "vfmul.vv       v22, v22, v1                                         \n\t" \
    "vfmul.vv       v23, v23, v2                                         \n\t"

// e16/m1, VL=64: pack rows 0..15 from v8..v15 into v24..v31.
// Every vpack writes a register pair; preserve v0 and the next payload in v4..v7.
#define PACK_CAST_PACK_F16_M16K32_LO                                           \
    "vpack.vv       v24, v8, v9, 1                                       \n\t" \
    "vpack.vv       v26, v10, v11, 1                                     \n\t" \
    "vpack.vv       v8, v24, v25, 1                                      \n\t" \
    "vpack.vv       v10, v26, v27, 1                                     \n\t" \
    "vpack.vv       v24, v8, v10, 3                                      \n\t" \
    "vpack.vv       v26, v9, v11, 3                                      \n\t" \
    "vpack.vv       v28, v12, v13, 1                                     \n\t" \
    "vpack.vv       v30, v14, v15, 1                                     \n\t" \
    "vpack.vv       v12, v28, v29, 1                                     \n\t" \
    "vpack.vv       v14, v30, v31, 1                                     \n\t" \
    "vpack.vv       v28, v12, v14, 3                                     \n\t" \
    "vpack.vv       v30, v13, v15, 3                                     \n\t"

// e16/m1, VL=64: pack rows 16..31 from v16..v23 into v24..v31.
// Every vpack writes a register pair; preserve v0 and the next payload in v4..v7.
#define PACK_CAST_PACK_F16_M16K32_HI                                           \
    "vpack.vv       v24, v16, v17, 1                                     \n\t" \
    "vpack.vv       v26, v18, v19, 1                                     \n\t" \
    "vpack.vv       v16, v24, v25, 1                                     \n\t" \
    "vpack.vv       v18, v26, v27, 1                                     \n\t" \
    "vpack.vv       v24, v16, v18, 3                                     \n\t" \
    "vpack.vv       v26, v17, v19, 3                                     \n\t" \
    "vpack.vv       v28, v20, v21, 1                                     \n\t" \
    "vpack.vv       v30, v22, v23, 1                                     \n\t" \
    "vpack.vv       v20, v28, v29, 1                                     \n\t" \
    "vpack.vv       v22, v30, v31, 1                                     \n\t" \
    "vpack.vv       v28, v20, v22, 3                                     \n\t" \
    "vpack.vv       v30, v21, v23, 3                                     \n\t"

// Store sixteen-row slices of four MB32 tiles. t3 starts eight rows after t2.
// The two stores per tile write consecutive 128-byte halves; tile_stride is 512.
#define PACK_CAST_STORE_F16_M16K32                                             \
    "vse16.v        v24, (t2)                                            \n\t" \
    "vse16.v        v28, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v25, (t2)                                            \n\t" \
    "vse16.v        v29, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v26, (t2)                                            \n\t" \
    "vse16.v        v30, (t3)                                            \n\t" \
    "addi           t2, t2, %[tile_stride]                               \n\t" \
    "addi           t3, t3, %[tile_stride]                               \n\t" \
    "vse16.v        v27, (t2)                                            \n\t" \
    "vse16.v        v31, (t3)                                            \n\t"

// clang-format on

// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef MMT4D_ASM_IME2
#define MMT4D_ASM_IME2
// clang-format off


/**
 * Operator function: Compute mmt4d (FP16 Matrix Multiplication with 4D Tiling)
 * Supported arch info: rvv1.0, vlen=1024bit
 * Data layout: mb=16, nb=32, kb=8
 * A: [kc][mb][kb] - Left-hand side matrix (activations)
 * B: [kc][nb][kb] - Right-hand side matrix (weights)
 * C: Output matrix
 *
 * Vector register allocation:
 * A (activations):  v0-v7 (8 registers for loading A tiles)
 * B (weights):      v8-v15 (8 registers for loading B tiles)
 * C (accumulators): v16-v31 (16 registers for FP32 accumulation results)
 *   - v16,v17: acc[0][0], v18,v19: acc[0][1], v20,v21: acc[0][2], v22,v23: acc[0][3]
 *   - v24,v25: acc[1][0], v26,v27: acc[1][1], v28,v29: acc[1][2], v30,v31: acc[1][3]
 *
 * Supported epilogues:
 * (1) EPILOGUE_TYPE_NONE
 * (2) EPILOGUE_TYPE_ADD_BIAS_MINMAXCLIP
 * (3) EPILOGUE_TYPE_HP_ADD_BIAS_MINMAXCLIP
 * (4) EPILOGUE_TYPE_ADD_BIAS_MINMAXCLIP_INPLACE_ADD
 * (5) EPILOGUE_TYPE_ADD_BIAS_PRELU_MINMAXCLIP
 * (6) EPILOGUE_TYPE_ADD_BIAS_LEAKYRELU_MINMAXCLIP
 */

/* ============================================================================
 * Scalar Register Aliases
 * ============================================================================
 * These macros define aliases for scalar registers used in assembly code.
 * Using aliases makes the code more readable and reduces register conflicts.
 */

/* Pointer registers for matrix A */
#define _A0         "a1"    // Pointer to A tile row 0
#define _A1         "a2"    // Pointer to A tile row 1

/* Pointer registers for matrix B */
#define _B0         "s1"    // Pointer to B tile column 0
#define _B1         "s2"    // Pointer to B tile column 1
#define _B2         "s3"    // Pointer to B tile column 2
#define _B3         "s4"    // Pointer to B tile column 3

/* Temporary registers */
#define _TMP_VL     "t0"    // Temporary for vsetvli result
#define _TMP_KC     "t2"    // Loop counter for K dimension

/* Input/Output parameter aliases (lowercase for inline asm operands) */
#define _A          "%[a]"      // Base pointer to A matrix
#define _B          "%[b]"      // Base pointer to B matrix
#define _C          "%[c]"      // Base pointer to C matrix (output)
#define _M          "%[m]"      // Number of rows in output
#define _N          "%[n]"      // Number of columns in output
#define _KC         "%[kc]"     // K dimension iteration count
#define _CS         "%[cs]"     // C stride (row stride of output)
#define _MIN        "%[min]"    // Min value for clipping
#define _MAX        "%[max]"    // Max value for clipping
#define _BIAS       "%[bias]"   // Pointer to bias vector

#define _FSCALAR    "%[fscalar]"
#define _ALPHA      "%[alpha]"
#define _ZERO       "%[zero]"

/* ============================================================================
 * Initialization Stage: Zero accumulators and setup pointers
 * ============================================================================
 * Vector registers used: v16-v31 (accumulators zeroed)
 */

/* Initialize: Zero accumulators and setup A/B pointers */
#define MMT4D_FP16_M16N32K8_INIT_IMPL(A, A0, A1, B, B0, B1, B2, B3, KC, TMP_VL, TMP_KC)         \
    /* Zero accumulator registers v16-v31 */                                                    \
    "vsetvli            " TMP_VL ", zero, e32, m8, tu, mu       \n\t"                           \
    "vxor.vv            v16, v16, v16                           \n\t"                           \
    "vxor.vv            v24, v24, v24                           \n\t"                           \
    /* Set vtype for FP16 operations */                                                         \
    "vsetvli            " TMP_VL ", zero, e16, m1               \n\t"                           \
    /* Initialize A pointers: A0 = A, A1 = A + 8*8*2 */                                         \
    "addi               " A0 ", " A ", 0                        \n\t"                           \
    "addi               " A1 ", " A ", 8*8*2                    \n\t"                           \
    /* Initialize B pointers: B0-B3 for 4 column blocks */                                      \
    "addi               " B0 ", " B ", 0                        \n\t"                           \
    "addi               " B1 ", " B ", 8*8*2                    \n\t"                           \
    "addi               " B2 ", " B ", 8*8*4                    \n\t"                           \
    "addi               " B3 ", " B ", 8*8*6                    \n\t"                           \
    /* Calculate KC/4 for main loop */                                                          \
    "srli               " TMP_KC ", " KC ", 2                   \n\t"

#define MMT4D_FP16_M16N32K8_INIT                                                                \
    MMT4D_FP16_M16N32K8_INIT_IMPL(_A, _A0, _A1, _B, _B0, _B1, _B2, _B3, _KC, _TMP_VL, _TMP_KC)

/* ============================================================================
 * Main Compute Loop: Handles KC iterations with software pipelining
 * ============================================================================
 */

/* Main compute loop with KC/4 iterations, handles KC%4 tail */
#define MMT4D_FP16_M16N32K8_COMPUTE_LOOP_IMPL(KC, A0, A1, B0, B1, B2, B3, TMP_VL, TMP_KC)       \
    /* Check if KC >= 4, otherwise skip to tail */                                              \
    "blez               " TMP_KC ", TAIL_KC1%=                  \n\t"                           \
                                                                                                \
    /* Preload first tiles for software pipelining */                                           \
    MMT4D_FP16_PRELOAD_M16N32_IMPL(_A, A0, A1, _B, B0, B1, B2, B3, TMP_VL)                      \
                                                                                                \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", TAIL_KC4%=                  \n\t"                           \
                                                                                                \
    /* Main loop: process 4 K iterations per loop */                                            \
    "LOOP_KC4%=:                                                \n\t"                           \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M2_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M3_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_M4_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "bnez               " TMP_KC ", LOOP_KC4%=                  \n\t"                           \
                                                                                                \
    /* Process last 4 iterations (epilogue without next tile preload) */                        \
    "TAIL_KC4%=:                                                \n\t"                           \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                      \
    MMT4D_FP16_COMP_M16N32_E2_IMPL(B0, B1, B2, B3)                                              \
    MMT4D_FP16_COMP_M16N32_E3_IMPL(B3)                                                          \
    MMT4D_FP16_COMP_M16N32_E4                                                                   \
                                                                                                \
    /* Handle KC % 4 tail iterations */                                                         \
    "TAIL_KC1%=:                                                \n\t"                           \
    "andi               " TMP_KC ", " KC ", 3                   \n\t"                           \
    "blez               " TMP_KC ", POST_PROCESS%=              \n\t"                           \
    "vsetvli            " TMP_VL ", zero, e16, m1               \n\t"                           \
                                                                                                \
    /* Single K iteration loop for tail */                                                      \
    "LOOP_KC1%=:                                                \n\t"                           \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v7, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    MMT4D_FP16_COMP_M16N32_E4                                                                   \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "bnez               " TMP_KC ", LOOP_KC1%=                  \n\t"                           \
                                                                                                \
    "POST_PROCESS%=:                                            \n\t"

#define MMT4D_FP16_M16N32K8_COMPUTE_LOOP                                                        \
    MMT4D_FP16_M16N32K8_COMPUTE_LOOP_IMPL(_KC, _A0, _A1, _B0, _B1, _B2, _B3, _TMP_VL, _TMP_KC)

/* ============================================================================
 * Postprocess Stage: Unpack accumulators to output layout
 * ============================================================================
 */

/* Unpack only (for NONE epilogue) */
#define MMT4D_FP16_M16N32K8_POSTPROCESS_UNPACK_IMPL(TMP_VL)                                     \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32

#define MMT4D_FP16_M16N32K8_POSTPROCESS_UNPACK                                                  \
    MMT4D_FP16_M16N32K8_POSTPROCESS_UNPACK_IMPL(_TMP_VL)

/* Postprocess with FP32 bias and minmax clip */
#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)            \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    /* Check if bias is null */                                                                 \
    "beq                " BIAS ", zero, CLIP%=                  \n\t"                           \
    /* Load FP32 bias vector */                                                                 \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "CLIP%=:                                                    \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX                                             \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_NEG_SLOPE_ONE(REG, ALPHA, ZERO)                                              \
    "vmflt.vf           v0, " REG ", " ZERO "                   \n\t"                           \
    "vfmul.vf           " REG ", " REG ", " ALPHA ", v0.t       \n\t"

#define MMT4D_FP16_NEG_SLOPE_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)                                \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    MMT4D_FP16_NEG_SLOPE_ONE("v16", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v17", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v18", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v19", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v20", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v21", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v22", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v23", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v24", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v25", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v26", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v27", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v28", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v29", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v30", ALPHA, ZERO)                                                \
    MMT4D_FP16_NEG_SLOPE_ONE("v31", ALPHA, ZERO)

#define MMT4D_FP16_PRELU_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)                                    \
    MMT4D_FP16_NEG_SLOPE_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)

#define MMT4D_FP16_PRELU_M16N32                                                                 \
    MMT4D_FP16_PRELU_M16N32_IMPL(_ALPHA, _ZERO, _N, _TMP_VL)

#define MMT4D_FP16_LEAKYRELU_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)                                \
    MMT4D_FP16_NEG_SLOPE_M16N32_IMPL(ALPHA, ZERO, N, TMP_VL)

#define MMT4D_FP16_LEAKYRELU_M16N32                                                             \
    MMT4D_FP16_LEAKYRELU_M16N32_IMPL(_ALPHA, _ZERO, _N, _TMP_VL)

/* Postprocess with FP32 bias, scalar FP32 PReLU, and minmax clip */
#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)      \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "beq                " BIAS ", zero, PRELU%=                 \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "PRELU%=:                                                   \n\t"                           \
    MMT4D_FP16_PRELU_M16N32                                                                     \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX                                       \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

/* Postprocess with FP32 bias, scalar FP32 LeakyRelu, and minmax clip */
#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)  \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "beq                " BIAS ", zero, LEAKYRELU%=             \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "LEAKYRELU%=:                                               \n\t"                           \
    MMT4D_FP16_LEAKYRELU_M16N32                                                                 \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX                                   \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

/* Postprocess with FP16 bias and minmax clip (HP = half precision bias) */
#define MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX_IMPL(BIAS, N, M, TMP_VL, TMP_KC)         \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    /* Convert FP32 accumulators to FP16 first */                                               \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    /* Check if bias is null */                                                                 \
    "beq                " BIAS ", zero, MINMAX%=                \n\t"                           \
    /* Load FP16 bias vector */                                                                 \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "vle16.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "MINMAX%=:                                                  \n\t"                           \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX                                          \
    MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX_IMPL(_BIAS, _N, _M, _TMP_VL, _TMP_KC)

/* Postprocess for none epilogue (no bias, no clip) */
#define MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(N, M, TMP_VL, TMP_KC)                         \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "CLIP%=:                                                    \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32

#define MMT4D_FP16_M16N32K8_POSTPROCESS_NONE                                                    \
    MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(_N, _M, _TMP_VL, _TMP_KC)

#define MMT4D_FP16_MUL_SCALAR_M16N32_IMPL(FSCALAR, N, TMP_VL)                                   \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "vfmul.vf           v16, v16, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v17, v17, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v18, v18, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v19, v19, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v20, v20, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v21, v21, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v22, v22, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v23, v23, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v24, v24, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v25, v25, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v26, v26, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v27, v27, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v28, v28, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v29, v29, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v30, v30, " FSCALAR "                   \n\t"                           \
    "vfmul.vf           v31, v31, " FSCALAR "                   \n\t"
#define MMT4D_FP16_M16N32K8_POSTPROCESS_MULSCALAR                                               \
    MMT4D_FP16_M16N32K8_POSTPROCESS_NONE_IMPL(_N, _M, _TMP_VL, _TMP_KC) \
MMT4D_FP16_MUL_SCALAR_M16N32_IMPL(_FSCALAR, _N,  _TMP_VL)

/* Postprocess with bias, minmax, and inplace add */
#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE_IMPL(BIAS, C, CS, N, M, TMP_VL, TMP_KC, A0) \
    "vsetvli            " TMP_VL ", zero, e32, m1               \n\t"                           \
    MMT4D_FP16_UNPACK_M16N32                                                                    \
    /* Check if bias is null */                                                                 \
    "beq                " BIAS ", zero, INPLACE_ADD%=           \n\t"                           \
    /* Load FP32 bias vector */                                                                 \
    "vsetvli            " TMP_VL ", " N ", e32, m1              \n\t"                           \
    "vle32.v            v0, (" BIAS ")                          \n\t"                           \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                       \
    "INPLACE_ADD%=:                                             \n\t"                           \
    MMT4D_FP16_INPLACE_ADD_LOAD_LT16_IMPL(C, CS, N, M, TMP_VL, TMP_KC, A0)                      \
    "CLIP%=:                                                    \n\t"                           \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "srli               " TMP_KC ", " M ", 4                    \n\t"                           \
    MMT4D_FP16_CONVERT_M16N32                                                                   \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

#define MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE                                     \
    MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE_IMPL(_BIAS, _C, _CS, _N, _M, _TMP_VL, _TMP_KC, _A0)

/* ============================================================================
 * Store Stage: Save results to memory with M bounds checking
 * ============================================================================
 */

/* Store output with M bounds check */
#define MMT4D_FP16_M16N32K8_STORE_IMPL(C, CS, M, TMP_KC)                                        \
    "bnez               " TMP_KC ", STORE_M16%=                 \n\t"                           \
    MMT4D_FP16_SAVING_LT16_IMPL(C, CS, M)                                                       \
    "jal                x0, END%=                               \n\t"                           \
    "STORE_M16%=:                                               \n\t"                           \
    MMT4D_FP16_SAVING_M16_IMPL(C, CS)                                                           \
    "END%=:                                                     \n\t"

#define MMT4D_FP16_M16N32K8_STORE                                                               \
    MMT4D_FP16_M16N32K8_STORE_IMPL(_C, _CS, _M, _TMP_KC)

/* ============================================================================
 * Preprocess Stage: Initialize pointers and load first tiles
 * ============================================================================
 * Registers used (read):   A, B
 * Registers used (write):  A0, A1, B0, B1, B2, B3, TMP_VL
 * Vector registers used:   v0-v5 (A tiles), v8-v14 (B tiles)
 */
#define MMT4D_FP16_PRELOAD_M16N32_IMPL(A, A0, A1, B, B0, B1, B2, B3, TMP_VL)                    \
    /* Load first A tile: v0, v1 from A[0:8] and A[8:16] */                                     \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v1, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
                                                                                                \
    /* Load second A tile: v2, v3 */                                                            \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v3, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
                                                                                                \
    /* Load third A tile: v4, v5 */                                                             \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v5, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
                                                                                                \
    /* Load first B tile: v8, v9, v10, v11 for 4 columns */                                     \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
                                                                                                \
    /* Load second B tile: v12, v13, v14 (v15 loaded in compute) */                             \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"

#define MMT4D_FP16_PRELOAD_M16N32                                                               \
    MMT4D_FP16_PRELOAD_M16N32_IMPL(_A, _A0, _A1, _B, _B0, _B1, _B2, _B3, _TMP_VL)

/* ============================================================================
 * Compute Stage: Main computation loops with software pipelining
 * ============================================================================
 * These macros implement the matrix multiply-accumulate operations.
 * Each M1-M4 macro computes one K iteration and preloads next data.
 * E2-E4 macros are for the epilogue (last iteration, no preload).
 *
 * Vector registers used:
 * - v0-v7:   A tiles (activations)
 * - v8-v15:  B tiles (weights)
 * - v16-v31: C accumulators (FP32, widened from FP16)
 */

/* Compute macro M1: Process A[v0,v1] x B[v8-v11], preload next A[v6,v7] and B[v15,v8-v10] */
#define MMT4D_FP16_COMP_M16N32_M1_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    /* Load next A tile for pipelining */                                                       \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v7, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
                                                                                                \
    /* vfwmadot: v16 += v0 * v8 (widening FP16->FP32) */                                        \
    "vfwmadot           v16, v0, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v1, v8                             \n\t"                           \
                                                                                                \
    "vfwmadot           v18, v0, v9                             \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v1, v9                             \n\t"                           \
                                                                                                \
    "vfwmadot           v20, v0, v10                            \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v1, v10                            \n\t"                           \
                                                                                                \
    "vfwmadot           v22, v0, v11                            \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v1, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M1                                                               \
    MMT4D_FP16_COMP_M16N32_M1_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

/* Compute macro M2: Process A[v2,v3] x B[v12-v15], preload next A[v0,v1] and B[v11-v14] */
#define MMT4D_FP16_COMP_M16N32_M2_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    /* Load next A tile for pipelining */                                                       \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v1, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
                                                                                                \
    "vfwmadot           v16, v2, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v3, v12                            \n\t"                           \
                                                                                                \
    "vfwmadot           v18, v2, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v3, v13                            \n\t"                           \
                                                                                                \
    "vfwmadot           v20, v2, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v3, v14                            \n\t"                           \
                                                                                                \
    "vfwmadot           v22, v2, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v3, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M2                                                               \
    MMT4D_FP16_COMP_M16N32_M2_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

/* Compute macro M3: Process A[v4,v5] x B[v8-v11], preload next A[v2,v3] and B[v15,v8-v10] */
#define MMT4D_FP16_COMP_M16N32_M3_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    /* Load next A tile for pipelining */                                                       \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v3, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
                                                                                                \
    "vfwmadot           v16, v4, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v5, v8                             \n\t"                           \
                                                                                                \
    "vfwmadot           v18, v4, v9                             \n\t"                           \
    "vle16.v            v8, (" B0 ")                            \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v5, v9                             \n\t"                           \
                                                                                                \
    "vfwmadot           v20, v4, v10                            \n\t"                           \
    "vle16.v            v9, (" B1 ")                            \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v5, v10                            \n\t"                           \
                                                                                                \
    "vfwmadot           v22, v4, v11                            \n\t"                           \
    "vle16.v            v10, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v5, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M3                                                               \
    MMT4D_FP16_COMP_M16N32_M3_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

/* Compute macro M4: Process A[v6,v7] x B[v12-v15], preload next A[v4,v5] and B[v11-v14] */
#define MMT4D_FP16_COMP_M16N32_M4_IMPL(A0, A1, B0, B1, B2, B3)                                  \
    /* Load next A tile for pipelining */                                                       \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "addi               " A0 ", " A0 ", 16*8*2                  \n\t"                           \
    "vle16.v            v5, (" A1 ")                            \n\t"                           \
    "addi               " A1 ", " A1 ", 16*8*2                  \n\t"                           \
                                                                                                \
    "vfwmadot           v16, v6, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v7, v12                            \n\t"                           \
                                                                                                \
    "vfwmadot           v18, v6, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v7, v13                            \n\t"                           \
                                                                                                \
    "vfwmadot           v20, v6, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v7, v14                            \n\t"                           \
                                                                                                \
    "vfwmadot           v22, v6, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v7, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_M4                                                               \
    MMT4D_FP16_COMP_M16N32_M4_IMPL(_A0, _A1, _B0, _B1, _B2, _B3)

/* Compute macro E2: Epilogue version of M2 (no next A preload, only B preload) */
#define MMT4D_FP16_COMP_M16N32_E2_IMPL(B0, B1, B2, B3)                                          \
    "vfwmadot           v16, v2, v12                            \n\t"                           \
    "vle16.v            v11, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v3, v12                            \n\t"                           \
                                                                                                \
    "vfwmadot           v18, v2, v13                            \n\t"                           \
    "vle16.v            v12, (" B0 ")                           \n\t"                           \
    "addi               " B0 ", " B0 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v26, v3, v13                            \n\t"                           \
                                                                                                \
    "vfwmadot           v20, v2, v14                            \n\t"                           \
    "vle16.v            v13, (" B1 ")                           \n\t"                           \
    "addi               " B1 ", " B1 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v28, v3, v14                            \n\t"                           \
                                                                                                \
    "vfwmadot           v22, v2, v15                            \n\t"                           \
    "vle16.v            v14, (" B2 ")                           \n\t"                           \
    "addi               " B2 ", " B2 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v30, v3, v15                            \n\t"

#define MMT4D_FP16_COMP_M16N32_E2                                                               \
    MMT4D_FP16_COMP_M16N32_E2_IMPL(_B0, _B1, _B2, _B3)

/* Compute macro E3: Epilogue version of M3 (only B[v15] preload) */
#define MMT4D_FP16_COMP_M16N32_E3_IMPL(B3)                                                      \
    "vfwmadot           v16, v4, v8                             \n\t"                           \
    "vle16.v            v15, (" B3 ")                           \n\t"                           \
    "addi               " B3 ", " B3 ", 32*8*2                  \n\t"                           \
    "vfwmadot           v24, v5, v8                             \n\t"                           \
                                                                                                \
    "vfwmadot           v18, v4, v9                             \n\t"                           \
    "vfwmadot           v26, v5, v9                             \n\t"                           \
                                                                                                \
    "vfwmadot           v20, v4, v10                            \n\t"                           \
    "vfwmadot           v28, v5, v10                            \n\t"                           \
                                                                                                \
    "vfwmadot           v22, v4, v11                            \n\t"                           \
    "vfwmadot           v30, v5, v11                            \n\t"

#define MMT4D_FP16_COMP_M16N32_E3                                                               \
    MMT4D_FP16_COMP_M16N32_E3_IMPL(_B3)

/* Compute macro E4: Final epilogue (no preload at all) */
#define MMT4D_FP16_COMP_M16N32_E4                                                               \
    "vfwmadot           v16, v6, v12                            \n\t"                           \
    "vfwmadot           v24, v7, v12                            \n\t"                           \
                                                                                                \
    "vfwmadot           v18, v6, v13                            \n\t"                           \
    "vfwmadot           v26, v7, v13                            \n\t"                           \
                                                                                                \
    "vfwmadot           v20, v6, v14                            \n\t"                           \
    "vfwmadot           v28, v7, v14                            \n\t"                           \
                                                                                                \
    "vfwmadot           v22, v6, v15                            \n\t"                           \
    "vfwmadot           v30, v7, v15                            \n\t"

/* ============================================================================
 * Post-process Stage: Unpack, add bias, convert, and clip
 * ============================================================================
 */

/* Unpack: Reorganize accumulator registers from compute layout to output layout */
#define MMT4D_FP16_UNPACK_M16N32                                                                \
    /* First level pack: group adjacent accumulators */                                         \
    "vpack.vv           v0, v16, v18, 2                         \n\t"                           \
    "vpack.vv           v2, v20, v22, 2                         \n\t"                           \
                                                                                                \
    "vpack.vv           v4, v17, v19, 2                         \n\t"                           \
    "vpack.vv           v6, v21, v23, 2                         \n\t"                           \
                                                                                                \
    "vpack.vv           v8, v24, v26, 2                         \n\t"                           \
    "vpack.vv           v10, v28, v30, 2                        \n\t"                           \
                                                                                                \
    "vpack.vv           v12, v25, v27, 2                        \n\t"                           \
    "vpack.vv           v14, v29, v31, 2                        \n\t"                           \
                                                                                                \
    /* Second level pack: final layout */                                                       \
    "vpack.vv           v16, v0, v2, 3                          \n\t"                           \
    "vpack.vv           v18, v1, v3, 3                          \n\t"                           \
                                                                                                \
    "vpack.vv           v20, v4, v6, 3                          \n\t"                           \
    "vpack.vv           v22, v5, v7, 3                          \n\t"                           \
                                                                                                \
    "vpack.vv           v24, v8, v10, 3                         \n\t"                           \
    "vpack.vv           v26, v9, v11, 3                         \n\t"                           \
                                                                                                \
    "vpack.vv           v28, v12, v14, 3                        \n\t"                           \
    "vpack.vv           v30, v13, v15, 3                        \n\t"

/* Add bias: Add FP32 bias vector to all accumulator rows */
#define MMT4D_FP16_ADD_BIAS_M16N32_IMPL(BIAS)                                                   \
    /* v0 contains the bias vector loaded before calling this macro */                          \
    "vfadd.vv           v16, v16, v0                            \n\t"                           \
    "vfadd.vv           v17, v17, v0                            \n\t"                           \
    "vfadd.vv           v18, v18, v0                            \n\t"                           \
    "vfadd.vv           v19, v19, v0                            \n\t"                           \
    "vfadd.vv           v20, v20, v0                            \n\t"                           \
    "vfadd.vv           v21, v21, v0                            \n\t"                           \
    "vfadd.vv           v22, v22, v0                            \n\t"                           \
    "vfadd.vv           v23, v23, v0                            \n\t"                           \
    "vfadd.vv           v24, v24, v0                            \n\t"                           \
    "vfadd.vv           v25, v25, v0                            \n\t"                           \
    "vfadd.vv           v26, v26, v0                            \n\t"                           \
    "vfadd.vv           v27, v27, v0                            \n\t"                           \
    "vfadd.vv           v28, v28, v0                            \n\t"                           \
    "vfadd.vv           v29, v29, v0                            \n\t"                           \
    "vfadd.vv           v30, v30, v0                            \n\t"                           \
    "vfadd.vv           v31, v31, v0                            \n\t"

#define MMT4D_FP16_ADD_BIAS_M16N32                                                              \
    MMT4D_FP16_ADD_BIAS_M16N32_IMPL(_BIAS)

/* Convert: Narrow FP32 accumulators to FP16 output */
#define MMT4D_FP16_CONVERT_M16N32                                                               \
    "vfncvt.f.f.w       v16, v16                                \n\t"                           \
    "vfncvt.f.f.w       v17, v17                                \n\t"                           \
    "vfncvt.f.f.w       v18, v18                                \n\t"                           \
    "vfncvt.f.f.w       v19, v19                                \n\t"                           \
    "vfncvt.f.f.w       v20, v20                                \n\t"                           \
    "vfncvt.f.f.w       v21, v21                                \n\t"                           \
    "vfncvt.f.f.w       v22, v22                                \n\t"                           \
    "vfncvt.f.f.w       v23, v23                                \n\t"                           \
    "vfncvt.f.f.w       v24, v24                                \n\t"                           \
    "vfncvt.f.f.w       v25, v25                                \n\t"                           \
    "vfncvt.f.f.w       v26, v26                                \n\t"                           \
    "vfncvt.f.f.w       v27, v27                                \n\t"                           \
    "vfncvt.f.f.w       v28, v28                                \n\t"                           \
    "vfncvt.f.f.w       v29, v29                                \n\t"                           \
    "vfncvt.f.f.w       v30, v30                                \n\t"                           \
    "vfncvt.f.f.w       v31, v31                                \n\t"

/* MinMax clip: Clamp FP16 values to [MIN, MAX] range */
#define MMT4D_FP16_MINMAX_M16N32_IMPL(MIN, MAX)                                                 \
    /* First apply max clipping */                                                              \
    "vfmin.vf           v16, v16, " MAX "                       \n\t"                           \
    "vfmin.vf           v17, v17, " MAX "                       \n\t"                           \
    "vfmin.vf           v18, v18, " MAX "                       \n\t"                           \
    "vfmin.vf           v19, v19, " MAX "                       \n\t"                           \
    "vfmin.vf           v20, v20, " MAX "                       \n\t"                           \
    "vfmin.vf           v21, v21, " MAX "                       \n\t"                           \
    "vfmin.vf           v22, v22, " MAX "                       \n\t"                           \
    "vfmin.vf           v23, v23, " MAX "                       \n\t"                           \
    "vfmin.vf           v24, v24, " MAX "                       \n\t"                           \
    "vfmin.vf           v25, v25, " MAX "                       \n\t"                           \
    "vfmin.vf           v26, v26, " MAX "                       \n\t"                           \
    "vfmin.vf           v27, v27, " MAX "                       \n\t"                           \
    "vfmin.vf           v28, v28, " MAX "                       \n\t"                           \
    "vfmin.vf           v29, v29, " MAX "                       \n\t"                           \
    "vfmin.vf           v30, v30, " MAX "                       \n\t"                           \
    "vfmin.vf           v31, v31, " MAX "                       \n\t"                           \
                                                                                                \
    /* Then apply min clipping */                                                               \
    "vfmax.vf           v16, v16, " MIN "                       \n\t"                           \
    "vfmax.vf           v17, v17, " MIN "                       \n\t"                           \
    "vfmax.vf           v18, v18, " MIN "                       \n\t"                           \
    "vfmax.vf           v19, v19, " MIN "                       \n\t"                           \
    "vfmax.vf           v20, v20, " MIN "                       \n\t"                           \
    "vfmax.vf           v21, v21, " MIN "                       \n\t"                           \
    "vfmax.vf           v22, v22, " MIN "                       \n\t"                           \
    "vfmax.vf           v23, v23, " MIN "                       \n\t"                           \
    "vfmax.vf           v24, v24, " MIN "                       \n\t"                           \
    "vfmax.vf           v25, v25, " MIN "                       \n\t"                           \
    "vfmax.vf           v26, v26, " MIN "                       \n\t"                           \
    "vfmax.vf           v27, v27, " MIN "                       \n\t"                           \
    "vfmax.vf           v28, v28, " MIN "                       \n\t"                           \
    "vfmax.vf           v29, v29, " MIN "                       \n\t"                           \
    "vfmax.vf           v30, v30, " MIN "                       \n\t"                           \
    "vfmax.vf           v31, v31, " MIN "                       \n\t"

#define MMT4D_FP16_MINMAX_M16N32                                                                \
    MMT4D_FP16_MINMAX_M16N32_IMPL(_MIN, _MAX)

/* ============================================================================
 * Store Stage: Save results to memory
 * ============================================================================
 */

/* Store M16: Store all 16 rows for full M=16 case */
#define MMT4D_FP16_SAVING_M16_IMPL(C, CS)                                                       \
    "vse16.v            v16, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v17, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v18, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v19, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v20, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v21, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v22, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v23, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v24, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v25, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v26, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v27, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v28, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v29, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v30, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "vse16.v            v31, (" C ")                            \n\t"

#define MMT4D_FP16_SAVING_M16                                                                   \
    MMT4D_FP16_SAVING_M16_IMPL(_C, _CS)

/* Store LT16: Store with M < 16, checking bounds on each row */
#define MMT4D_FP16_SAVING_LT16_IMPL(C, CS, M)                                                   \
    "vse16.v            v16, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v17, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v18, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v19, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v20, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v21, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v22, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v23, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v24, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v25, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v26, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v27, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v28, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v29, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v30, (" C ")                            \n\t"                           \
    "add                " C ", " C ", " CS "                    \n\t"                           \
    "addi               " M ", " M ", -1                        \n\t"                           \
    "blez               " M ", END%=                            \n\t"                           \
                                                                                                \
    "vse16.v            v31, (" C ")                            \n\t"                           \
    "jal                x0, END%=                               \n\t"

#define MMT4D_FP16_SAVING_LT16                                                                  \
    MMT4D_FP16_SAVING_LT16_IMPL(_C, _CS, _M)

/* ============================================================================
 * Inplace Add: Load existing C values and add to accumulators
 * ============================================================================
 * Used for residual/skip connections
 */

/* Load C values and add to accumulators (FP16 input, FP32 accumulator) */
#define MMT4D_FP16_INPLACE_ADD_LOAD_LT16_IMPL(C, CS, N, M, TMP_VL, TMP_KC, A0)                  \
    "vsetvli            " TMP_VL ", " N ", e16, mf2             \n\t"                           \
    "addi               " A0 ", " C ", 0                        \n\t"                           \
    "addi               " TMP_KC ", " M ", 0                    \n\t"                           \
                                                                                                \
    "vle16.v            v0, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v1, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v2, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v3, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v4, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v5, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v6, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v7, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v8, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v9, (" A0 ")                            \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v10, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v11, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v12, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v13, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v14, (" A0 ")                           \n\t"                           \
    "add                " A0 ", " A0 ", " CS "                  \n\t"                           \
    "addi               " TMP_KC ", " TMP_KC ", -1              \n\t"                           \
    "blez               " TMP_KC ", LOAD_END%=                  \n\t"                           \
                                                                                                \
    "vle16.v            v15, (" A0 ")                           \n\t"                           \
    "LOAD_END%=:                                                \n\t"                           \
                                                                                                \
    /* Add loaded FP16 values to FP32 accumulators (widening add) */                            \
    "vfwadd.wv          v16, v16, v0                            \n\t"                           \
    "vfwadd.wv          v17, v17, v1                            \n\t"                           \
    "vfwadd.wv          v18, v18, v2                            \n\t"                           \
    "vfwadd.wv          v19, v19, v3                            \n\t"                           \
    "vfwadd.wv          v20, v20, v4                            \n\t"                           \
    "vfwadd.wv          v21, v21, v5                            \n\t"                           \
    "vfwadd.wv          v22, v22, v6                            \n\t"                           \
    "vfwadd.wv          v23, v23, v7                            \n\t"                           \
    "vfwadd.wv          v24, v24, v8                            \n\t"                           \
    "vfwadd.wv          v25, v25, v9                            \n\t"                           \
    "vfwadd.wv          v26, v26, v10                           \n\t"                           \
    "vfwadd.wv          v27, v27, v11                           \n\t"                           \
    "vfwadd.wv          v28, v28, v12                           \n\t"                           \
    "vfwadd.wv          v29, v29, v13                           \n\t"                           \
    "vfwadd.wv          v30, v30, v14                           \n\t"                           \
    "vfwadd.wv          v31, v31, v15                           \n\t"

#define MMT4D_FP16_INPLACE_ADD_LOAD_LT16                                                        \
    MMT4D_FP16_INPLACE_ADD_LOAD_LT16_IMPL(_C, _CS, _N, _M, _TMP_VL, _TMP_KC, _A0)

// clang-format on

// ---- src/ime2/asm/spe_pack_cast_inner_kernel.h (lines 10-538, namespace verbatim) ----
namespace mlir::speir::spekernel {
// The Q41 repack layout and its scalar types are defined in repack.h
// (ggml::cpu::riscv64_spacemit); pull them into the vendored kernel namespace.
using ggml::cpu::riscv64_spacemit::Q41Block;
using ggml::cpu::riscv64_spacemit::_spe_fp16_;
using ggml::cpu::riscv64_spacemit::uint4x2_t;
}  // namespace mlir::speir::spekernel

namespace mlir::speir::spekernel::ime2 {

// Q41Block<32,32,2> stores 32x32 U4 values and one scale per row.
// v0 selects the second K32 row in each FP16 register; v4..v7 carry the next block.
// v8..v23 hold the current FP16 rows, and v24..v31 receive each packed 16-row half.
// zero_point comes from the first input block of the entire kernel call.
SPINE_FORCE_INLINE void _spe_pack_cast_q41_f16_m32k32(const void * in, void * out, int32_t zero_point, int64_t blocks) {
    using block_type = Q41Block<32, 32, 2>;
    static_assert(sizeof(uint4x2_t) == 1 && sizeof(_spe_fp16_) == 2);
    if (blocks == 0) {
        return;
    }
    const auto * src = reinterpret_cast<const uint8_t *>(in);
    auto *       dst = reinterpret_cast<uint8_t *>(out);

    // A finite FP16 scale times an integer of magnitude <=255 is exact in FP32.
    // Direct FP16 multiplication retains that final rounding without FP32 intermediates.
    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        "vid.v          v1                                                       \n\t"
        "li             t0, 31                                                   \n\t"
        "vmsgtu.vx      v0, v1, t0                                               \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "LOOP_Q41_BLOCK%=:                                                       \n\t"
        "addi           t1, %[in], %[scale_offset]                               \n\t"
        "vsetvli        t0, zero, e16, m8, tu, mu                                \n\t"
        PACK_CAST_UNPACK_U4_F16_M32K32
        // v4..v7 are free: preload only if another complete K block exists.
        "addi           %[blocks], %[blocks], -1                                 \n\t"
        "beqz           %[blocks], Q41_SCALE_BLOCK%=                             \n\t"
        "addi           %[in], %[in], %[block_bytes]                             \n\t"
        "vl4re8.v       v4, (%[in])                                              \n\t"
        "Q41_SCALE_BLOCK%=:                                                      \n\t"
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        PACK_CAST_LOAD_SCALE_F16_M16_LO
        PACK_CAST_SCALE_F16_M16K32_LO
        PACK_CAST_PACK_F16_M16K32_LO
        "mv             t2, %[out]                                               \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32
        // The final sixteen rows and their own scales use the same vtype.
        PACK_CAST_LOAD_SCALE_F16_M16_HI
        PACK_CAST_SCALE_F16_M16K32_HI
        PACK_CAST_PACK_F16_M16K32_HI
        "addi           t2, %[out], 256                                          \n\t"
        "addi           t3, t2, 128                                              \n\t"
        PACK_CAST_STORE_F16_M16K32
        "addi           %[out], %[out], 1024                                     \n\t"
        "addi           %[out], %[out], 1024                                     \n\t"
        "bnez           %[blocks], LOOP_Q41_BLOCK%=                              \n\t"
        : [in] "+&r"(src), [out] "+&r"(dst), [blocks] "+&r"(blocks)
        : [zero_point] "r"(zero_point), [tile_stride] "i"(32 * 8 * sizeof(_spe_fp16_)),
          [block_bytes] "i"(sizeof(block_type)), [scale_offset] "i"(offsetof(block_type, scale))
        : "cc", "memory", "t0", "t1", "t2", "t3",
          "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
          "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7");
    // clang-format on
}

// With one contiguous K tile, the packed layout is already row-major. Load
// fp32 with EMUL=8 and narrow to LMUL=4 without changing vtype in the loop.
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_contiguous(const void * in, void * out, int64_t count) {
    // clang-format off
    __asm__ volatile(
        "LOOP_CONTIGUOUS%=:                                                      \n\t"
        "vsetvli        t0, %[count], e16, m4, tu, mu                            \n\t"
        "vle32.v        v0, (%[in])                                              \n\t"
        "sh2add         %[in], t0, %[in]                                         \n\t"
        "sub            %[count], %[count], t0                                   \n\t"
        "vfncvt.f.f.w   v8, v0                                                   \n\t"
        "vse16.v        v8, (%[out])                                             \n\t"
        "sh1add         %[out], t0, %[out]                                       \n\t"
        "bnez           %[count], LOOP_CONTIGUOUS%=                              \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [count] "+&r"(count)
        :
        : "cc", "memory", "t0");
    // clang-format on
}

// A single K block needs no register permutation. Contiguous eight-column
// rows can be converted together; strided or partial rows retain row spacing.
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k8(const void * in,
                                                     int64_t      in_stride,
                                                     void *       out,
                                                     int64_t      rows,
                                                     int64_t      cols) {
    if (rows < 16 || cols < 8) {
        // clang-format off
        __asm__ volatile(
            "vsetvli        t0, zero, e16, m2, tu, mu                            \n\t"
            "vxor.vv        v4, v4, v4                                           \n\t"
            "vse16.v        v4, (%[out])                                         \n\t"
            :
            : [out] "r"(out)
            : "cc", "memory", "t0");
        // clang-format on
    }
    if (cols == 8 && in_stride == 8 * sizeof(float)) {
        const int64_t count = rows * 8;
        // clang-format off
        __asm__ volatile(
            "vsetvli        t0, %[count], e16, m2, tu, mu                        \n\t"
            "vle32.v        v0, (%[in])                                          \n\t"
            "vfncvt.f.f.w   v4, v0                                               \n\t"
            "vse16.v        v4, (%[out])                                         \n\t"
            :
            : [in] "r"(in), [out] "r"(out), [count] "r"(count)
            : "cc", "memory", "t0");
        // clang-format on
        return;
    }
    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, %[cols], e16, m1, tu, mu                             \n\t"
        "LOOP_ROW%=:                                                             \n\t"
        "vle32.v        v0, (%[in])                                              \n\t"
        "add            %[in], %[in], %[stride]                                  \n\t"
        "addi           %[rows], %[rows], -1                                     \n\t"
        "vfncvt.f.f.w   v2, v0                                                   \n\t"
        "vse16.v        v2, (%[out])                                             \n\t"
        "addi           %[out], %[out], 16                                       \n\t"
        "bnez           %[rows], LOOP_ROW%=                                      \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [rows] "+&r"(rows)
        : [stride] "r"(in_stride), [cols] "r"(cols)
        : "cc", "memory", "t0");
    // clang-format on
}

// VLEN1024 only. vle32 uses EMUL=2 under e16,m1, so conversion and the
// original 16x64 pack network share one vtype. Stage eight rows at a time:
// f32 in v16..31 -> f16 in v0..7, then f32 in v16..31 -> f16 in v8..15.
// dst_mb sets the destination K-tile spacing; only 16 rows are written.
template <size_t dst_mb = 16>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k64(const void * in, int64_t in_stride, void * out, int64_t blocks) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    constexpr int64_t kTileCols        = 8;
    constexpr int64_t kPanelCols       = 64;
    constexpr int64_t kHalfTileRows    = 8;
    constexpr int64_t kInputPanelBytes = kPanelCols * sizeof(float);
    constexpr int64_t kHalfTileBytes   = kHalfTileRows * kTileCols * sizeof(_spe_fp16_);
    constexpr int64_t kTileStride      = dst_mb * kTileCols * sizeof(_spe_fp16_);
    // clang-format off
    __asm__ volatile(
        "vsetvli        t1, zero, e16, m1, tu, mu                                \n\t"
        "LOOP_K%=:                                                               \n\t"
        "mv             t0, %[in]                                                \n\t"
        "add            t1, t0, %[stride]                                        \n\t"
        "sh1add         t5, %[stride], t0                                        \n\t"
        "add            t6, t5, %[stride]                                        \n\t"
        PACK_CAST_LOAD_F32_M8K64
        "sh2add         t0, %[stride], t0                                        \n\t"
        "sh2add         t1, %[stride], t1                                        \n\t"
        "sh2add         t5, %[stride], t5                                        \n\t"
        "sh2add         t6, %[stride], t6                                        \n\t"
        PACK_CAST_NARROW_F32_F16_M8_LO
        PACK_CAST_LOAD_F32_M8K64
        PACK_CAST_NARROW_F32_F16_M8_HI
        PACK_X16_M16N8
        "mv             t0, %[out]                                               \n\t"
        "addi           t1, t0, %[half_tile_bytes]                               \n\t"
        PACK_CAST_STORE_F16_M16K64
        "addi           %[in], %[in], %[input_panel_bytes]                       \n\t"
        "addi           %[out], t0, %[tile_stride]                               \n\t"
        "addi           %[blocks], %[blocks], -1                                 \n\t"
        "bnez           %[blocks], LOOP_K%=                                      \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [blocks] "+&r"(blocks)
        : [stride] "r"(in_stride), [tile_stride] "i"(kTileStride), [input_panel_bytes] "i"(kInputPanelBytes),
          [half_tile_bytes] "i"(kHalfTileBytes)
        : "cc", "memory", "t0", "t1", "t5", "t6");
    // clang-format on
}

// VLEN1024, e16/m1: one batch is 8x64, stored as eight 8x8 fp16 tiles.
// dst_mb=16/32 sets tile spacing; Rows=64/128 groups complete M blocks.
// Traverse col += 64 outside row += 8 over Rows rows. Tile j writes
// out[row / dst_mb, col / 8 + j, row % dst_mb : row % dst_mb + 8, :].
// Current fp16 and pack scratch use v0..15; next fp32 uses v16..31 (EMUL=2).
// Prologue narrows the first batch. Steady state loads next, packs/stores
// current, then narrows next; the drain skips its next load. No vtype change.
// t0/t1/t5/t6 retain four row addresses across packing and narrowing.
// The caller provides complete rows/columns and a tightly packed output.
template <size_t dst_mb, size_t Rows>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m8k64_rows(const void * in,
                                                          int64_t      in_stride,
                                                          void *       out,
                                                          int64_t      k_blocks) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    static_assert(Rows == 64 || Rows == 128);
    if (k_blocks == 0) {
        return;
    }
    constexpr int64_t kBatchRows                 = 8;
    constexpr int64_t kTileCols                  = 8;
    constexpr int64_t kPanelCols                 = 64;
    constexpr int64_t kOutputElementBytes        = sizeof(_spe_fp16_);
    constexpr int64_t kInputPanelBytes           = kPanelCols * sizeof(float);
    constexpr int64_t kOutputBatchBytes          = kBatchRows * kTileCols * kOutputElementBytes;
    constexpr int64_t kOutputPanelRemainderBytes = (kPanelCols - kTileCols) * kOutputElementBytes;
    constexpr int64_t kTileStride                = dst_mb * kTileCols * kOutputElementBytes;
    int64_t           remaining                  = k_blocks * (Rows / kBatchRows);
    const int64_t     k                          = k_blocks * kPanelCols;
    // After the last batch, each pointer advances to row Rows + i (i=0..3).
    // Adding in_col_fix rewinds it to row i in the next 64-column panel.
    const int64_t     in_col_fix                 = kInputPanelBytes - int64_t(Rows) * in_stride;
    const int64_t     out_mb_advance             = int64_t(dst_mb) * (k - kTileCols) * kOutputElementBytes;
    const int64_t     out_col_fix =
        int64_t(dst_mb) * kOutputPanelRemainderBytes - int64_t(Rows - dst_mb) * k * kOutputElementBytes;
    // clang-format off
    __asm__ volatile(
        "vsetvli        t1, zero, e16, m1, tu, mu                                \n\t"
        // t3: remaining 8-row batches in the current K64 panel.
        // t4: remaining 8-row batches in the current M block.
        "li             t3, %[group_segments]                                    \n\t"
        "li             t4, %[mb_segments]                                       \n\t"
        "mv             t0, %[in]                                                \n\t"
        "add            t1, t0, %[stride]                                        \n\t"
        "sh1add         t5, %[stride], t0                                        \n\t"
        "add            t6, t5, %[stride]                                        \n\t"
        // Prologue: load and narrow the first batch.
        PACK_CAST_LOAD_F32_M8K64
        PACK_CAST_NARROW_F32_F16_M8_LO
        "LOOP_GROUPED%=:                                                         \n\t"
        "mv             t2, %[out]                                               \n\t"
        "addi           %[remaining], %[remaining], -1                           \n\t"
        "beqz           %[remaining], PACK_GROUPED%=                             \n\t"
        "sh2add         t0, %[stride], t0                                        \n\t"
        "sh2add         t1, %[stride], t1                                        \n\t"
        "sh2add         t5, %[stride], t5                                        \n\t"
        "sh2add         t6, %[stride], t6                                        \n\t"
        "addi           %[out], %[out], %[batch_out_bytes]                       \n\t"
        "addi           t3, t3, -1                                               \n\t"
        "addi           t4, t4, -1                                               \n\t"
        "beqz           t3, COLUMN_GROUPED%=                                     \n\t"
        "bnez           t4, LOAD_GROUPED%=                                       \n\t"
        "add            %[out], %[out], %[out_mb_advance]                        \n\t"
        "li             t4, %[mb_segments]                                       \n\t"
        "j              LOAD_GROUPED%=                                           \n\t"
        "COLUMN_GROUPED%=:                                                       \n\t"
        "add            t0, t0, %[in_col_fix]                                    \n\t"
        "add            t1, t1, %[in_col_fix]                                    \n\t"
        "add            t5, t5, %[in_col_fix]                                    \n\t"
        "add            t6, t6, %[in_col_fix]                                    \n\t"
        "add            %[out], %[out], %[out_col_fix]                           \n\t"
        "li             t3, %[group_segments]                                    \n\t"
        "li             t4, %[mb_segments]                                       \n\t"
        // Load the next batch before packing/storing the current one.
        "LOAD_GROUPED%=:                                                         \n\t"
        PACK_CAST_LOAD_F32_M8K64
        "PACK_GROUPED%=:                                                         \n\t"
        PACK_CAST_PACK_F16_M8K64
        PACK_CAST_STORE_F16_M8K64
        // Drain: the last iteration has no next batch to convert.
        "beqz           %[remaining], END_GROUPED%=                              \n\t"
        PACK_CAST_NARROW_F32_F16_M8_LO
        "j              LOOP_GROUPED%=                                           \n\t"
        "END_GROUPED%=:                                                          \n\t"
        : [out] "+&r"(out), [remaining] "+&r"(remaining)
        : [in] "r"(in), [stride] "r"(in_stride), [group_segments] "i"(Rows / kBatchRows),
          [mb_segments] "i"(dst_mb / kBatchRows), [in_col_fix] "r"(in_col_fix), [out_mb_advance] "r"(out_mb_advance),
          [out_col_fix] "r"(out_col_fix), [tile_stride] "i"(kTileStride), [batch_out_bytes] "i"(kOutputBatchBytes)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6");
    // clang-format on
}

// rows in [1,16], cols in [1,64], tiles=ceil(cols/8). Missing rows and
// inactive columns stay +0 until the full-register pack network consumes them.
template <size_t dst_mb = 16>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_m16k64_tail(const void * in,
                                                           int64_t      in_stride,
                                                           void *       out,
                                                           int64_t      rows,
                                                           int64_t      cols,
                                                           int64_t      tiles) {
    static_assert(dst_mb == 16 || dst_mb == 32);
    constexpr int64_t kTileCols      = 8;
    constexpr int64_t kHalfTileRows  = 8;
    constexpr int64_t kHalfTileBytes = kHalfTileRows * kTileCols * sizeof(_spe_fp16_);
    constexpr int64_t kTileStride    = dst_mb * kTileCols * sizeof(_spe_fp16_);
    // clang-format off
    __asm__ volatile(
        "vsetvli        t0, zero, e16, m8, tu, mu                                \n\t"
        "vxor.vv        v0, v0, v0                                               \n\t"
        "vxor.vv        v8, v8, v8                                               \n\t"
        "vsetvli        t0, %[cols], e16, m1, tu, mu                             \n\t"
        "mv             t0, %[in]                                                \n\t"
        "mv             t2, %[rows]                                              \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v0, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v1, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v2, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v3, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v4, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v5, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v6, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v7, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v8, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v9, v16                                                  \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v10, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v11, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v12, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v13, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v14, v16                                                 \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, PACK%=                                               \n\t"
        "add            t0, t0, %[stride]                                        \n\t"
        "vle32.v        v16, (t0)                                                \n\t"
        "vfncvt.f.f.w   v15, v16                                                 \n\t"
        "PACK%=:                                                                 \n\t"
        "vsetvli        t0, zero, e16, m1, tu, mu                                \n\t"
        PACK_X16_M16N8
        "mv             t0, %[out]                                               \n\t"
        "addi           t1, t0, %[half_tile_bytes]                               \n\t"
        "mv             t2, %[tiles]                                             \n\t"
        "vse16.v        v16, (t0)                                                \n\t"
        "vse16.v        v24, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v17, (t0)                                                \n\t"
        "vse16.v        v25, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v18, (t0)                                                \n\t"
        "vse16.v        v26, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v19, (t0)                                                \n\t"
        "vse16.v        v27, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v20, (t0)                                                \n\t"
        "vse16.v        v28, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v21, (t0)                                                \n\t"
        "vse16.v        v29, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v22, (t0)                                                \n\t"
        "vse16.v        v30, (t1)                                                \n\t"
        "addi           t2, t2, -1                                               \n\t"
        "beqz           t2, END%=                                                \n\t"
        "addi           t0, t0, %[tile_stride]                                   \n\t"
        "addi           t1, t1, %[tile_stride]                                   \n\t"
        "vse16.v        v23, (t0)                                                \n\t"
        "vse16.v        v31, (t1)                                                \n\t"
        "END%=:                                                                  \n\t"
        :
        : [in] "r"(in), [stride] "r"(in_stride), [out] "r"(out), [rows] "r"(rows), [cols] "r"(cols), [tiles] "r"(tiles),
          [tile_stride] "i"(kTileStride), [half_tile_bytes] "i"(kHalfTileBytes)
        : "cc", "memory", "t0", "t1", "t2");
    // clang-format on
}

// Process complete M blocks. K may be cropped or padded; every output tile
// contains mb rows and eight columns. The caller handles incomplete M blocks.
template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_full_blocks(const void * in,
                                                           int64_t      in_stride,
                                                           void *       out,
                                                           int64_t      out_stride,
                                                           int64_t      mc,
                                                           int64_t      k,
                                                           int64_t      dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb         = 8;
    constexpr int64_t panel_k    = 64;
    constexpr int64_t inner_rows = 16;
    const int64_t     rows       = mc * mb;
    if (k == kb && dst_kc == 1 && in_stride == kb * sizeof(float) && out_stride == mb * kb * sizeof(_spe_fp16_)) {
        _spe_pack_cast_f32_f16_contiguous(in, out, rows * kb);
        return;
    }

    auto *  src      = reinterpret_cast<const int8_t *>(in);
    auto *  dst      = reinterpret_cast<int8_t *>(out);
    int64_t first_mo = 0;
    // Tightly packed full panels keep the existing 64/128-row pipeline.
    if (k % panel_k == 0 && dst_kc == k / kb && in_stride == k * sizeof(float) &&
        out_stride == k * mb * sizeof(_spe_fp16_)) {
        int64_t row = 0;
        for (; row + 128 <= rows; row += 128) {
            _spe_pack_cast_f32_f16_m8k64_rows<mb, 128>(src + row * in_stride, in_stride, dst + (row / mb) * out_stride,
                                                       k / panel_k);
        }
        if (row + 64 <= rows) {
            _spe_pack_cast_f32_f16_m8k64_rows<mb, 64>(src + row * in_stride, in_stride, dst + (row / mb) * out_stride,
                                                      k / panel_k);
            row += 64;
        }
        first_mo = row / mb;
    }

    const int64_t panels      = k / panel_k;
    const int64_t col         = panels * panel_k;
    const int64_t cols        = k - col;
    const int64_t valid_tiles = (k + kb - 1) / kb;
    for (int64_t mo = first_mo; mo < mc; ++mo) {
        auto * src_block = src + mo * mb * in_stride;
        auto * dst_block = reinterpret_cast<_spe_fp16_ *>(dst + mo * out_stride);
        // MB32 uses two 16-row halves with the full MB32 K-tile stride.
        for (int64_t mi = 0; mi < mb; mi += inner_rows) {
            auto * src_half = src_block + mi * in_stride;
            auto * dst_half = dst_block + mi * kb;
            if (panels != 0) {
                _spe_pack_cast_f32_f16_m16k64<mb>(src_half, in_stride, dst_half, panels);
            }
            if (cols != 0) {
                if (cols <= kb) {
                    _spe_pack_cast_f32_f16_m16k8(src_half + col * sizeof(float), in_stride, dst_half + col * mb,
                                                 inner_rows, cols);
                } else {
                    _spe_pack_cast_f32_f16_m16k64_tail<mb>(src_half + col * sizeof(float), in_stride,
                                                           dst_half + col * mb, inner_rows, cols, (cols + kb - 1) / kb);
                }
            }
        }
        std::fill_n(dst_block + valid_tiles * mb * kb, (dst_kc - valid_tiles) * mb * kb, static_cast<_spe_fp16_>(0.0f));
    }
}

// One valid row of the final M block. Consecutive K tiles are mb*8 elements
// apart in the output, so a whole input row cannot be stored contiguously.
template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f32_f16_row(const void * in, void * out, int64_t k, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            src = reinterpret_cast<const float *>(in);
    auto *            dst = reinterpret_cast<_spe_fp16_ *>(out);
    for (int64_t ko = 0; ko < dst_kc; ++ko) {
        const int64_t cols     = std::max<int64_t>(0, std::min<int64_t>(kb, k - ko * kb));
        auto *        dst_tile = dst + ko * mb * kb;
        if (cols != 0) {
            _spe_pack_cast_f32_f16_contiguous(src + ko * kb, dst_tile, cols);
        }
        std::fill_n(dst_tile + cols, kb - cols, static_cast<_spe_fp16_>(0.0f));
    }
}

template <size_t mb> SPINE_FORCE_INLINE void _spe_pack_cast_f16_pad_row(void * out, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            dst = reinterpret_cast<_spe_fp16_ *>(out);
    for (int64_t ko = 0; ko < dst_kc; ++ko) {
        std::fill_n(dst + ko * mb * kb, kb, static_cast<_spe_fp16_>(0.0f));
    }
}

// Zero only logical output elements; preserve the gap between M blocks.
template <size_t mb>
SPINE_FORCE_INLINE void _spe_pack_cast_f16_pad_blocks(void * out, int64_t out_stride, int64_t mc, int64_t dst_kc) {
    static_assert(mb == 16 || mb == 32);
    constexpr int64_t kb  = 8;
    auto *            dst = reinterpret_cast<int8_t *>(out);
    for (int64_t mo = 0; mo < mc; ++mo) {
        std::fill_n(reinterpret_cast<_spe_fp16_ *>(dst + mo * out_stride), dst_kc * mb * kb,
                    static_cast<_spe_fp16_>(0.0f));
    }
}

}  // namespace mlir::speir::spekernel::ime2

// ---- src/ime2/spe_mmt4d_inner_kernel.h (lines 10-297, namespace verbatim) ----
namespace mlir::speir::spekernel::ime2 {

/**
 * @brief FP16 GEMM kernel with FP32 bias and minmax clipping
 *
 * Computes: C = clip(A @ B + bias, min, max)
 * Where bias is FP32 format
 *
 * @param m     Number of output rows (up to 16)
 * @param n     Number of output columns (up to 32)
 * @param kc    Number of K iterations
 * @param a     Pointer to A matrix (packed format)
 * @param b     Pointer to B matrix (packed format)
 * @param bias  Pointer to FP32 bias vector (can be nullptr)
 * @param c     Pointer to output C matrix
 * @param c_stride  Row stride of C matrix in bytes
 * @param min_value Minimum clipping value (FP16)
 * @param max_value Maximum clipping value (FP16)
 */
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_minmax(size_t       m,
                                                              size_t       n,
                                                              size_t       kc,
                                                              const void * a,
                                                              const void * b,
                                                              const void * bias,
                                                              int8_t *     c,
                                                              size_t       c_stride,
                                                              _spe_fp16_   min_value,
                                                              _spe_fp16_   max_value) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [min] "f"(min_value),
          [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

/**
 * @brief FP16 GEMM kernel with FP32 bias, scalar FP32 PReLU, and minmax clipping
 *
 * Computes: C = clip(prelu(A @ B + bias, alpha), min, max)
 * Where bias and alpha are FP32 format
 *
 * @param m     Number of output rows (up to 16)
 * @param n     Number of output columns (up to 32)
 * @param kc    Number of K iterations
 * @param a     Pointer to A matrix (packed format)
 * @param b     Pointer to B matrix (packed format)
 * @param bias  Pointer to FP32 bias vector (can be nullptr)
 * @param c     Pointer to output C matrix
 * @param c_stride  Row stride of C matrix in bytes
 * @param alpha Scalar PReLU alpha
 * @param min_value Minimum clipping value (FP16)
 * @param max_value Maximum clipping value (FP16)
 */
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_prelu_minmax(size_t       m,
                                                                    size_t       n,
                                                                    size_t       kc,
                                                                    const void * a,
                                                                    const void * b,
                                                                    const void * bias,
                                                                    int8_t *     c,
                                                                    size_t       c_stride,
                                                                    float        alpha,
                                                                    _spe_fp16_   min_value,
                                                                    _spe_fp16_   max_value) {
    const float zero_value = 0.0f;
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_PRELU_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [alpha] "f"(alpha),
          [zero] "f"(zero_value), [min] "f"(min_value), [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

/**
 * @brief FP16 GEMM kernel with FP32 bias, scalar FP32 LeakyRelu, and minmax clipping
 *
 * Computes: C = clip(leakyrelu(A @ B + bias, alpha), min, max)
 * Where bias and alpha are FP32 format
 */
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_leakyrelu_minmax(size_t       m,
                                                                        size_t       n,
                                                                        size_t       kc,
                                                                        const void * a,
                                                                        const void * b,
                                                                        const void * bias,
                                                                        int8_t *     c,
                                                                        size_t       c_stride,
                                                                        float        alpha,
                                                                        _spe_fp16_   min_value,
                                                                        _spe_fp16_   max_value) {
    const float zero_value = 0.0f;
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_LEAKYRELU_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [alpha] "f"(alpha),
          [zero] "f"(zero_value), [min] "f"(min_value), [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

/**
 * @brief FP16 GEMM kernel with no epilogue (NONE)
 *
 * Computes: C = A @ B
 * No bias addition, no clipping
 *
 * @param m     Number of output rows (up to 16)
 * @param n     Number of output columns (up to 32)
 * @param kc    Number of K iterations
 * @param a     Pointer to A matrix (packed format)
 * @param b     Pointer to B matrix (packed format)
 * @param c     Pointer to output C matrix
 * @param c_stride  Row stride of C matrix in bytes
 */
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_none(size_t       m,
                                                       size_t       n,
                                                       size_t       kc,
                                                       const void * a,
                                                       const void * b,
                                                       int8_t *     c,
                                                       size_t       c_stride) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_NONE

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [n] "r"(n), [kc] "r"(kc), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

/**
 * @brief FP16 GEMM kernel with FP16 bias (half precision) and minmax clipping
 *
 * Computes: C = clip(A @ B + bias, min, max)
 * Where bias is FP16 format (half precision)
 *
 * @param m     Number of output rows (up to 16)
 * @param n     Number of output columns (up to 32)
 * @param kc    Number of K iterations
 * @param a     Pointer to A matrix (packed format)
 * @param b     Pointer to B matrix (packed format)
 * @param bias  Pointer to FP16 bias vector (can be nullptr)
 * @param c     Pointer to output C matrix
 * @param c_stride  Row stride of C matrix in bytes
 * @param min_value Minimum clipping value (FP16)
 * @param max_value Maximum clipping value (FP16)
 */
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_hp_bias_minmax(size_t       m,
                                                                 size_t       n,
                                                                 size_t       kc,
                                                                 const void * a,
                                                                 const void * b,
                                                                 const void * bias,
                                                                 int8_t *     c,
                                                                 size_t       c_stride,
                                                                 _spe_fp16_   min_value,
                                                                 _spe_fp16_   max_value) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_HP_BIAS_MINMAX

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [min] "f"(min_value),
          [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

/**
 * @brief FP16 GEMM kernel with FP32 bias, minmax clipping, and inplace add
 *
 * Computes: C = clip(A @ B + bias + C_old, min, max)
 * Where C_old is the existing value at output location (residual connection)
 *
 * @param m     Number of output rows (up to 16)
 * @param n     Number of output columns (up to 32)
 * @param kc    Number of K iterations
 * @param a     Pointer to A matrix (packed format)
 * @param b     Pointer to B matrix (packed format)
 * @param bias  Pointer to FP32 bias vector (can be nullptr)
 * @param c     Pointer to output C matrix (also contains input for inplace add)
 * @param c_stride  Row stride of C matrix in bytes
 * @param min_value Minimum clipping value (FP16)
 * @param max_value Maximum clipping value (FP16)
 */
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_bias_minmax_inplace_add(size_t       m,
                                                                          size_t       n,
                                                                          size_t       kc,
                                                                          const void * a,
                                                                          const void * b,
                                                                          const void * bias,
                                                                          int8_t *     c,
                                                                          size_t       c_stride,
                                                                          _spe_fp16_   min_value,
                                                                          _spe_fp16_   max_value) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_BIAS_MINMAX_INPLACE

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [bias] "r"(bias), [n] "r"(n), [kc] "r"(kc), [min] "f"(min_value),
          [max] "f"(max_value), [cs] "r"(c_stride)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

/**
 * @brief FP16 GEMM kernel with no epilogue (NONE)
 *
 * Computes: C = A @ B * f_scalar
 * No bias addition, no clipping
 *
 * @param m     Number of output rows (up to 16)
 * @param n     Number of output columns (up to 32)
 * @param kc    Number of K iterations
 * @param a     Pointer to A matrix (packed format)
 * @param b     Pointer to B matrix (packed format)
 * @param c     Pointer to output C matrix
 * @param c_stride  Row stride of C matrix in bytes
 * @param f_scalar  f_scalar
 */
SPINE_FORCE_INLINE void _spe_hpgemm_impl_m16n32k8_mulscalar(size_t       m,
                                                            size_t       n,
                                                            size_t       kc,
                                                            const void * a,
                                                            const void * b,
                                                            int8_t *     c,
                                                            size_t       c_stride,
                                                            _spe_fp16_   f_scalar) {
    __asm__ volatile(
        // clang-format off
        MMT4D_FP16_M16N32K8_INIT

        MMT4D_FP16_M16N32K8_COMPUTE_LOOP

        MMT4D_FP16_M16N32K8_POSTPROCESS_MULSCALAR

        MMT4D_FP16_M16N32K8_STORE
        // clang-format on
        : [c] "+r"(c), [m] "+r"(m)
        : [a] "r"(a), [b] "r"(b), [n] "r"(n), [kc] "r"(kc), [cs] "r"(c_stride), [fscalar] "f"(f_scalar)
        : "memory", "cc", "t0", "t2", "a1", "a2", "s1", "s2", "s3", "s4");
}

}  // namespace mlir::speir::spekernel::ime2


// ---- src/rvv/asm/spe_reducesum_mxn_inner_kernel.h (lines 6-113, namespace verbatim) ----
namespace mlir::speir::spekernel::rvv {

// Reduce one [nth, src_nb] row across nth, preserving n_size adjacent columns.
SPINE_FORCE_INLINE void _spe_reducesum_mxn_f16_f32_16x32_1x1(const void * in, void * out, int64_t n_size, int64_t nth) {
    // e16,mf2 and e32,m1 both hold 32 elements with VLEN=1024.
    constexpr int64_t src_nb = 32;
    __asm__ volatile(
        "L_N%=:                                               \n\t"
        "vsetvli                t0, %[nsize], e32, m1, tu, mu \n\t"
        "vxor.vv                v8, v8, v8                    \n\t"
        "vsetvli                zero, t0, e16, mf2, tu, mu    \n\t"
        "mv                     t1, %[nth]                    \n\t"
        "mv                     a1, %[in]                     \n\t"

        "L_THREADS%=:                                         \n\t"
        "vle16.v                v0, (a1)                      \n\t"
        "vfwadd.wv              v8, v8, v0                    \n\t"
        // Each split is 32 fp16 elements apart, independently of the current vl.
        "sh1add                 a1, %[snb], a1                \n\t"
        "addi                   t1, t1, -1                    \n\t"
        "bnez                   t1, L_THREADS%=               \n\t"

        // The widened accumulator occupies v8 (e32, m1).
        "vsetvli                zero, t0, e32, m1, tu, mu     \n\t"
        "vse32.v                v8, (%[out])                  \n\t"
        "sh1add                 %[in], t0, %[in]              \n\t"
        "sh2add                 %[out], t0, %[out]            \n\t"
        "sub                    %[nsize], %[nsize], t0        \n\t"
        "bnez                   %[nsize], L_N%=               \n\t"
        : [in] "+&r"(in), [out] "+&r"(out), [nsize] "+&r"(n_size)
        : [nth] "r"(nth), [snb] "r"(src_nb)
        : "cc", "memory", "t0", "t1", "a1");
}

// Requires VLEN=1024 and a complete [16, 8, 32] tile; out_stride is in bytes.
// Each input register holds two splits at VL=64. Conversion separates them into
// two FP32 registers; the two row sums then consume these registers at VL=32.
SPINE_FORCE_INLINE void _spe_reducesum_mxn_f16_f32_16x32_1x1_a100_nth8(const void * in,
                                                                       void *       out,
                                                                       int64_t      out_stride) {
    constexpr int64_t src_nb       = 32;
    constexpr int64_t load_nb      = 2 * src_nb;
    constexpr int64_t row_elements = 8 * src_nb;
    __asm__ volatile(
        "li                     t1, 8                         \n\t"
        "L_PAIR%=:                                            \n\t"
        "mv                     a1, %[in]                     \n\t"
        "sh1add                 a2, %[row_elements], %[in]    \n\t"
        "vsetvli                zero, %[lnb], e16, m1, tu, mu \n\t"
        "vle16.v                v0, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v4, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v1, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v5, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v2, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v6, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        "vle16.v                v3, (a1)                      \n\t"
        "sh2add                 a1, %[snb], a1                \n\t"
        "vle16.v                v7, (a2)                      \n\t"
        "sh2add                 a2, %[snb], a2                \n\t"
        // All FP16 inputs are live until their widening conversions finish.
        "vfwcvt.f.f.v           v16, v0                       \n\t"
        "vfwcvt.f.f.v           v24, v4                       \n\t"
        "vfwcvt.f.f.v           v18, v1                       \n\t"
        "vfwcvt.f.f.v           v26, v5                       \n\t"
        "vfwcvt.f.f.v           v20, v2                       \n\t"
        "vfwcvt.f.f.v           v28, v6                       \n\t"
        "vfwcvt.f.f.v           v22, v3                       \n\t"
        "vfwcvt.f.f.v           v30, v7                       \n\t"
        // Restore VL=32 before clearing dead input registers and adding splits.
        "vsetvli                zero, %[snb], e32, m1, tu, mu \n\t"
        "vxor.vv                v0, v0, v0                    \n\t"
        "vxor.vv                v1, v1, v1                    \n\t"
        // Preserve the original t=0..7 FP32 addition order independently per row.
        "vfadd.vv               v0, v0, v16                   \n\t"
        "vfadd.vv               v1, v1, v24                   \n\t"
        "vfadd.vv               v0, v0, v17                   \n\t"
        "vfadd.vv               v1, v1, v25                   \n\t"
        "vfadd.vv               v0, v0, v18                   \n\t"
        "vfadd.vv               v1, v1, v26                   \n\t"
        "vfadd.vv               v0, v0, v19                   \n\t"
        "vfadd.vv               v1, v1, v27                   \n\t"
        "vfadd.vv               v0, v0, v20                   \n\t"
        "vfadd.vv               v1, v1, v28                   \n\t"
        "vfadd.vv               v0, v0, v21                   \n\t"
        "vfadd.vv               v1, v1, v29                   \n\t"
        "vfadd.vv               v0, v0, v22                   \n\t"
        "vfadd.vv               v1, v1, v30                   \n\t"
        "vfadd.vv               v0, v0, v23                   \n\t"
        "vfadd.vv               v1, v1, v31                   \n\t"
        "vse32.v                v0, (%[out])                  \n\t"
        "add                    a1, %[out], %[os]             \n\t"
        "vse32.v                v1, (a1)                      \n\t"
        "sh1add                 %[out], %[os], %[out]         \n\t"
        "mv                     %[in], a2                     \n\t"
        "addi                   t1, t1, -1                    \n\t"
        "bnez                   t1, L_PAIR%=                  \n\t"
        : [in] "+&r"(in), [out] "+&r"(out)
        : [snb] "r"(src_nb), [lnb] "r"(load_nb), [row_elements] "r"(row_elements), [os] "r"(out_stride)
        : "cc", "memory", "t1", "a1", "a2");
}

}  // namespace mlir::speir::spekernel::rvv

namespace mlir::speir::spekernel {

// ---- inc/spe_kernel_func_types.h: the function typedefs (verbatim) ----
using spe_pack_cast_function_def = std::function<void(void *  in,
                                                      int64_t in_offset,
                                                      int64_t in_stride,
                                                      void *  out,
                                                      int64_t out_offset,
                                                      int64_t out_stride,
                                                      int64_t mc,
                                                      int64_t kc,
                                                      int64_t dst_mc,
                                                      int64_t dst_kc)>;

using spe_reducesum_mxn_function_def = std::function<void(void *  in,
                                                          int64_t lhs_offset,
                                                          int64_t lhs_stride0,
                                                          int64_t lhs_stride1,
                                                          void *  out,
                                                          int64_t out_offset,
                                                          int64_t out_stride,
                                                          int64_t src_mc,
                                                          int64_t src_nc,
                                                          int64_t nth,
                                                          int64_t dst_mc,
                                                          int64_t dst_nc)>;

// ---- inc/spe_kernel_func_types.h: primary template declarations (macros expanded) ----
template <typename in_type, typename out_type, size_t src_mb, size_t src_kb, size_t dst_mb, size_t dst_kb>
void spe_pack_cast(void * in, int64_t in_offset, int64_t in_stride, void * out, int64_t out_offset,
                   int64_t out_stride, int64_t mc, int64_t kc, int64_t dst_mc, int64_t dst_kc);

template <typename in_type, typename out_type, size_t src_mb, size_t src_nb, size_t dst_mb, size_t dst_nb>
void spe_reducesum_mxn(void * in, int64_t lhs_offset, int64_t lhs_stride0, int64_t lhs_stride1, void * out,
                       int64_t out_offset, int64_t out_stride, int64_t src_mc, int64_t src_nc, int64_t nth,
                       int64_t dst_mc, int64_t dst_nc);

}  // namespace mlir::speir::spekernel

// ---- inc/spe_kernel_cxx_helper.h: dispatch getter declarations (macros
// expanded; spe_kernel_dispatch.cpp defines them). ----
namespace mlir::speir::spekernel::ime2 {
spe_pack_cast_function_def get_spe_pack_cast_function(const uint64_t         march,
                                                      const std::type_info & in_type,
                                                      const std::type_info & out_type,
                                                      const size_t           src_mb,
                                                      const size_t           src_kb,
                                                      const size_t           dst_mb,
                                                      const size_t           dst_kb);
}  // namespace mlir::speir::spekernel::ime2

namespace mlir::speir::spekernel::rvv {
spe_reducesum_mxn_function_def get_spe_reducesum_mxn_function(const uint64_t         march,
                                                              const std::type_info & in_type,
                                                              const std::type_info & out_type,
                                                              const size_t           src_mb,
                                                              const size_t           src_nb,
                                                              const size_t           dst_mb,
                                                              const size_t           dst_nb);
}  // namespace mlir::speir::spekernel::rvv

#endif  // SPE_KERNELS_H

