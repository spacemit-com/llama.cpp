// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#include <riscv_vector.h>
inline void tg_copy_weights(uint8_t * dst,const uint8_t * src,size_t bytes) {
    while(bytes){
        const size_t vl=__riscv_vsetvl_e8m8(bytes);
        auto data=__riscv_vle8_v_u8m8(src,vl);
        __riscv_vse8_v_u8m8(dst,data,vl);
        dst+=vl;src+=vl;bytes-=vl;
    }
}
