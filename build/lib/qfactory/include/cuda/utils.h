#pragma once

#include "cute/tensor.hpp"

using namespace cute;

#define make_right_layout(...) make_layout(__VA_ARGS__, LayoutRight{})

template<int BITS> 
requires (BITS == 4 || BITS == 8 || BITS == 16 || BITS == 32)
constexpr int get_swizzle_MBase() {
    if constexpr (BITS == 4)  return 5;
    if constexpr (BITS == 8)  return 4;
    if constexpr (BITS == 16) return 3;
    if constexpr (BITS == 32) return 2;
}

template<int K, int BITS>
requires (K % (256 / BITS) == 0)
constexpr auto get_best_swizzle() {
    constexpr int mbase = get_swizzle_MBase<BITS>();
    if constexpr (K % (1024 / BITS) == 0) return Swizzle<3, mbase, 3>{};
    else if constexpr (K % (512 / BITS) == 0) return Swizzle<2, mbase, 3>{};
    else return Swizzle<1, mbase, 3>{};
}
