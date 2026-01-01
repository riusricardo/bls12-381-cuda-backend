/*
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This file is part of BLS12-381 CUDA Backend.
 *
 * BLS12-381 CUDA Backend is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BLS12-381 CUDA Backend is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with BLS12-381 CUDA Backend.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @file params.cu
 * @brief BLS12-381 Device Constant Definitions
 * 
 * =============================================================================
 * PURPOSE
 * =============================================================================
 * 
 * This file defines the __device__ __constant__ arrays for GPU constant memory.
 * Values are sourced from bls12_381_constants.h (single source of truth).
 * 
 * The header (bls12_381_params.cuh) declares these as extern. This file
 * provides the actual definitions that the device linker resolves.
 * 
 * =============================================================================
 * BUILD REQUIREMENTS
 * =============================================================================
 * 
 * This file must be compiled with CUDA separate compilation:
 *   nvcc -dc -rdc=true params.cu -o params.o
 * 
 * And linked with device linking:
 *   nvcc -dlink params.o other.o -o device_link.o
 * 
 * =============================================================================
 */

#include <cstdint>
#include "bls12_381_constants.h"

namespace bls12_381 {

// =============================================================================
// Base Field Fq Constants
// =============================================================================

__device__ __constant__ uint64_t FQ_MODULUS[BLS12_381_FP_LIMBS_64] = FQ_MODULUS_LIMBS;
__device__ __constant__ uint64_t FQ_ONE[BLS12_381_FP_LIMBS_64]     = FQ_ONE_LIMBS;
__device__ __constant__ uint64_t FQ_R2[BLS12_381_FP_LIMBS_64]      = FQ_R2_LIMBS;

// =============================================================================
// Scalar Field Fr Constants
// =============================================================================

__device__ __constant__ uint64_t FR_MODULUS[BLS12_381_FR_LIMBS_64] = FR_MODULUS_LIMBS;
__device__ __constant__ uint64_t FR_ONE[BLS12_381_FR_LIMBS_64]     = FR_ONE_LIMBS;
__device__ __constant__ uint64_t FR_R2[BLS12_381_FR_LIMBS_64]      = FR_R2_LIMBS;

} // namespace bls12_381
