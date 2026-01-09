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

//! Type-safe dispatch helpers for GPU acceleration
//!
//! This module provides ergonomic utilities that encapsulate the common pattern
//! of checking field/curve types at runtime and dispatching to GPU or CPU paths.
//!
//! # Problem
//!
//! GPU acceleration in midnight-proofs requires runtime type checks because:
//! - Generic code works with `F: PrimeField` or `C: CurveAffine`
//! - GPU only supports BLS12-381 (`Fq`, `G1Affine`)
//! - We need to transmute generic types to concrete types for GPU calls
//!
//! This leads to repetitive, error-prone code:
//!
//! ```rust,ignore
//! // This pattern repeats 8+ times across proofs/
//! if TypeId::of::<F>() == TypeId::of::<Fq>() {
//!     let fq_slice: &mut [Fq] = unsafe {
//!         std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut Fq, n)
//!     };
//!     // ... GPU path
//! } else {
//!     // ... CPU fallback
//! }
//! ```
//!
//! # Solution
//!
//! This module provides type-safe wrappers that:
//! 1. Perform the type check at runtime
//! 2. Handle the unsafe transmute internally (with SAFETY comments)
//! 3. Call the appropriate GPU or CPU function
//! 4. Return a unified result type
//!
//! # Usage
//!
//! ```rust,ignore
//! use midnight_bls12_381_cuda::dispatch::{dispatch_msm, MsmResult};
//!
//! // Instead of manual type checks, just call:
//! let result: C::Curve = dispatch_msm::<C>(
//!     coeffs,
//!     |fq_coeffs| gpu_msm(fq_coeffs, &device_bases),
//!     || cpu_msm(coeffs, bases),
//! );
//! ```

use std::any::TypeId;
use std::fmt::Debug;

use ff::PrimeField;
use group::prime::PrimeCurveAffine;
use midnight_curves::{Fq, G1Affine, G1Projective};

#[cfg(feature = "gpu")]
use crate::config::{should_use_gpu, should_use_gpu_ntt};
#[cfg(not(feature = "gpu"))]
fn should_use_gpu(_size: usize) -> bool { false }
#[cfg(not(feature = "gpu"))]
fn should_use_gpu_ntt(_size: usize) -> bool { false }

// =============================================================================
// Type Checking Utilities
// =============================================================================

/// Check if a generic field type `F` is `Fq` (BLS12-381 scalar field).
///
/// This is a compile-time-inlinable check that enables runtime dispatch
/// to GPU-accelerated code paths.
#[inline]
pub fn is_fq<F: PrimeField + 'static>() -> bool {
    TypeId::of::<F>() == TypeId::of::<Fq>()
}

/// Check if a generic curve affine type `C` is `G1Affine` (BLS12-381 G1).
#[inline]
pub fn is_g1_affine<C: PrimeCurveAffine + 'static>() -> bool {
    TypeId::of::<C>() == TypeId::of::<G1Affine>()
}

/// Check if a generic curve projective type is `G1Projective`.
#[inline]
pub fn is_g1_projective<G: group::Group + 'static>() -> bool {
    TypeId::of::<G>() == TypeId::of::<G1Projective>()
}

/// Check if GPU should be used for field operations of the given size.
///
/// Returns true only if:
/// - The field type is `Fq` (GPU only supports BLS12-381)
/// - GPU is available and not disabled
/// - Size is above the threshold (default: 16384 for MSM, 4096 for NTT)
#[inline]
pub fn should_dispatch_to_gpu_field<F: PrimeField + 'static>(size: usize) -> bool {
    is_fq::<F>() && should_use_gpu(size)
}

/// Check if GPU should be used for curve operations of the given size.
#[inline]
pub fn should_dispatch_to_gpu_curve<C: PrimeCurveAffine + 'static>(size: usize) -> bool {
    is_g1_affine::<C>() && should_use_gpu(size)
}

/// Check if GPU should be used for NTT of the given size.
#[inline]
pub fn should_dispatch_to_gpu_ntt<F: PrimeField + 'static>(size: usize) -> bool {
    is_fq::<F>() && should_use_gpu_ntt(size)
}

// =============================================================================
// Safe Slice Conversion Utilities
// =============================================================================

/// Safely view a generic field slice as `&[Fq]` if the types match.
///
/// This is the core type-erasure mechanism. It performs a runtime type check
/// and returns `None` if the types don't match, avoiding undefined behavior.
///
/// # Safety
///
/// The transmute is safe because:
/// - We verify `F == Fq` via `TypeId` comparison
/// - Both types have identical memory representation (256-bit field element)
/// - Rust's type system ensures the input slice is valid `F` elements
#[inline]
pub fn try_as_fq_slice<F: PrimeField + 'static>(slice: &[F]) -> Option<&[Fq]> {
    if is_fq::<F>() {
        // SAFETY: TypeId check guarantees F == Fq, so memory layout is identical
        Some(unsafe { &*(slice as *const [F] as *const [Fq]) })
    } else {
        None
    }
}

/// Safely view a generic field slice as `&mut [Fq]` if the types match.
#[inline]
pub fn try_as_fq_slice_mut<F: PrimeField + 'static>(slice: &mut [F]) -> Option<&mut [Fq]> {
    if is_fq::<F>() {
        // SAFETY: TypeId check guarantees F == Fq
        Some(unsafe { &mut *(slice as *mut [F] as *mut [Fq]) })
    } else {
        None
    }
}

/// Safely view a generic curve point slice as `&[G1Affine]` if the types match.
#[inline]
pub fn try_as_g1_affine_slice<C: PrimeCurveAffine + 'static>(slice: &[C]) -> Option<&[G1Affine]> {
    if is_g1_affine::<C>() {
        // SAFETY: TypeId check guarantees C == G1Affine
        Some(unsafe { &*(slice as *const [C] as *const [G1Affine]) })
    } else {
        None
    }
}

/// Safely view a generic projective point slice as `&[G1Projective]`.
#[inline]
pub fn try_as_g1_projective_slice<G: group::Group + 'static>(slice: &[G]) -> Option<&[G1Projective]> {
    if is_g1_projective::<G>() {
        // SAFETY: TypeId check guarantees G == G1Projective
        Some(unsafe { &*(slice as *const [G] as *const [G1Projective]) })
    } else {
        None
    }
}

/// Convert a `G1Projective` result back to generic curve type.
///
/// # Safety
///
/// Caller must ensure `C` is actually `G1Affine` (or compatible with the
/// output projective type).
#[inline]
pub fn projective_to_curve<C: PrimeCurveAffine + 'static>(result: G1Projective) -> C::Curve {
    debug_assert!(is_g1_affine::<C>(), "projective_to_curve requires G1Affine");
    // SAFETY: We verified C == G1Affine, so C::Curve == G1Projective
    unsafe { std::mem::transmute_copy(&result) }
}

// =============================================================================
// High-Level Dispatch Functions
// =============================================================================

/// Result of a dispatched operation.
#[derive(Debug)]
pub enum DispatchResult<T, E> {
    /// GPU path was taken and succeeded
    Gpu(T),
    /// GPU path was taken but failed, includes fallback result
    GpuFailed { error: E, fallback: T },
    /// CPU path was taken (GPU not beneficial or not available)
    Cpu(T),
}

impl<T, E> DispatchResult<T, E> {
    /// Unwrap the result, regardless of which path was taken.
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            DispatchResult::Gpu(t) => t,
            DispatchResult::GpuFailed { fallback, .. } => fallback,
            DispatchResult::Cpu(t) => t,
        }
    }
    
    /// Check if GPU path was used (successfully).
    #[inline]
    pub fn used_gpu(&self) -> bool {
        matches!(self, DispatchResult::Gpu(_))
    }
}

/// Dispatch an MSM operation to GPU or CPU based on type and size.
///
/// This is the primary dispatch function for MSM operations. It:
/// 1. Checks if the curve type is `G1Affine` (GPU-compatible)
/// 2. Checks if the size is above the GPU threshold
/// 3. Calls the appropriate function
/// 4. Handles fallback if GPU fails
///
/// # Type Parameters
///
/// - `C`: Curve affine type (typically generic in caller)
/// - `E`: Error type from GPU operation
///
/// # Arguments
///
/// - `coeffs`: Scalar coefficients (will be converted to `&[Fq]` if possible)
/// - `size`: Operation size (for threshold check)
/// - `gpu_fn`: Closure that performs GPU MSM (receives `&[Fq]`)
/// - `cpu_fn`: Closure that performs CPU MSM (receives original `&[C::Scalar]`)
///
/// # Example
///
/// ```rust,ignore
/// let result = dispatch_msm::<G1Affine, MsmError, _>(
///     coeffs,
///     |fq_coeffs| ctx.msm_with_cached_bases(fq_coeffs, &device_bases),
///     || G1Projective::multi_exp(&proj_bases, coeffs),
/// );
/// ```
#[inline]
pub fn dispatch_msm<C, E, GpuFn, CpuFn>(
    coeffs: &[C::Scalar],
    gpu_fn: GpuFn,
    cpu_fn: CpuFn,
) -> DispatchResult<C::Curve, E>
where
    C: PrimeCurveAffine + 'static,
    C::Scalar: PrimeField + 'static,
    E: Debug,
    GpuFn: FnOnce(&[Fq]) -> Result<G1Projective, E>,
    CpuFn: FnOnce() -> C::Curve,
{
    // Check if GPU path is viable
    if !should_dispatch_to_gpu_curve::<C>(coeffs.len()) {
        return DispatchResult::Cpu(cpu_fn());
    }
    
    // Convert coefficients (safe because we checked the type)
    let fq_coeffs = try_as_fq_slice::<C::Scalar>(coeffs)
        .expect("Type check passed but conversion failed - this is a bug");
    
    // Try GPU
    match gpu_fn(fq_coeffs) {
        Ok(result) => DispatchResult::Gpu(projective_to_curve::<C>(result)),
        Err(e) => {
            tracing::warn!("GPU MSM failed, using CPU fallback: {:?}", e);
            DispatchResult::GpuFailed {
                error: e,
                fallback: cpu_fn(),
            }
        }
    }
}

/// Dispatch an in-place FFT/NTT operation to GPU or CPU.
///
/// # Arguments
///
/// - `data`: Mutable slice of field elements (modified in-place)
/// - `gpu_fn`: Closure that performs GPU NTT (receives `&mut [Fq]`)
/// - `cpu_fn`: Closure that performs CPU FFT (receives `&mut [F]`)
///
/// # Returns
///
/// `true` if GPU path was taken, `false` if CPU.
#[inline]
pub fn dispatch_ntt_inplace<F, E, GpuFn, CpuFn>(
    data: &mut [F],
    gpu_fn: GpuFn,
    cpu_fn: CpuFn,
) -> bool
where
    F: PrimeField + 'static,
    E: Debug,
    GpuFn: FnOnce(&mut [Fq]) -> Result<(), E>,
    CpuFn: FnOnce(&mut [F]),
{
    // Check if GPU path is viable
    if !should_dispatch_to_gpu_ntt::<F>(data.len()) {
        cpu_fn(data);
        return false;
    }
    
    // Convert to Fq (safe because we checked the type)
    let fq_data = try_as_fq_slice_mut::<F>(data)
        .expect("Type check passed but conversion failed - this is a bug");
    
    // Try GPU
    match gpu_fn(fq_data) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!("GPU NTT failed, using CPU fallback: {:?}", e);
            cpu_fn(data);
            false
        }
    }
}

/// Dispatch batch MSM operations with automatic GPU/CPU selection.
///
/// Optimized for committing multiple polynomials that share the same bases.
#[inline]
pub fn dispatch_batch_msm<C, E, GpuFn, CpuFn>(
    coeffs_batch: &[&[C::Scalar]],
    individual_size: usize,
    gpu_fn: GpuFn,
    cpu_fn: CpuFn,
) -> DispatchResult<Vec<C::Curve>, E>
where
    C: PrimeCurveAffine + 'static,
    C::Scalar: PrimeField + 'static,
    E: Debug,
    GpuFn: FnOnce(Vec<&[Fq]>) -> Result<Vec<G1Projective>, E>,
    CpuFn: FnOnce() -> Vec<C::Curve>,
{
    use crate::config::should_use_gpu_batch;
    
    // Check if GPU batch is beneficial
    let batch_count = coeffs_batch.len();
    if !is_g1_affine::<C>() || !should_use_gpu_batch(individual_size, batch_count) {
        return DispatchResult::Cpu(cpu_fn());
    }
    
    // Convert all coefficient slices
    let fq_batch: Vec<&[Fq]> = coeffs_batch
        .iter()
        .map(|coeffs| {
            try_as_fq_slice::<C::Scalar>(coeffs)
                .expect("Type check passed but conversion failed")
        })
        .collect();
    
    // Try GPU batch
    match gpu_fn(fq_batch) {
        Ok(results) => {
            let converted: Vec<C::Curve> = results
                .into_iter()
                .map(|r| projective_to_curve::<C>(r))
                .collect();
            DispatchResult::Gpu(converted)
        }
        Err(e) => {
            tracing::warn!("GPU batch MSM failed, using CPU fallback: {:?}", e);
            DispatchResult::GpuFailed {
                error: e,
                fallback: cpu_fn(),
            }
        }
    }
}

// =============================================================================
// With-Slice Pattern (for callbacks that need concrete types)
// =============================================================================

/// Execute a closure with the slice viewed as `&[Fq]` if types match.
///
/// This is the "with-pattern" that avoids unsafe code in caller:
///
/// ```rust,ignore
/// // Instead of:
/// if TypeId::of::<F>() == TypeId::of::<Fq>() {
///     let fq_slice = unsafe { transmute... };
///     do_gpu_stuff(fq_slice);
/// }
///
/// // Use:
/// with_fq_slice(slice, |fq_slice| {
///     do_gpu_stuff(fq_slice);
///     SomeResult
/// })
/// ```
#[inline]
pub fn with_fq_slice<F, T, Fn>(slice: &[F], f: Fn) -> Option<T>
where
    F: PrimeField + 'static,
    Fn: FnOnce(&[Fq]) -> T,
{
    try_as_fq_slice(slice).map(f)
}

/// Execute a closure with the slice viewed as `&mut [Fq]` if types match.
#[inline]
pub fn with_fq_slice_mut<F, T, Fn>(slice: &mut [F], f: Fn) -> Option<T>
where
    F: PrimeField + 'static,
    Fn: FnOnce(&mut [Fq]) -> T,
{
    try_as_fq_slice_mut(slice).map(f)
}

/// Execute a closure with the slice viewed as `&[G1Affine]` if types match.
#[inline]
pub fn with_g1_affine_slice<C, T, Fn>(slice: &[C], f: Fn) -> Option<T>
where
    C: PrimeCurveAffine + 'static,
    Fn: FnOnce(&[G1Affine]) -> T,
{
    try_as_g1_affine_slice(slice).map(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_is_fq() {
        assert!(is_fq::<Fq>());
        // We can't easily test non-Fq fields without importing them
    }

    #[test]
    fn test_is_g1_affine() {
        assert!(is_g1_affine::<G1Affine>());
    }

    #[test]
    fn test_try_as_fq_slice() {
        let values: Vec<Fq> = (0..10).map(|_| Fq::random(OsRng)).collect();
        
        // Should succeed for Fq
        let result = try_as_fq_slice(&values);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn test_with_fq_slice() {
        let values: Vec<Fq> = (0..10).map(|_| Fq::random(OsRng)).collect();
        
        let result = with_fq_slice(&values, |fq| fq.len());
        assert_eq!(result, Some(10));
    }

    #[test]
    fn test_dispatch_result_unwrap() {
        let gpu_result: DispatchResult<i32, ()> = DispatchResult::Gpu(42);
        assert_eq!(gpu_result.unwrap(), 42);
        
        let cpu_result: DispatchResult<i32, ()> = DispatchResult::Cpu(99);
        assert_eq!(cpu_result.unwrap(), 99);
    }
}
