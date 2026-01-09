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

//! Trait-based abstraction layer for GPU acceleration
//!
//! This module provides production-ready traits that abstract over GPU acceleration,
//! allowing `midnight-proofs` to depend only on this crate without needing direct
//! access to ICICLE types or low-level GPU management.
//!
//! # Design Goals
//!
//! 1. **Minimal Integration Surface**: Consumers only interact with traits
//! 2. **Type Safety**: No unsafe transmutes in consumer code
//! 3. **Zero-Cost Abstractions**: Traits are #[inline] for monomorphization
//! 4. **Single Dependency**: ICICLE is encapsulated within this crate
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Consumer (midnight-proofs)                                      │
//! │                                                                  │
//! │  - Uses MsmBackend trait for MSM operations                      │
//! │  - Uses NttBackend trait for FFT operations                      │
//! │  - Uses GpuCachedBases for SRS caching                          │
//! │  - No ICICLE imports, no unsafe transmutes                       │
//! └────────────────────────────────────────┬─────────────────────────┘
//!                                          │
//!                                          ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  midnight-bls12-381-cuda (this crate)                            │
//! │                                                                  │
//! │  - Implements traits with ICICLE backend                         │
//! │  - Handles all type conversions internally                       │
//! │  - Manages GPU context and resources                             │
//! │  - Provides CPU fallback when GPU unavailable                    │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use midnight_bls12_381_cuda::{MsmBackend, GpuCachedBases};
//! use midnight_curves::{Fq, G1Projective, G1Affine};
//!
//! // Get global accelerator
//! let accel = midnight_bls12_381_cuda::global_accelerator();
//!
//! // Upload SRS bases once
//! let cached_bases = accel.upload_bases(&srs_bases)?;
//!
//! // Compute MSM using cached bases
//! let result: G1Projective = accel.msm_with_cached_bases(&scalars, &cached_bases)?;
//! ```

use std::fmt::Debug;

// Re-export types that consumers will use
pub use midnight_curves::{Fq, G1Affine, G1Projective, G2Affine, G2Projective};

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during GPU-accelerated operations.
///
/// Provides a unified error type for all GPU operations, abstracting over
/// backend-specific errors.
#[derive(Debug, Clone)]
pub enum AcceleratorError {
    /// GPU backend failed to initialize or load.
    BackendNotAvailable(String),
    
    /// Failed to allocate GPU memory.
    AllocationFailed(String),
    
    /// GPU operation (MSM, NTT, etc.) failed.
    OperationFailed(String),
    
    /// Invalid input (size mismatch, empty, etc.).
    InvalidInput(String),
    
    /// Async operation failed to complete.
    AsyncError(String),
    
    /// GPU is available but was explicitly disabled via configuration.
    GpuDisabled,
    
    /// Operation not supported (e.g., wrong field type).
    NotSupported(String),
}

impl std::fmt::Display for AcceleratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotAvailable(msg) => write!(f, "GPU backend not available: {}", msg),
            Self::AllocationFailed(msg) => write!(f, "GPU allocation failed: {}", msg),
            Self::OperationFailed(msg) => write!(f, "GPU operation failed: {}", msg),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::AsyncError(msg) => write!(f, "Async operation failed: {}", msg),
            Self::GpuDisabled => write!(f, "GPU is disabled via configuration"),
            Self::NotSupported(msg) => write!(f, "Operation not supported: {}", msg),
        }
    }
}

impl std::error::Error for AcceleratorError {}

// Conversion from internal error types
#[cfg(feature = "gpu")]
impl From<crate::GpuError> for AcceleratorError {
    fn from(e: crate::GpuError) -> Self {
        AcceleratorError::BackendNotAvailable(e.to_string())
    }
}

#[cfg(feature = "gpu")]
impl From<crate::MsmError> for AcceleratorError {
    fn from(e: crate::MsmError) -> Self {
        AcceleratorError::OperationFailed(e.to_string())
    }
}

#[cfg(feature = "gpu")]
impl From<crate::NttError> for AcceleratorError {
    fn from(e: crate::NttError) -> Self {
        AcceleratorError::OperationFailed(e.to_string())
    }
}

/// Result type for accelerator operations.
pub type AcceleratorResult<T> = Result<T, AcceleratorError>;

// =============================================================================
// Async Handle Traits
// =============================================================================

/// Handle for an in-flight async operation.
///
/// Allows launching GPU work and waiting for it later, enabling CPU/GPU overlap.
pub trait AsyncHandle {
    /// The result type when the operation completes.
    type Output;
    
    /// Wait for the operation to complete and return the result.
    fn wait(self) -> AcceleratorResult<Self::Output>;
    
    /// Check if the operation has completed without blocking.
    fn is_ready(&self) -> bool;
}

// =============================================================================
// GPU Cached Bases Trait
// =============================================================================

/// Opaque handle to bases cached in GPU memory.
///
/// This trait abstracts over the internal GPU memory representation,
/// allowing consumers to hold references to uploaded SRS bases without
/// knowing about ICICLE's DeviceVec or other implementation details.
///
/// # Lifecycle
///
/// 1. Created via `MsmBackend::upload_bases()` or `MsmBackend::upload_bases_precomputed()`
/// 2. Held for the lifetime of the proof system (typically entire program)
/// 3. Automatically freed when dropped
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow sharing across threads.
/// The underlying GPU memory is safely accessible from multiple threads
/// when using CUDA's default stream synchronization.
pub trait GpuCachedBases: Send + Sync + Debug {
    /// Returns the number of bases stored (original count, before any precomputation).
    fn len(&self) -> usize;
    
    /// Returns true if no bases are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Returns the precomputation factor (1 = no precomputation).
    fn precompute_factor(&self) -> i32;
    
    /// Returns the actual GPU memory size in bytes.
    fn gpu_memory_bytes(&self) -> usize;
}

// =============================================================================
// MSM Backend Trait
// =============================================================================

/// Multi-Scalar Multiplication (MSM) backend trait.
///
/// Provides GPU-accelerated MSM operations with both synchronous and asynchronous
/// APIs. Implementations handle type conversions, GPU resource management, and
/// automatic fallback to CPU when appropriate.
///
/// # Type Parameters
///
/// The trait is designed for BLS12-381 curve operations with:
/// - `Scalar = Fq` (scalar field)
/// - `Point = G1Projective` or `G2Projective`
/// - `Affine = G1Affine` or `G2Affine`
///
/// # Usage Pattern
///
/// ```rust,ignore
/// let backend = GlobalMsmBackend::get();
///
/// // One-time: upload SRS bases to GPU
/// let cached_bases = backend.upload_bases(&srs_points)?;
///
/// // Per-proof: compute MSM using cached bases (fast!)
/// let commitment = backend.msm_with_cached_bases(&scalars, &cached_bases)?;
/// ```
pub trait MsmBackend: Send + Sync {
    /// Opaque type for cached GPU bases.
    type CachedBases: GpuCachedBases;
    
    /// Handle type for async MSM operations.
    type AsyncHandle: AsyncHandle<Output = G1Projective>;
    
    /// Handle type for async batch MSM operations.
    type BatchAsyncHandle: AsyncHandle<Output = Vec<G1Projective>>;
    
    // =========================================================================
    // Base Uploading
    // =========================================================================
    
    /// Upload affine bases to GPU memory for reuse across multiple MSMs.
    ///
    /// This is the critical optimization: SRS bases are uploaded once at startup,
    /// eliminating per-MSM data transfer overhead. All subsequent MSMs with these
    /// bases execute entirely on GPU.
    ///
    /// # Arguments
    /// * `bases` - Affine points to upload (typically from SRS)
    ///
    /// # Returns
    /// Handle to GPU-resident bases that can be used with `msm_with_cached_bases()`
    fn upload_bases(&self, bases: &[G1Affine]) -> AcceleratorResult<Self::CachedBases>;
    
    /// Upload bases with precomputation for faster MSM.
    ///
    /// Precomputes point multiples on GPU, trading memory for ~20-30% MSM speedup.
    /// The precompute_factor determines how many multiples to store.
    ///
    /// # Arguments
    /// * `bases` - Affine points to upload
    /// * `precompute_factor` - Precomputation level (1-8, higher = faster but more memory)
    fn upload_bases_precomputed(
        &self,
        bases: &[G1Affine],
        precompute_factor: i32,
    ) -> AcceleratorResult<Self::CachedBases>;
    
    // =========================================================================
    // Synchronous MSM Operations
    // =========================================================================
    
    /// Compute MSM using bases already cached on GPU (primary API).
    ///
    /// This is the **hot path** for proof generation. Uses bases uploaded via
    /// `upload_bases()`, avoiding per-call data transfer.
    ///
    /// # Arguments
    /// * `scalars` - Scalar coefficients
    /// * `bases` - Pre-uploaded GPU bases
    ///
    /// # Returns
    /// The MSM result: sum(scalars[i] * bases[i])
    fn msm_with_cached_bases(
        &self,
        scalars: &[Fq],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<G1Projective>;
    
    /// Compute MSM with host-resident bases (convenience API).
    ///
    /// Converts and uploads bases per-call. Use for non-SRS MSMs or when
    /// bases vary between calls.
    ///
    /// # Performance Note
    /// For repeated MSMs with same bases, prefer `upload_bases()` + `msm_with_cached_bases()`.
    fn msm(&self, scalars: &[Fq], bases: &[G1Affine]) -> AcceleratorResult<G1Projective>;
    
    /// Compute MSM with projective bases.
    ///
    /// Handles conversion to affine internally. Useful when bases are already
    /// in projective form.
    fn msm_projective(
        &self,
        scalars: &[Fq],
        bases: &[G1Projective],
    ) -> AcceleratorResult<G1Projective>;
    
    // =========================================================================
    // Batch MSM Operations
    // =========================================================================
    
    /// Compute multiple MSMs with shared cached bases.
    ///
    /// More efficient than individual calls due to:
    /// - Single kernel launch overhead
    /// - Better GPU utilization
    /// - Pipelined data transfers
    ///
    /// # Arguments
    /// * `scalars_batch` - Slice of scalar slices, one per MSM
    /// * `bases` - Shared GPU bases for all MSMs
    fn msm_batch_with_cached_bases(
        &self,
        scalars_batch: &[&[Fq]],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Vec<G1Projective>>;
    
    // =========================================================================
    // Asynchronous MSM Operations (for pipelining)
    // =========================================================================
    
    /// Launch async MSM with cached bases.
    ///
    /// Returns immediately with a handle that can be waited on later,
    /// enabling CPU work during GPU computation.
    fn msm_with_cached_bases_async(
        &self,
        scalars: &[Fq],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Self::AsyncHandle>;
    
    /// Launch async batch MSM.
    fn msm_batch_async(
        &self,
        scalars_batch: &[&[Fq]],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Self::BatchAsyncHandle>;
    
    // =========================================================================
    // Status and Configuration
    // =========================================================================
    
    /// Check if GPU acceleration is available and enabled.
    fn is_gpu_available(&self) -> bool;
    
    /// Check if GPU should be used for the given operation size.
    ///
    /// Returns true if:
    /// - GPU is available
    /// - Device mode is not `Cpu`
    /// - Size is above the threshold (for Auto mode)
    fn should_use_gpu(&self, size: usize) -> bool;
    
    /// Warmup the GPU backend.
    ///
    /// Runs a small MSM to trigger CUDA JIT compilation and memory allocation.
    /// Call at application startup to avoid first-request latency.
    fn warmup(&self) -> AcceleratorResult<std::time::Duration>;
}

// =============================================================================
// NTT Backend Trait
// =============================================================================

/// Number Theoretic Transform (NTT) backend trait.
///
/// Provides GPU-accelerated FFT/IFFT operations for polynomial transformations.
///
/// # Automatic Backend Selection
///
/// The implementation automatically chooses between GPU and CPU based on:
/// - Input size (GPU is beneficial for size >= 4096)
/// - Device configuration (`MIDNIGHT_DEVICE` environment variable)
/// - GPU availability
pub trait NttBackend: Send + Sync {
    // =========================================================================
    // Forward NTT (Coefficients → Evaluations)
    // =========================================================================
    
    /// Perform forward NTT (coefficient form to evaluation form).
    ///
    /// Allocates a new output vector.
    fn forward_ntt(&self, coeffs: &[Fq]) -> AcceleratorResult<Vec<Fq>>;
    
    /// Perform forward NTT in-place.
    fn forward_ntt_inplace(&self, data: &mut [Fq]) -> AcceleratorResult<()>;
    
    // =========================================================================
    // Inverse NTT (Evaluations → Coefficients)
    // =========================================================================
    
    /// Perform inverse NTT (evaluation form to coefficient form).
    ///
    /// Includes the 1/n scaling factor automatically.
    fn inverse_ntt(&self, evals: &[Fq]) -> AcceleratorResult<Vec<Fq>>;
    
    /// Perform inverse NTT in-place.
    fn inverse_ntt_inplace(&self, data: &mut [Fq]) -> AcceleratorResult<()>;
    
    // =========================================================================
    // Batch NTT Operations
    // =========================================================================
    
    /// Perform forward NTT on multiple polynomials.
    ///
    /// More efficient than individual calls for batch operations.
    ///
    /// # Arguments
    /// * `batch` - Concatenated polynomial coefficients
    /// * `poly_size` - Size of each individual polynomial
    fn forward_ntt_batch(&self, batch: &[Fq], poly_size: usize) -> AcceleratorResult<Vec<Fq>>;
    
    /// Perform inverse NTT on multiple polynomials.
    fn inverse_ntt_batch(&self, batch: &[Fq], poly_size: usize) -> AcceleratorResult<Vec<Fq>>;
    
    // =========================================================================
    // Status and Configuration
    // =========================================================================
    
    /// Check if GPU NTT is available for this backend.
    fn is_gpu_available(&self) -> bool;
    
    /// Check if GPU should be used for the given size.
    fn should_use_gpu(&self, size: usize) -> bool;
}

// =============================================================================
// Global Accelerator Access
// =============================================================================

/// Combined interface for all GPU acceleration operations.
///
/// Provides unified access to MSM and NTT backends through a single type.
pub trait GpuAccelerator: MsmBackend + NttBackend {
    /// Initialize the accelerator, loading GPU backends if available.
    ///
    /// Called automatically on first use, but can be called explicitly
    /// at startup to detect issues early and trigger JIT compilation.
    fn initialize(&self) -> AcceleratorResult<()>;
    
    /// Get human-readable backend information.
    fn backend_info(&self) -> String;
}

// =============================================================================
// Concrete Implementation Placeholder
// =============================================================================

#[cfg(feature = "gpu")]
mod gpu_impl;

#[cfg(feature = "gpu")]
pub use gpu_impl::{GlobalAccelerator, GLOBAL_ACCELERATOR};

#[cfg(not(feature = "gpu"))]
mod cpu_impl;

#[cfg(not(feature = "gpu"))]
pub use cpu_impl::{GlobalAccelerator, GLOBAL_ACCELERATOR};

/// Get the global GPU accelerator instance.
///
/// This is the main entry point for GPU-accelerated operations.
/// The accelerator is lazily initialized on first use.
///
/// # Example
///
/// ```rust,ignore
/// use midnight_bls12_381_cuda::traits::global_accelerator;
///
/// let accel = global_accelerator();
/// if accel.is_gpu_available() {
///     println!("GPU acceleration enabled!");
/// }
/// ```
#[inline]
pub fn global_accelerator() -> &'static GlobalAccelerator {
    &GLOBAL_ACCELERATOR
}
