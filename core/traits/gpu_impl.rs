/*
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! GPU implementation of accelerator traits using ICICLE CUDA backend.

use std::sync::OnceLock;
use std::time::Duration;
use tracing::{info, warn};

use crate::traits::{
    AcceleratorError, AcceleratorResult, AsyncHandle, GpuAccelerator, GpuCachedBases,
    MsmBackend, NttBackend,
};
use crate::{
    ensure_backend_loaded, GpuMsmContext, PrecomputedBases, TypeConverter,
    config::{should_use_gpu, should_use_gpu_ntt, min_gpu_size, min_ntt_gpu_size},
    ntt::{forward_ntt_auto, forward_ntt_inplace_auto, inverse_ntt_auto, inverse_ntt_inplace_auto},
};
use midnight_curves::{Fq, G1Affine, G1Projective};

use group::prime::PrimeCurveAffine;
use group::Curve;
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
    Device, set_device,
};

// =============================================================================
// Global Accelerator Singleton
// =============================================================================

/// Global accelerator instance.
pub static GLOBAL_ACCELERATOR: GlobalAccelerator = GlobalAccelerator::new();

/// GPU-accelerated implementation of the accelerator traits.
///
/// Wraps ICICLE CUDA backend with a safe, ergonomic API.
pub struct GlobalAccelerator {
    /// Lazily initialized MSM context.
    msm_context: OnceLock<Result<GpuMsmContext, String>>,
    
    /// Track if we've already logged initialization.
    initialized: OnceLock<()>,
}

impl GlobalAccelerator {
    /// Create a new accelerator (const, for static initialization).
    pub const fn new() -> Self {
        Self {
            msm_context: OnceLock::new(),
            initialized: OnceLock::new(),
        }
    }
    
    /// Get or initialize the MSM context.
    fn get_msm_context(&self) -> AcceleratorResult<&GpuMsmContext> {
        let result = self.msm_context.get_or_init(|| {
            // Ensure backend is loaded first
            if let Err(e) = ensure_backend_loaded() {
                return Err(e.to_string());
            }
            
            match GpuMsmContext::new() {
                Ok(ctx) => {
                    info!("GPU MSM context initialized");
                    Ok(ctx)
                }
                Err(e) => {
                    warn!("Failed to create GPU MSM context: {:?}", e);
                    Err(e.to_string())
                }
            }
        });
        
        match result {
            Ok(ctx) => Ok(ctx),
            Err(msg) => Err(AcceleratorError::BackendNotAvailable(msg.clone())),
        }
    }
}

// =============================================================================
// Cached Bases Implementation
// =============================================================================

/// GPU-cached bases wrapper implementing the trait.
pub struct CachedGpuBases {
    inner: PrecomputedBases,
}

impl std::fmt::Debug for CachedGpuBases {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedGpuBases")
            .field("len", &self.inner.original_size())
            .field("precompute_factor", &self.inner.factor())
            .field("buffer_size", &self.inner.buffer_size())
            .finish()
    }
}

impl CachedGpuBases {
    /// Create from internal PrecomputedBases.
    pub fn new(inner: PrecomputedBases) -> Self {
        Self { inner }
    }
    
    /// Get reference to internal bases (for use within this crate).
    pub(crate) fn inner(&self) -> &PrecomputedBases {
        &self.inner
    }
}

unsafe impl Send for CachedGpuBases {}
unsafe impl Sync for CachedGpuBases {}

impl GpuCachedBases for CachedGpuBases {
    fn len(&self) -> usize {
        self.inner.original_size()
    }
    
    fn precompute_factor(&self) -> i32 {
        self.inner.factor()
    }
    
    fn gpu_memory_bytes(&self) -> usize {
        // Each G1Affine point is 96 bytes (2 x 48-byte Fp coordinates)
        self.inner.buffer_size() * 96
    }
}

// =============================================================================
// Async Handle Implementations
// =============================================================================

/// Wrapper for MSM async handle.
pub struct MsmAsyncHandle {
    inner: crate::msm::MsmHandle,
}

impl AsyncHandle for MsmAsyncHandle {
    type Output = G1Projective;
    
    fn wait(self) -> AcceleratorResult<Self::Output> {
        self.inner
            .wait()
            .map_err(|e| AcceleratorError::AsyncError(e.to_string()))
    }
    
    fn is_ready(&self) -> bool {
        // ICICLE doesn't provide a non-blocking check, so always return false
        false
    }
}

/// Wrapper for batch MSM async handle.
pub struct BatchMsmAsyncHandle {
    inner: crate::msm::BatchMsmHandle,
}

impl AsyncHandle for BatchMsmAsyncHandle {
    type Output = Vec<G1Projective>;
    
    fn wait(self) -> AcceleratorResult<Self::Output> {
        self.inner
            .wait()
            .map_err(|e| AcceleratorError::AsyncError(e.to_string()))
    }
    
    fn is_ready(&self) -> bool {
        false
    }
}

// =============================================================================
// MsmBackend Implementation
// =============================================================================

impl MsmBackend for GlobalAccelerator {
    type CachedBases = CachedGpuBases;
    type AsyncHandle = MsmAsyncHandle;
    type BatchAsyncHandle = BatchMsmAsyncHandle;
    
    fn upload_bases(&self, bases: &[G1Affine]) -> AcceleratorResult<Self::CachedBases> {
        ensure_backend_loaded().map_err(|e| AcceleratorError::BackendNotAvailable(e.to_string()))?;
        
        #[cfg(feature = "trace-msm")]
        let start = Instant::now();
        #[cfg(feature = "trace-msm")]
        debug!("Uploading {} bases to GPU", bases.len());
        
        // Set device before allocation
        let device = Device::new("CUDA", 0);
        set_device(&device).map_err(|e| AcceleratorError::AllocationFailed(format!("{:?}", e)))?;
        
        // Convert to ICICLE format (zero-copy)
        let icicle_points = TypeConverter::g1_slice_as_icicle(bases);
        
        // Upload to GPU
        let stream = IcicleStream::default();
        let mut device_bases = DeviceVec::device_malloc_async(icicle_points.len(), &stream)
            .map_err(|e| AcceleratorError::AllocationFailed(format!("{:?}", e)))?;
        
        device_bases
            .copy_from_host_async(HostSlice::from_slice(icicle_points), &stream)
            .map_err(|e| AcceleratorError::AllocationFailed(format!("{:?}", e)))?;
        
        stream.synchronize().map_err(|e| AcceleratorError::AllocationFailed(format!("{:?}", e)))?;
        
        let precomputed = PrecomputedBases::new(device_bases, icicle_points.len());
        
        #[cfg(feature = "trace-msm")]
        debug!("Bases uploaded in {:?}", start.elapsed());
        
        Ok(CachedGpuBases::new(precomputed))
    }
    
    fn upload_bases_precomputed(
        &self,
        bases: &[G1Affine],
        precompute_factor: i32,
    ) -> AcceleratorResult<Self::CachedBases> {
        let ctx = self.get_msm_context()?;
        
        #[cfg(feature = "trace-msm")]
        let start = Instant::now();
        #[cfg(feature = "trace-msm")]
        debug!("Uploading {} bases with precompute factor {}", bases.len(), precompute_factor);
        
        let precomputed = ctx
            .upload_g1_bases_with_precompute(bases, precompute_factor)
            .map_err(|e| AcceleratorError::AllocationFailed(e.to_string()))?;
        
        #[cfg(feature = "trace-msm")]
        debug!("Precomputed bases uploaded in {:?}", start.elapsed());
        
        Ok(CachedGpuBases::new(precomputed))
    }
    
    fn msm_with_cached_bases(
        &self,
        scalars: &[Fq],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<G1Projective> {
        let ctx = self.get_msm_context()?;
        
        ctx.msm_with_device_bases(scalars, bases.inner())
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn msm(&self, scalars: &[Fq], bases: &[G1Affine]) -> AcceleratorResult<G1Projective> {
        let ctx = self.get_msm_context()?;
        
        ctx.msm(scalars, bases)
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn msm_projective(
        &self,
        scalars: &[Fq],
        bases: &[G1Projective],
    ) -> AcceleratorResult<G1Projective> {
        // Convert projective to affine
        let mut affine_bases = vec![G1Affine::identity(); bases.len()];
        for (i, proj) in bases.iter().enumerate() {
            affine_bases[i] = proj.to_affine();
        }
        
        self.msm(scalars, &affine_bases)
    }
    
    fn msm_batch_with_cached_bases(
        &self,
        scalars_batch: &[&[Fq]],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Vec<G1Projective>> {
        let ctx = self.get_msm_context()?;
        
        ctx.msm_batch_with_device_bases(scalars_batch, bases.inner())
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn msm_with_cached_bases_async(
        &self,
        scalars: &[Fq],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Self::AsyncHandle> {
        let ctx = self.get_msm_context()?;
        
        ctx.msm_with_device_bases_async(scalars, bases.inner())
            .map(|h| MsmAsyncHandle { inner: h })
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn msm_batch_async(
        &self,
        scalars_batch: &[&[Fq]],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Self::BatchAsyncHandle> {
        let ctx = self.get_msm_context()?;
        
        ctx.msm_batch_with_device_bases_async(scalars_batch, bases.inner())
            .map(|h| BatchMsmAsyncHandle { inner: h })
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn is_gpu_available(&self) -> bool {
        crate::is_gpu_available()
    }
    
    fn should_use_gpu(&self, size: usize) -> bool {
        should_use_gpu(size)
    }
    
    fn warmup(&self) -> AcceleratorResult<Duration> {
        let ctx = self.get_msm_context()?;
        
        ctx.warmup()
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
}

// =============================================================================
// NttBackend Implementation
// =============================================================================

impl NttBackend for GlobalAccelerator {
    fn forward_ntt(&self, coeffs: &[Fq]) -> AcceleratorResult<Vec<Fq>> {
        forward_ntt_auto(coeffs).map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn forward_ntt_inplace(&self, data: &mut [Fq]) -> AcceleratorResult<()> {
        forward_ntt_inplace_auto(data).map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn inverse_ntt(&self, evals: &[Fq]) -> AcceleratorResult<Vec<Fq>> {
        inverse_ntt_auto(evals).map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn inverse_ntt_inplace(&self, data: &mut [Fq]) -> AcceleratorResult<()> {
        inverse_ntt_inplace_auto(data).map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn forward_ntt_batch(&self, batch: &[Fq], poly_size: usize) -> AcceleratorResult<Vec<Fq>> {
        use crate::ntt::forward_ntt_batch_auto;
        forward_ntt_batch_auto(batch, poly_size)
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn inverse_ntt_batch(&self, batch: &[Fq], poly_size: usize) -> AcceleratorResult<Vec<Fq>> {
        use crate::ntt::inverse_ntt_batch_auto;
        inverse_ntt_batch_auto(batch, poly_size)
            .map_err(|e| AcceleratorError::OperationFailed(e.to_string()))
    }
    
    fn is_gpu_available(&self) -> bool {
        crate::is_gpu_available()
    }
    
    fn should_use_gpu(&self, size: usize) -> bool {
        should_use_gpu_ntt(size)
    }
}

// =============================================================================
// GpuAccelerator Implementation
// =============================================================================

impl GpuAccelerator for GlobalAccelerator {
    fn initialize(&self) -> AcceleratorResult<()> {
        // Mark as initialized
        self.initialized.get_or_init(|| {
            info!("Initializing GPU accelerator");
        });
        
        // Ensure backend is loaded
        ensure_backend_loaded().map_err(|e| AcceleratorError::BackendNotAvailable(e.to_string()))?;
        
        // Create MSM context to validate GPU is working
        let _ = self.get_msm_context()?;
        
        Ok(())
    }
    
    fn backend_info(&self) -> String {
        if MsmBackend::is_gpu_available(self) {
            format!(
                "ICICLE CUDA backend (device: CUDA:0, msm_threshold: {}, ntt_threshold: {})",
                min_gpu_size(),
                min_ntt_gpu_size()
            )
        } else {
            "CPU fallback (BLST)".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_accelerator_creation() {
        let accel = &GLOBAL_ACCELERATOR;
        // This should work regardless of GPU availability
        let info = accel.backend_info();
        assert!(!info.is_empty());
    }
    
    #[test]
    #[ignore] // Requires GPU
    fn test_msm_with_cached_bases() {
        let accel = &GLOBAL_ACCELERATOR;
        
        // Skip if GPU not available
        if !accel.is_gpu_available() {
            return;
        }
        
        // Create test data
        let n = 1024;
        let scalars: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();
        let bases: Vec<G1Affine> = (0..n)
            .map(|_| (G1Projective::random(OsRng)).to_affine())
            .collect();
        
        // Upload bases
        let cached = accel.upload_bases(&bases).expect("upload failed");
        assert_eq!(cached.len(), n);
        
        // Compute MSM
        let result = accel.msm_with_cached_bases(&scalars, &cached);
        assert!(result.is_ok());
    }
}
