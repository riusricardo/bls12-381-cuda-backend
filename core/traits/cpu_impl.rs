/*
 * Copyright (C) 2026 BLS12-381 CUDA Backend Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

//! CPU fallback implementation when GPU feature is disabled.
//!
//! Uses BLST for MSM operations and midnight_curves::fft for NTT.

use std::time::Duration;

use crate::traits::{
    AcceleratorError, AcceleratorResult, AsyncHandle, GpuAccelerator, GpuCachedBases,
    MsmBackend, NttBackend,
};
use midnight_curves::{Fq, G1Affine, G1Projective};

use ff::{Field, PrimeField};

// =============================================================================
// Global Accelerator (CPU-only)
// =============================================================================

/// Global accelerator instance (CPU fallback).
pub static GLOBAL_ACCELERATOR: GlobalAccelerator = GlobalAccelerator;

/// CPU-only accelerator using BLST for MSM.
pub struct GlobalAccelerator;

// =============================================================================
// CPU Cached Bases (just stores the bases in RAM)
// =============================================================================

/// CPU-side bases storage (no GPU upload).
#[derive(Debug)]
pub struct CachedCpuBases {
    bases: Vec<G1Affine>,
}

impl GpuCachedBases for CachedCpuBases {
    fn len(&self) -> usize {
        self.bases.len()
    }
    
    fn precompute_factor(&self) -> i32 {
        1 // No precomputation on CPU
    }
    
    fn gpu_memory_bytes(&self) -> usize {
        0 // No GPU memory used
    }
}

// =============================================================================
// Dummy Async Handles (CPU operations are synchronous)
// =============================================================================

/// Immediate async handle - result is already computed.
pub struct ImmediateMsmHandle {
    result: AcceleratorResult<G1Projective>,
}

impl AsyncHandle for ImmediateMsmHandle {
    type Output = G1Projective;
    
    fn wait(self) -> AcceleratorResult<Self::Output> {
        self.result
    }
    
    fn is_ready(&self) -> bool {
        true // Always ready since computation is immediate
    }
}

/// Immediate batch async handle.
pub struct ImmediateBatchMsmHandle {
    results: AcceleratorResult<Vec<G1Projective>>,
}

impl AsyncHandle for ImmediateBatchMsmHandle {
    type Output = Vec<G1Projective>;
    
    fn wait(self) -> AcceleratorResult<Self::Output> {
        self.results
    }
    
    fn is_ready(&self) -> bool {
        true
    }
}

// =============================================================================
// MsmBackend Implementation (CPU via BLST)
// =============================================================================

impl MsmBackend for GlobalAccelerator {
    type CachedBases = CachedCpuBases;
    type AsyncHandle = ImmediateMsmHandle;
    type BatchAsyncHandle = ImmediateBatchMsmHandle;
    
    fn upload_bases(&self, bases: &[G1Affine]) -> AcceleratorResult<Self::CachedBases> {
        Ok(CachedCpuBases {
            bases: bases.to_vec(),
        })
    }
    
    fn upload_bases_precomputed(
        &self,
        bases: &[G1Affine],
        _precompute_factor: i32,
    ) -> AcceleratorResult<Self::CachedBases> {
        // No precomputation on CPU, just store bases
        self.upload_bases(bases)
    }
    
    fn msm_with_cached_bases(
        &self,
        scalars: &[Fq],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<G1Projective> {
        use midnight_curves::G1Projective;
        
        if scalars.len() != bases.len() {
            return Err(AcceleratorError::InvalidInput(format!(
                "Scalar count {} != base count {}",
                scalars.len(),
                bases.len()
            )));
        }
        
        // Convert affine to projective for BLST multi_exp
        let proj_bases: Vec<G1Projective> = bases.bases.iter().map(|a| a.into()).collect();
        
        Ok(G1Projective::multi_exp(&proj_bases, scalars))
    }
    
    fn msm(&self, scalars: &[Fq], bases: &[G1Affine]) -> AcceleratorResult<G1Projective> {
        if scalars.len() != bases.len() {
            return Err(AcceleratorError::InvalidInput(format!(
                "Scalar count {} != base count {}",
                scalars.len(),
                bases.len()
            )));
        }
        
        let proj_bases: Vec<G1Projective> = bases.iter().map(|a| a.into()).collect();
        Ok(G1Projective::multi_exp(&proj_bases, scalars))
    }
    
    fn msm_projective(
        &self,
        scalars: &[Fq],
        bases: &[G1Projective],
    ) -> AcceleratorResult<G1Projective> {
        if scalars.len() != bases.len() {
            return Err(AcceleratorError::InvalidInput(format!(
                "Scalar count {} != base count {}",
                scalars.len(),
                bases.len()
            )));
        }
        
        Ok(G1Projective::multi_exp(bases, scalars))
    }
    
    fn msm_batch_with_cached_bases(
        &self,
        scalars_batch: &[&[Fq]],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Vec<G1Projective>> {
        scalars_batch
            .iter()
            .map(|scalars| self.msm_with_cached_bases(scalars, bases))
            .collect()
    }
    
    fn msm_with_cached_bases_async(
        &self,
        scalars: &[Fq],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Self::AsyncHandle> {
        let result = self.msm_with_cached_bases(scalars, bases);
        Ok(ImmediateMsmHandle { result })
    }
    
    fn msm_batch_async(
        &self,
        scalars_batch: &[&[Fq]],
        bases: &Self::CachedBases,
    ) -> AcceleratorResult<Self::BatchAsyncHandle> {
        let results = self.msm_batch_with_cached_bases(scalars_batch, bases);
        Ok(ImmediateBatchMsmHandle { results })
    }
    
    fn is_gpu_available(&self) -> bool {
        false
    }
    
    fn should_use_gpu(&self, _size: usize) -> bool {
        false
    }
    
    fn warmup(&self) -> AcceleratorResult<Duration> {
        // No warmup needed for CPU
        Ok(Duration::ZERO)
    }
}

// =============================================================================
// NttBackend Implementation (CPU via midnight_curves::fft)
// =============================================================================

impl NttBackend for GlobalAccelerator {
    fn forward_ntt(&self, coeffs: &[Fq]) -> AcceleratorResult<Vec<Fq>> {
        use ff::PrimeField;
        
        let n = coeffs.len();
        if !n.is_power_of_two() {
            return Err(AcceleratorError::InvalidInput(
                "NTT size must be power of two".to_string(),
            ));
        }
        
        let k = n.ilog2();
        let mut result = coeffs.to_vec();
        
        // Compute omega
        let mut omega = Fq::ROOT_OF_UNITY;
        for _ in k..Fq::S {
            omega = omega.square();
        }
        
        midnight_curves::fft::best_fft(&mut result, omega, k);
        Ok(result)
    }
    
    fn forward_ntt_inplace(&self, data: &mut [Fq]) -> AcceleratorResult<()> {
        use ff::PrimeField;
        
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(AcceleratorError::InvalidInput(
                "NTT size must be power of two".to_string(),
            ));
        }
        
        let k = n.ilog2();
        let mut omega = Fq::ROOT_OF_UNITY;
        for _ in k..Fq::S {
            omega = omega.square();
        }
        
        midnight_curves::fft::best_fft(data, omega, k);
        Ok(())
    }
    
    fn inverse_ntt(&self, evals: &[Fq]) -> AcceleratorResult<Vec<Fq>> {
        use ff::{Field, PrimeField};
        
        let n = evals.len();
        if !n.is_power_of_two() {
            return Err(AcceleratorError::InvalidInput(
                "NTT size must be power of two".to_string(),
            ));
        }
        
        let k = n.ilog2();
        let mut result = evals.to_vec();
        
        // Compute omega and its inverse
        let mut omega = Fq::ROOT_OF_UNITY;
        for _ in k..Fq::S {
            omega = omega.square();
        }
        let omega_inv = omega.invert().unwrap();
        let n_inv = Fq::from(n as u64).invert().unwrap();
        
        midnight_curves::fft::best_fft(&mut result, omega_inv, k);
        
        // Apply 1/n scaling
        for val in result.iter_mut() {
            *val *= n_inv;
        }
        
        Ok(result)
    }
    
    fn inverse_ntt_inplace(&self, data: &mut [Fq]) -> AcceleratorResult<()> {
        use ff::{Field, PrimeField};
        
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(AcceleratorError::InvalidInput(
                "NTT size must be power of two".to_string(),
            ));
        }
        
        let k = n.ilog2();
        
        let mut omega = Fq::ROOT_OF_UNITY;
        for _ in k..Fq::S {
            omega = omega.square();
        }
        let omega_inv = omega.invert().unwrap();
        let n_inv = Fq::from(n as u64).invert().unwrap();
        
        midnight_curves::fft::best_fft(data, omega_inv, k);
        
        for val in data.iter_mut() {
            *val *= n_inv;
        }
        
        Ok(())
    }
    
    fn forward_ntt_batch(&self, batch: &[Fq], poly_size: usize) -> AcceleratorResult<Vec<Fq>> {
        if batch.len() % poly_size != 0 {
            return Err(AcceleratorError::InvalidInput(
                "Batch size must be multiple of polynomial size".to_string(),
            ));
        }
        
        let mut result = batch.to_vec();
        for chunk in result.chunks_mut(poly_size) {
            self.forward_ntt_inplace(chunk)?;
        }
        Ok(result)
    }
    
    fn inverse_ntt_batch(&self, batch: &[Fq], poly_size: usize) -> AcceleratorResult<Vec<Fq>> {
        if batch.len() % poly_size != 0 {
            return Err(AcceleratorError::InvalidInput(
                "Batch size must be multiple of polynomial size".to_string(),
            ));
        }
        
        let mut result = batch.to_vec();
        for chunk in result.chunks_mut(poly_size) {
            self.inverse_ntt_inplace(chunk)?;
        }
        Ok(result)
    }
    
    fn is_gpu_available(&self) -> bool {
        false
    }
    
    fn should_use_gpu(&self, _size: usize) -> bool {
        false
    }
}

// =============================================================================
// GpuAccelerator Implementation
// =============================================================================

impl GpuAccelerator for GlobalAccelerator {
    fn initialize(&self) -> AcceleratorResult<()> {
        // Nothing to initialize for CPU
        Ok(())
    }
    
    fn backend_info(&self) -> String {
        "CPU (BLST multi_exp, no GPU support compiled)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use rand_core::OsRng;
    use group::{Group, Curve};

    #[test]
    fn test_cpu_msm() {
        let accel = &GLOBAL_ACCELERATOR;
        
        let n = 64;
        let scalars: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();
        let bases: Vec<G1Affine> = (0..n)
            .map(|_| G1Projective::random(OsRng).to_affine())
            .collect();
        
        let cached = accel.upload_bases(&bases).unwrap();
        let result = accel.msm_with_cached_bases(&scalars, &cached);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_cpu_ntt_roundtrip() {
        let accel = &GLOBAL_ACCELERATOR;
        
        let n = 64;
        let original: Vec<Fq> = (0..n).map(|i| Fq::from(i as u64 + 1)).collect();
        
        let evals = accel.forward_ntt(&original).unwrap();
        let recovered = accel.inverse_ntt(&evals).unwrap();
        
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert_eq!(*a, *b);
        }
    }
}
