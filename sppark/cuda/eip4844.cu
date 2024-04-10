// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ff/bls12-381-fp2.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

typedef jacobian_t<fp2_t> point_fp2_t;
typedef xyzz_t<fp2_t> bucket_fp2_t;
typedef bucket_fp2_t::affine_inf_t affine_fp2_t;

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <msm/pippenger.cuh>
#include <util/gpu_t.cuh>
#include <util/all_gpus.cpp>

#include "blst.h"

/** The number of bytes in a BLS scalar field element. */
#define BYTES_PER_FIELD_ELEMENT 32

/** The number of field elements in a blob. */
#define FIELD_ELEMENTS_PER_BLOB 4096

/** The number of bytes in a blob. */
#define BYTES_PER_BLOB (FIELD_ELEMENTS_PER_BLOB * BYTES_PER_FIELD_ELEMENT)

/**
 * An array of 48 bytes. Represents an untrusted
 * (potentially invalid) commitment/proof.
 */
typedef struct {
    uint8_t bytes[48];
} Bytes48;

/**
 * An array of 32 bytes. Represents an untrusted
 * (potentially invalid) field element.
 */
typedef struct {
    uint8_t bytes[32];
} Bytes32;

/**
 * A trusted (valid) KZG commitment.
 */
typedef Bytes48 KZGCommitment;

/**
 * A basic blob data.
 */
typedef struct {
    uint8_t bytes[BYTES_PER_BLOB];
} Blob;

/**
 * Stores the setup and parameters needed for computing KZG proofs.
 */
typedef struct {
    /** The length of `roots_of_unity`, a power of 2. */
    uint64_t max_width;
    /** Powers of the primitive root of unity determined by
     * `SCALE2_ROOT_OF_UNITY` in bit-reversal permutation order,
     * length `max_width`. */
    scalar_t *roots_of_unity;
    /** G1 group elements from the trusted setup,
     * in Lagrange form bit-reversal permutation. */
    point_t *g1_values;
    /** G2 group elements from the trusted setup. */
    point_fp2_t *g2_values;
} KZGSettings;

// __device__ __forceinline__
// static void blst_scalar_from_bendian() {

// }

// __device__ __forceinline__ 
// static void bytes_to_bls_field(scalar_t *out, const Bytes32 *b) {
//     blst_scalar tmp;
//     blst_scalar_from_bendian(&tmp, b->bytes);
//     // if (!blst_scalar_fr_check(&tmp)) return C_KZG_BADARGS;
//     blst_fr_from_scalar((blst_fr*)out, &tmp);
// }

typedef struct {
    fr_t evals[FIELD_ELEMENTS_PER_BLOB];
} Polynomial;

// __global__ void blob_to_polynomial_kernel(Polynomial *p, const Blob *blob) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     // C_KZG_RET ret;
//     // for (size_t i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
//     //     ret = bytes_to_bls_field(
//     //         &p->evals[i], (Bytes32 *)&blob->bytes[i * BYTES_PER_FIELD_ELEMENT]
//     //     );
//     //     if (ret != C_KZG_OK) return ret;
//     // }
//     // return C_KZG_OK;
//     bytes_to_bls_field(&p->evals[idx], (Bytes32 *)&blob->bytes[idx * BYTES_PER_FIELD_ELEMENT]);
// }

void bytes_from_g1(Bytes48 *out, const point_t *in) {
    blst_p1_compress(out->bytes, (blst_p1*) in);
}

void bytes_to_bls_field(scalar_t *out, const Bytes32 *b) {
    blst_scalar tmp;
    blst_scalar_from_bendian(&tmp, b->bytes);
    // if (!blst_scalar_fr_check(&tmp)) return C_KZG_BADARGS;
    blst_fr_from_scalar((blst_fr*)out, &tmp);
    // return C_KZG_OK;
}

void blob_to_polynomial(Polynomial *p, const Blob *blob) {
    // C_KZG_RET ret;
    for (size_t i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
        bytes_to_bls_field(
            &p->evals[i], (Bytes32 *)&blob->bytes[i * BYTES_PER_FIELD_ELEMENT]
        );
        // if (ret != C_KZG_OK) return ret;
    }
    // return C_KZG_OK;
}

void poly_to_kzg_commitment(
    point_t *out, const Polynomial *p, const KZGSettings *s
) {
    const blst_p1 *p_arg[2] = {(blst_p1*)s->g1_values, NULL};
    blst_p1_affine *p_affine = (blst_p1_affine *) calloc(FIELD_ELEMENTS_PER_BLOB, sizeof(blst_p1_affine));

    blst_p1s_to_affine(p_affine, p_arg, FIELD_ELEMENTS_PER_BLOB);

    mult_pippenger<bucket_t>(
        out, (affine_t*) p_affine, FIELD_ELEMENTS_PER_BLOB, (const scalar_t *)(&p->evals), false, sizeof(blst_p1_affine)
    );
}

extern "C"
void sppark_blob_to_kzg_commitment(KZGCommitment *out, const Blob *blob, const KZGSettings *s) {
    // C_KZG_RET ret;
    Polynomial p;
    point_t commitment;

    blob_to_polynomial(&p, blob);
    // if (ret != C_KZG_OK) return ret;
    poly_to_kzg_commitment(&commitment, &p, s);
    // if (ret != C_KZG_OK) return ret;
    bytes_from_g1(out, &commitment);
    // return C_KZG_OK;
    // const gpu_t &gpu = select_gpu(-1);

    // Blob *d_blob = (Blob*) gpu.Dmalloc(sizeof(Blob));
    // gpu.HtoD(d_blob, (void*)blob, 1);

    // Polynomial *d_poly = (Polynomial*) gpu.Dmalloc(sizeof(Polynomial));

    // dim3 threadsPerBlock(256);
    // dim3 numBlocks(FIELD_ELEMENTS_PER_BLOB / threadsPerBlock.x);
    // blob_to_polynomial_kernel<<<numBlocks, threadsPerBlock>>>(d_poly, d_blob);
}


// extern "C"
// RustError::by_value mult_pippenger_inf(point_t* out, const affine_t points[],
//                                        size_t npoints, const scalar_t scalars[],
//                                        size_t ffi_affine_sz)
// {
//     return mult_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);
// }

// extern "C"
// RustError::by_value mult_pippenger_fp2_inf(point_fp2_t* out, const affine_fp2_t points[],
//                                            size_t npoints, const scalar_t scalars[],
//                                            size_t ffi_affine_sz)
// {
//     return mult_pippenger<bucket_fp2_t>(out, points, npoints, scalars, false, ffi_affine_sz);
// }