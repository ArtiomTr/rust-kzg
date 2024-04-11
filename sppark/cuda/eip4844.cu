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

#  ifdef _LP64
#   define LIMB_T_BITS   64
#  else
#   define LIMB_T_BITS   32
#  endif

#define NLIMBS(bits)   (bits/LIMB_T_BITS)

typedef limb_t d_vec256[NLIMBS(256)];
typedef limb_t d_vec384[NLIMBS(384)];
typedef byte d_pow256[256/8];

__device__ __forceinline__
static void d_limbs_from_le_bytes(limb_t * ret,
                                       const unsigned char *in, size_t n)
{
    limb_t limb = 0;

    while(n--) {
        limb <<= 8;
        limb |= in[n];
        /*
         * 'if (n % sizeof(limb_t) == 0)' is omitted because it's cheaper
         * to perform redundant stores than to pay penalty for
         * mispredicted branch. Besides, some compilers unroll the
         * loop and remove redundant stores to 'restrict'-ed storage...
         */
        ret[n / sizeof(limb_t)] = limb;
    }
}

__device__ __forceinline__
static void d_limbs_from_be_bytes(limb_t *ret, const unsigned char *in, size_t n) {
    limb_t limb = 0;

    while(n--) {
        limb <<= 8;
        limb |= *in++;
        /*
         * 'if (n % sizeof(limb_t) == 0)' is omitted because it's cheaper
         * to perform redundant stores than to pay penalty for
         * mispredicted branch. Besides, some compilers unroll the
         * loop and remove redundant stores to 'restrict'-ed storage...
         */
        ret[n / sizeof(limb_t)] = limb;
    }
}

#ifdef __UINTPTR_TYPE__
typedef __UINTPTR_TYPE__ uptr_t;
#else
typedef const void *uptr_t;
#endif

__device__ __forceinline__
static void d_le_bytes_from_limbs(unsigned char *out, const limb_t *in,
                                       size_t n)
{
    const union {
        long one;
        char little;
    } is_endian = { 1 };
    limb_t limb;
    size_t i, j, r;

    if ((uptr_t)out == (uptr_t)in && is_endian.little)
        return;

    r = n % sizeof(limb_t);
    n /= sizeof(limb_t);

    for(i = 0; i < n; i++) {
        for (limb = in[i], j = 0; j < sizeof(limb_t); j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }
    if (r) {
        for (limb = in[i], j = 0; j < r; j++, limb >>= 8)
            *out++ = (unsigned char)limb;
    }
}

__device__ __forceinline__
static void d_vec_zero(void *ret, size_t num)
{
    volatile limb_t *rp = (volatile limb_t *)ret;
    size_t i;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = 0;
}

__device__ __forceinline__
static void d_blst_scalar_from_bendian(d_pow256 ret, const unsigned char a[32]) {
    d_vec256 out;
    d_limbs_from_be_bytes(out, a, sizeof(out));
    d_le_bytes_from_limbs(ret, out, sizeof(out));
    d_vec_zero(out, sizeof(out));
}

typedef unsigned long long llimb_t;
#define MUL_MONT_IMPL(bits) \
__device__ __forceinline__ \
static void d_mul_mont_base_##bits(limb_t ret[], const limb_t a[], const limb_t b[], \
                       const limb_t p[], limb_t n0) \
{ \
    llimb_t limbx; \
    limb_t mask, borrow, mx, hi, tmp[NLIMBS(bits)+1], carry; \
    size_t i, j; \
 \
    for (mx=b[0], hi=0, i=0; i<bits; i++) { \
        limbx = (mx * (llimb_t)a[i]) + hi; \
        tmp[i] = (limb_t)limbx; \
        hi = (limb_t)(limbx >> LIMB_T_BITS); \
    } \
    mx = n0*tmp[0]; \
    tmp[i] = hi; \
 \
    for (carry=0, j=0; ; ) { \
        limbx = (mx * (llimb_t)p[0]) + tmp[0]; \
        hi = (limb_t)(limbx >> LIMB_T_BITS); \
        for (i=1; i<bits; i++) { \
            limbx = (mx * (llimb_t)p[i] + hi) + tmp[i]; \
            tmp[i-1] = (limb_t)limbx; \
            hi = (limb_t)(limbx >> LIMB_T_BITS); \
        } \
        limbx = tmp[i] + (hi + (llimb_t)carry); \
        tmp[i-1] = (limb_t)limbx; \
        carry = (limb_t)(limbx >> LIMB_T_BITS); \
 \
        if (++j==bits) \
            break; \
 \
        for (mx=b[j], hi=0, i=0; i<bits; i++) { \
            limbx = (mx * (llimb_t)a[i] + hi) + tmp[i]; \
            tmp[i] = (limb_t)limbx; \
            hi = (limb_t)(limbx >> LIMB_T_BITS); \
        } \
        mx = n0*tmp[0]; \
        limbx = hi + (llimb_t)carry; \
        tmp[i] = (limb_t)limbx; \
        carry = (limb_t)(limbx >> LIMB_T_BITS); \
    } \
 \
    for (borrow=0, i=0; i<bits; i++) { \
        limbx = tmp[i] - (p[i] + (llimb_t)borrow); \
        ret[i] = (limb_t)limbx; \
        borrow = (limb_t)(limbx >> LIMB_T_BITS) & 1; \
    } \
 \
    mask = carry - borrow; \
 \
    for(i=0; i<bits; i++) \
        ret[i] = (ret[i] & ~mask) | (tmp[i] & mask); \
} \
__device__ __forceinline__ \
void d_mul_mont_##bits(d_vec##bits ret, const d_vec##bits a, \
                            const d_vec##bits b, const d_vec##bits p, limb_t n0) \
{   d_mul_mont_base_##bits(ret, a, b, p, n0);   } \
\
__device__ __forceinline__ \
void d_sqr_mont_##bits(d_vec##bits ret, const d_vec##bits a, \
                            const d_vec##bits p, limb_t n0) \
{   d_mul_mont_base_##bits(ret, a, a, p, n0);   }

/*
 * 256-bit subroutines can handle arbitrary modulus, even non-"sparse",
 * but we have to harmonize the naming with assembly.
 */
#define d_mul_mont_256 d_mul_mont_sparse_256
#define d_sqr_mont_256 d_sqr_mont_sparse_256
MUL_MONT_IMPL(256)
#undef d_mul_mont_256
#undef d_sqr_mont_256
MUL_MONT_IMPL(384)

static const limb_t r0 = (limb_t)0xfffffffeffffffff; 

__device__ __forceinline__
void d_blst_fr_from_scalar(d_vec256 ret, const d_pow256 a)
{
    const union {
        long one;
        char little;
    } is_endian = { 1 };

    if ((uptr_t)ret == (uptr_t)a && is_endian.little) {
        d_mul_mont_sparse_256(ret, (const limb_t *)a, (limb_t*)device::BLS12_381_rRR,
                                                    (limb_t*)device::BLS12_381_r, r0);
    } else {
        d_vec256 out;
        d_limbs_from_le_bytes(out, a, 32);
        d_mul_mont_sparse_256(ret, out, (limb_t*) device::BLS12_381_rRR, (limb_t*) device::BLS12_381_r, r0);
        d_vec_zero(out, sizeof(out));
    }
}

__device__ __forceinline__ 
static void d_bytes_to_bls_field(scalar_t *out, const Bytes32 *b) {
    blst_scalar tmp;
    d_blst_scalar_from_bendian((byte*)&tmp, b->bytes);
    // if (!blst_scalar_fr_check(&tmp)) return C_KZG_BADARGS;
    d_blst_fr_from_scalar((limb_t*)out, (const byte*)&tmp);
}

typedef struct {
    fr_t evals[FIELD_ELEMENTS_PER_BLOB];
} Polynomial;

__global__ void blob_to_polynomial_kernel(Polynomial *p, const Blob *blob) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // C_KZG_RET ret;
    // for (size_t i = 0; i < FIELD_ELEMENTS_PER_BLOB; i++) {
    //     ret = bytes_to_bls_field(
    //         &p->evals[i], (Bytes32 *)&blob->bytes[i * BYTES_PER_FIELD_ELEMENT]
    //     );
    //     if (ret != C_KZG_OK) return ret;
    // }
    // return C_KZG_OK;
    d_bytes_to_bls_field(&p->evals[idx], (Bytes32 *)&blob->bytes[idx * BYTES_PER_FIELD_ELEMENT]);
}

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
        out, (affine_t*) p_affine, FIELD_ELEMENTS_PER_BLOB, (const scalar_t *)(&p->evals), true, sizeof(blst_p1_affine)
    );
}

extern "C"
void sppark_blob_to_kzg_commitment(KZGCommitment *out, const Blob *blob, const KZGSettings *s) {
    // C_KZG_RET ret;
    point_t commitment;

    // blob_to_polynomial(&p, blob);
    // if (ret != C_KZG_OK) return ret;
    // return C_KZG_OK;
    const gpu_t &gpu = select_gpu(-1);

    Blob *d_blob = (Blob*) gpu.Dmalloc(sizeof(Blob));
    gpu.HtoD(d_blob, (void*)blob, 1);

    Polynomial *d_poly = (Polynomial*) gpu.Dmalloc(sizeof(Polynomial));

    dim3 threadsPerBlock(256);
    dim3 numBlocks(FIELD_ELEMENTS_PER_BLOB / threadsPerBlock.x);
    blob_to_polynomial_kernel<<<numBlocks, threadsPerBlock>>>(d_poly, d_blob);

    Polynomial poly;
    gpu.DtoH(&poly, d_poly, 1);

    poly_to_kzg_commitment(&commitment, &poly, s);
    // if (ret != C_KZG_OK) return ret;
    bytes_from_g1(out, &commitment);
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