#include <iostream>
#include <random>
#include <cmath>
#include <blst.h>
#include <chrono>


// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <ff/bls12-381.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>


typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

RustError mult_pippenger(point_t* out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[])
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
}

int main() {

    point_t out;

#define NPOINTS (size_t) 4096

    affine_t *points = (affine_t *) calloc(NPOINTS, sizeof(affine_t));
    scalar_t *scalars = (scalar_t *) calloc(NPOINTS, sizeof(scalar_t));
    
    const blst_p1 *generator = blst_p1_generator();

    std::random_device rd;

    std::mt19937_64 e2(rd());

    std::uniform_int_distribution<uint64_t> dist{};

    for (int i = 0; i < NPOINTS; ++i) {
        blst_scalar mult;
        {
            blst_fr fr;
            uint64_t vals[] = {dist(e2), dist(e2), dist(e2), dist(e2)};
            blst_fr_from_uint64(&fr, vals);
            blst_scalar_from_fr(&mult, &fr);
        }
        blst_p1 point;
        blst_p1_mult(&point, generator, mult.b, 255);
        point_t::affine_t f = point_t(fp_t(fp_mont(point.x.l)), fp_t(fp_mont(point.y.l)), fp_t(fp_mont(point.z.l)));
        points[i] = *(affine_t*)((void*)&f);

        blst_fr scalar;
        {
            uint64_t vals[] = {dist(e2), dist(e2), dist(e2), dist(e2)};
            blst_fr_from_uint64(&scalar, vals);
        }
        scalars[i] = scalar_t(fr_mont(scalar.l));
    }

    std::cout << "begin pippenger..." << std::endl;
    std::cout.flush();

    auto start = std::chrono::high_resolution_clock::now();
    RustError err = mult_pippenger(&out, points, NPOINTS, scalars);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Error code: " << err.code << std::endl;
    std::cout << "Elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::cout.flush();

#undef NPOINTS

    return 0;
}
#endif