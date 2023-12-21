#![allow(non_camel_case_types)]
use crate::consts::{G1_GENERATOR, G2_GENERATOR};
use crate::kzg_types::{MclFr as BlstFr, MclG1, MclG2};
use crate::poly::PolyData;
use kzg::eip_4844::hash_to_bls_field;
use kzg::{Fr as FrTrait, G1Mul, G2Mul};
use mcl_rust::{G2, G1, miller_loop, final_exp, GT};
use std::ops::{Add, Neg};

#[derive(Debug, Clone)]
pub struct FFTSettings {
    pub max_width: usize,
    pub root_of_unity: BlstFr,
    pub expanded_roots_of_unity: Vec<BlstFr>,
    pub reverse_roots_of_unity: Vec<BlstFr>,
    pub roots_of_unity: Vec<BlstFr>,
}

pub fn expand_root_of_unity(root: &BlstFr, width: usize) -> Result<Vec<BlstFr>, String> {
    let mut generated_powers = vec![BlstFr::one(), *root];

    while !(generated_powers.last().unwrap().is_one()) {
        if generated_powers.len() > width {
            return Err(String::from("Root of unity multiplied for too long"));
        }

        generated_powers.push(generated_powers.last().unwrap().mul(root));
    }

    if generated_powers.len() != width + 1 {
        return Err(String::from("Root of unity has invalid scale"));
    }

    Ok(generated_powers)
}

#[derive(Debug, Clone, Default)]
pub struct KZGSettings {
    pub fs: FFTSettings,
    pub secret_g1: Vec<MclG1>,
    pub secret_g2: Vec<MclG2>,
}

pub fn generate_trusted_setup(len: usize, secret: [u8; 32usize]) -> (Vec<MclG1>, Vec<MclG2>) {
    let s = hash_to_bls_field::<BlstFr>(&secret);
    let mut s_pow = BlstFr::one();

    let mut s1 = Vec::with_capacity(len);
    let mut s2 = Vec::with_capacity(len);

    for _ in 0..len {
        s1.push(G1_GENERATOR.mul(&s_pow));
        s2.push(G2_GENERATOR.mul(&s_pow));

        s_pow = s_pow.mul(&s);
    }

    (s1, s2)
}

pub fn eval_poly(p: &PolyData, x: &BlstFr) -> BlstFr {
    if p.coeffs.is_empty() {
        return BlstFr::zero();
    } else if x.is_zero() {
        return p.coeffs[0];
    }

    let mut out = p.coeffs[p.coeffs.len() - 1];
    let mut i = p.coeffs.len() - 2;

    loop {
        let temp = out.mul(x);
        out = temp.add(&p.coeffs[i]);

        if i == 0 {
            break;
        }
        i -= 1;
    }
    out
}

pub fn pairings_verify(a1: &MclG1, a2: &MclG2, b1: &MclG1, b2: &MclG2) -> bool {
    let a1neg = a1.proj.neg();

    let aa1 = G1::from(&a1neg);
    let bb1 = G1::from(b1.proj);
    let aa2 = G2::from(a2.proj);
    let bb2 = G2::from(b2.proj);

    let aa2_prepared = G2::from(aa2);
    let bb2_prepared = G2::from(bb2);

    let loop0 = miller_loop(&[(&aa1, &aa2_prepared)]);
    let loop1 = miller_loop(&[(&bb1, &bb2_prepared)]);

    let gt_point = loop0.add(loop1);

    let new_point = final_exp(&gt_point);

    GT::eq(&GT::one(), &new_point.0)
}
