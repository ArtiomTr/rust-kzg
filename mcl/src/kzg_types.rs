use crate::consts::{
    G1_GENERATOR, G1_IDENTITY, G1_NEGATIVE_GENERATOR, G2_GENERATOR, G2_NEGATIVE_GENERATOR,
    SCALE2_ROOT_OF_UNITY, R2,
};
use crate::fft_g1::g1_linear_combination;
use crate::kzg_proofs::{
    expand_root_of_unity, pairings_verify, FFTSettings as ZFFTSettings, KZGSettings as ZKZGSettings,
};
use crate::poly::PolyData;
use crate::utils::{blst_fr_into_mcl_fr, mcl_fr_into_blst_fr, sbb, blst_p1_into_mcl_g1projective, mcl_g1projective_into_blst_p1, blst_p2_into_mcl_g2projective};
use kzg::common_utils::reverse_bit_order;
use kzg::eip_4844::{BYTES_PER_FIELD_ELEMENT, BYTES_PER_G1, BYTES_PER_G2};
use kzg::{
    FFTFr, FFTSettings, Fr as KzgFr, G1Mul, G2Mul, KZGSettings, PairingVerify, Poly, G1 as KzgG1, G2 as KzgG2,
};
use std::ops::{Mul, Sub, Add};
use mcl_rust::{Fr, G1, G2};
use blst::{blst_fr, blst_p1};
use subtle::{ConstantTimeEq, CtOption};

pub const MODULUS: Fr = Fr{d:[
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
]};


fn bigint_check_mod_256(a: &[u64; 4]) -> bool {
    let (_, overflow) = a[0].overflowing_sub(MODULUS.0[0]);
    let (_, overflow) = a[1].overflowing_sub(MODULUS.0[1] + overflow as u64);
    let (_, overflow) = a[2].overflowing_sub(MODULUS.0[2] + overflow as u64);
    let (_, overflow) = a[3].overflowing_sub(MODULUS.0[3] + overflow as u64);
    overflow
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MclFr {
    pub fr: Fr,
}

impl MclFr {
    pub fn from_blst_fr(fr: blst_fr) -> Self {
        Self {
            fr: blst_fr_into_mcl_fr(fr),
        }
    }
    pub fn to_blst_fr(&self) -> blst_fr {
        mcl_fr_into_blst_fr(self.fr)
    }

    pub fn converter(points: &[MclFr]) -> Vec<Fr> {
        let mut result = Vec::new();

        for i in points {
            result.push(i.fr);
        }
        result
    }
}

impl KzgFr for MclFr {
    fn null() -> Self {
        Self {
            fr: Fr{ d: [u64::MAX, u64::MAX, u64::MAX, u64::MAX]},
        }
    }
    fn zero() -> Self {
        Self::from_u64(0)
    }

    fn one() -> Self {
        Self::from_u64(1)
    }

    #[cfg(feature = "rand")]
    fn rand() -> Self {
        let val: [u64; 4] = [
            rand::random(),
            rand::random(),
            rand::random(),
            rand::random(),
        ];

        Self::from_u64_arr(&val)
    }
    #[allow(clippy::bind_instead_of_map)]
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_FIELD_ELEMENT,
                    bytes.len()
                )
            })
            .and_then(|bytes: &[u8; BYTES_PER_FIELD_ELEMENT]| {
                let mut tmp = Fr { d: [0, 0, 0, 0]};

                tmp.d[0] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap());
                tmp.d[1] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap());
                tmp.d[2] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap());
                tmp.d[3] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap());

                // Try to subtract the modulus
                let (_, borrow) = sbb(tmp.d[0], MODULUS.d[0], 0);
                let (_, borrow) = sbb(tmp.d[1], MODULUS.d[1], borrow);
                let (_, borrow) = sbb(tmp.d[2], MODULUS.d[2], borrow);
                let (_, _borrow) = sbb(tmp.d[3], MODULUS.d[3], borrow);
                let mut tmp2 = Fr::default();

                tmp2.d[0] = tmp.d[3];
                tmp2.d[1] = tmp.d[2];
                tmp2.d[2] = tmp.d[1];
                tmp2.d[3] = tmp.d[0];

                let is_zero: bool = tmp2.is_zero().into();
                if !is_zero && !bigint_check_mod_256(&tmp2.0) {
                    return Err("Invalid Fr".to_string());
                }

                tmp2 *= &R2;
                Ok(Self { fr: tmp2 })
            })
    }
    fn from_bytes_unchecked(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_FIELD_ELEMENT,
                    bytes.len()
                )
            })
            .map(|bytes: &[u8; BYTES_PER_FIELD_ELEMENT]| {
                let mut tmp = Fr { d: [0, 0, 0, 0]};

                tmp.d[0] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap());
                tmp.d[1] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap());
                tmp.d[2] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap());
                tmp.d[3] = u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap());

                // Try to subtract the modulus
                let (_, borrow) = sbb(tmp.d[0], MODULUS.d[0], 0);
                let (_, borrow) = sbb(tmp.d[1], MODULUS.d[1], borrow);
                let (_, borrow) = sbb(tmp.d[2], MODULUS.d[2], borrow);
                let (_, _borrow) = sbb(tmp.d[3], MODULUS.d[3], borrow);
                let mut tmp2 = Fr::default();

                tmp2.d[0] = tmp.d[3];
                tmp2.d[1] = tmp.d[2];
                tmp2.d[2] = tmp.d[1];
                tmp2.d[3] = tmp.d[0];

                tmp2 *= &R2;
                Self { fr: tmp2 }
            })
    }

    fn from_hex(hex: &str) -> Result<Self, String> {
        let bytes = hex::decode(&hex[2..]).unwrap();
        Self::from_bytes(&bytes)
    }

    fn from_u64_arr(u: &[u64; 4]) -> Self {
        Self {
            fr: Fr { d: [u[0], u[1], u[2], u[3]]}
        }
    }

    fn from_u64(val: u64) -> Self {
        let mut tmp: Fr;
        tmp.set_int(val.try_into().unwrap());
        Self {
            fr: tmp,
        }
    }

    fn to_bytes(&self) -> [u8; 32] {
        let mut res = vec![u8::default(); 32usize];
        res = self.fr.serialize();
        res.try_into().unwrap()
    }

    //testuoti
    fn to_u64_arr(&self) -> [u64; 4] {
        let bytes = self.to_bytes();
        [
            u64::from_be_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_be_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_be_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_be_bytes(bytes[0..8].try_into().unwrap()),
        ]
    }

    fn is_one(&self) -> bool {
        self.fr.is_one()
    }

    fn is_zero(&self) -> bool {
        self.fr.is_zero()
    }

    fn is_null(&self) -> bool {
        self.fr.is_valid()
    }

    fn sqr(&self) -> Self {
        Fr::sqr(&mut self.fr, &self.fr);
        self.to_owned()
    }

    fn mul(&self, b: &Self) -> Self {
        Self{
            fr: self.fr.mul(&b.fr),
        }
    }

    fn add(&self, b: &Self) -> Self {
        Self { fr: self.fr.add(&b.fr) }
    }

    fn sub(&self, b: &Self) -> Self {
        Self { fr: self.fr.sub(&b.fr) }
    }

    fn eucl_inverse(&self) -> Self {
        Fr::inv(&mut self.fr, &self.fr);
        self.to_owned()
    }

    fn negate(&self) -> Self {
        Fr::neg(&mut self.fr, &self.fr);
        self.to_owned()
    }

    fn inverse(&self) -> Self {
        Fr::inv(&mut self.fr, &self.fr);
        self.to_owned()
    }

    fn pow(&self, n: usize) -> Self {
        let mut tmp = *self;
        let mut out = Self::one();
        let mut n2 = n;

        loop {
            if n2 & 1 == 1 {
                out = out.mul(&tmp);
            }
            n2 >>= 1;
            if n2 == 0 {
                break;
            }
            tmp = tmp.sqr();
        }

        out
    }

    fn div(&self, b: &Self) -> Result<Self, String> {
        if <Fr>::is_zero(&b.fr) {
            return Err("Cannot divide by zero".to_string());
        }
        let tmp = b.eucl_inverse();
        let out = self.mul(&tmp);
        Ok(out)
    }

    fn equals(&self, b: &Self) -> bool {
        self.fr == b.fr
    }
}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct MclG1 {
    pub proj: G1,
}

impl MclG1 {
    pub const fn from_blst_p1(p1: blst_p1) -> Self {
        Self {
            proj: blst_p1_into_mcl_g1projective(&p1),
        }
    }

    pub const fn to_blst_p1(&self) -> blst_p1 {
        mcl_g1projective_into_blst_p1(self.proj)
    }
    pub const fn from_g1_projective(proj: G1) -> Self {
        Self { proj }
    }

    fn affine_to_projective(p: G1) -> Self {
        Self {
            proj: p,
        }
    }
    pub fn converter(points: &[G1]) -> Vec<G1> {
        let mut result = Vec::new();

        for MclG1 in points {
            result.push(MclG1.proj);
        }
        result
    }
}

impl From<blst_p1> for MclG1 {
    fn from(p1: blst_p1) -> Self {
        let proj = blst_p1_into_mcl_g1projective(&p1);
        Self { proj }
    }
}

impl KzgG1 for MclG1 {
    fn identity() -> Self {
        Self{
            proj: G1_IDENTITY
        }
    }

    fn generator() -> Self {
        Self{
            proj: G1_GENERATOR
        }
    }

    fn negative_generator() -> Self {
        Self{
            proj: G1_NEGATIVE_GENERATOR
        }
    }

    #[cfg(feature = "rand")]
    fn rand() -> Self {
        let mut bytes: [u8; 128 * 3];
        let mut rng = rand::thread_rng();
        rand::RngCore::fill_bytes(&mut rng, &mut bytes);
        let mut ret: G1;
        ret.set_hash_of(&bytes);
        Self {
            proj: ret,
        }
    }

    #[allow(clippy::bind_instead_of_map)]
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_G1,
                    bytes.len()
                )
            })
            .and_then(|bytes: &[u8; BYTES_PER_G1]| {
                let affine: CtOption<G1>;
                G1::deserialize(&mut affine.value, bytes);
                match affine.into() {
                    Some(x) => Ok(MclG1::affine_to_projective(x)),
                    None => Err("Failed to deserialize G1: Affine not available".to_string()),
                }
            })
    }

    fn from_hex(hex: &str) -> Result<Self, String> {
        let bytes = hex::decode(&hex[2..]).unwrap();
        Self::from_bytes(&bytes)
    }

    fn to_bytes(&self) -> [u8; 48] {
        let g1_affine = G1::from(self.proj);
        g1_affine.to_compressed()
    }
    //zyme
    fn add_or_dbl(&mut self, b: &Self) -> Self {
        Self {
            proj: self.proj + b.proj,
        }
    }
    fn is_inf(&self) -> bool {
        bool::from(self.proj.is_identity())
    }
    fn is_valid(&self) -> bool {
        bool::from(self.proj.is_on_curve())
    }

    fn dbl(&self) -> Self {
        Self {
            proj: self.proj.double(),
        }
    }
    fn add(&self, b: &Self) -> Self {
        Self {
            proj: self.proj + b.proj,
        }
    }

    fn sub(&self, b: &Self) -> Self {
        Self {
            proj: self.proj.sub(&b.proj),
        }
    }

    fn equals(&self, b: &Self) -> bool {
        self.proj.eq(&b.proj)
    }
}

impl G1Mul<MclFr> for MclG1 {
    fn mul(&self, b: &MclFr) -> Self {
        Self {
            proj: self.proj.mul(b.fr),
        }
    }

    fn g1_lincomb(points: &[Self], Frs: &[MclFr], len: usize) -> Self {
        let mut out = Self::default();
        g1_linear_combination(&mut out, points, Frs, len);
        out
    }
}

impl PairingVerify<MclG1, MclG2> for MclG1 {
    fn verify(a1: &MclG1, a2: &MclG2, b1: &MclG1, b2: &MclG2) -> bool {
        pairings_verify(a1, a2, b1, b2)
    }
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct MclG2 {
    pub proj: G2,
}

impl MclG2 {
    pub const fn from_blst_p2(p2: blst::blst_p2) -> Self {
        Self {
            proj: blst_p2_into_mcl_g2projective(&p2),
        }
    }
    pub const fn from_g2_projective(proj: G2) -> Self {
        Self { proj }
    }
    pub const fn to_blst_p2(&self) -> blst::blst_p2 {
        mcl_fr_into_blst_fr(self.proj)
    }
}

impl KzgG2 for MclG2 {
    fn generator() -> Self {
        G2_GENERATOR
    }

    fn negative_generator() -> Self {
        G2_NEGATIVE_GENERATOR
    }

    #[allow(clippy::bind_instead_of_map)]
    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        bytes
            .try_into()
            .map_err(|_| {
                format!(
                    "Invalid byte length. Expected {}, got {}",
                    BYTES_PER_G2,
                    bytes.len()
                )
            })
            .and_then(|bytes: &[u8; BYTES_PER_G2]| {
                let affine = G2::from_compressed(bytes).unwrap();
                Ok(MclG2::from_g2_projective(G2::from(affine)))
            })
    }

    fn to_bytes(&self) -> [u8; 96] {
        let g2_affine = G2::from(self.proj);
        g2_affine.to_compressed()
    }

    fn add_or_dbl(&mut self, b: &Self) -> Self {
        Self {
            proj: self.proj + b.proj,
        }
    }

    fn dbl(&self) -> Self {
        Self {
            proj: self.proj.double(),
        }
    }

    fn sub(&self, b: &Self) -> Self {
        Self {
            proj: self.proj - b.proj,
        }
    }

    fn equals(&self, b: &Self) -> bool {
        self.proj.eq(&b.proj)
    }
}

impl G2Mul<MclFr> for MclG2 {
    fn mul(&self, b: &MclFr) -> Self {
        // FIXME: Is this right?
        Self {
            proj: self.proj.mul(b.fr),
        }
    }
}

impl Default for ZFFTSettings {
    fn default() -> Self {
        Self {
            max_width: 0,
            root_of_unity: MclFr::zero(),
            expanded_roots_of_unity: Vec::new(),
            reverse_roots_of_unity: Vec::new(),
            roots_of_unity: Vec::new(),
        }
    }
}

impl FFTSettings<MclFr> for ZFFTSettings {
    fn new(scale: usize) -> Result<Self, String> {
        if scale >= SCALE2_ROOT_OF_UNITY.len() {
            return Err(String::from(
                "Scale is expected to be within root of unity matrix row size",
            ));
        }

        // max_width = 2 ^ max_scale
        let max_width: usize = 1 << scale;
        let root_of_unity = MclFr::from_u64_arr(&SCALE2_ROOT_OF_UNITY[scale]);

        // create max_width of roots & store them reversed as well
        let expanded_roots_of_unity = expand_root_of_unity(&root_of_unity, max_width).unwrap();
        let mut reverse_roots_of_unity = expanded_roots_of_unity.clone();
        reverse_roots_of_unity.reverse();

        // Permute the roots of unity
        let mut roots_of_unity = expanded_roots_of_unity.clone();
        roots_of_unity.pop();
        reverse_bit_order(&mut roots_of_unity)?;

        Ok(Self {
            max_width,
            root_of_unity,
            expanded_roots_of_unity,
            reverse_roots_of_unity,
            roots_of_unity,
        })
    }

    fn get_max_width(&self) -> usize {
        self.max_width
    }

    fn get_expanded_roots_of_unity_at(&self, i: usize) -> MclFr {
        self.expanded_roots_of_unity[i]
    }

    fn get_expanded_roots_of_unity(&self) -> &[MclFr] {
        &self.expanded_roots_of_unity
    }

    fn get_reverse_roots_of_unity_at(&self, i: usize) -> MclFr {
        self.reverse_roots_of_unity[i]
    }

    fn get_reversed_roots_of_unity(&self) -> &[MclFr] {
        &self.reverse_roots_of_unity
    }

    fn get_roots_of_unity_at(&self, i: usize) -> MclFr {
        self.roots_of_unity[i]
    }

    fn get_roots_of_unity(&self) -> &[MclFr] {
        &self.roots_of_unity
    }
}

impl KZGSettings<MclFr, MclG1, MclG2, ZFFTSettings, PolyData> for ZKZGSettings {
    fn new(
        secret_g1: &[MclG1],
        secret_g2: &[MclG2],
        _length: usize,
        fft_settings: &ZFFTSettings,
    ) -> Result<ZKZGSettings, String> {
        Ok(Self {
            secret_g1: secret_g1.to_vec(),
            secret_g2: secret_g2.to_vec(),
            fs: fft_settings.clone(),
        })
    }

    fn commit_to_poly(&self, p: &PolyData) -> Result<MclG1, String> {
        if p.coeffs.len() > self.secret_g1.len() {
            return Err(String::from("Polynomial is longer than secret g1"));
        }

        let mut out = MclG1::default();
        g1_linear_combination(&mut out, &self.secret_g1, &p.coeffs, p.coeffs.len());

        Ok(out)
    }

    fn compute_proof_single(&self, p: &PolyData, x: &MclFr) -> Result<MclG1, String> {
        if p.coeffs.is_empty() {
            return Err(String::from("Polynomial must not be empty"));
        }

        // `-(x0^n)`, where `n` is `1`
        let divisor_0 = x.negate();

        // Calculate `q = p / (x^n - x0^n)` for our reduced case (see `compute_proof_multi` for
        // generic implementation)
        let mut out_coeffs = Vec::from(&p.coeffs[1..]);
        for i in (1..out_coeffs.len()).rev() {
            let tmp = out_coeffs[i].mul(&divisor_0);
            out_coeffs[i - 1] = out_coeffs[i - 1].sub(&tmp);
        }

        let q = PolyData { coeffs: out_coeffs };
        let ret = self.commit_to_poly(&q)?;
        Ok(ret)
    }

    fn check_proof_single(&self, com: &MclG1, proof: &MclG1, x: &MclFr, y: &MclFr) -> Result<bool, String> {
        let x_g2 = G2_GENERATOR.mul(x);
        let s_minus_x: MclG2 = self.secret_g2[1].sub(&x_g2);
        let y_g1 = G1_GENERATOR.mul(y);
        let commitment_minus_y: MclG1 = com.sub(&y_g1);

        Ok(pairings_verify(
            &commitment_minus_y,
            &G2_GENERATOR,
            proof,
            &s_minus_x,
        ))
    }

    fn compute_proof_multi(&self, p: &PolyData, x: &MclFr, n: usize) -> Result<MclG1, String> {
        if p.coeffs.is_empty() {
            return Err(String::from("Polynomial must not be empty"));
        }

        if !n.is_power_of_two() {
            return Err(String::from("n must be a power of two"));
        }

        // Construct x^n - x0^n = (x - x0.w^0)(x - x0.w^1)...(x - x0.w^(n-1))
        let mut divisor = PolyData {
            coeffs: Vec::with_capacity(n + 1),
        };

        // -(x0^n)
        let x_pow_n = x.pow(n);

        divisor.coeffs.push(x_pow_n.negate());

        // Zeros
        for _ in 1..n {
            divisor.coeffs.push(MclFr { fr: Fr::zero() });
        }

        // x^n
        divisor.coeffs.push(MclFr { fr: Fr::one() });

        let mut new_polina = p.clone();

        // Calculate q = p / (x^n - x0^n)
        // let q = p.div(&divisor).unwrap();
        let q = new_polina.div(&divisor)?;
        let ret = self.commit_to_poly(&q)?;
        Ok(ret)
    }

    fn check_proof_multi(
        &self,
        com: &MclG1,
        proof: &MclG1,
        x: &MclFr,
        ys: &[MclFr],
        n: usize,
    ) -> Result<bool, String> {
        if !n.is_power_of_two() {
            return Err(String::from("n is not a power of two"));
        }

        // Interpolate at a coset.
        let mut interp = PolyData {
            coeffs: self.fs.fft_fr(ys, true)?,
        };

        let inv_x = x.inverse(); // Not euclidean?
        let mut inv_x_pow = inv_x;
        for i in 1..n {
            interp.coeffs[i] = interp.coeffs[i].mul(&inv_x_pow);
            inv_x_pow = inv_x_pow.mul(&inv_x);
        }

        // [x^n]_2
        let x_pow = inv_x_pow.inverse();

        let xn2 = G2_GENERATOR.mul(&x_pow);

        // [s^n - x^n]_2
        let xn_minus_yn = self.secret_g2[n].sub(&xn2);

        // [interpolation_polynomial(s)]_1
        let is1 = self.commit_to_poly(&interp).unwrap();

        // [commitment - interpolation_polynomial(s)]_1 = [commit]_1 - [interpolation_polynomial(s)]_1
        let commit_minus_interp = com.sub(&is1);
        let ret = pairings_verify(&commit_minus_interp, &G2_GENERATOR, proof, &xn_minus_yn);

        Ok(ret)
    }

    fn get_expanded_roots_of_unity_at(&self, i: usize) -> MclFr {
        self.fs.get_expanded_roots_of_unity_at(i)
    }

    fn get_roots_of_unity_at(&self, i: usize) -> MclFr {
        self.fs.get_roots_of_unity_at(i)
    }

    fn get_fft_settings(&self) -> &ZFFTSettings {
        &self.fs
    }

    fn get_g1_secret(&self) -> &[MclG1] {
        &self.secret_g1
    }

    fn get_g2_secret(&self) -> &[MclG2] {
        &self.secret_g2
    }
}
