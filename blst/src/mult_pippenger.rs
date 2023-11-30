// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
extern crate alloc;
extern crate blst;
extern crate core;
extern crate threadpool;
use alloc::{boxed::Box, vec, vec::Vec};
use blst::{
    blst_p1, blst_p1_add_or_double, blst_p1_affine, blst_p1_double, blst_p1s_add,
    blst_p1s_mult_pippenger_scratch_sizeof, blst_p1s_to_affine,
};
use core::mem::MaybeUninit;
use core::num::Wrapping;
use core::ops::{Index, IndexMut};
use core::ptr;
use core::slice::SliceIndex;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc::channel, Arc, Barrier};

struct Tile {
    x: usize,
    dx: usize,
    y: usize,
    dy: usize,
}

trait ThreadPoolExt {
    fn joined_execute<'any, F>(&self, job: F)
    where
        F: FnOnce() + Send + 'any;
}

use core::mem::transmute;
use std::sync::{Mutex, Once};
use threadpool::ThreadPool;

use crate::msm::{p1s_tile_pippenger_pub, pippenger};
use crate::types::g1::FsG1;

pub fn da_pool() -> ThreadPool {
    static INIT: Once = Once::new();
    static mut POOL: *const Mutex<ThreadPool> = 0 as *const Mutex<ThreadPool>;

    INIT.call_once(|| {
        let pool = Mutex::new(ThreadPool::default());
        unsafe { POOL = transmute(Box::new(pool)) };
    });
    unsafe { (*POOL).lock().unwrap().clone() }
}

type Thunk<'any> = Box<dyn FnOnce() + Send + 'any>;

impl ThreadPoolExt for ThreadPool {
    fn joined_execute<'scope, F>(&self, job: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        // Bypass 'lifetime limitations by brute force. It works,
        // because we explicitly join the threads...
        self.execute(unsafe { transmute::<Thunk<'scope>, Thunk<'static>>(Box::new(job)) })
    }
}

// Minimalist core::cell::Cell stand-in, but with Sync marker, which
// makes it possible to pass it to multiple threads. It works, because
// *here* each Cell is written only once and by just one thread.
#[repr(transparent)]
struct Cell<T: ?Sized> {
    value: T,
}
unsafe impl<T: ?Sized + Sync> Sync for Cell<T> {}
impl<T> Cell<T> {
    pub fn as_ptr(&self) -> *mut T {
        &self.value as *const T as *mut T
    }
}

//MULT IMPL
pub struct P1Affines {
    points: Vec<blst_p1_affine>,
}

impl<I: SliceIndex<[blst_p1_affine]>> Index<I> for P1Affines {
    type Output = I::Output;

    #[inline]
    fn index(&self, i: I) -> &Self::Output {
        &self.points[i]
    }
}
impl<I: SliceIndex<[blst_p1_affine]>> IndexMut<I> for P1Affines {
    #[inline]
    fn index_mut(&mut self, i: I) -> &mut Self::Output {
        &mut self.points[i]
    }
}

impl P1Affines {
    #[inline]
    pub fn as_slice(&self) -> &[blst_p1_affine] {
        self.points.as_slice()
    }

    pub fn from(points: &[FsG1]) -> Self {
        let npoints = points.len();
        let mut ret = Self {
            points: Vec::with_capacity(npoints),
        };
        unsafe { ret.points.set_len(npoints) };

        let pool = da_pool();
        let ncpus = pool.max_count();
        if ncpus < 2 || npoints < 768 {
            let p: [*const blst_p1; 2] = [&points[0].0, ptr::null()];
            unsafe { blst_p1s_to_affine(&mut ret.points[0], &p[0], npoints) };
            return ret;
        }

        let mut nslices = (npoints + 511) / 512;
        nslices = core::cmp::min(nslices, ncpus);
        let wg = Arc::new((Barrier::new(2), AtomicUsize::new(nslices)));

        let (mut delta, mut rem) = (npoints / nslices + 1, Wrapping(npoints % nslices));
        let mut x = 0usize;
        while x < npoints {
            let out = &mut ret.points[x];
            let inp = &points[x].0;

            delta -= (rem == Wrapping(0)) as usize;
            rem -= Wrapping(1);
            x += delta;

            let wg = wg.clone();
            pool.joined_execute(move || {
                let p: [*const blst_p1; 2] = [inp, ptr::null()];
                unsafe { blst_p1s_to_affine(out, &p[0], delta) };
                if wg.1.fetch_sub(1, Ordering::AcqRel) == 1 {
                    wg.0.wait();
                }
            });
        }
        wg.0.wait();

        ret
    }

    pub fn mult(&self, scalars: &[u8], nbits: usize) -> blst_p1 {
        let npoints = self.points.len();
        let nbytes = (nbits + 7) / 8;

        if scalars.len() < nbytes * npoints {
            panic!("scalars length mismatch");
        }

        let pool = da_pool();
        let ncpus = pool.max_count();
        if ncpus < 2 || npoints < 32 {
            let p: [*const blst_p1_affine; 2] = [&self.points[0], ptr::null()];
            let s: [*const u8; 2] = [&scalars[0], ptr::null()];

            unsafe {
                let mut scratch: Vec<u64> =
                    Vec::with_capacity(blst_p1s_mult_pippenger_scratch_sizeof(npoints) / 8);
                #[allow(clippy::uninit_vec)]
                scratch.set_len(scratch.capacity());
                let mut ret = <blst_p1>::default();
                pippenger(&mut ret, &p[0], npoints, &s[0], nbits, &mut scratch[0], 0);
                return ret;
            }
        }

        let (nx, ny, window) = breakdown(nbits, pippenger_window_size(npoints), ncpus);

        // |grid[]| holds "coordinates" and place for result
        let mut grid: Vec<(Tile, Cell<blst_p1>)> = Vec::with_capacity(nx * ny);
        #[allow(clippy::uninit_vec)]
        unsafe {
            grid.set_len(grid.capacity())
        };
        let dx = npoints / nx;
        let mut y = window * (ny - 1);
        let mut total = 0usize;

        while total < nx {
            grid[total].0.x = total * dx;
            grid[total].0.dx = dx;
            grid[total].0.y = y;
            grid[total].0.dy = nbits - y;
            total += 1;
        }
        grid[total - 1].0.dx = npoints - grid[total - 1].0.x;
        while y != 0 {
            y -= window;
            for i in 0..nx {
                grid[total].0.x = grid[i].0.x;
                grid[total].0.dx = grid[i].0.dx;
                grid[total].0.y = y;
                grid[total].0.dy = window;
                total += 1;
            }
        }
        let grid = &grid[..];

        let points = &self.points[..];
        let sz = unsafe { blst_p1s_mult_pippenger_scratch_sizeof(0) / 8 };

        let mut row_sync: Vec<AtomicUsize> = Vec::with_capacity(ny);
        row_sync.resize_with(ny, Default::default);
        let row_sync = Arc::new(row_sync);
        let counter = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = channel();
        let n_workers = core::cmp::min(ncpus, total);
        for _ in 0..n_workers {
            let tx = tx.clone();
            let counter = counter.clone();
            let row_sync = row_sync.clone();

            pool.joined_execute(move || {
                let mut scratch = vec![0u64; sz << (window - 1)];
                let mut p: [*const blst_p1_affine; 2] = [ptr::null(), ptr::null()];
                let mut s: [*const u8; 2] = [ptr::null(), ptr::null()];

                loop {
                    let work = counter.fetch_add(1, Ordering::Relaxed);
                    if work >= total {
                        break;
                    }
                    let x = grid[work].0.x;
                    let y = grid[work].0.y;

                    p[0] = &points[x];
                    s[0] = &scalars[x * nbytes];
                    unsafe {
                        p1s_tile_pippenger_pub(
                            grid[work].1.as_ptr(),
                            &p[0],
                            grid[work].0.dx,
                            &s[0],
                            nbits,
                            &mut scratch[0],
                            y,
                            window,
                        );
                    }
                    if row_sync[y / window].fetch_add(1, Ordering::AcqRel) == nx - 1 {
                        tx.send(y).expect("disaster");
                    }
                }
            });
        }

        let mut ret = <blst_p1>::default();
        let mut rows = vec![false; ny];
        let mut row = 0usize;
        for _ in 0..ny {
            let mut y = rx.recv().unwrap();
            rows[y / window] = true;
            while grid[row].0.y == y {
                while row < total && grid[row].0.y == y {
                    unsafe {
                        blst_p1_add_or_double(&mut ret, &ret, grid[row].1.as_ptr());
                    }
                    row += 1;
                }
                if y == 0 {
                    break;
                }
                for _ in 0..window {
                    unsafe { blst_p1_double(&mut ret, &ret) };
                }
                y -= window;
                if !rows[y / window] {
                    break;
                }
            }
        }
        ret
    }

    pub fn add(&self) -> blst_p1 {
        let npoints = self.points.len();

        let pool = da_pool();
        let ncpus = pool.max_count();
        if ncpus < 2 || npoints < 384 {
            let p: [*const _; 2] = [&self.points[0], ptr::null()];
            let mut ret = <blst_p1>::default();
            unsafe { blst_p1s_add(&mut ret, &p[0], npoints) };
            return ret;
        }

        let (tx, rx) = channel();
        let counter = Arc::new(AtomicUsize::new(0));
        let nchunks = (npoints + 255) / 256;
        let chunk = npoints / nchunks + 1;

        let n_workers = core::cmp::min(ncpus, nchunks);
        for _ in 0..n_workers {
            let tx = tx.clone();
            let counter = counter.clone();

            pool.joined_execute(move || {
                let mut acc = <blst_p1>::default();
                let mut chunk = chunk;
                let mut p: [*const _; 2] = [ptr::null(), ptr::null()];

                loop {
                    let work = counter.fetch_add(chunk, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    p[0] = &self.points[work];
                    if work + chunk > npoints {
                        chunk = npoints - work;
                    }
                    unsafe {
                        let mut t = MaybeUninit::<blst_p1>::uninit();
                        blst_p1s_add(t.as_mut_ptr(), &p[0], chunk);
                        blst_p1_add_or_double(&mut acc, &acc, t.as_ptr());
                    };
                }
                tx.send(acc).expect("disaster");
            });
        }

        let mut ret = rx.recv().unwrap();
        for _ in 1..n_workers {
            unsafe { blst_p1_add_or_double(&mut ret, &ret, &rx.recv().unwrap()) };
        }

        ret
    }
}

fn num_bits(l: usize) -> usize {
    8 * core::mem::size_of_val(&l) - l.leading_zeros() as usize
}

fn breakdown(nbits: usize, window: usize, ncpus: usize) -> (usize, usize, usize) {
    let mut nx: usize;
    let mut wnd: usize;

    if nbits > window * ncpus {
        nx = 1;
        wnd = num_bits(ncpus / 4);
        if (window + wnd) > 18 {
            wnd = window - wnd;
        } else {
            wnd = (nbits / window + ncpus - 1) / ncpus;
            if (nbits / (window + 1) + ncpus - 1) / ncpus < wnd {
                wnd = window + 1;
            } else {
                wnd = window;
            }
        }
    } else {
        nx = 2;
        wnd = window - 2;
        while (nbits / wnd + 1) * nx < ncpus {
            nx += 1;
            wnd = window - num_bits(3 * nx / 2);
        }
        nx -= 1;
        wnd = window - num_bits(3 * nx / 2);
    }
    let ny = nbits / wnd + 1;
    wnd = nbits / ny + 1;

    (nx, ny, wnd)
}

fn pippenger_window_size(npoints: usize) -> usize {
    let wbits = num_bits(npoints);

    if wbits > 13 {
        return wbits - 4;
    }
    if wbits > 5 {
        return wbits - 3;
    }
    2
}
