// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
extern crate alloc;
extern crate blst;
extern crate core;
extern crate threadpool;
use alloc::{boxed::Box, vec, vec::Vec};
use blst::{blst_p1, blst_p1_add_or_double, blst_p1_affine};
use core::ops::{Index, IndexMut};
use core::slice::SliceIndex;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc::channel, Arc};

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

use super::bgmw::{bgmw_tile_pub, p1_integrate_buckets};
use super::pippenger::P1XYZZ;
use super::BGMWTable;

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

pub fn multiply(
    table: &BGMWTable,
    br: (usize, usize, usize),
    npoints: usize,
    scalars: &[u8],
    nbits: usize,
    pool: ThreadPool,
) -> blst_p1 {
    let ncpus = pool.max_count();
    let nbytes = (nbits + 7) / 8;
    let (nx, ny, window) = br;

    // |grid[]| holds "coordinates" and place for result
    let mut grid: Vec<Tile> = Vec::with_capacity(nx * ny);
    #[allow(clippy::uninit_vec)]
    unsafe {
        grid.set_len(grid.capacity())
    };
    let dx = npoints / nx;
    let mut y = window * (ny - 1);
    let mut total = 0usize;

    while total < nx {
        grid[total].x = total * dx;
        grid[total].dx = dx;
        grid[total].y = y;
        grid[total].dy = nbits - y;
        total += 1;
    }
    grid[total - 1].dx = npoints - grid[total - 1].x;
    while y != 0 {
        y -= window;
        for i in 0..nx {
            grid[total].x = grid[i].x;
            grid[total].dx = grid[i].dx;
            grid[total].y = y;
            grid[total].dy = window;
            total += 1;
        }
    }
    let grid = &grid[..];

    let mut row_sync: Vec<AtomicUsize> = Vec::with_capacity(ny);
    row_sync.resize_with(ny, Default::default);
    let counter = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = channel();
    let n_workers = core::cmp::min(ncpus, total);

    let mut results: Vec<Cell<blst_p1>> = Vec::with_capacity(n_workers);
    #[allow(clippy::uninit_vec)]
    unsafe {
        results.set_len(results.capacity())
    }

    let results = &results[..];

    #[allow(clippy::needless_range_loop)]
    for worker_index in 0..n_workers {
        let tx = tx.clone();
        let counter = counter.clone();

        pool.joined_execute(move || {
            let mut scratch = vec![P1XYZZ::default(); 1usize << (window - 1)];

            loop {
                let work = counter.fetch_add(1, Ordering::Relaxed);
                if work >= total {
                    p1_integrate_buckets(
                        unsafe { results[worker_index].as_ptr().as_mut() }.unwrap(),
                        &scratch,
                        window - 1,
                    );
                    tx.send(worker_index).expect("disaster");

                    break;
                }
                let x = grid[work].x;
                let y = grid[work].y;
                let dx = grid[work].dx;

                let points = &table.precomputed[((y / window) * table.numpoints + x)
                    ..((y / window) * table.numpoints + x + dx)];

                bgmw_tile_pub(
                    points,
                    dx,
                    &scalars[x * nbytes..],
                    nbits,
                    &mut scratch,
                    y,
                    window,
                );
            }
        });
    }

    let mut ret = <blst_p1>::default();
    for _ in 0..n_workers {
        let idx = rx.recv().unwrap();
        unsafe {
            blst_p1_add_or_double(&mut ret, &ret, results[idx].as_ptr() as *const blst_p1);
        }
    }
    ret
}

fn num_bits(l: usize) -> usize {
    8 * core::mem::size_of_val(&l) - l.leading_zeros() as usize
}

pub fn breakdown(nbits: usize, window: usize, ncpus: usize) -> (usize, usize, usize) {
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
