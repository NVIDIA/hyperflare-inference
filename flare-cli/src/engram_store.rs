// SPDX-License-Identifier: Apache-2.0
//! `flare engram-store`: serve a verified core's Engram embedding tables over RDMA.

use crate::cli::EngramStore;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "engram-rdma")]
pub fn run(args: EngramStore) -> Result<()> {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Duration,
    };
    let mut progress = flare_helios::startup::StartupProgress::stderr();
    let (core, _publisher) = crate::trust::load_store(&args.core, &args, &mut progress)?;
    progress.finish_animation();
    // Installed after the publisher prompt and before loading: a stop request while planes load
    // and register is honored when loading returns, through the store's drop, instead of
    // ending the process mid-registration.
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        libc::signal(libc::SIGINT, crate::stop as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, crate::stop as *const () as libc::sighandler_t);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let store = flare_helios::engram::fabric::store::start(
        core,
        &flare_helios::engram::fabric::store::Options { listen: args.listen },
        stop.clone(),
        &mut |line| eprintln!("{line}"),
    )?;
    eprintln!(
        "Engram store listening on {}; stop with Ctrl-C or SIGTERM.",
        store.address
    );
    while !crate::STOP.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(250));
    }
    stop.store(true, Ordering::Release);
    drop(store);
    Ok(())
}

#[cfg(not(feature = "engram-rdma"))]
pub fn run(_args: EngramStore) -> Result<()> {
    Err("flare engram-store requires a Flare build with the engram-rdma feature; this build has no RDMA support".into())
}
