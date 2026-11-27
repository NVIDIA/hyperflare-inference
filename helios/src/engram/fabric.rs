/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Engram tables held in the RAM of a store host and read over one-sided RDMA READ.
//!
//! Fabric storage replaces only the row cache's miss read. The cache still coalesces
//! duplicate IDs and serves hits; each table worker thread reads its misses from the store
//! into the cache's registered miss workspace before the staged rows are uploaded, exactly
//! where the disk reader would have read them. Staging frames, the per-rank frame handshake
//! and the captured graphs are unchanged.
//!
//! A store that cannot be reached is a transient condition: the affected gather fails with
//! [`AdmissionError::EngramUnavailable`], the request that consumed it fails, the layer stays
//! usable, and the session reconnects in the background. Admission rejects new requests until
//! every session is ready again. A store that answers with the wrong core, a plane layout that
//! disagrees, or a protocol violation is fatal and poisons the layer like a disk read error.

mod plan;
mod protocol;

#[cfg(feature = "engram-rdma")]
mod client;
#[cfg(feature = "engram-rdma")]
pub mod store;
#[cfg(feature = "engram-rdma")]
mod verbs;

use crate::{AdmissionError, Error};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

/// Queue pairs between one serving process and its store when the flag is absent.
pub const DEFAULT_LANES: usize = 4;
pub const MAX_LANES: usize = protocol::MAX_LANES;
pub const DEFAULT_TIMEOUT: Duration = Duration::from_millis(5000);
pub(crate) const NO_RDMA: &str = "fabric embedding-table storage requires a Flare build with the engram-rdma feature; this build has no RDMA support";

/// How a serving process reaches its Engram store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    /// `HOST:PORT` of the store's control listener.
    pub address: String,
    pub lanes: usize,
    /// Bound on connecting, each control exchange, and each gather's RDMA reads.
    pub timeout: Duration,
}

/// The identity and rank a session presents in `HELLO`.
#[derive(Clone)]
pub(crate) struct Connect {
    pub endpoint: Endpoint,
    /// SHA-256 of the signed core envelope; the store refuses any other core.
    pub identity: [u8; 32],
    pub rank: u32,
    pub world: u32,
    pub status: Arc<Status>,
}

/// Failure that leaves the layer usable: the request fails and admission waits for the store.
pub(crate) fn unavailable(message: impl Into<String>) -> Error {
    Error::admission(AdmissionError::EngramUnavailable, message)
}

/// Whether a gather error leaves the layer usable. Every other reader error poisons it.
pub(crate) fn transient(error: &Error) -> bool {
    matches!(error, Error::Admission(AdmissionError::EngramUnavailable(_)))
}

/// Store reachability for every session of one serving process. Admission accepts requests
/// only while every session is ready; health reports this state without failing.
pub struct Status {
    endpoint: String,
    sessions: AtomicUsize,
    ready: AtomicUsize,
    reconnects: AtomicU64,
    last_error: Mutex<Option<String>>,
    failed: Mutex<Option<String>>,
}

/// One health sample of [`Status`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusSnapshot {
    /// `ready`, `down`, or `failed`.
    pub state: &'static str,
    pub endpoint: String,
    pub last_error: Option<String>,
    pub reconnects: u64,
}

impl Status {
    pub(crate) fn new(endpoint: &str) -> Arc<Self> {
        Arc::new(Self {
            endpoint: endpoint.to_owned(),
            sessions: AtomicUsize::new(0),
            ready: AtomicUsize::new(0),
            reconnects: AtomicU64::new(0),
            last_error: Mutex::new(None),
            failed: Mutex::new(None),
        })
    }

    pub fn ready(&self) -> bool {
        let sessions = self.sessions.load(Ordering::Acquire);
        sessions != 0
            && self.ready.load(Ordering::Acquire) == sessions
            && self.failed.lock().unwrap_or_else(|e| e.into_inner()).is_none()
    }

    pub fn snapshot(&self) -> StatusSnapshot {
        let failed = self.failed.lock().unwrap_or_else(|e| e.into_inner()).clone();
        StatusSnapshot {
            state: if failed.is_some() {
                "failed"
            } else if self.ready() {
                "ready"
            } else {
                "down"
            },
            endpoint: self.endpoint.clone(),
            last_error: failed
                .or_else(|| self.last_error.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            reconnects: self.reconnects.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn reconnects(&self) -> u64 {
        self.reconnects.load(Ordering::Relaxed)
    }

    /// A fatal session failure, after which gathers poison their layers.
    pub(crate) fn failure(&self) -> Option<String> {
        self.failed.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    #[cfg_attr(not(feature = "engram-rdma"), allow(dead_code))]
    fn register(&self) {
        self.sessions.fetch_add(1, Ordering::AcqRel);
    }

    #[cfg_attr(not(feature = "engram-rdma"), allow(dead_code))]
    fn unregister(&self, ready: bool) {
        if ready {
            self.ready.fetch_sub(1, Ordering::AcqRel);
        }
        self.sessions.fetch_sub(1, Ordering::AcqRel);
    }

    #[cfg_attr(not(feature = "engram-rdma"), allow(dead_code))]
    fn up(&self, reconnect: bool) {
        if reconnect {
            self.reconnects.fetch_add(1, Ordering::Relaxed);
        }
        self.ready.fetch_add(1, Ordering::AcqRel);
    }

    #[cfg_attr(not(feature = "engram-rdma"), allow(dead_code))]
    fn down(&self, was_ready: bool, reason: String) {
        if was_ready {
            self.ready.fetch_sub(1, Ordering::AcqRel);
        }
        *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(reason);
    }

    #[cfg_attr(not(feature = "engram-rdma"), allow(dead_code))]
    fn fail(&self, reason: String) {
        self.failed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert(reason);
    }
}

#[cfg(feature = "engram-rdma")]
pub(crate) use client::{Reader, Session};

/// Builds without `engram-rdma` keep these types so the reader compiles, but neither can be
/// constructed: [`Session::connect`] always fails and a [`Reader`] needs a session.
#[cfg(not(feature = "engram-rdma"))]
mod unsupported {
    use super::{Connect, NO_RDMA};
    use crate::{Result, engram::{Plane, Stats, reader::Resident}};
    use std::{convert::Infallible, sync::Arc};

    pub(crate) struct Session(Infallible);

    impl Session {
        pub(crate) fn connect(_connect: Connect, _tables: &[[Plane; 2]]) -> Result<Arc<Self>> {
            Err(NO_RDMA.into())
        }
    }

    pub(crate) struct Reader(Infallible);

    impl Reader {
        pub(crate) fn new(
            session: Arc<Session>,
            _table: usize,
            _resident_scales: Option<Resident>,
            _stats: Arc<Stats>,
        ) -> Self {
            match session.0 {}
        }
        pub(crate) fn register(&mut self, _workspace: &mut [u8]) -> Result<()> {
            match self.0 {}
        }
        pub(crate) fn gather(&mut self, _planes: &[Plane; 2], _ids: &[i64], _output: &mut [u8]) -> Result<()> {
            match self.0 {}
        }
    }
}

#[cfg(not(feature = "engram-rdma"))]
pub(crate) use unsupported::{Reader, Session};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_waits_for_every_session_and_a_fatal_failure_is_reported_first() {
        let status = Status::new("10.0.0.5:7700");
        assert!(!status.ready());
        assert_eq!(status.snapshot().state, "down");
        status.register();
        status.register();
        status.up(false);
        assert!(!status.ready());
        status.up(false);
        assert!(status.ready());
        assert_eq!(status.snapshot().state, "ready");
        status.down(true, "heartbeat timeout".into());
        let snapshot = status.snapshot();
        assert_eq!((snapshot.state, snapshot.last_error.as_deref()), ("down", Some("heartbeat timeout")));
        status.up(true);
        assert!(status.ready());
        assert_eq!(status.snapshot().reconnects, 1);
        status.fail("store holds another core".into());
        status.fail("a later failure does not replace the first".into());
        let snapshot = status.snapshot();
        assert!(!status.ready());
        assert_eq!(snapshot.state, "failed");
        assert_eq!(snapshot.last_error.as_deref(), Some("store holds another core"));
        status.unregister(true);
        status.unregister(true);
        assert!(!status.ready());
    }

    #[test]
    fn only_store_unavailability_is_transient() {
        assert!(transient(&unavailable("embedding table store is unreachable")));
        assert!(!transient(&Error::from("Engram row exceeds its table")));
        assert!(!transient(&Error::admission(AdmissionError::Unavailable, "runtime is unavailable")));
    }
}
