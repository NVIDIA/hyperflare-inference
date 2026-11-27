/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Serving-process side of the Engram store: one control session per table group, and the
//! miss reader that replaces the disk read under the row cache.
//!
//! A session owns the TCP control connection on a background thread: `HELLO`, the
//! queue-pair exchange, a one-second heartbeat, and reconnection with capped exponential
//! backoff. The connected queue pairs form a link that table workers borrow. A worker that
//! sees a READ fail marks the link broken; the session withdraws it, reports the store down,
//! and builds a new link. Workers never block on reconnection: without a ready link their
//! gather fails at once as unavailable.

use super::{
    Connect,
    plan,
    protocol::{
        self, Hello, HelloOk, Message, PlaneLayout, Receiver, RemoteTable, STORE_ABI, TableLayout,
        VERSION,
    },
    unavailable,
    verbs::{Access, Device, Failure, Lane, Region, Role},
};
use crate::{
    Result,
    engram::{Plane, Stats, reader::{self, Resident}},
};
use std::{
    io,
    net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs, UdpSocket},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const HEARTBEAT: Duration = Duration::from_secs(1);
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(3);
/// Read timeout of the supervising loop, which bounds how long a broken link or a stop
/// request waits to be noticed.
const POLL: Duration = Duration::from_millis(100);
const MIN_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
/// Client cap on one READ; longer runs split.
const MAX_READ: u32 = 1 << 20;
/// Empty completion polls before the table worker yields its CPU.
const SPIN: u32 = 1024;
/// Bound on flushing posted READs after a failure. A READ that neither completes nor flushes
/// may still write the miss workspace, so that outcome poisons the layer.
const FLUSH: Duration = Duration::from_secs(1);

/// Why a connection attempt failed: a store that may come back, or a disagreement that persists.
enum Attempt {
    Transient(String),
    Fatal(String),
}

fn attempt(what: &str, error: io::Error) -> Attempt {
    if error.kind() == io::ErrorKind::InvalidData {
        Attempt::Fatal(format!("{what}: {error}"))
    } else {
        Attempt::Transient(format!("{what}: {error}"))
    }
}

/// Connected queue pairs and the registered planes they read.
struct Link {
    lanes: Vec<Mutex<Lane>>,
    tables: Vec<RemoteTable>,
    max_read: usize,
    broken: Mutex<Option<String>>,
}

impl Link {
    fn broken(&self) -> Option<String> {
        self.broken.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    fn mark_broken(&self, reason: &str) {
        self.broken
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(|| reason.to_owned());
    }
}

struct Inner {
    connect: Connect,
    address: SocketAddr,
    tables: Vec<TableLayout>,
    device: Arc<Device>,
    link: Mutex<Option<Arc<Link>>>,
    stop: AtomicBool,
    signal: (Mutex<()>, Condvar),
}

impl Inner {
    fn link(&self) -> Option<Arc<Link>> {
        self.link.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Sleep for `duration` unless a stop request arrives first; true when stopping.
    fn wait(&self, duration: Duration) -> bool {
        let guard = self.signal.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self
            .signal
            .1
            .wait_timeout_while(guard, duration, |_| !self.stop.load(Ordering::Acquire));
        self.stop.load(Ordering::Acquire)
    }
}

/// One table group's connection to the store.
pub(crate) struct Session {
    inner: Arc<Inner>,
    thread: Option<JoinHandle<()>>,
}

/// The local address the kernel would use to reach `address`. A connected UDP socket selects
/// the route without sending a packet.
fn route(address: SocketAddr) -> io::Result<IpAddr> {
    let socket = UdpSocket::bind(if address.is_ipv4() { "0.0.0.0:0" } else { "[::]:0" })?;
    socket.connect(address)?;
    Ok(socket.local_addr()?.ip())
}

impl Session {
    /// Open the RDMA device that routes to the store and make the first connection attempt.
    /// A store that refuses this core fails here; an unreachable store leaves the session
    /// down and reconnecting.
    pub(crate) fn connect(connect: Connect, tables: &[[Plane; 2]]) -> Result<Arc<Self>> {
        let endpoint = &connect.endpoint;
        if !(1..=protocol::MAX_LANES).contains(&endpoint.lanes) {
            return Err("Engram store lanes must be 1 through 8".into());
        }
        let address = endpoint
            .address
            .to_socket_addrs()
            .map_err(|error| format!("resolve Engram store {}: {error}", endpoint.address))?
            .next()
            .ok_or_else(|| format!("Engram store {} has no address", endpoint.address))?;
        let local = route(address)
            .map_err(|error| format!("find the route to Engram store {address}: {error}"))?;
        let device = Device::open(local)
            .map_err(|error| format!("open the RDMA device for Engram store {address}: {error}"))?;
        let tables = tables
            .iter()
            .enumerate()
            .map(|(id, planes)| -> Result<TableLayout> {
                let layout = |plane: &Plane| -> Result<PlaneLayout> {
                    Ok(PlaneLayout {
                        row_bytes: u32::try_from(plane.row_bytes)
                            .map_err(|_| "Engram row exceeds the store protocol range")?,
                        nbytes: plane.nbytes as u64,
                    })
                };
                Ok(TableLayout {
                    id: u32::try_from(id).map_err(|_| "Engram table catalog is too large")?,
                    planes: [layout(&planes[0])?, layout(&planes[1])?],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if tables.len() > protocol::MAX_TABLES {
            return Err("Engram table catalog exceeds the store protocol limit".into());
        }
        let inner = Arc::new(Inner {
            connect,
            address,
            tables,
            device,
            link: Mutex::new(None),
            stop: AtomicBool::new(false),
            signal: (Mutex::new(()), Condvar::new()),
        });
        let status = inner.connect.status.clone();
        status.register();
        let (stream, ready) = match establish(&inner) {
            Ok((stream, link)) => {
                *inner.link.lock().unwrap() = Some(Arc::new(link));
                status.up(false);
                (Some(stream), true)
            }
            Err(Attempt::Fatal(reason)) => {
                status.unregister(false);
                return Err(format!("Engram store {address}: {reason}").into());
            }
            Err(Attempt::Transient(reason)) => {
                status.down(false, reason);
                (None, false)
            }
        };
        let worker = inner.clone();
        let thread = thread::Builder::new()
            .name("flare-engram-store".into())
            .spawn(move || run(worker, stream, ready))
            .map_err(|error| {
                status.unregister(ready);
                format!("start the Engram store session: {error}")
            })?;
        Ok(Arc::new(Self {
            inner,
            thread: Some(thread),
        }))
    }

    /// The ready link, an unavailable error while the store is down, or a fatal error after
    /// the store refused this core.
    fn link(&self) -> Result<Arc<Link>> {
        if let Some(reason) = self.inner.connect.status.failure() {
            return Err(format!("embedding table store failed: {reason}").into());
        }
        match self.inner.link() {
            Some(link) if link.broken().is_none() => Ok(link),
            _ => Err(unavailable(format!(
                "embedding table store {} is unreachable",
                self.inner.connect.endpoint.address
            ))),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::Release);
        {
            let _guard = self.inner.signal.0.lock().unwrap_or_else(|e| e.into_inner());
            self.inner.signal.1.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(inner: Arc<Inner>, mut stream: Option<TcpStream>, mut ready: bool) {
    let status = inner.connect.status.clone();
    let mut connected = ready;
    let mut backoff = MIN_BACKOFF;
    loop {
        if let Some(stream) = stream.take() {
            let reason = supervise(&inner, stream);
            if let Some(link) = inner.link.lock().unwrap_or_else(|e| e.into_inner()).take() {
                link.mark_broken(&reason);
            }
            status.down(ready, reason);
            ready = false;
            backoff = MIN_BACKOFF;
        }
        if inner.wait(backoff) {
            break;
        }
        match establish(&inner) {
            Ok((connection, link)) => {
                *inner.link.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(link));
                status.up(connected);
                connected = true;
                ready = true;
                stream = Some(connection);
            }
            Err(Attempt::Fatal(reason)) => {
                status.fail(reason);
                break;
            }
            Err(Attempt::Transient(reason)) => {
                status.down(false, reason);
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        }
    }
    if let Some(link) = inner.link.lock().unwrap_or_else(|e| e.into_inner()).take() {
        link.mark_broken("session closed");
    }
    status.unregister(ready);
}

/// Hold one connection until it fails, the link breaks, or the session stops.
fn supervise(inner: &Inner, mut stream: TcpStream) -> String {
    if let Err(error) = stream.set_read_timeout(Some(POLL)) {
        return format!("control connection failed: {error}");
    }
    let mut receiver = Receiver::default();
    let mut sequence = 0u64;
    let mut sent: Option<Instant> = None;
    let mut heard = Instant::now();
    loop {
        if inner.stop.load(Ordering::Acquire) {
            let _ = protocol::write(&mut stream, &Message::Bye);
            return "session closed".into();
        }
        if let Some(reason) = inner.link().and_then(|link| link.broken()) {
            let _ = protocol::write(&mut stream, &Message::Bye);
            return reason;
        }
        if sent.is_none_or(|at| at.elapsed() >= HEARTBEAT) {
            if let Err(error) = protocol::write(&mut stream, &Message::Heartbeat(sequence)) {
                return format!("control connection failed: {error}");
            }
            sequence += 1;
            sent = Some(Instant::now());
        }
        match receiver.poll(&mut stream) {
            Ok(Some(Message::Heartbeat(_))) => heard = Instant::now(),
            Ok(Some(Message::Bye)) => return "the store closed the session".into(),
            Ok(Some(Message::Error { code, message })) => {
                return format!("the store reported {}: {message}", code.name());
            }
            Ok(Some(_)) => return "the store sent an unexpected control message".into(),
            Ok(None) if heard.elapsed() > HEARTBEAT_TIMEOUT => {
                return format!(
                    "no heartbeat from the store for {} ms",
                    heard.elapsed().as_millis()
                );
            }
            Ok(None) => {}
            Err(error) => return format!("control connection failed: {error}"),
        }
    }
}

/// Check a `HELLO_OK` against the request. Returns the store's READ depth and the READ size.
fn validate(ok: &HelloOk, tables: &[TableLayout], lanes: u8, mtu: u32) -> std::result::Result<(u8, usize), String> {
    if ok.abi != STORE_ABI {
        return Err(format!("the store describes its planes with ABI {}; this build reads ABI {STORE_ABI}", ok.abi));
    }
    if ok.lanes == 0 || ok.lanes > lanes {
        return Err(format!("the store confirmed {} lanes for a request of {lanes}", ok.lanes));
    }
    if ok.depth == 0 || ok.max_read_bytes == 0 {
        return Err("the store accepts no RDMA READs".into());
    }
    if !protocol::valid_mtu(ok.mtu) || ok.mtu > mtu {
        return Err(format!("the store selected path MTU {} for a port MTU of {mtu}", ok.mtu));
    }
    if ok.tables.len() != tables.len() {
        return Err(format!(
            "plane_layout_mismatch: the store returned {} tables for {} requested",
            ok.tables.len(),
            tables.len()
        ));
    }
    for (remote, local) in ok.tables.iter().zip(tables) {
        if remote.id != local.id
            || remote.planes.iter().zip(&local.planes).any(|(remote, local)| {
                remote.row_bytes != local.row_bytes
                    || remote.nbytes != local.nbytes
                    || remote.address.checked_add(remote.nbytes).is_none()
            })
        {
            return Err(format!("plane_layout_mismatch: table {} differs from this core's catalog", local.id));
        }
    }
    Ok((ok.depth, ok.max_read_bytes.min(MAX_READ) as usize))
}

fn refused(code: protocol::Code, message: &str) -> Attempt {
    let reason = format!("the store refused this session ({}): {message}", code.name());
    if code.retryable() {
        Attempt::Transient(reason)
    } else {
        Attempt::Fatal(reason)
    }
}

/// One connection attempt: `HELLO`, validation, queue-pair exchange, and RTS on every lane.
fn establish(inner: &Inner) -> std::result::Result<(TcpStream, Link), Attempt> {
    let endpoint = &inner.connect.endpoint;
    let timeout = endpoint.timeout;
    inner
        .device
        .refresh()
        .map_err(|error| Attempt::Transient(format!("local RoCE interface: {error}")))?;
    let mut stream = TcpStream::connect_timeout(&inner.address, timeout)
        .map_err(|error| Attempt::Transient(format!("connect to {}: {error}", inner.address)))?;
    stream
        .set_nodelay(true)
        .and_then(|()| stream.set_read_timeout(Some(timeout)))
        .and_then(|()| stream.set_write_timeout(Some(timeout)))
        .map_err(|error| Attempt::Transient(format!("configure the control connection: {error}")))?;
    let lanes = endpoint.lanes as u8;
    let hello = Hello {
        identity: inner.connect.identity,
        rank: inner.connect.rank,
        world: inner.connect.world,
        lanes,
        mtu: inner.device.mtu,
        tables: inner.tables.clone(),
    };
    protocol::write(&mut stream, &Message::Hello(hello)).map_err(|e| attempt("send HELLO", e))?;
    let ok = match protocol::read(&mut stream).map_err(|e| attempt("read HELLO_OK", e))? {
        Message::HelloOk(ok) => ok,
        Message::Error { code, message } => return Err(refused(code, &message)),
        Message::HelloOkVersion(version) => {
            return Err(Attempt::Fatal(format!(
                "the store speaks control protocol version {version}; this build speaks version {VERSION}"
            )));
        }
        _ => return Err(Attempt::Fatal("the store answered HELLO with an unexpected message".into())),
    };
    let (depth, max_read) =
        validate(&ok, &inner.tables, lanes, inner.device.mtu).map_err(Attempt::Fatal)?;
    let depth = depth.min(inner.device.initiator_depth);
    let mut created = Vec::with_capacity(usize::from(ok.lanes));
    for _ in 0..ok.lanes {
        created.push(
            Lane::new(&inner.device, Role::Requester, depth)
                .map_err(|error| Attempt::Transient(format!("create a queue pair: {error}")))?,
        );
    }
    let local = created.iter().map(Lane::local).collect();
    protocol::write(&mut stream, &Message::QpExchange(local))
        .map_err(|e| attempt("send the queue-pair exchange", e))?;
    let peers = match protocol::read(&mut stream).map_err(|e| attempt("read the queue-pair exchange", e))? {
        Message::QpExchange(peers) if peers.len() == created.len() => peers,
        Message::Error { code, message } => return Err(refused(code, &message)),
        _ => {
            return Err(Attempt::Fatal(
                "the store answered the queue-pair exchange with an unexpected message".into(),
            ));
        }
    };
    for (lane, peer) in created.iter_mut().zip(&peers) {
        lane.connect(peer, ok.mtu, depth)
            .map_err(|error| Attempt::Transient(format!("connect a queue pair: {error}")))?;
    }
    Ok((
        stream,
        Link {
            lanes: created.into_iter().map(Mutex::new).collect(),
            tables: ok.tables,
            max_read,
            broken: Mutex::new(None),
        },
    ))
}

/// Reads one table's row-cache misses from the store.
pub(crate) struct Reader {
    session: Arc<Session>,
    table: usize,
    resident_scales: Option<Resident>,
    /// Registration of the row cache's miss workspace, the only destination of READs.
    region: Option<Region>,
    reads: Vec<plan::Read>,
    stats: Arc<Stats>,
}

impl Reader {
    pub(crate) fn new(
        session: Arc<Session>,
        table: usize,
        resident_scales: Option<Resident>,
        stats: Arc<Stats>,
    ) -> Self {
        Self {
            session,
            table,
            resident_scales,
            region: None,
            reads: Vec::new(),
            stats,
        }
    }

    /// Register the row cache's miss workspace. The cache allocates and touches it before
    /// sizing and declares this reader first, so the registration is released before the
    /// workspace is freed.
    pub(crate) fn register(&mut self, workspace: &mut [u8]) -> Result<()> {
        // SAFETY: the workspace outlives this reader's registration, as documented above.
        let region = unsafe {
            Region::register(
                &self.session.inner.device,
                workspace.as_mut_ptr(),
                workspace.len(),
                Access::LocalWrite,
            )
        }
        .map_err(|error| {
            format!(
                "register the {}-byte Engram miss workspace for RDMA: {error}; the locked-memory limit (ulimit -l) must cover every table's miss workspace",
                workspace.len()
            )
        })?;
        self.region = Some(region);
        Ok(())
    }

    pub(crate) fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        let region = self
            .region
            .as_ref()
            .ok_or("Engram store reader has no registered miss workspace")?;
        let start = (output.as_ptr() as u64)
            .checked_sub(region.address())
            .filter(|offset| offset + output.len() as u64 <= region.len() as u64)
            .ok_or("Engram miss output lies outside its registered workspace")? as usize;
        let link = self.session.link()?;
        let remote = link
            .tables
            .get(self.table)
            .ok_or("Engram store link lacks this table")?;
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let count = ids.len();
        self.reads.clear();
        let weight_plane = plan::Plane {
            row_bytes: weights,
            rkey: remote.planes[0].rkey,
            remote: remote.planes[0].address,
            local: start,
        };
        if let Some(resident) = &self.resident_scales {
            reader::gather_resident(resident, scales, ids, &mut output[count * weights..]);
            plan::plan(ids, &[weight_plane], link.max_read, &mut self.reads)?;
        } else {
            let scale_plane = plan::Plane {
                row_bytes: scales,
                rkey: remote.planes[1].rkey,
                remote: remote.planes[1].address,
                local: start + count * weights,
            };
            plan::plan(ids, &[weight_plane, scale_plane], link.max_read, &mut self.reads)?;
        }
        execute(
            &link,
            self.table,
            &self.reads,
            region,
            self.session.inner.connect.endpoint.timeout,
            &self.stats,
        )
    }
}

/// Post `reads` on the table's lane within its confirmed depth and poll to completion on the
/// calling worker thread: spin, then yield after [`SPIN`] empty polls. On any failure the lane
/// is moved to ERR and drained, so no READ can write the workspace after this returns.
fn execute(
    link: &Link,
    table: usize,
    reads: &[plan::Read],
    region: &Region,
    timeout: Duration,
    stats: &Stats,
) -> Result<()> {
    if reads.is_empty() {
        return Ok(());
    }
    let begin = Instant::now();
    let deadline = begin + timeout;
    // Concurrent table workers spread over the lanes; workers mapped to one lane take turns.
    let mut lane = link.lanes[table % link.lanes.len()]
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (base, lkey) = (region.address(), region.lkey());
    let mut next = 0;
    let mut posted_bytes = 0u64;
    let mut failure = None;
    let mut flush_deadline = None;
    let mut empty = 0u32;
    loop {
        if failure.is_none() && next < reads.len() {
            let take = lane.credit().min(reads.len() - next);
            if take > 0 {
                let before = lane.outstanding();
                let result = lane.post(&reads[next..next + take], base, lkey);
                let posted = lane.outstanding() - before;
                posted_bytes += reads[next..next + posted]
                    .iter()
                    .map(|read| u64::from(read.nbytes))
                    .sum::<u64>();
                next += posted;
                if let Err(error) = result {
                    failure = Some(Failure::Transient(format!("post RDMA READ: {error}")));
                }
            }
        }
        let progressed = match lane.poll() {
            Ok((count, error)) => {
                if let Some(error) = error {
                    failure.get_or_insert(error);
                }
                count > 0
            }
            Err(error) => {
                failure.get_or_insert(Failure::Fatal(format!("poll RDMA completions: {error}")));
                false
            }
        };
        if failure.is_none() {
            if next == reads.len() && lane.outstanding() == 0 {
                break;
            }
            if Instant::now() >= deadline {
                failure = Some(Failure::Transient(format!(
                    "Engram store READs did not complete within {} ms",
                    timeout.as_millis()
                )));
            }
        }
        if failure.is_some() {
            if flush_deadline.is_none() {
                let _ = lane.fail();
                flush_deadline = Some(Instant::now() + FLUSH);
            }
            if lane.outstanding() == 0 {
                break;
            }
            if flush_deadline.is_some_and(|at| Instant::now() >= at) {
                link.mark_broken("RDMA READs did not flush");
                return Err(format!(
                    "{} Engram store READs neither completed nor flushed; the miss workspace may still be written",
                    lane.outstanding()
                )
                .into());
            }
        }
        if progressed {
            empty = 0;
        } else {
            empty += 1;
            if empty >= SPIN {
                empty = 0;
                thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }
    stats.fabric_reads.fetch_add(next as u64, Ordering::Relaxed);
    stats.fabric_bytes.fetch_add(posted_bytes, Ordering::Relaxed);
    stats
        .fabric_wait_ns
        .fetch_add(begin.elapsed().as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
    match failure {
        None => Ok(()),
        Some(Failure::Transient(reason)) => {
            link.mark_broken(&reason);
            stats.fabric_failures.fetch_add(1, Ordering::Relaxed);
            Err(unavailable(format!("embedding table store read failed: {reason}")))
        }
        Some(Failure::Fatal(reason)) => {
            link.mark_broken(&reason);
            Err(format!("embedding table store read failed: {reason}").into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::fabric::{Endpoint, Status, protocol::RemotePlane};

    fn layouts() -> Vec<TableLayout> {
        vec![TableLayout {
            id: 0,
            planes: [
                PlaneLayout { row_bytes: 256, nbytes: 256 * 64 },
                PlaneLayout { row_bytes: 8, nbytes: 8 * 64 },
            ],
        }]
    }

    fn answer() -> HelloOk {
        HelloOk {
            abi: STORE_ABI,
            gid: [0; 16],
            mtu: 1024,
            max_read_bytes: 4 << 20,
            lanes: 2,
            depth: 16,
            tables: vec![RemoteTable {
                id: 0,
                planes: [
                    RemotePlane { rkey: 1, address: 0x1000, nbytes: 256 * 64, row_bytes: 256 },
                    RemotePlane { rkey: 2, address: 0x9000, nbytes: 8 * 64, row_bytes: 8 },
                ],
            }],
        }
    }

    #[test]
    fn hello_ok_must_match_the_requested_catalog_and_limits() {
        assert_eq!(validate(&answer(), &layouts(), 4, 4096).unwrap(), (16, MAX_READ as usize));
        let cases: [fn(&mut HelloOk); 8] = [
            |ok| ok.abi += 1,
            |ok| ok.lanes = 5,
            |ok| ok.depth = 0,
            |ok| ok.mtu = 1500,
            |ok| ok.tables.clear(),
            |ok| ok.tables[0].planes[0].nbytes -= 256,
            |ok| ok.tables[0].planes[1].row_bytes = 10,
            |ok| ok.tables[0].planes[0].address = u64::MAX,
        ];
        for (index, change) in cases.into_iter().enumerate() {
            let mut ok = answer();
            change(&mut ok);
            assert!(validate(&ok, &layouts(), 4, 4096).is_err(), "case {index}");
        }
        assert!(validate(&answer(), &layouts(), 4, 512).is_err());
    }

    fn rdma_tests() -> bool {
        std::env::var_os("FLARE_RUN_RDMA_TESTS").is_some()
    }

    /// The first address with an active RoCE v2 GID, such as `rdma_rxe` over loopback's peer.
    fn loopback_address() -> Option<IpAddr> {
        ["127.0.0.1", "::1"]
            .into_iter()
            .filter_map(|address| address.parse().ok())
            .find(|address| Device::open(*address).is_ok())
    }

    #[test]
    fn fabric_session_requires_lanes_in_range() {
        let status = Status::new("127.0.0.1:1");
        for lanes in [0, 9] {
            let connect = Connect {
                endpoint: Endpoint {
                    address: "127.0.0.1:1".into(),
                    lanes,
                    timeout: Duration::from_millis(100),
                },
                identity: [0; 32],
                rank: 0,
                world: 1,
                status: status.clone(),
            };
            assert!(Session::connect(connect, &[]).is_err());
        }
        if !rdma_tests() {
            eprintln!("skipping the RDMA half: set FLARE_RUN_RDMA_TESTS=1 with a RoCE v2 device");
            return;
        }
        assert!(loopback_address().is_some(), "FLARE_RUN_RDMA_TESTS requires a RoCE v2 GID on loopback");
    }
}
