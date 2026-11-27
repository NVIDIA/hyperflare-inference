/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The Engram store: the planes of a verified core held in RAM on a host without a GPU and
//! served to Flare processes through registered memory regions.
//!
//! The store reads every plane of every served rank once through the resident-table loader,
//! registers each distinct plane once for remote READ, and answers control connections. A
//! `HELLO` must name the store's exact signed core, its tensor-parallel size, a served rank,
//! and that rank's table layout; only then does the store return keys and addresses. The data
//! path carries no commands: peers can only READ the registered planes. Planes that ranks
//! share are registered once, and row-sharded planes together cover the full table set, so the
//! store holds every table byte of the deployment, page-locked by registration.

use super::{
    protocol::{
        self, Code, HelloOk, MAX_LANES, Message, PlaneLayout, Receiver, RemotePlane, RemoteTable,
        STORE_ABI, TableLayout,
    },
    verbs::{Access, Device, Lane, Region, Role},
};
use crate::{
    Result,
    engram::{Plane, reader::Resident},
    runtime_core::AuthorizedCore,
};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io,
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Largest READ a client may post.
const MAX_READ: u32 = 1 << 20;
const HANDSHAKE: Duration = Duration::from_secs(10);
const POLL: Duration = Duration::from_millis(100);
/// A client that sends no heartbeat for this long is dropped and its queue pairs destroyed.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Options {
    /// Address of the control listener. It must be an address of the RoCE interface whose
    /// GID the store's queue pairs use.
    pub listen: SocketAddr,
    /// Ranks to serve; all ranks of the deployment when absent.
    pub ranks: Option<Vec<usize>>,
}

/// One distinct plane registered for remote READ. The registration is declared first so it is
/// released before its memory.
struct Served {
    region: Region,
    _memory: Resident,
}

/// A served rank's tables: for each plane, the distinct plane it maps to and its layout.
type RankTables = Vec<[(usize, PlaneLayout); 2]>;

struct State {
    identity: [u8; 32],
    world: u32,
    device: Arc<Device>,
    ranks: HashMap<u32, RankTables>,
    served: RwLock<Option<Vec<Served>>>,
    stop: Arc<AtomicBool>,
}

/// Planes to load: a source file, and per rank the distinct planes of each table.
struct Layout {
    sources: Vec<(Arc<File>, Plane)>,
    ranks: HashMap<u32, RankTables>,
}

fn layout(files: &[Arc<File>], ranks: &[(usize, Vec<[Plane; 2]>)]) -> Result<Layout> {
    let mut distinct = BTreeMap::new();
    let mut sources = Vec::new();
    let mut served = HashMap::new();
    for (rank, tables) in ranks {
        let mut planes = Vec::with_capacity(tables.len());
        for table in tables {
            let mut entry = [(0usize, PlaneLayout { row_bytes: 0, nbytes: 0 }); 2];
            for (slot, plane) in entry.iter_mut().zip(table) {
                let file = files
                    .get(plane.file)
                    .ok_or("Engram store plane references an unknown file")?;
                let key = (plane.file, plane.offset, plane.nbytes, plane.row_bytes);
                let index = *distinct.entry(key).or_insert_with(|| {
                    sources.push((file.clone(), *plane));
                    sources.len() - 1
                });
                *slot = (
                    index,
                    PlaneLayout {
                        row_bytes: u32::try_from(plane.row_bytes)
                            .map_err(|_| "Engram row exceeds the store protocol range")?,
                        nbytes: plane.nbytes as u64,
                    },
                );
            }
            planes.push(entry);
        }
        let rank = u32::try_from(*rank).map_err(|_| "Engram rank exceeds the protocol range")?;
        served.insert(rank, planes);
    }
    Ok(Layout { sources, ranks: served })
}

/// A running store; dropping it stops the listener and every session.
pub struct Running {
    state: Arc<State>,
    accept: Option<JoinHandle<()>>,
    pub address: SocketAddr,
}

impl Running {
    /// Block until `stop` is set.
    pub fn wait(mut self) {
        while !self.state.stop.load(Ordering::Acquire) {
            thread::sleep(POLL);
        }
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Release);
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
    }
}

/// Verify-time identity and catalog of the core, then load, register and serve its planes.
/// `report` receives the startup lines. Returns once serving has started.
pub fn start(
    core: AuthorizedCore,
    options: &Options,
    stop: Arc<AtomicBool>,
    report: &mut dyn FnMut(String),
) -> Result<Running> {
    let catalog = crate::runtime_core::engram_catalog(core)?;
    let world = u32::try_from(catalog.world).map_err(|_| "tensor-parallel size exceeds range")?;
    let ranks = catalog
        .ranks
        .into_iter()
        .filter(|(rank, _)| options.ranks.as_ref().is_none_or(|selected| selected.contains(rank)))
        .map(|(rank, index)| {
            (rank, index.tables.iter().map(|table| table.planes).collect::<Vec<[Plane; 2]>>())
        })
        .collect::<Vec<_>>();
    if let Some(selected) = &options.ranks {
        if let Some(missing) = selected.iter().find(|rank| !ranks.iter().any(|(r, _)| r == *rank)) {
            return Err(format!("the core has no rank {missing}; its deployment has {world} ranks").into());
        }
    }
    serve(options.listen, catalog.identity, world, &catalog.files, &ranks, stop, report)
}

fn serve(
    listen: SocketAddr,
    identity: [u8; 32],
    world: u32,
    files: &[Arc<File>],
    ranks: &[(usize, Vec<[Plane; 2]>)],
    stop: Arc<AtomicBool>,
    report: &mut dyn FnMut(String),
) -> Result<Running> {
    if listen.ip().is_unspecified() {
        return Err("the store's listen address must be the address of its RoCE interface, not a wildcard".into());
    }
    let layout = layout(files, ranks)?;
    let device = Device::open(listen.ip())
        .map_err(|error| format!("open the RDMA device for {}: {error}", listen.ip()))?;
    let listener = TcpListener::bind(listen).map_err(|error| format!("listen on {listen}: {error}"))?;
    let address = listener.local_addr().map_err(|error| format!("listen on {listen}: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("configure the store listener: {error}"))?;
    let state = Arc::new(State {
        identity,
        world,
        device: device.clone(),
        ranks: layout.ranks,
        served: RwLock::new(None),
        stop,
    });
    let accepting = state.clone();
    let accept = thread::Builder::new()
        .name("flare-engram-store-accept".into())
        .spawn(move || accept_loop(listener, accepting))
        .map_err(|error| format!("start the store listener: {error}"))?;
    let running = Running { state: state.clone(), accept: Some(accept), address };
    let total: u64 = layout.sources.iter().map(|(_, plane)| plane.nbytes as u64).sum();
    report(format!(
        "Engram store: {} distinct planes, {:.1} GiB, for {} of {world} ranks on {} at {address}",
        layout.sources.len(),
        total as f64 / (1u64 << 30) as f64,
        state.ranks.len(),
        device.name,
    ));
    let loading = Instant::now();
    let mut registration = Duration::ZERO;
    let mut served = Vec::with_capacity(layout.sources.len());
    for (file, plane) in &layout.sources {
        let memory = Resident::load(file, *plane)?;
        let bytes = memory.bytes();
        let begin = Instant::now();
        // SAFETY: `Served` keeps the sealed mapping alive for as long as the registration.
        let region = unsafe { Region::register(&device, bytes.as_ptr(), bytes.len(), Access::RemoteRead) }
            .map_err(|error| {
                format!(
                    "register a {}-byte Engram plane for remote READ: {error}; the store page-locks every plane, so its locked-memory limit (ulimit -l, or LimitMEMLOCK) must be at least {} KiB or unlimited",
                    bytes.len(),
                    total.div_ceil(1024),
                )
            })?;
        registration += begin.elapsed();
        served.push(Served { region, _memory: memory });
    }
    *state.served.write().unwrap_or_else(|e| e.into_inner()) = Some(served);
    report(format!(
        "Engram store ready: loaded in {:.1} s, registration {:.2} s",
        loading.elapsed().as_secs_f64(),
        registration.as_secs_f64(),
    ));
    Ok(running)
}

fn accept_loop(listener: TcpListener, state: Arc<State>) {
    let mut sessions: Vec<JoinHandle<()>> = Vec::new();
    while !state.stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, peer)) => {
                let session = state.clone();
                match thread::Builder::new()
                    .name("flare-engram-store-session".into())
                    .spawn(move || {
                        if let Err(error) = handle(stream, &session) {
                            eprintln!("Engram store session from {peer} ended: {error}");
                        }
                    }) {
                    Ok(handle) => sessions.push(handle),
                    Err(error) => eprintln!("warning: Engram store session thread: {error}"),
                }
                sessions.retain(|handle| !handle.is_finished());
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => thread::sleep(POLL / 2),
            Err(error) => {
                eprintln!("warning: Engram store accept: {error}");
                thread::sleep(POLL);
            }
        }
    }
    for session in sessions {
        let _ = session.join();
    }
}

fn refuse(stream: &mut TcpStream, code: Code, message: String) -> io::Result<()> {
    let _ = protocol::write(stream, &Message::Error { code, message: message.clone() });
    Err(io::Error::other(format!("refused ({}): {message}", code.name())))
}

fn handle(mut stream: TcpStream, state: &State) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(HANDSHAKE))?;
    stream.set_write_timeout(Some(HANDSHAKE))?;
    let hello = match protocol::read(&mut stream)? {
        Message::Hello(hello) => hello,
        Message::HelloVersion(version) => {
            return refuse(
                &mut stream,
                Code::VersionUnsupported,
                format!("this store speaks control protocol version {}, not {version}", protocol::VERSION),
            );
        }
        _ => return Err(protocol::invalid("the first message is not HELLO")),
    };
    if !(1..=MAX_LANES).contains(&usize::from(hello.lanes)) || !protocol::valid_mtu(hello.mtu) {
        return Err(protocol::invalid("HELLO requests an invalid lane count or MTU"));
    }
    if hello.identity != state.identity {
        return refuse(&mut stream, Code::CoreMismatch, "the store holds a different signed core".into());
    }
    if hello.world != state.world {
        return refuse(
            &mut stream,
            Code::CoreMismatch,
            format!("the store holds the {}-rank deployment, not {} ranks", state.world, hello.world),
        );
    }
    let Some(tables) = state.ranks.get(&hello.rank) else {
        return refuse(&mut stream, Code::PlaneLayoutMismatch, format!("the store does not serve rank {}", hello.rank));
    };
    let expected = tables
        .iter()
        .enumerate()
        .map(|(id, planes)| TableLayout { id: id as u32, planes: planes.map(|(_, layout)| layout) })
        .collect::<Vec<_>>();
    if hello.tables != expected {
        return refuse(
            &mut stream,
            Code::PlaneLayoutMismatch,
            format!("rank {}'s requested tables differ from the store's catalog", hello.rank),
        );
    }
    let served = state.served.read().unwrap_or_else(|e| e.into_inner());
    let Some(served) = served.as_ref() else {
        return refuse(&mut stream, Code::NotReady, "the store is still loading its planes".into());
    };
    let mtu = hello.mtu.min(state.device.mtu);
    let depth = state.device.responder_depth;
    let remote = tables
        .iter()
        .enumerate()
        .map(|(id, planes)| RemoteTable {
            id: id as u32,
            planes: planes.map(|(index, layout)| RemotePlane {
                rkey: served[index].region.rkey(),
                address: served[index].region.address(),
                nbytes: layout.nbytes,
                row_bytes: layout.row_bytes,
            }),
        })
        .collect();
    protocol::write(
        &mut stream,
        &Message::HelloOk(HelloOk {
            abi: STORE_ABI,
            gid: state.device.gid(),
            mtu,
            max_read_bytes: MAX_READ,
            lanes: hello.lanes,
            depth,
            tables: remote,
        }),
    )?;
    let peers = match protocol::read(&mut stream)? {
        Message::QpExchange(peers) if peers.len() == usize::from(hello.lanes) => peers,
        Message::Bye => return Ok(()),
        _ => return Err(protocol::invalid("expected the client's queue-pair exchange")),
    };
    let mut lanes = Vec::with_capacity(peers.len());
    for peer in &peers {
        let mut lane = match Lane::new(&state.device, Role::Responder, depth) {
            Ok(lane) => lane,
            Err(error) if error.raw_os_error() == Some(libc::ENOMEM) => {
                return refuse(&mut stream, Code::InsufficientMemory, format!("create a queue pair: {error}"));
            }
            Err(error) => return Err(error),
        };
        lane.connect(peer, mtu, depth)?;
        lanes.push(lane);
    }
    protocol::write(&mut stream, &Message::QpExchange(lanes.iter().map(Lane::local).collect()))?;
    stream.set_read_timeout(Some(POLL))?;
    let mut receiver = Receiver::default();
    let mut heard = Instant::now();
    loop {
        if state.stop.load(Ordering::Acquire) {
            let _ = protocol::write(&mut stream, &Message::Bye);
            return Ok(());
        }
        match receiver.poll(&mut stream)? {
            Some(Message::Heartbeat(sequence)) => {
                heard = Instant::now();
                protocol::write(&mut stream, &Message::Heartbeat(sequence))?;
            }
            Some(Message::Bye) => return Ok(()),
            Some(_) => return Err(protocol::invalid("unexpected message in an established session")),
            None if heard.elapsed() > CLIENT_TIMEOUT => {
                return Err(io::Error::other("no heartbeat from the client"));
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! Loopback tests need a RoCE v2 device with a loopback GID, for example
    //! `rdma link add rxe0 type rxe netdev lo`, and run with `FLARE_RUN_RDMA_TESTS=1`.

    use super::*;
    use crate::engram::{
        Stats,
        fabric::{Connect, Endpoint, Session, Status, transient},
        reader::Reader,
    };
    use std::{io::Write, path::PathBuf};

    fn enabled() -> bool {
        if std::env::var_os("FLARE_RUN_RDMA_TESTS").is_none() {
            eprintln!("skipping: set FLARE_RUN_RDMA_TESTS=1 with a RoCE v2 device on loopback");
            return false;
        }
        true
    }

    struct Fixture {
        path: PathBuf,
        file: Arc<File>,
        planes: [Plane; 2],
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// A container holding an 80-byte weight plane and a 10-byte scale plane at offsets that
    /// are not block aligned, as NVFP4 tables are stored.
    fn fixture(rows: usize) -> Fixture {
        let path = std::env::temp_dir().join(format!(
            "flare-engram-store-{}-{rows}.bin",
            std::process::id()
        ));
        let (weights, scales) = (80usize, 10usize);
        let weight_offset = 123u64;
        let scale_offset = weight_offset + (rows * weights) as u64 + 77;
        let mut bytes = vec![0u8; scale_offset as usize + rows * scales + 51];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index * 31 % 251) as u8;
        }
        std::fs::File::create(&path).unwrap().write_all(&bytes).unwrap();
        let file = Arc::new(crate::engram::reader::open_file(&path).unwrap());
        Fixture {
            path,
            file,
            planes: [
                Plane { file: 0, offset: weight_offset, nbytes: rows * weights, row_bytes: weights },
                Plane { file: 0, offset: scale_offset, nbytes: rows * scales, row_bytes: scales },
            ],
        }
    }

    fn store(fixture: &Fixture, listen: SocketAddr, identity: [u8; 32]) -> Running {
        serve(
            listen,
            identity,
            1,
            std::slice::from_ref(&fixture.file),
            &[(0, vec![fixture.planes])],
            Arc::new(AtomicBool::new(false)),
            &mut |line| eprintln!("{line}"),
        )
        .unwrap()
    }

    fn connect(address: SocketAddr, identity: [u8; 32], status: &Arc<Status>) -> Connect {
        Connect {
            endpoint: Endpoint { address: address.to_string(), lanes: 2, timeout: Duration::from_millis(2000) },
            identity,
            rank: 0,
            world: 1,
            status: status.clone(),
        }
    }

    fn wait_ready(status: &Status) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !status.ready() {
            assert!(Instant::now() < deadline, "store session did not become ready: {:?}", status.snapshot());
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    #[test]
    fn fabric_gather_equals_disk_gather_with_duplicates_padding_and_unaligned_planes() {
        if !enabled() {
            return;
        }
        let rows = 4096;
        let fixture = fixture(rows);
        let identity = [3u8; 32];
        let running = store(&fixture, loopback(), identity);
        let status = Status::new(&running.address.to_string());
        let session = Session::connect(connect(running.address, identity, &status), &[fixture.planes]).unwrap();
        wait_ready(&status);
        let files = [fixture.file.clone()];
        let stats = Arc::new(Stats::default());
        let max_rows = 1024;
        for resident_scales in [false, true] {
            let mut disk = Reader::from_files(&files, &fixture.planes, false, 4 << 20, max_rows, stats.clone(), 1, resident_scales).unwrap();
            let mut fabric = Reader::from_fabric(&files, &fixture.planes, session.clone(), 0, 4 << 20, max_rows, stats.clone(), resident_scales).unwrap();
            let batches: [Vec<i64>; 4] = [
                vec![5, 5, 6, 7, -1, 4095, 0, 6, 1000, 1001, 1003],
                (0..1024).map(|i| (i * 37 % rows) as i64).collect(),
                (0..1024).map(|i| if i % 3 == 0 { -1 } else { (i / 2) as i64 }).collect(),
                vec![5, 6, 7, 2000],
            ];
            for ids in &batches {
                let size = ids.len() * 90;
                let (mut expected, mut actual) = (vec![0xa5u8; size], vec![0x5au8; size]);
                disk.gather(&fixture.planes, ids, &mut expected).unwrap();
                fabric.gather(&fixture.planes, ids, &mut actual).unwrap();
                assert_eq!(actual, expected, "resident scales {resident_scales}, ids {:?}", &ids[..ids.len().min(8)]);
            }
        }
        assert!(stats.fabric_reads.load(Ordering::Relaxed) > 0);
        drop(running);
    }

    #[test]
    fn fabric_store_death_fails_the_gather_transiently_and_a_restart_reconnects() {
        if !enabled() {
            return;
        }
        let fixture = fixture(512);
        let identity = [4u8; 32];
        let running = store(&fixture, loopback(), identity);
        let address = running.address;
        let status = Status::new(&address.to_string());
        let session = Session::connect(connect(address, identity, &status), &[fixture.planes]).unwrap();
        wait_ready(&status);
        let files = [fixture.file.clone()];
        let stats = Arc::new(Stats::default());
        let mut fabric = Reader::from_fabric(&files, &fixture.planes, session.clone(), 0, 1 << 20, 64, stats.clone(), false).unwrap();
        let ids: Vec<i64> = (0..64).collect();
        let mut output = vec![0u8; ids.len() * 90];
        fabric.gather(&fixture.planes, &ids, &mut output).unwrap();
        drop(running);
        // Every attempt uses IDs the row cache has not seen, so each one must reach the store.
        let mut next = 64i64;
        let mut fresh = || {
            next += 8;
            (next - 8..next).collect::<Vec<i64>>()
        };
        let mut small = vec![0u8; 8 * 90];
        let deadline = Instant::now() + Duration::from_secs(20);
        let error = loop {
            match fabric.gather(&fixture.planes, &fresh(), &mut small) {
                Err(error) => break error,
                Ok(()) => {
                    assert!(Instant::now() < deadline, "gathers kept succeeding after the store stopped");
                    thread::sleep(Duration::from_millis(50));
                }
            }
        };
        assert!(transient(&error), "store loss must not poison: {error}");
        let restarted = store(&fixture, address, identity);
        wait_ready(&status);
        assert!(status.snapshot().reconnects >= 1);
        fabric.gather(&fixture.planes, &fresh(), &mut small).unwrap();
        drop(restarted);
    }

    #[test]
    fn fabric_store_refuses_another_core_and_an_unknown_rank() {
        if !enabled() {
            return;
        }
        let fixture = fixture(64);
        let running = store(&fixture, loopback(), [5u8; 32]);
        let status = Status::new(&running.address.to_string());
        let error = Session::connect(connect(running.address, [6u8; 32], &status), &[fixture.planes]).err().unwrap();
        assert!(error.to_string().contains("core_mismatch"), "{error}");
        let mut other = connect(running.address, [5u8; 32], &status);
        other.rank = 3;
        let error = Session::connect(other, &[fixture.planes]).err().unwrap();
        assert!(error.to_string().contains("plane_layout_mismatch"), "{error}");
        let mut planes = fixture.planes;
        planes[0].nbytes -= planes[0].row_bytes;
        let error = Session::connect(connect(running.address, [5u8; 32], &status), &[planes]).err().unwrap();
        assert!(error.to_string().contains("plane_layout_mismatch"), "{error}");
    }

    #[test]
    fn distinct_planes_are_registered_once_across_ranks() {
        let file = Arc::new(std::fs::File::open(std::env::current_exe().unwrap()).unwrap());
        let shared = Plane { file: 0, offset: 0, nbytes: 64, row_bytes: 8 };
        let scales = Plane { file: 0, offset: 64, nbytes: 16, row_bytes: 2 };
        let sharded = |rank: usize| Plane { file: 0, offset: 4096 * (rank as u64 + 1), nbytes: 32, row_bytes: 8 };
        let ranks = (0..4).map(|rank| (rank, vec![[shared, scales], [sharded(rank), scales]])).collect::<Vec<_>>();
        let planned = layout(&[file], &ranks).unwrap();
        assert_eq!(planned.sources.len(), 2 + 4);
        assert_eq!(planned.ranks[&0][0][0].0, planned.ranks[&3][0][0].0);
        assert_ne!(planned.ranks[&0][1][0].0, planned.ranks[&1][1][0].0);
        assert_eq!(planned.ranks[&2][1][1], planned.ranks[&2][0][1]);
        assert!(layout(&[], &ranks).is_err());
    }
}
