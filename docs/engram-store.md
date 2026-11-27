# Engram store

An Engram store holds the n-gram embedding tables of a signed core in the RAM
of a host without a GPU and serves them to Flare processes over one-sided RDMA
READ. A serving host selects it with `--embedding-table-storage fabric` and
`--embedding-table-store HOST:PORT`. The serving host keeps its GPU memory and
host RAM plan of disk storage; the store host holds the table bytes.

Fabric storage is implemented behind the `engram-rdma` Cargo feature of
`flare-helios`, `flare`, and `flare-cli`, which links `libibverbs`. Release
builds do not enable it. It is not qualified on any hardware or core; see
[Qualification](#qualification).

## Dataflow

Each Engram table has a delivery worker thread on the serving host. The
worker gathers the rows a step requests into one of the table's two staging
frames, and the CUDA upload and the captured graphs consume that frame. With
disk storage the worker's row cache coalesces duplicate IDs, copies cache
hits, and reads the misses from NVMe in one call that fills a miss workspace
laid out as `[count × weight rows][count × scale rows]`.

Fabric storage replaces only that miss read. The row cache, the staging
frames, the per-rank frame handshake, the upload, and the graphs are
unchanged; no prepared program, core format, or captured-graph operation
changes, and an existing core is served without re-export or re-signing. The
fetch runs on the worker thread before the graph's upload waits for the frame,
where the disk read runs.

RDMA READs land in the row cache's miss workspace, which the cache allocates
and touches before the memory plan is sized. Each table registers its
workspace once. The CUDA-mapped staging frames are never registered; rows
reach them through the same copy the disk path performs. Resident scales,
enabled by default on discrete GPUs as for disk storage, keep the scale plane
in local RAM, so only weight rows cross the fabric.

```
serving host                                        store host
table worker ── row cache ── miss workspace (MR) <── RDMA READ ── plane MRs (locked RAM)
      │                                                             ▲
      └── staging frame ── CUDA upload ── graph       control TCP ──┘
```

## Control protocol

The control connection is TCP. Every message is a big-endian `u32` length
followed by a kind byte and a big-endian body; a message longer than 1 MiB is
rejected. `HELLO` and `HELLO_OK` begin with the magic `FEGS` and protocol
version 1. A store refuses a `HELLO` of another version with
`version_unsupported`, and a client refuses a `HELLO_OK` of another version,
without interpreting the rest of the body.

1. `HELLO` (client to store): core identity, rank, tensor-parallel world,
   requested lanes, the client port's active MTU, and per table the weight and
   scale plane layout `{row_bytes, nbytes}`.
2. `HELLO_OK` (store to client): store ABI 1, the store's GID, the selected MTU
   (the smaller of both ports), the largest READ the store accepts (1 MiB), the
   confirmed lanes, the READ depth each store queue pair accepts, and per table
   and plane `{rkey, remote_addr, nbytes, row_bytes}`. The store may instead
   answer `ERROR {code, message}` with `version_unsupported`, `core_mismatch`,
   `plane_layout_mismatch`, `insufficient_memory`, or `not_ready`.
3. `QP_EXCHANGE`: the client sends `{gid, qpn, psn}` for each lane; the store
   creates and connects its queue pairs and answers with its own. Both sides
   then hold reliable-connected queue pairs in RTS.
4. `HEARTBEAT`: the client sends one each second and the store echoes it. The
   client treats three seconds without an echo as a lost store; the store drops
   a client after ten seconds without one. `BYE` ends a session from either
   side.

## Catalog and identity

The store opens the core through the same publisher review, signature
verification, and consent as `flare run`, then decodes every rank's Engram
catalog with the loader's own decoder, without a GPU. Planes that ranks share
are loaded and registered once; row-sharded planes are rank-specific, and
together they cover every table row, so the store holds the full table set
whichever ranks connect.

The core identity is the SHA-256 digest of the signed core envelope, which the
signature covers along with the digest of every companion file. A client
presents the identity of the core it loaded. The store answers `core_mismatch`
for another identity or another tensor-parallel size, and
`plane_layout_mismatch` for a rank it does not serve or a table layout that
differs from its catalog, so it cannot serve bytes that do not belong to the
client's core. A byte-identical core published under two revision names has one
identity.

## Data plane

The store registers each plane for remote READ only. Queue pairs carry no
commands; a client can only READ registered planes at the keys and addresses
`HELLO_OK` returned, and the store returns them only after validating the
client's identity and layout.

The row cache passes misses as sorted, unique row IDs. A run of consecutive IDs
is one READ of `run_length × row_bytes` from `remote_addr + first_id ×
row_bytes` into the run's offset in the miss workspace: weights at
`row × weight_bytes`, scales at `count × weight_bytes + row × scale_bytes`. A
READ longer than 1 MiB is split. Every READ is signalled.

A gather takes the lane `table mod lanes`, so concurrent table workers spread
over the lanes and workers mapped to one lane take turns. A lane's READs in
flight never exceed its confirmed depth: the client requests the smaller of its
device's `max_qp_init_rd_atom` and the store's `max_qp_rd_atom`, and reads back
the depth the device set. The worker polls completions on its own thread,
spinning and yielding after 1024 empty polls.

## Failure states

| Failure | Classification | Effect |
|---|---|---|
| Store unreachable, control connection closed, heartbeat timeout, store restart | Transient | The session reports `down` and reconnects with exponential backoff from 100 ms to 5 s; admission rejects requests |
| READ completion `RETRY_EXC_ERR`, `RNR_RETRY_EXC_ERR`, or `WR_FLUSH_ERR`, or a READ exceeding the timeout | Transient | The gather fails, the link is withdrawn, and the session reconnects |
| `version_unsupported`, `core_mismatch`, `plane_layout_mismatch`, a malformed message during connection setup, or a `HELLO_OK` that disagrees with the request | Fatal | At startup, loading fails; later, the store reports `failed` and the layer is poisoned |
| A malformed or unexpected message in an established session | Transient | The link is withdrawn and the next connection validates the store again |
| Invalid rkey or range, local protection error, or any other READ completion status | Fatal | The layer is poisoned, as for a disk read error |
| `insufficient_memory`, `not_ready` | Transient | The session retries |

A transient gather failure does not disable the table. The worker zero-fills
the frame only so the frame handshake can drain, and stores the error for that
sequence. The executor reports the frame as failed instead of poisoning
itself. The runtime replaces every record of the frame, and of any frame
already submitted for the same requests, with a device-error record at the
request's committed progress, so no token computed from the zeroed rows is
emitted and no snapshot is taken; each affected request fails with
`embedding_store_unavailable`. The next sequence that reuses the frame clears
the stored error. A rank process of a multi-host core cannot report a failed
frame to rank zero, so it fails as it does for any reader error.

On a transient failure the worker moves its lane to the error state and waits
up to one second for every posted READ to complete or flush before returning,
so no READ writes the miss workspace afterwards. A READ that neither completes
nor flushes in that time poisons the layer.

Flare starts with an unreachable store and never substitutes local reads.
While any session is not ready, requests are rejected with HTTP 503, error code
`embedding_store_unavailable`, and `Retry-After: 1`; a request whose response
has started streaming receives the same error as a terminal stream event.
`GET /health` stays 200 and reports
`engram_store: {state, endpoint, last_error, reconnects}`, where `state` is
`ready`, `down`, or `failed`, so an orchestrator does not restart a healthy
process for a remote outage. Metrics add `flare_engram_fabric_reads_total`,
`flare_engram_fabric_bytes_total`, `flare_engram_fabric_wait_nanoseconds_total`,
`flare_engram_fabric_failures_total`, `flare_engram_fabric_reconnects_total`,
and the gauge `flare_engram_fabric_ready`. Other storage modes render none of
them.

## Operator requirements

```sh
flare engram-store --package /models/deepseek-v41 --listen 10.0.0.5:7700 --tensor-parallel 4
flare run CORE --gpus 0,1,2,3 --embedding-table-storage fabric --embedding-table-store 10.0.0.5:7700
```

- **Memory.** The store reads every served plane into RAM and registers it,
  which page-locks it; registered memory cannot be reclaimed under pressure.
  The store needs the table bytes plus process overhead, and its locked-memory
  limit (`ulimit -l`, or `LimitMEMLOCK` for a service) must cover every plane,
  normally `unlimited`. Registration failure reports the required limit. The
  serving host needs no memory beyond its disk plan; its locked-memory limit
  must cover every table's miss workspace, which is part of the row-cache
  budget.
- **Huge pages.** Planes are sealed shared-memory files mapped read-only, as
  RAM table storage maps them. They use 2 MiB pages only when the host enables
  transparent huge pages for shared memory
  (`/sys/kernel/mm/transparent_hugepage/shmem_enabled`).
- **Address and GID.** `--listen` must be an address of the store's RoCE
  interface; the store's queue pairs use the RoCE v2 GID of that address. The
  serving host selects the device and GID of the address its kernel routes to
  the store, and looks the GID up again on each connection.
- **MTU.** The path MTU is the smaller of both ports' active MTUs.
- **Lossless fabric.** Queue pairs use GRH traffic class 0; PFC or ECN on the
  path must cover that class.
- **Security.** The RDMA fabric is trusted, as for the RoCE ring: control
  messages and READs are neither encrypted nor authenticated. The store holds
  bytes of a core whose signature it verified.
- **Payload files.** The serving host still stores and verifies the core's
  companion files, including the Engram planes, and loads resident scales from
  them.

## Performance model

Per prefill chunk, a table looks up `tokens × heads` rows before cache hits
and duplicate coalescing; DeepSeek Engram tables use 24 heads for host-hashed
order-4 tables and 16 for graph-requested order-3 tables. FP8 tables store
256-byte weight rows and 8-byte scale rows; NVFP4 tables store 80 and 10. With
resident scales, a miss moves one weight row:

- wire bytes per chunk ≈ `tokens × heads × tables × weight_row_bytes`, reduced
  by the hit rate and duplicate coalescing;
- READs per chunk ≈ coalesced runs, which for hashed IDs is close to the number
  of distinct misses;
- wall time per lane ≈ `max(bytes / bandwidth, READs / depth × latency)`.

For 1,000 tokens of a 24-head FP8 table with no hits, the wire carries about
6.1 MB, 0.12 ms at 400 Gb/s, while 24,000 READs at a depth of 16 and an
assumed 3 µs round trip take about 4.5 ms on one lane. The depth term dominates
by more than an order of magnitude, so the lane count, and with it the READs in
flight per table, sets fabric gather time. The disk reader issues one 4 KiB
direct read per scattered miss through io_uring, coalescing adjacent blocks up
to 64 KiB at a depth of 128. These figures are an expectation from assumed
latencies, not a measurement.

## Qualification

Not qualified. The control framing, READ planning, store-status accounting,
configuration validation, memory plan, admission mapping, and failed-frame
retirement have unit tests. Loopback tests of the data path, store loss and
restart, and identity refusal run with `FLARE_RUN_RDMA_TESTS=1` on a host with
a RoCE v2 device on loopback, such as `rdma_rxe`. No core has been served with
fabric storage, and no serving measurement exists; [the TODO list](TODO.txt)
records the pending checks.
