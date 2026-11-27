/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! libibverbs objects of the Engram store data path: the device whose RoCE v2 GID carries a
//! local address, memory registrations, and reliable-connected queue pairs for RDMA READ.
//!
//! Calls follow `ibverbs-sys` 0.3.2 as `flare_selene::shared::rdma` uses it. Posting and
//! polling go through the context's provider operations, which libibverbs' inline
//! `ibv_post_send` and `ibv_poll_cq` call.

use super::{plan::Read, protocol::QueuePair};
pub(crate) use ibverbs_sys as ffi;
use std::{ffi::CStr, io, mem, net::IpAddr, ptr, sync::Arc};

/// Completions taken per poll call.
const POLL_BATCH: usize = 32;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_owned())
}

/// An open device, the port and GID that carry the selected local address, and its PD.
pub(crate) struct Device {
    context: *mut ffi::ibv_context,
    pub(crate) pd: *mut ffi::ibv_pd,
    port: u8,
    gid: [u8; 16],
    /// GID table index of `gid`. The kernel rebuilds the table when the interface or its
    /// address changes, so connections look it up again with [`Device::refresh`].
    gid_index: std::sync::atomic::AtomicU8,
    /// The port's active MTU in bytes.
    pub(crate) mtu: u32,
    /// RDMA READs one queue pair may have in flight as requester (`max_qp_init_rd_atom`).
    pub(crate) initiator_depth: u8,
    /// RDMA READs one queue pair may serve concurrently as responder (`max_qp_rd_atom`).
    pub(crate) responder_depth: u8,
    pub(crate) max_wr: u32,
    pub(crate) name: String,
}

// SAFETY: libibverbs handles may be used from any thread and stay live until the last owner
// drops; no field changes after `open`.
unsafe impl Send for Device {}
// SAFETY: as above.
unsafe impl Sync for Device {}

fn mtu_bytes(mtu: ffi::ibv_mtu) -> u32 {
    match mtu {
        ffi::IBV_MTU_256 => 256,
        ffi::IBV_MTU_512 => 512,
        ffi::IBV_MTU_1024 => 1024,
        ffi::IBV_MTU_2048 => 2048,
        ffi::IBV_MTU_4096 => 4096,
        _ => 0,
    }
}

fn mtu_value(bytes: u32) -> io::Result<ffi::ibv_mtu> {
    Ok(match bytes {
        256 => ffi::IBV_MTU_256,
        512 => ffi::IBV_MTU_512,
        1024 => ffi::IBV_MTU_1024,
        2048 => ffi::IBV_MTU_2048,
        4096 => ffi::IBV_MTU_4096,
        _ => return Err(invalid("RoCE path MTU must be 256, 512, 1024, 2048, or 4096 bytes")),
    })
}

fn depth(value: i32) -> u8 {
    value.clamp(1, u8::MAX as i32) as u8
}

impl Device {
    /// Open the device and port whose RoCE v2 GID is `local`, the address that routes to the
    /// peer, so RDMA traffic leaves on the interface the control connection uses.
    pub(crate) fn open(local: IpAddr) -> io::Result<Arc<Self>> {
        let wanted = match local {
            IpAddr::V4(address) => address.to_ipv6_mapped().octets(),
            IpAddr::V6(address) => address.octets(),
        };
        let mut count = 0;
        // SAFETY: libibverbs owns the returned list until it is freed below.
        let list = unsafe { ffi::ibv_get_device_list(&mut count) };
        if list.is_null() {
            return Err(io::Error::last_os_error());
        }
        let result = (|| {
            for index in 0..count.max(0) as usize {
                // SAFETY: the list has `count` entries.
                let device = unsafe { *list.add(index) };
                if device.is_null() {
                    continue;
                }
                // SAFETY: device belongs to the live list.
                let context = unsafe { ffi::ibv_open_device(device) };
                if context.is_null() {
                    continue;
                }
                let mut attr = ffi::ibv_device_attr::default();
                // SAFETY: the context is open and attr is writable.
                if unsafe { ffi::ibv_query_device(context, &mut attr) } != 0 {
                    unsafe { ffi::ibv_close_device(context) };
                    continue;
                }
                let mut selected = None;
                for port in 1..=attr.phys_port_cnt {
                    let mut port_attr = ffi::ibv_port_attr::default();
                    // SAFETY: the compat cast matches libibverbs' inline wrapper.
                    let status = unsafe {
                        ffi::ibv_query_port(
                            context,
                            port,
                            &mut port_attr as *mut ffi::ibv_port_attr
                                as *mut ffi::_compat_ibv_port_attr,
                        )
                    };
                    if status != 0 || port_attr.state != ffi::ibv_port_state::IBV_PORT_ACTIVE {
                        continue;
                    }
                    for gid_index in 0..port_attr.gid_tbl_len.max(0) as u32 {
                        let mut entry = unsafe { mem::zeroed::<ffi::ibv_gid_entry>() };
                        // SAFETY: the context is open and entry has the supplied size.
                        let status = unsafe {
                            ffi::_ibv_query_gid_ex(
                                context,
                                u32::from(port),
                                gid_index,
                                &mut entry,
                                0,
                                mem::size_of::<ffi::ibv_gid_entry>(),
                            )
                        };
                        // SAFETY: libibverbs filled the union.
                        if status == 0
                            && entry.gid_type == ffi::IBV_GID_TYPE_ROCE_V2
                            && unsafe { entry.gid.raw } == wanted
                            && gid_index <= u32::from(u8::MAX)
                        {
                            selected = Some((port, gid_index as u8, mtu_bytes(port_attr.active_mtu)));
                            break;
                        }
                    }
                    if selected.is_some() {
                        break;
                    }
                }
                let Some((port, gid_index, mtu)) = selected else {
                    // SAFETY: no PD or registration uses this context.
                    unsafe { ffi::ibv_close_device(context) };
                    continue;
                };
                if mtu == 0 {
                    unsafe { ffi::ibv_close_device(context) };
                    return Err(invalid("the RoCE port reports no supported active MTU"));
                }
                // SAFETY: the context is open.
                let pd = unsafe { ffi::ibv_alloc_pd(context) };
                if pd.is_null() {
                    let error = io::Error::last_os_error();
                    unsafe { ffi::ibv_close_device(context) };
                    return Err(error);
                }
                // SAFETY: the name belongs to the live device list.
                let name = unsafe { CStr::from_ptr(ffi::ibv_get_device_name(device)) }
                    .to_string_lossy()
                    .into_owned();
                return Ok(Arc::new(Self {
                    context,
                    pd,
                    port,
                    gid: wanted,
                    gid_index: std::sync::atomic::AtomicU8::new(gid_index),
                    mtu,
                    initiator_depth: depth(attr.max_qp_init_rd_atom),
                    responder_depth: depth(attr.max_qp_rd_atom),
                    max_wr: attr.max_qp_wr.max(1) as u32,
                    name,
                }));
            }
            Err(invalid(&format!(
                "no active RDMA port has a RoCE v2 GID for {local}; the Engram store must be reached through a RoCE interface"
            )))
        })();
        // SAFETY: every context opened from the list was opened before it is freed.
        unsafe { ffi::ibv_free_device_list(list) };
        result
    }

    pub(crate) fn gid(&self) -> [u8; 16] {
        self.gid
    }

    /// Find the selected address in the port's GID table again. Fails while the interface is
    /// down or no longer holds the address, which a later connection attempt retries.
    pub(crate) fn refresh(&self) -> io::Result<()> {
        let mut port_attr = ffi::ibv_port_attr::default();
        // SAFETY: the compat cast matches libibverbs' inline wrapper; the context is open.
        let status = unsafe {
            ffi::ibv_query_port(
                self.context,
                self.port,
                &mut port_attr as *mut ffi::ibv_port_attr as *mut ffi::_compat_ibv_port_attr,
            )
        };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        if port_attr.state != ffi::ibv_port_state::IBV_PORT_ACTIVE {
            return Err(io::Error::other("the RoCE port is not active"));
        }
        for gid_index in 0..port_attr.gid_tbl_len.clamp(0, 256) as u32 {
            let mut entry = unsafe { mem::zeroed::<ffi::ibv_gid_entry>() };
            // SAFETY: the context is open and entry has the supplied size.
            let status = unsafe {
                ffi::_ibv_query_gid_ex(
                    self.context,
                    u32::from(self.port),
                    gid_index,
                    &mut entry,
                    0,
                    mem::size_of::<ffi::ibv_gid_entry>(),
                )
            };
            // SAFETY: libibverbs filled the union.
            if status == 0
                && entry.gid_type == ffi::IBV_GID_TYPE_ROCE_V2
                && unsafe { entry.gid.raw } == self.gid
            {
                self.gid_index
                    .store(gid_index as u8, std::sync::atomic::Ordering::Release);
                return Ok(());
            }
        }
        Err(io::Error::other("the RoCE port no longer holds the local address's GID"))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: registrations and queue pairs hold an `Arc<Device>` and release first.
        unsafe {
            ffi::ibv_dealloc_pd(self.pd);
            ffi::ibv_close_device(self.context);
        }
    }
}

/// A registered memory range. The caller keeps the memory mapped until this value drops.
pub(crate) struct Region {
    mr: *mut ffi::ibv_mr,
    _device: Arc<Device>,
}

// SAFETY: the registration is immutable after creation and released once.
unsafe impl Send for Region {}
// SAFETY: as above.
unsafe impl Sync for Region {}

pub(crate) enum Access {
    /// Local writes by RDMA READ responses into the miss workspace.
    LocalWrite,
    /// Remote READ of a store plane.
    RemoteRead,
}

impl Region {
    /// # Safety
    /// `address..address + nbytes` must stay mapped, and for [`Access::LocalWrite`] writable,
    /// until the returned region drops.
    pub(crate) unsafe fn register(
        device: &Arc<Device>,
        address: *const u8,
        nbytes: usize,
        access: Access,
    ) -> io::Result<Self> {
        if nbytes == 0 || address.is_null() {
            return Err(invalid("an RDMA registration needs a nonempty mapped range"));
        }
        let access = (match access {
            Access::LocalWrite => ffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0,
            Access::RemoteRead => ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0,
        }) as i32;
        // SAFETY: the caller guarantees the range stays mapped for this registration.
        let mr = unsafe { ffi::ibv_reg_mr(device.pd, address as *mut _, nbytes, access) };
        if mr.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { mr, _device: device.clone() })
    }

    pub(crate) fn address(&self) -> u64 {
        // SAFETY: the registration is live.
        unsafe { (*self.mr).addr as u64 }
    }
    pub(crate) fn len(&self) -> usize {
        unsafe { (*self.mr).length }
    }
    pub(crate) fn lkey(&self) -> u32 {
        unsafe { (*self.mr).lkey }
    }
    pub(crate) fn rkey(&self) -> u32 {
        unsafe { (*self.mr).rkey }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: the registration is live and released once.
        unsafe { ffi::ibv_dereg_mr(self.mr) };
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// Posts RDMA READs; needs no remote access to its own memory.
    Requester,
    /// Serves RDMA READs of its registered planes and posts nothing.
    Responder,
}

/// How a work completion failed. A retry or flush follows a lost peer or link and may succeed
/// on a new connection; every other status means local or remote state disagrees.
pub(crate) enum Failure {
    Transient(String),
    Fatal(String),
}

fn classify(status: ffi::ibv_wc_status::Type, vendor: u32) -> Failure {
    let name = match status {
        ffi::ibv_wc_status::IBV_WC_RETRY_EXC_ERR => {
            return Failure::Transient("RDMA READ retries exhausted; the store did not answer".into());
        }
        ffi::ibv_wc_status::IBV_WC_RNR_RETRY_EXC_ERR => {
            return Failure::Transient("RDMA receiver-not-ready retries exhausted".into());
        }
        ffi::ibv_wc_status::IBV_WC_WR_FLUSH_ERR => {
            return Failure::Transient("RDMA READ flushed by a failed queue pair".into());
        }
        ffi::ibv_wc_status::IBV_WC_REM_ACCESS_ERR => "remote access error (invalid rkey or range)",
        ffi::ibv_wc_status::IBV_WC_LOC_PROT_ERR => "local protection error (unregistered destination)",
        ffi::ibv_wc_status::IBV_WC_LOC_LEN_ERR => "local length error",
        ffi::ibv_wc_status::IBV_WC_REM_INV_REQ_ERR => "remote invalid request",
        ffi::ibv_wc_status::IBV_WC_REM_OP_ERR => "remote operation error",
        _ => "RDMA work completion failed",
    };
    Failure::Fatal(format!("{name} (status {status}, vendor {vendor:#x})"))
}

/// One reliable-connected queue pair and its completion queue.
pub(crate) struct Lane {
    qp: *mut ffi::ibv_qp,
    cq: *mut ffi::ibv_cq,
    device: Arc<Device>,
    role: Role,
    psn: u32,
    /// Confirmed RDMA READ depth after connection; the requester never exceeds it.
    depth: usize,
    outstanding: usize,
    wrs: Vec<ffi::ibv_send_wr>,
    sges: Vec<ffi::ibv_sge>,
    wcs: Vec<ffi::ibv_wc>,
}

// SAFETY: a lane is owned by one thread at a time (behind a mutex on the requester).
unsafe impl Send for Lane {}

fn psn() -> u32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.subsec_nanos());
    (nanos ^ std::process::id().rotate_left(13)) & 0x00ff_ffff
}

impl Lane {
    /// Create a queue pair in INIT for `role`, sized for `depth` READs in flight.
    pub(crate) fn new(device: &Arc<Device>, role: Role, depth: u8) -> io::Result<Self> {
        let depth = usize::from(depth.max(1));
        let send_wr = match role {
            Role::Requester => (depth as u32).min(device.max_wr),
            Role::Responder => 1,
        };
        let cqe = (send_wr as i32).saturating_mul(2).max(16);
        // SAFETY: the context is open; no completion channel is used.
        let cq = unsafe {
            ffi::ibv_create_cq(device.context, cqe, ptr::null_mut(), ptr::null_mut(), 0)
        };
        if cq.is_null() {
            return Err(io::Error::last_os_error());
        }
        let mut lane = Self {
            qp: ptr::null_mut(),
            cq,
            device: device.clone(),
            role,
            psn: psn(),
            depth: send_wr as usize,
            outstanding: 0,
            // SAFETY: work requests, scatter entries and completions are plain C records for
            // which all-zero bytes are valid; each is filled before libibverbs reads it.
            wrs: (0..send_wr).map(|_| unsafe { mem::zeroed() }).collect(),
            sges: (0..send_wr).map(|_| unsafe { mem::zeroed() }).collect(),
            wcs: (0..POLL_BATCH).map(|_| unsafe { mem::zeroed() }).collect(),
        };
        let mut init = ffi::ibv_qp_init_attr {
            send_cq: cq,
            recv_cq: cq,
            cap: ffi::ibv_qp_cap {
                max_send_wr: send_wr,
                max_recv_wr: 1,
                max_send_sge: 1,
                max_recv_sge: 1,
                ..Default::default()
            },
            qp_type: ffi::ibv_qp_type::IBV_QPT_RC,
            sq_sig_all: 0,
            ..Default::default()
        };
        // SAFETY: the PD is live and `init` describes a valid RC queue pair.
        lane.qp = unsafe { ffi::ibv_create_qp(device.pd, &mut init) };
        if lane.qp.is_null() {
            return Err(io::Error::last_os_error());
        }
        let access = match role {
            Role::Requester => 0,
            Role::Responder => ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0,
        };
        let mut attr = ffi::ibv_qp_attr {
            qp_state: ffi::ibv_qp_state::IBV_QPS_INIT,
            pkey_index: 0,
            port_num: device.port,
            qp_access_flags: access,
            ..Default::default()
        };
        let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE.0
            | ffi::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX.0
            | ffi::ibv_qp_attr_mask::IBV_QP_PORT.0
            | ffi::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS.0;
        // SAFETY: the queue pair is in RESET and the attributes are valid for INIT.
        let status = unsafe { ffi::ibv_modify_qp(lane.qp, &mut attr, mask as i32) };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        Ok(lane)
    }

    pub(crate) fn local(&self) -> QueuePair {
        QueuePair {
            gid: self.device.gid,
            // SAFETY: the queue pair is live.
            qpn: unsafe { (*self.qp).qp_num },
            psn: self.psn,
        }
    }

    /// Move to RTS against `peer`. A requester reads with at most `depth` READs in flight; a
    /// responder accepts that many. Returns the depth the device confirmed.
    pub(crate) fn connect(&mut self, peer: &QueuePair, mtu: u32, depth: u8) -> io::Result<usize> {
        let depth = depth.max(1);
        let (initiator, responder) = match self.role {
            Role::Requester => (depth, 1),
            Role::Responder => (1, depth),
        };
        let mut attr = ffi::ibv_qp_attr {
            qp_state: ffi::ibv_qp_state::IBV_QPS_RTR,
            path_mtu: mtu_value(mtu)?,
            dest_qp_num: peer.qpn,
            rq_psn: peer.psn,
            max_dest_rd_atomic: responder,
            min_rnr_timer: 12,
            ah_attr: ffi::ibv_ah_attr {
                dlid: 0,
                port_num: self.device.port,
                is_global: 1,
                grh: ffi::ibv_global_route {
                    dgid: ffi::ibv_gid { raw: peer.gid },
                    sgid_index: self
                        .device
                        .gid_index
                        .load(std::sync::atomic::Ordering::Acquire),
                    hop_limit: 64,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE.0
            | ffi::ibv_qp_attr_mask::IBV_QP_AV.0
            | ffi::ibv_qp_attr_mask::IBV_QP_PATH_MTU.0
            | ffi::ibv_qp_attr_mask::IBV_QP_DEST_QPN.0
            | ffi::ibv_qp_attr_mask::IBV_QP_RQ_PSN.0
            | ffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC.0
            | ffi::ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER.0;
        // SAFETY: the queue pair is in INIT and the RTR attributes are valid.
        let status = unsafe { ffi::ibv_modify_qp(self.qp, &mut attr, mask as i32) };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        let mut attr = ffi::ibv_qp_attr {
            qp_state: ffi::ibv_qp_state::IBV_QPS_RTS,
            timeout: 14,
            retry_cnt: 7,
            rnr_retry: 7,
            sq_psn: self.psn,
            max_rd_atomic: initiator,
            ..Default::default()
        };
        let mask = ffi::ibv_qp_attr_mask::IBV_QP_STATE.0
            | ffi::ibv_qp_attr_mask::IBV_QP_TIMEOUT.0
            | ffi::ibv_qp_attr_mask::IBV_QP_RETRY_CNT.0
            | ffi::ibv_qp_attr_mask::IBV_QP_RNR_RETRY.0
            | ffi::ibv_qp_attr_mask::IBV_QP_SQ_PSN.0
            | ffi::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC.0;
        // SAFETY: the queue pair is in RTR and the RTS attributes are valid.
        let status = unsafe { ffi::ibv_modify_qp(self.qp, &mut attr, mask as i32) };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        let mut current = ffi::ibv_qp_attr::default();
        let mut init = ffi::ibv_qp_init_attr::default();
        let query = (ffi::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC.0
            | ffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC.0) as i32;
        // SAFETY: the queue pair is live and both outputs are writable.
        let status = unsafe { ffi::ibv_query_qp(self.qp, &mut current, query, &mut init) };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        let confirmed = match self.role {
            Role::Requester => current.max_rd_atomic,
            Role::Responder => current.max_dest_rd_atomic,
        };
        if confirmed == 0 {
            return Err(invalid("the device confirmed no RDMA READ depth"));
        }
        if self.role == Role::Requester {
            self.depth = self.depth.min(usize::from(confirmed));
        }
        Ok(usize::from(confirmed))
    }

    pub(crate) fn outstanding(&self) -> usize {
        self.outstanding
    }

    pub(crate) fn credit(&self) -> usize {
        self.depth - self.outstanding
    }

    /// Post `reads` as one chain of signalled READs into the region at `base` under `lkey`.
    /// The caller keeps `reads.len()` within [`Lane::credit`].
    pub(crate) fn post(&mut self, reads: &[Read], base: u64, lkey: u32) -> io::Result<()> {
        if reads.is_empty() {
            return Ok(());
        }
        if reads.len() > self.credit() {
            return Err(invalid("RDMA READ posting exceeds the confirmed depth"));
        }
        for (index, read) in reads.iter().enumerate() {
            self.sges[index] = ffi::ibv_sge {
                addr: base + read.local as u64,
                length: read.nbytes,
                lkey,
            };
            let wr = &mut self.wrs[index];
            *wr = unsafe { mem::zeroed() };
            wr.wr_id = index as u64;
            wr.num_sge = 1;
            wr.opcode = ffi::ibv_wr_opcode::IBV_WR_RDMA_READ;
            wr.send_flags = ffi::ibv_send_flags::IBV_SEND_SIGNALED.0;
            wr.wr.rdma = ffi::ibv_send_wr__bindgen_ty_2__bindgen_ty_1 {
                remote_addr: read.remote,
                rkey: read.rkey,
            };
        }
        let count = reads.len();
        let wrs = self.wrs.as_mut_ptr();
        let sges = self.sges.as_mut_ptr();
        for index in 0..count {
            // SAFETY: both vectors hold at least `count` entries and are not resized while
            // the chain is posted.
            unsafe {
                (*wrs.add(index)).sg_list = sges.add(index);
                (*wrs.add(index)).next =
                    if index + 1 < count { wrs.add(index + 1) } else { ptr::null_mut() };
            }
        }
        let mut bad: *mut ffi::ibv_send_wr = ptr::null_mut();
        // SAFETY: the context is open; the queue pair is RTS and every SGE refers to memory
        // registered under `lkey`, which the caller keeps registered until completion.
        let status = unsafe {
            let post = (*self.device.context)
                .ops
                .post_send
                .ok_or_else(|| invalid("libibverbs has no post_send"))?;
            post(self.qp, wrs, &mut bad)
        };
        if status != 0 {
            // Requests before the first rejected one were posted and will complete or flush.
            let posted = if bad.is_null() {
                0
            } else {
                // SAFETY: `bad` points into the posted chain.
                unsafe { bad.offset_from(wrs) }.clamp(0, count as isize) as usize
            };
            self.outstanding += posted;
            return Err(io::Error::from_raw_os_error(status));
        }
        self.outstanding += count;
        Ok(())
    }

    /// Take available completions. Returns the number completed and the first failure.
    pub(crate) fn poll(&mut self) -> io::Result<(usize, Option<Failure>)> {
        if self.outstanding == 0 {
            return Ok((0, None));
        }
        // SAFETY: the context is open and the CQ is live; `wcs` holds POLL_BATCH entries.
        let count = unsafe {
            let poll = (*self.device.context)
                .ops
                .poll_cq
                .ok_or_else(|| invalid("libibverbs has no poll_cq"))?;
            poll(self.cq, POLL_BATCH as i32, self.wcs.as_mut_ptr())
        };
        if count < 0 {
            return Err(io::Error::other("polling the RDMA completion queue failed"));
        }
        let count = count as usize;
        let mut failure = None;
        for completion in &self.wcs[..count] {
            if let Some((status, vendor)) = completion.error() {
                failure.get_or_insert_with(|| classify(status, vendor));
            }
        }
        self.outstanding = self.outstanding.saturating_sub(count);
        Ok((count, failure))
    }

    /// Move the queue pair to ERR so every posted READ completes, normally or flushed.
    pub(crate) fn fail(&mut self) -> io::Result<()> {
        let mut attr = ffi::ibv_qp_attr {
            qp_state: ffi::ibv_qp_state::IBV_QPS_ERR,
            ..Default::default()
        };
        // SAFETY: the queue pair is live.
        let status = unsafe {
            ffi::ibv_modify_qp(self.qp, &mut attr, ffi::ibv_qp_attr_mask::IBV_QP_STATE.0 as i32)
        };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        Ok(())
    }

    fn close(&mut self) -> io::Result<()> {
        if !self.qp.is_null() {
            self.fail()?;
            // SAFETY: owned here; destruction fences all access through this queue pair.
            let status = unsafe { ffi::ibv_destroy_qp(self.qp) };
            if status != 0 {
                return Err(io::Error::from_raw_os_error(status));
            }
            self.qp = ptr::null_mut();
        }
        if !self.cq.is_null() {
            // SAFETY: its queue pair is destroyed.
            let status = unsafe { ffi::ibv_destroy_cq(self.cq) };
            if status != 0 {
                return Err(io::Error::from_raw_os_error(status));
            }
            self.cq = ptr::null_mut();
        }
        Ok(())
    }
}

impl Drop for Lane {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            // A queue pair that cannot be destroyed may still access registered memory;
            // keep the device and its registrations alive for the rest of the process.
            eprintln!("warning: quarantining an unfenced Engram store queue pair: {error}");
            mem::forget(self.device.clone());
        }
    }
}
