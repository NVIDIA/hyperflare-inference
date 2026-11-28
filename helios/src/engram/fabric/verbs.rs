/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! libibverbs objects of the Engram store data path, on top of Selene's device and queue-pair
//! lifecycle (`flare_selene::shared::rdma`): the device limits RDMA READ needs, memory
//! registrations, and READ posting and polling on a reliable-connected queue pair.
//!
//! Posting and polling go through the context's provider operations, which libibverbs' inline
//! `ibv_post_send` and `ibv_poll_cq` call.

use super::{Failure, plan::Read, protocol::QueuePair};
use flare_selene::shared::rdma::{self, Handshake, Path, Qp, ffi};
use std::{ffi::CStr, io, mem, net::IpAddr, ptr, sync::Arc};

/// Completions taken per poll call.
const POLL_BATCH: usize = 32;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_owned())
}

/// The device and port whose RoCE v2 GID carries the selected local address, with the limits
/// of its RDMA READ queue pairs.
pub(crate) struct Device {
    inner: Arc<rdma::Device>,
    /// The port's active MTU in bytes.
    pub(crate) mtu: u32,
    /// RDMA READs one queue pair may have in flight as requester (`max_qp_init_rd_atom`).
    pub(crate) initiator_depth: u8,
    /// RDMA READs one queue pair may serve concurrently as responder (`max_qp_rd_atom`).
    pub(crate) responder_depth: u8,
    pub(crate) max_wr: u32,
    pub(crate) name: String,
}

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
        let inner = rdma::Device::open(local)?;
        let mut attr = ffi::ibv_device_attr::default();
        // SAFETY: the context is open and attr is writable.
        let status = unsafe { ffi::ibv_query_device(inner.context, &mut attr) };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        let mut port = ffi::ibv_port_attr::default();
        // SAFETY: the compat cast matches libibverbs' inline wrapper; the context is open.
        let status = unsafe {
            ffi::ibv_query_port(
                inner.context,
                inner.port(),
                &mut port as *mut ffi::ibv_port_attr as *mut ffi::_compat_ibv_port_attr,
            )
        };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        let mtu = mtu_bytes(port.active_mtu);
        if mtu == 0 {
            return Err(invalid("the RoCE port reports no supported active MTU"));
        }
        // SAFETY: an open context refers to its device, whose name libibverbs owns.
        let name = unsafe { CStr::from_ptr(ffi::ibv_get_device_name((*inner.context).device)) }
            .to_string_lossy()
            .into_owned();
        Ok(Arc::new(Self {
            inner,
            mtu,
            initiator_depth: depth(attr.max_qp_init_rd_atom),
            responder_depth: depth(attr.max_qp_rd_atom),
            max_wr: attr.max_qp_wr.max(1) as u32,
            name,
        }))
    }

    /// Find the local address in the port's GID table again; fails while the interface is down
    /// or no longer holds the address, which a later connection attempt retries.
    pub(crate) fn refresh(&self) -> io::Result<()> {
        self.inner.refresh()
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
        let mr = unsafe { ffi::ibv_reg_mr(device.inner.pd, address as *mut _, nbytes, access) };
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

/// One reliable-connected queue pair, with the READ accounting of its requester.
pub(crate) struct Lane {
    // Selene's pair destroys the queue pair before its completion queue, and quarantines its
    // device if destruction fails, when it drops.
    qp: Qp,
    device: Arc<Device>,
    role: Role,
    /// Confirmed RDMA READ depth after connection; the requester never exceeds it.
    depth: usize,
    outstanding: usize,
    wrs: Vec<ffi::ibv_send_wr>,
    sges: Vec<ffi::ibv_sge>,
    wcs: Vec<ffi::ibv_wc>,
}

// SAFETY: a lane is owned by one thread at a time (behind a mutex on the requester).
unsafe impl Send for Lane {}

impl Lane {
    /// Create a queue pair in INIT for `role`, sized for `depth` READs in flight.
    pub(crate) fn new(device: &Arc<Device>, role: Role, depth: u8) -> io::Result<Self> {
        let send_wr = match role {
            Role::Requester => u32::from(depth.max(1)).min(device.max_wr),
            Role::Responder => 1,
        };
        let access = match role {
            Role::Requester => 0,
            Role::Responder => ffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ.0,
        };
        let cqe = (send_wr as i32).saturating_mul(2).max(16);
        Ok(Self {
            qp: Qp::new(&device.inner, send_wr, cqe, access)?,
            device: device.clone(),
            role,
            depth: send_wr as usize,
            outstanding: 0,
            // SAFETY: work requests, scatter entries and completions are plain C records for
            // which all-zero bytes are valid; each is filled before libibverbs reads it.
            wrs: (0..send_wr).map(|_| unsafe { mem::zeroed() }).collect(),
            sges: (0..send_wr).map(|_| unsafe { mem::zeroed() }).collect(),
            wcs: (0..POLL_BATCH).map(|_| unsafe { mem::zeroed() }).collect(),
        })
    }

    pub(crate) fn local(&self) -> QueuePair {
        let handshake = self.qp.handshake(0, 0);
        QueuePair {
            gid: handshake.gid,
            qpn: handshake.qpn,
            psn: handshake.psn,
        }
    }

    /// Move to RTS against `peer`. A requester reads with at most `depth` READs in flight; a
    /// responder accepts that many. Returns the depth the device confirmed.
    pub(crate) fn connect(&mut self, peer: &QueuePair, mtu: u32, depth: u8) -> io::Result<usize> {
        let depth = depth.max(1);
        let (max_rd_atomic, max_dest_rd_atomic) = match self.role {
            Role::Requester => (depth, 1),
            Role::Responder => (1, depth),
        };
        self.qp.connect_path(
            &Handshake {
                qpn: peer.qpn,
                psn: peer.psn,
                gid: peer.gid,
                lid: 0,
                rkey_staging: 0,
                staging_base: 0,
            },
            Path {
                mtu: mtu_value(mtu)?,
                max_rd_atomic,
                max_dest_rd_atomic,
            },
        )?;
        let mut current = ffi::ibv_qp_attr::default();
        let mut init = ffi::ibv_qp_init_attr::default();
        let query = (ffi::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC.0
            | ffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC.0) as i32;
        // SAFETY: the queue pair is live and both outputs are writable.
        let status = unsafe { ffi::ibv_query_qp(self.qp.qp, &mut current, query, &mut init) };
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
    /// Returns how many were posted, which is fewer than requested only with an error; the
    /// posted ones complete or flush. The caller keeps `reads.len()` within [`Lane::credit`].
    pub(crate) fn post(&mut self, reads: &[Read], base: u64, lkey: u32) -> (usize, Option<io::Error>) {
        if reads.is_empty() {
            return (0, None);
        }
        if reads.len() > self.credit() {
            return (0, Some(invalid("RDMA READ posting exceeds the confirmed depth")));
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
        // SAFETY: the context is open.
        let Some(post) = (unsafe { (*self.device.inner.context).ops.post_send }) else {
            return (0, Some(invalid("libibverbs has no post_send")));
        };
        // SAFETY: the queue pair is RTS and every SGE refers to memory registered under
        // `lkey`, which the caller keeps registered until completion.
        let status = unsafe { post(self.qp.qp, wrs, &mut bad) };
        if status != 0 {
            // Requests before the first rejected one were posted and will complete or flush.
            let posted = if bad.is_null() {
                0
            } else {
                // SAFETY: `bad` points into the posted chain.
                unsafe { bad.offset_from(wrs) }.clamp(0, count as isize) as usize
            };
            self.outstanding += posted;
            return (posted, Some(io::Error::from_raw_os_error(status)));
        }
        self.outstanding += count;
        (count, None)
    }

    /// Take available completions. Returns the number completed and the first failure.
    pub(crate) fn poll(&mut self) -> io::Result<(usize, Option<Failure>)> {
        if self.outstanding == 0 {
            return Ok((0, None));
        }
        // SAFETY: the context is open and the CQ is live; `wcs` holds POLL_BATCH entries.
        let count = unsafe {
            let poll = (*self.device.inner.context)
                .ops
                .poll_cq
                .ok_or_else(|| invalid("libibverbs has no poll_cq"))?;
            poll(self.qp.cq, POLL_BATCH as i32, self.wcs.as_mut_ptr())
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
            ffi::ibv_modify_qp(self.qp.qp, &mut attr, ffi::ibv_qp_attr_mask::IBV_QP_STATE.0 as i32)
        };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
        Ok(())
    }
}
