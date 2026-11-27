/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Control messages between a serving process and its Engram store.
//!
//! Every message is a big-endian `u32` length followed by that many bytes: one kind byte and
//! the body. `HELLO` and `HELLO_OK` begin with a four-byte magic and a protocol version; a
//! peer that implements a different version is refused before the rest of its body is read.
//! Decoding failures are `InvalidData`; every other I/O error means the connection failed.

use std::io::{self, Read, Write};

pub(crate) const MAGIC: [u8; 4] = *b"FEGS";
pub(crate) const VERSION: u16 = 1;
/// Layout of the store's registered planes as `HELLO_OK` describes them.
pub(crate) const STORE_ABI: u16 = 1;
pub(crate) const MAX_MESSAGE: usize = 1 << 20;
pub(crate) const MAX_TABLES: usize = 4096;
pub(crate) const MAX_LANES: usize = 8;
const PLANES: u8 = 2;

const HELLO: u8 = 1;
const HELLO_OK: u8 = 2;
const ERROR: u8 = 3;
const QP_EXCHANGE: u8 = 4;
const HEARTBEAT: u8 = 5;
const BYE: u8 = 6;

/// One plane of a table as the client's core declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneLayout {
    pub row_bytes: u32,
    pub nbytes: u64,
}

/// One registered plane in the store: rows start at `address` under `rkey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RemotePlane {
    pub rkey: u32,
    pub address: u64,
    pub nbytes: u64,
    pub row_bytes: u32,
}

/// A table of the client's catalog. `id` is the table's position in the core's Engram catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TableLayout {
    pub id: u32,
    pub planes: [PlaneLayout; 2],
}

/// The store's registration of one requested table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RemoteTable {
    pub id: u32,
    pub planes: [RemotePlane; 2],
}

/// One end of a reliable-connected queue pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueuePair {
    pub gid: [u8; 16],
    pub qpn: u32,
    pub psn: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Code {
    VersionUnsupported,
    CoreMismatch,
    PlaneLayoutMismatch,
    InsufficientMemory,
    NotReady,
}

impl Code {
    fn wire(self) -> u16 {
        match self {
            Self::VersionUnsupported => 1,
            Self::CoreMismatch => 2,
            Self::PlaneLayoutMismatch => 3,
            Self::InsufficientMemory => 4,
            Self::NotReady => 5,
        }
    }

    fn from_wire(value: u16) -> io::Result<Self> {
        Ok(match value {
            1 => Self::VersionUnsupported,
            2 => Self::CoreMismatch,
            3 => Self::PlaneLayoutMismatch,
            4 => Self::InsufficientMemory,
            5 => Self::NotReady,
            _ => return Err(invalid("unknown Engram store error code")),
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::VersionUnsupported => "version_unsupported",
            Self::CoreMismatch => "core_mismatch",
            Self::PlaneLayoutMismatch => "plane_layout_mismatch",
            Self::InsufficientMemory => "insufficient_memory",
            Self::NotReady => "not_ready",
        }
    }

    /// A store that is still loading or short of queue-pair memory may accept a later attempt;
    /// a version, identity, or layout disagreement persists until an operator changes a host.
    pub(crate) fn retryable(self) -> bool {
        matches!(self, Self::InsufficientMemory | Self::NotReady)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Hello {
    pub identity: [u8; 32],
    pub rank: u32,
    pub world: u32,
    pub lanes: u8,
    /// The client's active path MTU in bytes; the store answers with the smaller of both.
    pub mtu: u32,
    pub tables: Vec<TableLayout>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HelloOk {
    pub abi: u16,
    pub gid: [u8; 16],
    pub mtu: u32,
    pub max_read_bytes: u32,
    pub lanes: u8,
    /// RDMA READs each store queue pair accepts in flight (its `max_dest_rd_atomic`).
    pub depth: u8,
    pub tables: Vec<RemoteTable>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Message {
    Hello(Hello),
    /// A `HELLO` whose version this side does not implement; its body is not interpreted.
    HelloVersion(u16),
    HelloOk(HelloOk),
    HelloOkVersion(u16),
    Error { code: Code, message: String },
    QpExchange(Vec<QueuePair>),
    Heartbeat(u64),
    Bye,
}

pub(crate) fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_owned())
}

/// MTU values a RoCE path can use, in bytes.
pub(crate) fn valid_mtu(mtu: u32) -> bool {
    matches!(mtu, 256 | 512 | 1024 | 2048 | 4096)
}

pub(crate) fn encode(message: &Message) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    match message {
        Message::Hello(hello) => {
            body.push(HELLO);
            body.extend_from_slice(&MAGIC);
            body.extend_from_slice(&VERSION.to_be_bytes());
            body.extend_from_slice(&hello.identity);
            body.extend_from_slice(&hello.rank.to_be_bytes());
            body.extend_from_slice(&hello.world.to_be_bytes());
            body.push(hello.lanes);
            body.extend_from_slice(&hello.mtu.to_be_bytes());
            count(&mut body, hello.tables.len(), MAX_TABLES)?;
            for table in &hello.tables {
                body.extend_from_slice(&table.id.to_be_bytes());
                body.push(PLANES);
                for plane in &table.planes {
                    body.extend_from_slice(&plane.row_bytes.to_be_bytes());
                    body.extend_from_slice(&plane.nbytes.to_be_bytes());
                }
            }
        }
        Message::HelloOk(ok) => {
            body.push(HELLO_OK);
            body.extend_from_slice(&MAGIC);
            body.extend_from_slice(&VERSION.to_be_bytes());
            body.extend_from_slice(&ok.abi.to_be_bytes());
            body.extend_from_slice(&ok.gid);
            body.extend_from_slice(&ok.mtu.to_be_bytes());
            body.extend_from_slice(&ok.max_read_bytes.to_be_bytes());
            body.push(ok.lanes);
            body.push(ok.depth);
            count(&mut body, ok.tables.len(), MAX_TABLES)?;
            for table in &ok.tables {
                body.extend_from_slice(&table.id.to_be_bytes());
                body.push(PLANES);
                for plane in &table.planes {
                    body.extend_from_slice(&plane.rkey.to_be_bytes());
                    body.extend_from_slice(&plane.address.to_be_bytes());
                    body.extend_from_slice(&plane.nbytes.to_be_bytes());
                    body.extend_from_slice(&plane.row_bytes.to_be_bytes());
                }
            }
        }
        Message::HelloVersion(_) | Message::HelloOkVersion(_) => {
            return Err(invalid("an unsupported version is decoded, never sent"));
        }
        Message::Error { code, message } => {
            body.push(ERROR);
            body.extend_from_slice(&code.wire().to_be_bytes());
            let text = message.as_bytes();
            let text = &text[..text.len().min(4096)];
            body.extend_from_slice(&(text.len() as u32).to_be_bytes());
            body.extend_from_slice(text);
        }
        Message::QpExchange(lanes) => {
            body.push(QP_EXCHANGE);
            if lanes.is_empty() || lanes.len() > MAX_LANES {
                return Err(invalid("queue-pair exchange lane count is out of range"));
            }
            body.push(lanes.len() as u8);
            for lane in lanes {
                body.extend_from_slice(&lane.gid);
                body.extend_from_slice(&lane.qpn.to_be_bytes());
                body.extend_from_slice(&lane.psn.to_be_bytes());
            }
        }
        Message::Heartbeat(sequence) => {
            body.push(HEARTBEAT);
            body.extend_from_slice(&sequence.to_be_bytes());
        }
        Message::Bye => body.push(BYE),
    }
    if body.len() > MAX_MESSAGE {
        return Err(invalid("Engram store control message exceeds its size limit"));
    }
    let mut framed = Vec::with_capacity(4 + body.len());
    framed.extend_from_slice(&(body.len() as u32).to_be_bytes());
    framed.extend_from_slice(&body);
    Ok(framed)
}

fn count(body: &mut Vec<u8>, value: usize, limit: usize) -> io::Result<()> {
    if value > limit {
        return Err(invalid("Engram store table count exceeds its limit"));
    }
    body.extend_from_slice(&(value as u32).to_be_bytes());
    Ok(())
}

pub(crate) fn write(stream: &mut impl Write, message: &Message) -> io::Result<()> {
    stream.write_all(&encode(message)?)?;
    stream.flush()
}

pub(crate) fn read(stream: &mut impl Read) -> io::Result<Message> {
    let mut prefix = [0u8; 4];
    stream.read_exact(&mut prefix)?;
    let length = u32::from_be_bytes(prefix) as usize;
    if length == 0 || length > MAX_MESSAGE {
        return Err(invalid("Engram store control message length is out of range"));
    }
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body)?;
    decode(&body)
}

/// Accumulates one connection's bytes across reads that time out, so a heartbeat loop with a
/// short read timeout never splits or drops a message.
#[derive(Default)]
pub(crate) struct Receiver {
    buffer: Vec<u8>,
}

impl Receiver {
    /// The next complete message, or `None` when the stream's read timeout expired first.
    pub(crate) fn poll(&mut self, stream: &mut impl Read) -> io::Result<Option<Message>> {
        loop {
            if self.buffer.len() >= 4 {
                let length = u32::from_be_bytes(self.buffer[..4].try_into().unwrap()) as usize;
                if length == 0 || length > MAX_MESSAGE {
                    return Err(invalid("Engram store control message length is out of range"));
                }
                if self.buffer.len() >= 4 + length {
                    let message = decode(&self.buffer[4..4 + length]);
                    self.buffer.drain(..4 + length);
                    return message.map(Some);
                }
            }
            let mut chunk = [0u8; 4096];
            match stream.read(&mut chunk) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Engram store control connection closed",
                    ));
                }
                Ok(count) => self.buffer.extend_from_slice(&chunk[..count]),
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    return Ok(None);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| invalid("truncated Engram store control message"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn array<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }
    fn tables(&mut self) -> io::Result<usize> {
        let count = self.u32()? as usize;
        if count > MAX_TABLES {
            return Err(invalid("Engram store table count exceeds its limit"));
        }
        Ok(count)
    }
    fn planes(&mut self) -> io::Result<()> {
        if self.u8()? != PLANES {
            return Err(invalid("an Engram table has exactly a weight and a scale plane"));
        }
        Ok(())
    }
    /// Read the magic and version; `None` when the version is foreign and the rest is opaque.
    fn version(&mut self) -> io::Result<Option<u16>> {
        if self.array::<4>()? != MAGIC {
            return Err(invalid("peer is not an Engram store endpoint"));
        }
        let version = self.u16()?;
        Ok((version != VERSION).then_some(version))
    }
    fn end(&self) -> io::Result<()> {
        if self.at != self.bytes.len() {
            return Err(invalid("trailing bytes in Engram store control message"));
        }
        Ok(())
    }
}

pub(crate) fn decode(body: &[u8]) -> io::Result<Message> {
    let mut cursor = Cursor { bytes: body, at: 0 };
    let message = match cursor.u8()? {
        HELLO => {
            if let Some(version) = cursor.version()? {
                return Ok(Message::HelloVersion(version));
            }
            let identity = cursor.array::<32>()?;
            let rank = cursor.u32()?;
            let world = cursor.u32()?;
            let lanes = cursor.u8()?;
            let mtu = cursor.u32()?;
            let count = cursor.tables()?;
            let mut tables = Vec::with_capacity(count);
            for _ in 0..count {
                let id = cursor.u32()?;
                cursor.planes()?;
                let mut plane = || -> io::Result<PlaneLayout> {
                    Ok(PlaneLayout { row_bytes: cursor.u32()?, nbytes: cursor.u64()? })
                };
                let planes = [plane()?, plane()?];
                tables.push(TableLayout { id, planes });
            }
            Message::Hello(Hello { identity, rank, world, lanes, mtu, tables })
        }
        HELLO_OK => {
            if let Some(version) = cursor.version()? {
                return Ok(Message::HelloOkVersion(version));
            }
            let abi = cursor.u16()?;
            let gid = cursor.array::<16>()?;
            let mtu = cursor.u32()?;
            let max_read_bytes = cursor.u32()?;
            let lanes = cursor.u8()?;
            let depth = cursor.u8()?;
            let count = cursor.tables()?;
            let mut tables = Vec::with_capacity(count);
            for _ in 0..count {
                let id = cursor.u32()?;
                cursor.planes()?;
                let mut plane = || -> io::Result<RemotePlane> {
                    Ok(RemotePlane {
                        rkey: cursor.u32()?,
                        address: cursor.u64()?,
                        nbytes: cursor.u64()?,
                        row_bytes: cursor.u32()?,
                    })
                };
                let planes = [plane()?, plane()?];
                tables.push(RemoteTable { id, planes });
            }
            Message::HelloOk(HelloOk { abi, gid, mtu, max_read_bytes, lanes, depth, tables })
        }
        ERROR => {
            let code = Code::from_wire(cursor.u16()?)?;
            let length = cursor.u32()? as usize;
            if length > 4096 {
                return Err(invalid("Engram store error message exceeds its size limit"));
            }
            let message = String::from_utf8_lossy(cursor.take(length)?).into_owned();
            Message::Error { code, message }
        }
        QP_EXCHANGE => {
            let count = cursor.u8()? as usize;
            if count == 0 || count > MAX_LANES {
                return Err(invalid("queue-pair exchange lane count is out of range"));
            }
            let mut lanes = Vec::with_capacity(count);
            for _ in 0..count {
                lanes.push(QueuePair {
                    gid: cursor.array::<16>()?,
                    qpn: cursor.u32()?,
                    psn: cursor.u32()?,
                });
            }
            Message::QpExchange(lanes)
        }
        HEARTBEAT => Message::Heartbeat(cursor.u64()?),
        BYE => Message::Bye,
        _ => return Err(invalid("unknown Engram store control message kind")),
    };
    cursor.end()?;
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello() -> Hello {
        Hello {
            identity: [7; 32],
            rank: 2,
            world: 4,
            lanes: 4,
            mtu: 4096,
            tables: (0..3)
                .map(|id| TableLayout {
                    id,
                    planes: [
                        PlaneLayout { row_bytes: 256, nbytes: 256 * 1000 + u64::from(id) * 256 },
                        PlaneLayout { row_bytes: 8, nbytes: 8 * 1000 + u64::from(id) * 8 },
                    ],
                })
                .collect(),
        }
    }

    fn hello_ok() -> HelloOk {
        HelloOk {
            abi: STORE_ABI,
            gid: [0xfe; 16],
            mtu: 1024,
            max_read_bytes: 1 << 20,
            lanes: 2,
            depth: 16,
            tables: vec![RemoteTable {
                id: 0,
                planes: [
                    RemotePlane { rkey: 0x1234, address: 0x7f00_0000_1000, nbytes: 4096, row_bytes: 256 },
                    RemotePlane { rkey: 0x5678, address: 0x7f00_2000_0000, nbytes: 128, row_bytes: 8 },
                ],
            }],
        }
    }

    #[test]
    fn every_message_round_trips_through_its_length_prefixed_frame() {
        let messages = [
            Message::Hello(hello()),
            Message::HelloOk(hello_ok()),
            Message::Error { code: Code::CoreMismatch, message: "store holds another core".into() },
            Message::QpExchange(vec![
                QueuePair { gid: [1; 16], qpn: 0x11, psn: 0x22 },
                QueuePair { gid: [2; 16], qpn: 0x33, psn: 0x44 },
            ]),
            Message::Heartbeat(u64::MAX - 3),
            Message::Bye,
        ];
        for message in messages {
            let framed = encode(&message).unwrap();
            assert_eq!(u32::from_be_bytes(framed[..4].try_into().unwrap()) as usize, framed.len() - 4);
            assert_eq!(read(&mut framed.as_slice()).unwrap(), message);
        }
    }

    #[test]
    fn fields_are_big_endian_after_the_kind_byte() {
        let framed = encode(&Message::Heartbeat(0x0102_0304_0506_0708)).unwrap();
        assert_eq!(framed, [0, 0, 0, 9, HEARTBEAT, 1, 2, 3, 4, 5, 6, 7, 8]);
        let framed = encode(&Message::Hello(hello())).unwrap();
        assert_eq!(&framed[5..9], b"FEGS");
        assert_eq!(&framed[9..11], &VERSION.to_be_bytes());
        assert_eq!(&framed[43..47], &2u32.to_be_bytes());
    }

    #[test]
    fn a_foreign_version_is_reported_without_reading_its_body() {
        for (kind, expected) in [(HELLO, Message::HelloVersion(9)), (HELLO_OK, Message::HelloOkVersion(9))] {
            let mut body = vec![kind];
            body.extend_from_slice(&MAGIC);
            body.extend_from_slice(&9u16.to_be_bytes());
            body.extend_from_slice(b"an unknown future layout");
            assert_eq!(decode(&body).unwrap(), expected);
        }
        let mut body = vec![HELLO];
        body.extend_from_slice(b"HTTP");
        body.extend_from_slice(&VERSION.to_be_bytes());
        assert_eq!(decode(&body).unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    /// Serves bytes in fixed slices and reports a read timeout between them.
    struct Trickle {
        bytes: Vec<u8>,
        at: usize,
        slice: usize,
        paused: bool,
    }

    impl Read for Trickle {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            self.paused = !self.paused;
            if self.paused {
                return Err(io::ErrorKind::WouldBlock.into());
            }
            let count = self.slice.min(out.len()).min(self.bytes.len() - self.at);
            out[..count].copy_from_slice(&self.bytes[self.at..self.at + count]);
            self.at += count;
            Ok(count)
        }
    }

    #[test]
    fn the_receiver_reassembles_messages_split_across_read_timeouts() {
        let messages = [Message::Heartbeat(1), Message::Hello(hello()), Message::Bye];
        let bytes = messages.iter().flat_map(|m| encode(m).unwrap()).collect::<Vec<_>>();
        for slice in [1, 3, 64, 4096] {
            let mut stream = Trickle { bytes: bytes.clone(), at: 0, slice, paused: false };
            let mut receiver = Receiver::default();
            let mut received = Vec::new();
            while received.len() < messages.len() {
                if let Some(message) = receiver.poll(&mut stream).unwrap() {
                    received.push(message);
                }
            }
            assert_eq!(received, messages, "slice {slice}");
            let error = loop {
                match receiver.poll(&mut stream) {
                    Ok(None) => continue,
                    Ok(Some(message)) => panic!("unexpected {message:?}"),
                    Err(error) => break error,
                }
            };
            assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        }
        let mut oversized = Trickle { bytes: u32::MAX.to_be_bytes().to_vec(), at: 0, slice: 4, paused: true };
        assert_eq!(Receiver::default().poll(&mut oversized).unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn malformed_frames_are_invalid_data_and_short_streams_are_connection_errors() {
        let framed = encode(&Message::Hello(hello())).unwrap();
        for cut in 5..framed.len() {
            let body = framed[4..cut].to_vec();
            assert_eq!(decode(&body).unwrap_err().kind(), io::ErrorKind::InvalidData, "cut {cut}");
        }
        let mut body = framed[4..].to_vec();
        body.push(0);
        assert_eq!(decode(&body).unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            read(&mut &framed[..framed.len() - 1]).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        for length in [0u32, MAX_MESSAGE as u32 + 1] {
            let prefix = length.to_be_bytes();
            assert_eq!(read(&mut prefix.as_slice()).unwrap_err().kind(), io::ErrorKind::InvalidData);
        }
        assert!(decode(&[0x7f]).is_err());
        let mut planes = encode(&Message::Hello(hello())).unwrap()[4..].to_vec();
        // The first table's plane count follows the fixed fields and the table id.
        let at = 1 + 4 + 2 + 32 + 4 + 4 + 1 + 4 + 4 + 4;
        assert_eq!(planes[at], PLANES);
        planes[at] = 3;
        assert!(decode(&planes).is_err());
        assert!(decode(&[ERROR, 0, 99, 0, 0, 0, 0]).is_err());
        assert!(encode(&Message::QpExchange(Vec::new())).is_err());
        assert!(encode(&Message::QpExchange(vec![QueuePair { gid: [0; 16], qpn: 0, psn: 0 }; 9])).is_err());
    }

    #[test]
    fn error_codes_keep_their_wire_values_and_retry_classes() {
        for (code, wire, name, retryable) in [
            (Code::VersionUnsupported, 1, "version_unsupported", false),
            (Code::CoreMismatch, 2, "core_mismatch", false),
            (Code::PlaneLayoutMismatch, 3, "plane_layout_mismatch", false),
            (Code::InsufficientMemory, 4, "insufficient_memory", true),
            (Code::NotReady, 5, "not_ready", true),
        ] {
            assert_eq!(code.wire(), wire);
            assert_eq!(Code::from_wire(wire).unwrap(), code);
            assert_eq!(code.name(), name);
            assert_eq!(code.retryable(), retryable);
        }
        assert!(Code::from_wire(0).is_err());
    }
}
