/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded protobuf indexes and resource reads inside a safetensors container.

use super::{
    format::{Core, invalid},
    protection::{Protection, Publisher, ResourceReader},
    protobuf::wire,
};
use crate::Result;
use crate::os::FileReadAt;
use prost::Message;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Take},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub const MAX_LEGACY_HEADER: u64 = 64 << 10;
pub const MAX_LEGACY_INDEX: u64 = 128 << 20;
pub const MAX_LEGACY_PAYLOAD: u64 = (64u64 << 30) - 4096;
const MAX_RECORD: usize = 1 << 20;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    #[serde(rename = "__metadata__")]
    format: Format,
    #[serde(rename = "flare.payload")]
    payload: ByteTensor,
    #[serde(rename = "flare.metadata", default)]
    index: Option<ByteTensor>,
    #[serde(rename = "flare.layout", default)]
    layout: Option<ByteTensor>,
}

/// Validated safetensors header for legacy (`flare`, `flare-staging`, `flare-snapshot`, `flare-data`) containers.
pub struct LegacyHeader {
    pub format: String,
    pub data_start: u64,
    pub payload_length: u64,
    pub index_length: Option<u64>,
    pub layout_span: Option<(u64, u64)>,
    pub raw_header: Vec<u8>,
}

/// Read and validate a legacy safetensors container header against file size, alignment,
/// zero-byte hygiene, and contiguous U8 tensor layout.
pub fn read_legacy_header(file: &File) -> Result<LegacyHeader> {
    let info = file.metadata().map_err(|e| e.to_string())?;
    if !info.is_file() {
        return Err(invalid("core file must be a regular file"));
    }
    let mut prefix = [0u8; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| e.to_string())?;
    let header_size = u64::from_le_bytes(prefix);
    let data_start = header_size
        .checked_add(8)
        .ok_or_else(|| invalid("invalid safetensors header size"))?;
    if header_size == 0 || header_size > MAX_LEGACY_HEADER {
        return Err(invalid("invalid safetensors header size"));
    }
    if data_start % 4096 != 0 {
        return Err(invalid("invalid or unaligned safetensors header"));
    }
    let mut raw_header = vec![0u8; header_size as usize];
    file.read_exact_at(&mut raw_header, 8)
        .map_err(|e| e.to_string())?;
    if raw_header[0] != b'{' || raw_header.contains(&0) {
        return Err(invalid("invalid safetensors header"));
    }
    let header: Header = serde_json::from_slice(&raw_header)
        .map_err(|e| invalid(format!("invalid safetensors header: {e}")))?;
    let payload_length = header.payload.size()?;
    if header.payload.data_offsets[0] != 0 {
        return Err(invalid(
            "core tensors must cover the data buffer without gaps or overlap",
        ));
    }
    let mut end = payload_length;
    let mut index_length = None;
    let mut layout_span = None;
    match header.format.format.as_str() {
        "flare-data" => {
            if header.index.is_some() || header.layout.is_some() {
                return Err(invalid("unexpected companion metadata"));
            }
        }
        "flare" | "flare-staging" | "flare-snapshot" => {
            let index = header
                .index
                .ok_or_else(|| invalid("missing model metadata"))?;
            let size = index.size()?;
            if size == 0 || size > MAX_LEGACY_INDEX {
                return Err(invalid("invalid model index size"));
            }
            if index.data_offsets[0] != payload_length {
                return Err(invalid(
                    "core tensors must cover the data buffer without gaps or overlap",
                ));
            }
            index_length = Some(size);
            end = index.data_offsets[1];
            if header.format.format == "flare-snapshot" {
                let tensor = header
                    .layout
                    .ok_or_else(|| invalid("snapshot lacks storage layout"))?;
                let count = tensor.size()?;
                if payload_length != 0
                    || tensor.data_offsets[0] != end
                    || count == 0
                    || count > MAX_LEGACY_INDEX
                {
                    return Err(invalid("invalid snapshot layout tensor"));
                }
                layout_span = Some((tensor.data_offsets[0], count));
                end = tensor.data_offsets[1];
            } else if header.layout.is_some() {
                return Err(invalid("unexpected snapshot layout"));
            }
        }
        _ => return Err(invalid("not a Flare safetensors core")),
    }
    if info.len().checked_sub(data_start) != Some(end) {
        return Err(invalid(
            "safetensors data length differs from model core",
        ));
    }
    Ok(LegacyHeader {
        format: header.format.format,
        data_start,
        payload_length,
        index_length,
        layout_span,
        raw_header,
    })
}

pub fn is_snapshot_blob_name(name: &str) -> bool {
    name.len() == 76
        && name.ends_with(".safetensors")
        && name.as_bytes()[..64]
            .iter()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
}

fn unique_files<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, Vec<(u64, String)>>, D::Error> {
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, Vec<(u64, String)>>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("unique logical core files")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut result = BTreeMap::new();
            while let Some((name, parts)) = map.next_entry::<String, Vec<(u64, String)>>()? {
                if result.insert(name, parts).is_some() {
                    return Err(serde::de::Error::custom("duplicate logical core file"));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Unique)
}

#[derive(Clone, Debug, serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyLayout {
    pub version: u32,
    #[serde(deserialize_with = "unique_files")]
    pub files: BTreeMap<String, Vec<(u64, String)>>,
}

type Layout = LegacyLayout;

impl LegacyLayout {
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 || !self.files.contains_key("") || self.files.len() > 100 {
            return Err(invalid("invalid snapshot layout"));
        }
        let mut count = 0;
        let mut lengths = BTreeMap::new();
        for (name, parts) in &self.files {
            if !name.is_empty() {
                validate_filename(name)?;
            }
            let mut total = 0u64;
            for (size, blob) in parts {
                count += 1;
                if *size == 0
                    || *size > MAX_LEGACY_PAYLOAD
                    || count > 65536
                    || (!blob.is_empty() && !is_snapshot_blob_name(blob))
                {
                    return Err(invalid("invalid snapshot blob"));
                }
                total = total
                    .checked_add(*size)
                    .ok_or_else(|| invalid("snapshot length overflow"))?;
                if !blob.is_empty()
                    && lengths
                        .insert(blob, size)
                        .is_some_and(|previous| previous != size)
                {
                    return Err(invalid("conflicting snapshot blob lengths"));
                }
            }
        }
        Ok(())
    }

    pub fn read_from_header(file: &File, header: &LegacyHeader) -> Result<Option<Self>> {
        let Some((offset, count)) = header.layout_span else {
            return Ok(None);
        };
        let mut bytes = vec![0u8; count as usize];
        file.read_exact_at(&mut bytes, header.data_start + offset)
            .map_err(|e| e.to_string())?;
        let layout: Self = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        layout.validate()?;
        Ok(Some(layout))
    }

    fn segments(&self, span: &wire::Span) -> Result<Vec<(Option<String>, u64, u64)>> {
        let parts = self
            .files
            .get(&span.filename)
            .ok_or_else(|| invalid("missing logical core file"))?;
        let end = span
            .offset
            .checked_add(span.nbytes)
            .ok_or_else(|| invalid("resource overflow"))?;
        let mut cursor = 0u64;
        let mut result = Vec::new();
        for (size, name) in parts {
            let stop = cursor + size;
            if cursor < end && span.offset < stop {
                let start = cursor.max(span.offset);
                result.push((
                    (!name.is_empty()).then_some(name.clone()),
                    start - cursor,
                    end.min(stop) - start,
                ));
            }
            cursor = stop;
        }
        if end > cursor {
            return Err(invalid("resource exceeds logical payload"));
        }
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Format {
    format: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ByteTensor {
    dtype: String,
    shape: [u64; 1],
    data_offsets: [u64; 2],
}
impl ByteTensor {
    fn size(&self) -> Result<u64> {
        let [begin, end] = self.data_offsets;
        if self.dtype != "U8" || end.checked_sub(begin) != Some(self.shape[0]) {
            return Err(invalid("invalid core byte tensor"));
        }
        Ok(self.shape[0])
    }
}

/// One core file retained while a peer receives selected byte ranges.
#[derive(Clone)]
pub struct CoreFile {
    /// Empty for the primary; companions retain their signed sibling basenames.
    pub filename: String,
    pub file: Arc<File>,
    pub ranges: Vec<(u64, u64)>,
}

fn merge_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.retain(|&(_, nbytes)| nbytes > 0);
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (offset, nbytes) in ranges {
        match merged.last_mut() {
            Some((first, extent)) if offset <= *first + *extent => {
                *extent = (*extent).max(offset + nbytes - *first);
            }
            _ => merged.push((offset, nbytes)),
        }
    }
    merged
}

pub fn validate_companion_filename(name: &str) -> Result<()> {
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    if name.len() > 255
        || !name.ends_with(".safetensors")
        || name.contains("..")
        || !name
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        || matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
    {
        return Err(invalid("invalid core companion filename"));
    }
    Ok(())
}

pub(super) fn validate_filename(name: &str) -> Result<()> {
    validate_companion_filename(name)
}

pub(super) struct PayloadFile {
    pub file: Arc<File>,
    pub start: u64,
    pub length: u64,
    direct: Arc<Mutex<Option<Arc<File>>>>,
}
impl PayloadFile {
    pub(super) fn new(file: Arc<File>, start: u64, length: u64) -> Self {
        Self::new_shared(file, start, length, Arc::new(Mutex::new(None)))
    }
    /// Share one cached direct-I/O handle with other payloads in the same container. Payloads that
    /// live in one packed file must not each reopen it.
    pub(super) fn new_shared(
        file: Arc<File>,
        start: u64,
        length: u64,
        direct: Arc<Mutex<Option<Arc<File>>>>,
    ) -> Self {
        Self {
            file,
            start,
            length,
            direct,
        }
    }
    pub fn direct(&self) -> Result<Arc<File>> {
        let mut cached = self.direct.lock().unwrap();
        if cached.is_none() {
            *cached = Some(Arc::new(crate::os::direct_reopen(&self.file).map_err(
                |e| invalid(format!("open core payload for direct I/O: {e}")),
            )?));
        }
        Ok(cached.as_ref().unwrap().clone())
    }
    fn companion(file: File) -> Result<Self> {
        let header = read_legacy_header(&file)?;
        if header.format != "flare-data" {
            return Err(invalid("invalid companion payload length or format"));
        }
        Ok(PayloadFile::new(
            Arc::new(file),
            header.data_start,
            header.payload_length,
        ))
    }
}

/// Inspect bounded metadata for sibling files before authentication, without opening any sibling.
/// Callers may fetch these names from the primary's pinned source; signatures still require verification.
pub fn core_companions(path: &Path) -> Result<Vec<String>> {
    // A distributed directory core names its companions through signed location
    // records; only a packed snapshot carries them in the archive layout.
    let primary = if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_owned()
    };
    if crate::core::reader::declared_format(&primary)?
        == Some(crate::core::container::PAYLOAD_FORMAT.to_owned())
    {
        let metadata = crate::core::reader::Metadata::open(&primary)?;
        let names: std::collections::BTreeSet<String> = metadata
            .locations
            .locations
            .iter()
            .filter(|location| !location.file.is_empty())
            .map(|location| location.file.clone())
            .collect();
        return Ok(names.into_iter().collect());
    }
    let (_, _, _, bytes, layout) = Archive::envelope(path)?;
    if let Some(layout) = layout {
        return Ok(layout
            .files
            .values()
            .flatten()
            .filter_map(|(_, name)| (!name.is_empty()).then_some(name.clone()))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect());
    }
    let envelope =
        wire::ProtectedIndex::decode(bytes.as_slice()).map_err(|e| invalid(e.to_string()))?;
    super::protection::ExecutableEncoding::from_version(envelope.version)?;
    if envelope.groups.len() != 3
        || envelope.groups[0].kind != 1
        || envelope.ranks.is_empty()
    {
        return Err(invalid("invalid protected core envelope"));
    }
    let metadata = wire::PublisherMetadata::decode(envelope.groups[0].body.as_slice())
        .map_err(|e| invalid(e.to_string()))?;
    let mut index = metadata
        .index
        .ok_or_else(|| invalid("missing model index"))?;
    if !matches!(index.version, 4 | 5) {
        return Err(invalid("unsupported model index"));
    }
    let mut parts = wire::StoredData::decode(envelope.groups[2].body.as_slice())
        .map_err(|e| invalid(e.to_string()))?
        .parts;
    for group in envelope.ranks {
        let rank = wire::RankProtection::decode(group.body.as_slice())
            .map_err(|e| invalid(e.to_string()))?;
        index
            .ranks
            .push(rank.bundle.ok_or_else(|| invalid("missing rank bundle"))?);
        parts.extend(rank.data.unwrap_or_default().parts);
    }
    let mut names = std::collections::BTreeSet::new();
    for span in index
        .ranks
        .iter()
        .flat_map(|rank| rank.resources.values())
        .chain(
            index
                .assets
                .iter()
                .filter_map(|asset| asset.payload.as_ref()),
        )
        .chain(
            super::engram::spans(&index, None)?
                .into_iter()
                .map(|(_, span)| span),
        )
        .chain(&parts)
    {
        if !span.filename.is_empty() {
            validate_filename(&span.filename)?;
            if index.version != 5 {
                return Err(invalid("companion references require model index 5"));
            }
            names.insert(span.filename.clone());
            if names.len() > 99 {
                return Err(invalid("core references more than 99 data files"));
            }
        }
    }
    Ok(names.into_iter().collect())
}

pub(super) struct Archive {
    pub path: PathBuf,
    pub start: u64,
    /// Payload bytes after `start`; the signed index follows them to the end of the file.
    pub payload: u64,
    pub file: Arc<File>,
    pub files: BTreeMap<String, PayloadFile>,
    layout: Option<Layout>,
    _directory: File,
    /// SHA-256 of the exact signed envelope bytes read at open.
    pub(super) envelope_digest: [u8; 32],
    pub(super) protection: Protection,
}
impl Archive {
    /// SHA-256 of the exact signed envelope bytes authenticated for this copy.
    pub(super) fn envelope_digest(&self) -> [u8; 32] {
        self.envelope_digest
    }

    /// Open the file and read its signed envelope: the file, the payload start and size, the
    /// envelope bytes, and any snapshot layout, none of them verified yet.
    pub(super) fn envelope(path: &Path) -> Result<(File, u64, u64, Vec<u8>, Option<Layout>)> {
        let primary_path = path.is_dir().then(|| path.join("model.safetensors"));
        let path = primary_path.as_deref().unwrap_or(path);
        #[cfg(unix)]
        let file = File::open(path).map_err(|e| e.to_string())?;
        #[cfg(windows)]
        let file = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
                .open(path)
                .map_err(|e| e.to_string())?
        };
        let header = read_legacy_header(&file)?;
        if !matches!(header.format.as_str(), "flare" | "flare-snapshot") {
            return Err(invalid("not a Flare safetensors core"));
        }
        let size = header
            .index_length
            .ok_or_else(|| invalid("missing model metadata"))?;
        let mut bytes = vec![0; size as usize];
        file.read_exact_at(&mut bytes, header.data_start + header.payload_length)
            .map_err(|e| e.to_string())?;
        let layout = Layout::read_from_header(&file, &header)?;
        let logical_payload = layout
            .as_ref()
            .map_or(header.payload_length, |l| l.files[""].iter().map(|p| p.0).sum());
        Ok((file, header.data_start, logical_payload, bytes, layout))
    }
    pub fn open(
        path: &Path,
        publishers: &[Publisher],
        rank: Option<usize>,
    ) -> Result<(Self, wire::ModelIndex)> {
        let primary_path = path.is_dir().then(|| path.join("model.safetensors"));
        let path = primary_path.as_deref().unwrap_or(path);
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        #[cfg(unix)]
        let directory = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_PATH | libc::O_DIRECTORY)
                .open(parent)
                .map_err(|e| e.to_string())?
        };
        #[cfg(windows)]
        let directory = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ)
                .custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_BACKUP_SEMANTICS)
                .open(parent)
                .map_err(|e| e.to_string())?
        };
        #[cfg(unix)]
        let primary = {
            use std::os::fd::AsRawFd;
            PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd())).join(
                path.file_name()
                    .ok_or_else(|| invalid("missing primary filename"))?,
            )
        };
        #[cfg(windows)]
        let primary = path.to_owned();
        let (file, start, payload, bytes, layout) = Self::envelope(&primary)?;
        let envelope_digest: [u8; 32] = super::protection::sha256(&bytes)
            .try_into()
            .map_err(|_| invalid("signed envelope digest has the wrong length"))?;
        let (protection, index) = Protection::verify(&bytes, publishers, payload, rank)?;
        let file = Arc::new(file);
        let mut files = BTreeMap::new();
        files.insert(
            String::new(),
            PayloadFile::new(
                file.clone(),
                start,
                if layout.is_some() { 0 } else { payload },
            ),
        );
        let mut names = std::collections::BTreeSet::new();
        let spans: Vec<_> = protection
            .data_parts()
            .into_iter()
            .chain(index.ranks.iter().flat_map(|r| r.resources.values()))
            .chain(index.assets.iter().filter_map(|a| a.payload.as_ref()))
            .chain(
                super::engram::spans(&index, rank)?
                    .into_iter()
                    .map(|(_, s)| s),
            )
            .collect();
        for span in &spans {
            if let Some(layout) = &layout {
                for (name, _, _) in layout.segments(span)? {
                    if let Some(name) = name {
                        names.insert(name);
                    }
                }
            } else if !span.filename.is_empty() {
                names.insert(span.filename.clone());
            }
        }
        for name in names {
            validate_filename(&name)?;
            if files.len() > if layout.is_some() { 65536 } else { 99 } {
                return Err(invalid("core references too many data files"));
            }
            #[cfg(unix)]
            let companion = {
                use std::os::fd::{AsRawFd, FromRawFd};
                let name = std::ffi::CString::new(name.as_str()).unwrap();
                // SAFETY: directory and name remain live; the returned descriptor has one owner.
                let fd = unsafe {
                    libc::openat(
                        directory.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NONBLOCK,
                    )
                };
                if fd < 0 {
                    return Err(invalid(format!(
                        "open companion {:?}: {}",
                        name,
                        std::io::Error::last_os_error()
                    )));
                }
                unsafe { File::from_raw_fd(fd) }
            };
            #[cfg(windows)]
            let companion = {
                use std::os::windows::fs::OpenOptionsExt;
                use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;
                std::fs::OpenOptions::new()
                    .read(true)
                    .share_mode(FILE_SHARE_READ)
                    .open(parent.join(&name))
                    .map_err(|e| e.to_string())?
            };
            let companion = PayloadFile::companion(companion)?;
            if let Some(layout) = &layout {
                let size = layout
                    .files
                    .values()
                    .flatten()
                    .find(|(_, blob)| blob == &name)
                    .map(|(size, _)| *size)
                    .ok_or_else(|| invalid("unlisted snapshot blob"))?;
                if companion.length != size || companion.start != 4096 {
                    return Err(invalid("snapshot blob length or alignment mismatch"));
                }
            }
            files.insert(name, companion);
        }
        let result = Self {
            #[cfg(unix)]
            path: {
                use std::os::fd::AsRawFd;
                PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()))
            },
            #[cfg(windows)]
            path: path.to_owned(),
            start,
            payload,
            file,
            files,
            layout,
            _directory: directory,
            envelope_digest,
            protection,
        };
        for part in result.protection.data_parts() {
            result.segments(part)?;
        }
        let spans = index
            .ranks
            .iter()
            .flat_map(|rank| rank.resources.values())
            .chain(
                index
                    .assets
                    .iter()
                    .filter_map(|asset| asset.payload.as_ref()),
            )
            .chain(
                super::engram::spans(&index, rank)?
                    .into_iter()
                    .map(|(_, span)| span),
            );
        let mut ends = BTreeMap::new();
        for span in spans {
            result.source(span)?;
            let end = ends.entry(&span.filename).or_insert(0);
            *end = (*end).max(span.offset + span.nbytes);
        }
        if rank.is_none()
            && ends
                .iter()
                .any(|(name, end)| result.logical_length(name) != Some(*end))
        {
            return Err(invalid("unreferenced core payload bytes"));
        }
        Ok((result, index))
    }
    fn logical_length(&self, name: &str) -> Option<u64> {
        match &self.layout {
            Some(layout) => layout
                .files
                .get(name)
                .map(|parts| parts.iter().map(|p| p.0).sum()),
            None => self.files.get(name).map(|f| f.length),
        }
    }
    fn segments(&self, span: &wire::Span) -> Result<Vec<(Option<String>, u64, u64)>> {
        let parts = if let Some(layout) = &self.layout {
            layout.segments(span)?
        } else {
            vec![(Some(span.filename.clone()), span.offset, span.nbytes)]
        };
        for (name, offset, size) in &parts {
            if let Some(name) = name {
                let file = self
                    .files
                    .get(name)
                    .ok_or_else(|| invalid("missing core blob"))?;
                if offset
                    .checked_add(*size)
                    .is_none_or(|end| end > file.length)
                {
                    return Err(invalid("resource exceeds core payload"));
                }
            }
        }
        Ok(parts)
    }
    pub fn source(&self, span: &wire::Span) -> Result<(&PayloadFile, u64)> {
        let parts = self.segments(span)?;
        if parts.len() != 1 || parts[0].0.is_none() {
            return Err(invalid("resource must occupy one contiguous payload blob"));
        }
        let (name, offset, _) = &parts[0];
        let file = &self.files[name.as_ref().unwrap()];
        Ok((file, file.start + offset))
    }
    #[cfg(windows)]
    pub fn source_name(&self, span: &wire::Span) -> Result<String> {
        let parts = self.segments(span)?;
        if parts.len() != 1 {
            return Err(invalid("resource crosses payload blobs"));
        }
        parts[0]
            .0
            .clone()
            .ok_or_else(|| invalid("resource names padding"))
    }
    pub fn source_index(&self, span: &wire::Span) -> Result<usize> {
        let parts = self.segments(span)?;
        if parts.len() != 1 || parts[0].0.is_none() {
            return Err(invalid("resource must occupy one contiguous payload blob"));
        }
        self.file_index(parts[0].0.as_ref().unwrap())
    }
    pub fn read_logical(&self, span: &wire::Span, offset: u64, output: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|n| n > span.nbytes)
        {
            return Err(invalid("logical read exceeds resource"));
        }
        let slice = wire::Span {
            filename: span.filename.clone(),
            offset: span.offset + offset,
            nbytes: output.len() as u64,
        };
        let mut cursor = 0;
        for (name, offset, size) in self.segments(&slice)? {
            let target = &mut output[cursor..cursor + size as usize];
            if let Some(name) = name {
                let file = &self.files[&name];
                file.file
                    .read_exact_at(target, file.start + offset)
                    .map_err(|e| e.to_string())?;
            } else {
                target.fill(0);
            }
            cursor += size as usize;
        }
        Ok(())
    }
    pub fn file_index(&self, filename: &str) -> Result<usize> {
        self.files
            .keys()
            .position(|name| name == filename)
            .ok_or_else(|| invalid("unknown core file"))
    }

    /// Per-file ranges that a rank needs, including signed stored-data padding.
    pub fn rank_ranges(&self, index: &wire::ModelIndex, rank: usize) -> Result<Vec<CoreFile>> {
        let position = self.protection.position(rank)?;
        let bundle = index
            .ranks
            .get(position)
            .ok_or_else(|| invalid("rank outside core"))?;
        let stored = self.protection.stored_ranges(position);
        let mut selected: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
        let length = self.file.metadata().map_err(|e| e.to_string())?.len();
        selected.insert(
            String::new(),
            if self.layout.is_some() {
                vec![(0, length)]
            } else {
                vec![
                    (0, self.start),
                    (
                        self.start + self.payload,
                        length - self.start - self.payload,
                    ),
                ]
            },
        );
        for span in bundle
            .resources
            .values()
            .chain(index.assets.iter().filter_map(|a| a.payload.as_ref()))
            .chain(
                super::engram::spans(&index, Some(rank))?
                    .into_iter()
                    .map(|(_, span)| span),
            )
            .chain(&stored)
        {
            for (name, offset, size) in self.segments(span)? {
                if let Some(name) = name {
                    let file = &self.files[&name];
                    let ranges = selected.entry(name).or_default();
                    if self.layout.is_some() {
                        // A content-addressed blob is transferred whole, so it can be reused.
                        ranges.push((0, file.start + file.length));
                    } else {
                        ranges.push((0, file.start));
                        ranges.push((file.start + offset, size));
                    }
                }
            }
        }
        Ok(selected
            .into_iter()
            .map(|(filename, ranges)| CoreFile {
                file: self.files[&filename].file.clone(),
                filename,
                ranges: merge_ranges(ranges),
            })
            .collect())
    }
    pub fn span<'a>(&self, core: &'a Core, key: &str) -> Result<&'a wire::Span> {
        core
            .resources
            .get(key)
            .ok_or_else(|| invalid(format!("missing core resource {key}")))
    }
    pub fn validate(&self, core: &Core) -> Result<()> {
        for allocation in &core.allocations {
            if allocation.is_payload_bound() {
                for (source, span) in allocation.sources.iter().zip(&allocation.data_spans) {
                    let payload = self.span(core, &source.resource)?;
                    if source.nbytes != span[1] as u64
                        || source
                            .offset
                            .checked_add(source.nbytes)
                            .is_none_or(|end| end > payload.nbytes)
                    {
                        return Err(invalid("allocation source exceeds its payload resource"));
                    }
                }
            } else if let Some(key) = &allocation.file {
                if self.span(core, key)?.nbytes != allocation.stored_bytes() as u64 {
                    return Err(invalid("allocation payload size differs from its spans"));
                }
            }
        }
        for module in &core.modules {
            if self.span(core, &module.file)?.nbytes < 4 {
                return Err(invalid("empty module image"));
            }
            for global in &module.globals {
                if self.span(core, &global.file)?.nbytes != global.nbytes as u64 {
                    return Err(invalid("module global payload size mismatch"));
                }
            }
        }
        for graph in &core.graphs {
            if graph
                .node_count
                .checked_add(graph.edge_count)
                .and_then(|n| n.checked_mul(4))
                .is_none_or(|n| {
                    n as u64 > self.span(core, &graph.resource).map_or(0, |s| s.nbytes)
                })
            {
                return Err(invalid(
                    "graph resource cannot contain its declared records",
                ));
            }
        }
        Ok(())
    }
    /// Drop clean archive pages before CUDA initialization and after payload loading.
    ///
    /// Caches are sized from the free device memory the driver reports. An integrated GPU such as
    /// GB10 allocates from system memory, where the driver does not count cached file pages as free,
    /// so distribution and verification reads can otherwise prevent context or cache allocation.
    pub fn release_page_cache(&self) {
        // SAFETY: an advisory call on an open descriptor; failure only leaves the pages cached.
        for file in self.files.values() {
            crate::os::release_page_cache(&file.file);
        }
    }
    fn reader(&self, core: &Core, key: &str) -> Result<ResourceReader> {
        let span = self.span(core, key)?;
        let (source, offset) = self.source(span)?;
        self.protection.reader(source.file.clone(), offset, span)
    }
    pub fn read(&self, core: &Core, key: &str) -> Result<Vec<u8>> {
        let mut reader = self.reader(core, key)?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    pub fn records(&self, core: &Core, key: &str) -> Result<Records> {
        let reader = self.reader(core, key)?;
        let nbytes = reader.len();
        Ok(Records {
            reader: BufReader::with_capacity(16 << 10, reader).take(nbytes),
            buffer: Vec::new(),
        })
    }
}

pub(super) struct Records {
    reader: Take<BufReader<ResourceReader>>,
    buffer: Vec<u8>,
}
impl Records {
    pub fn read<T: Message + Default>(&mut self) -> Result<T> {
        let mut prefix = [0u8; 4];
        self.reader
            .read_exact(&mut prefix)
            .map_err(|e| invalid(format!("truncated graph record: {e}")))?;
        let size = u32::from_le_bytes(prefix) as usize;
        if size > MAX_RECORD || size as u64 > self.reader.limit() {
            return Err(invalid("invalid graph record size"));
        }
        self.buffer.resize(size, 0);
        self.reader
            .read_exact(&mut self.buffer)
            .map_err(|e| e.to_string())?;
        T::decode(self.buffer.as_slice()).map_err(|e| invalid(format!("invalid graph record: {e}")))
    }
    pub fn finish(self) -> Result<()> {
        if self.reader.limit() != 0 {
            return Err(invalid("unexpected trailing graph records"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn transfer_ranges_merge_adjacent_and_overlapping_extents() {
        assert_eq!(
            merge_ranges(vec![(20, 3), (0, 10), (5, 10), (15, 1), (30, 0)]),
            vec![(0, 16), (20, 3)]
        );
    }

    #[test]
    fn header_and_index_lengths_are_bounded_before_reading() {
        let path =
            std::env::temp_dir().join(format!("flare-header-{}.safetensors", std::process::id()));
        for size in [0u64, MAX_LEGACY_HEADER + 1, u64::MAX] {
            std::fs::write(&path, size.to_le_bytes()).unwrap();
            assert!(
                Archive::open(&path, &[], None)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("header size")
            );
        }
        let header = serde_json::json!({
            "__metadata__": {"format": "flare"},
            "flare.payload": {"dtype": "U8", "shape": [0], "data_offsets": [0,0]},
            "flare.metadata": {"dtype": "U8", "shape": [MAX_LEGACY_INDEX + 1], "data_offsets": [0,MAX_LEGACY_INDEX + 1]},
        });
        let mut bytes = serde_json::to_vec(&header).unwrap();
        bytes.resize(4088, b' ');
        let mut file = File::create(&path).unwrap();
        file.write_all(&4088u64.to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        assert!(
            Archive::open(&path, &[], None)
                .err()
                .unwrap()
                .to_string()
                .contains("index size")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_layout_rejects_duplicate_logical_files() {
        let json = r#"{"version":1,"files":{"":[[4,""]],"":[[8,""]]}}"#;
        assert!(serde_json::from_str::<LegacyLayout>(json).is_err());
    }

    #[test]
    fn legacy_header_rejects_zero_bytes_in_header_padding() {
        let path = std::env::temp_dir().join(format!(
            "flare-header-nul-{}.safetensors",
            std::process::id()
        ));
        let raw = r#"{"__metadata__":{"format":"flare-data"},"flare.payload":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}"#;
        let header_size = 4096 - 8;
        let mut buf = Vec::with_capacity(4096);
        buf.extend_from_slice(&(header_size as u64).to_le_bytes());
        buf.extend_from_slice(raw.as_bytes());
        buf.resize(4096, 0);
        let mut file = File::create(&path).unwrap();
        file.write_all(&buf).unwrap();
        drop(file);

        let reopened = File::open(&path).unwrap();
        assert!(read_legacy_header(&reopened).is_err());
        let _ = std::fs::remove_file(path);
    }
}

fn retain(file: &Arc<File>) -> Result<Arc<File>> {
    #[cfg(windows)]
    {
        return Ok(Arc::new(crate::os::direct_read_guard(file).map_err(
            |error| invalid(format!("retain read-only core: {error}")),
        )?));
    }
    #[cfg(unix)]
    {
        Ok(file.clone())
    }
}

impl super::payload::Files for Archive {
    fn span<'a>(&self, core: &'a Core, key: &str) -> Result<&'a wire::Span> {
        Archive::span(self, core, key)
    }
    fn source(&self, span: &wire::Span) -> Result<(&PayloadFile, u64)> {
        Archive::source(self, span)
    }
    fn read(&self, core: &Core, key: &str) -> Result<Vec<u8>> {
        Archive::read(self, core, key)
    }
    fn records<'a>(
        &'a self,
        core: &Core,
        key: &str,
    ) -> Result<super::payload::RecordReader<'a>> {
        Ok(super::payload::RecordReader::Archive(Archive::records(
            self, core, key,
        )?))
    }
    fn path(&self) -> &std::path::Path {
        &self.path
    }
    fn source_index(&self, span: &wire::Span) -> Result<usize> {
        Archive::source_index(self, span)
    }
    fn asset_path(&self, span: &wire::Span) -> Result<std::path::PathBuf> {
        let (file, offset) = self.source(span)?;
        let _ = offset;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            Ok(std::path::PathBuf::from(format!(
                "/proc/self/fd/{}",
                file.file.as_raw_fd()
            )))
        }
        #[cfg(windows)]
        {
            let name = self.source_name(span)?;
            let root = self.path.as_path();
            Ok(if name.is_empty() {
                root.to_owned()
            } else {
                root.parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(name)
            })
        }
    }
    fn release_page_cache(&self) {
        Archive::release_page_cache(self)
    }
    fn direct_files(&self) -> Box<dyn Iterator<Item = Result<Arc<File>>> + '_> {
        Box::new(self.files.values().map(|file| file.direct()))
    }
    fn direct_file(&self, index: usize) -> Result<Arc<File>> {
        self.files
            .values()
            .nth(index)
            .ok_or_else(|| invalid("payload container index out of range"))?
            .direct()
    }
    fn retained_files(&self) -> Result<Vec<Arc<File>>> {
        self.files
            .values()
            .map(|payload| retain(&payload.file))
            .chain(std::iter::once(retain(&self.file)))
            .collect()
    }
    fn validate(&self, core: &Core) -> Result<()> {
        Archive::validate(self, core)
    }
    fn resource_reader<'a>(
        &'a self,
        core: &Core,
        key: &str,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        Ok(Box::new(self.reader(core, key)?))
    }
}
