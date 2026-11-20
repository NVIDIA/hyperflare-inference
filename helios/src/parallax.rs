//! Native virtual-memory expert residency mechanisms.

pub mod residency;

use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    any(feature = "server", feature = "runtime-core"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct ExpertId {
    pub layer: usize,
    pub expert: usize,
}

static ORIGIN: OnceLock<Instant> = OnceLock::new();

/// A single native-process monotonic clock domain, independent of GPU timestamps.
pub fn now_ns() -> u64 {
    ORIGIN
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
