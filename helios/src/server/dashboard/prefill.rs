// SPDX-License-Identifier: Apache-2.0
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};

const BUCKET_NS: u64 = super::REFRESH.as_nanos() as u64;
const HISTORY_NS: u64 = (super::HISTORY * 1e9) as u64;
const BUCKETS: usize = (HISTORY_NS / BUCKET_NS) as usize + 1;

#[derive(Default)]
struct Bucket {
    index: AtomicU64,
    tokens: AtomicU64,
}

impl Bucket {
    // Only the completion worker writes. Invalidate the old index before recycling its value.
    fn add(&self, index: u64, tokens: f64) {
        if self.index.load(SeqCst) == index {
            let previous = f64::from_bits(self.tokens.load(SeqCst));
            self.tokens.store((previous + tokens).to_bits(), SeqCst);
        } else {
            self.index.store(u64::MAX, SeqCst);
            self.tokens.store(tokens.to_bits(), SeqCst);
            self.index.store(index, SeqCst);
        }
    }

    fn tokens(&self, index: u64) -> f64 {
        if self.index.load(SeqCst) != index {
            return 0.0;
        }
        let value = f64::from_bits(self.tokens.load(SeqCst));
        if self.index.load(SeqCst) == index {
            value
        } else {
            0.0
        }
    }
}

pub(in crate::server) struct PrefillPlot([Bucket; BUCKETS]);

impl Default for PrefillPlot {
    fn default() -> Self {
        Self(std::array::from_fn(|_| Bucket::default()))
    }
}

impl PrefillPlot {
    pub(in crate::server) fn record(&self, end: u64, gpu_ns: u64, tokens: u64) {
        if gpu_ns == 0 {
            let index = end / BUCKET_NS;
            self.0[index as usize % BUCKETS].add(index, tokens as f64);
            return;
        }
        let start = end.saturating_sub(gpu_ns);
        let last = end.saturating_sub(1) / BUCKET_NS;
        let first = (start / BUCKET_NS).max(last.saturating_sub(BUCKETS as u64 - 1));
        for index in first..=last {
            let left = start.max(index * BUCKET_NS);
            let right = end.min((index + 1) * BUCKET_NS);
            let share = tokens as f64 * (right - left) as f64 / gpu_ns as f64;
            self.0[index as usize % BUCKETS].add(index, share);
        }
    }

    pub(super) fn points(&self, now: u64) -> Vec<(f64, f64)> {
        let visible_start = now.saturating_sub(HISTORY_NS);
        let first = visible_start / BUCKET_NS;
        let mut points = Vec::with_capacity(2 * BUCKETS);
        for index in first..now / BUCKET_NS {
            let start = index * BUCKET_NS;
            let end = start + BUCKET_NS;
            let tokens = self.0[index as usize % BUCKETS].tokens(index);
            let rate = tokens * 1e9 / BUCKET_NS as f64;
            points.push((-((now - start.max(visible_start)) as f64) / 1e9, rate));
            points.push((-((now - end) as f64) / 1e9, rate));
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_replays_fill_only_their_overlapping_buckets() {
        let plot = PrefillPlot::default();
        let second = 1_000_000_000;
        plot.record(3 * second + second / 4, 2 * second, 800);
        plot.record(3 * second + second / 2, second / 4, 50);
        plot.record(8 * second, second / 2, 64);
        let points = plot.points(8 * second);
        let rates: Vec<_> = points.chunks_exact(2).map(|pair| pair[0].1).collect();
        assert_eq!(
            rates,
            [
                0., 0., 200., 400., 400., 400., 300., 0., 0., 0., 0., 0., 0., 0., 0., 128.
            ]
        );
        let area: f64 = points
            .chunks_exact(2)
            .map(|p| (p[1].0 - p[0].0) * p[0].1)
            .sum();
        assert_eq!(area, 914.);
    }

    #[test]
    fn old_replays_expire_and_long_replays_keep_their_original_rate() {
        let plot = PrefillPlot::default();
        let second = 1_000_000_000;
        plot.record(second / 2, second / 2, 100);
        assert!(
            plot.points(200 * second)
                .iter()
                .all(|&(_, rate)| rate == 0.)
        );
        plot.record(200 * second, 150 * second, 3000);
        assert!(
            plot.points(200 * second)
                .iter()
                .all(|&(_, rate)| rate == 20.)
        );
        plot.record(200 * second + second / 4, second / 4, 10);
        assert_eq!(
            plot.points(200 * second + second / 4).last(),
            Some(&(-0.25, 20.))
        );
        assert_eq!(
            plot.points(200 * second + BUCKET_NS).last(),
            Some(&(0., 20.))
        );
    }
}
