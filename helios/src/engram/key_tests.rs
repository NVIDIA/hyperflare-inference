//! Prediction-store keying, exercised without a CUDA context.
//!
//! `record_prediction` and `take_prediction` keep one entry per
//! (`rank`, `sequence`) in the prediction store and hand every buffer they
//! displace back to the caller for recycling. These tests drive both functions
//! directly, so they need no `Shared`, no table, and no device.

use super::*;

/// Allocation identity of a row-id buffer: its heap pointer and capacity.
///
/// A buffer that has left the prediction vector stays alive inside the returned
/// value, so two buffers sharing an identity are the same allocation handed back
/// twice rather than two allocations holding equal rows.
fn buffer_identity(buffer: &Vec<i64>) -> (*const i64, usize) {
    (buffer.as_ptr(), buffer.capacity())
}

#[test]
fn four_ranks_predict_one_step_without_collision() {
    let ranks = 4;
    let bound = 2 * STAGING_FRAMES * ranks;
    let sequence = 9;
    let mut predicted: Vec<(usize, u64, Vec<i64>)> = Vec::new();

    for rank in 0..ranks {
        let rows = vec![10 + rank as i64];
        let (displaced, aged) = record_prediction(&mut predicted, rank, sequence, rows, bound);
        assert!(
            displaced.is_none(),
            "rank {rank} displaced an entry for its own sequence"
        );
        assert!(
            aged.is_empty(),
            "rank {rank} pushed the store past its bound of {bound}"
        );
    }

    assert_eq!(predicted.len(), ranks);
    for rank in 0..ranks {
        assert!(
            predicted
                .iter()
                .any(|(entry_rank, entry_sequence, rows)| *entry_rank == rank
                    && *entry_sequence == sequence
                    && *rows == vec![10 + rank as i64]),
            "rank {rank} has no entry holding its own rows at sequence {sequence}"
        );
    }

    assert_eq!(take_prediction(&mut predicted, 0, sequence), Some(vec![10]));
    assert_eq!(predicted.len(), ranks - 1);
    assert_eq!(take_prediction(&mut predicted, 1, sequence), Some(vec![11]));
    assert_eq!(take_prediction(&mut predicted, 2, sequence), Some(vec![12]));
    assert_eq!(take_prediction(&mut predicted, 3, sequence), Some(vec![13]));
    assert!(predicted.is_empty());
}

#[test]
fn one_rank_keeps_the_sequence_only_order() {
    // A run with one rank records every prediction at rank 0, so `expect` hands
    // `record_prediction` the bound 2 * STAGING_FRAMES, the depth a store keyed by
    // sequence alone holds. The assertions below pin the two observable effects
    // such a store has: the newest 2 * STAGING_FRAMES sequences survive in
    // recording order, and re-recording a live sequence removes it from its place
    // and re-appends it, which makes it the most recent entry. Because rank 0 is
    // the only rank, (0, sequence) and sequence select the same entries, so equal
    // behaviour here means the rank-and-sequence key is indistinguishable from
    // sequence-only keying for single-rank serving.
    let bound = 2 * STAGING_FRAMES;
    let mut predicted: Vec<(usize, u64, Vec<i64>)> = Vec::new();

    for sequence in 1..=4u64 {
        let rows = vec![sequence as i64];
        let (displaced, aged) = record_prediction(&mut predicted, 0, sequence, rows, bound);
        assert!(displaced.is_none(), "sequence {sequence} displaced another");
        assert!(aged.is_empty(), "sequence {sequence} aged another out");
    }
    let order: Vec<u64> = predicted.iter().map(|(_, sequence, _)| *sequence).collect();
    assert_eq!(order, vec![1, 2, 3, 4]);
    for sequence in 1..=4u64 {
        assert!(
            predicted
                .iter()
                .any(|(rank, entry_sequence, rows)| *rank == 0
                    && *entry_sequence == sequence
                    && *rows == vec![sequence as i64]),
            "sequence {sequence} is not retrievable at rank 0"
        );
    }

    let (displaced, aged) = record_prediction(&mut predicted, 0, 5, vec![5], bound);
    assert!(displaced.is_none());
    assert_eq!(aged, vec![vec![1]], "only the oldest sequence ages out");
    let order: Vec<u64> = predicted.iter().map(|(_, sequence, _)| *sequence).collect();
    assert_eq!(order, vec![2, 3, 4, 5]);

    let (displaced, aged) = record_prediction(&mut predicted, 0, 3, vec![3], bound);
    assert_eq!(displaced, Some(vec![3]));
    assert!(
        aged.is_empty(),
        "the re-recorded sequence stays inside the bound"
    );
    let order: Vec<u64> = predicted.iter().map(|(_, sequence, _)| *sequence).collect();
    assert_eq!(order, vec![2, 4, 5, 3]);
}

#[test]
fn each_predicted_buffer_is_returned_once() {
    let bound = 2;
    let mut predicted: Vec<(usize, u64, Vec<i64>)> = Vec::new();
    let mut returned: Vec<Vec<i64>> = Vec::new();
    let mut inserted: Vec<(*const i64, usize)> = Vec::new();

    // Same-key replacement: re-recording (0, 1) returns the buffer it replaces.
    let first = vec![1, 2, 3];
    inserted.push(buffer_identity(&first));
    let (displaced, aged) = record_prediction(&mut predicted, 0, 1, first, bound);
    assert!(displaced.is_none() && aged.is_empty());

    let second = vec![1, 2, 3, 4, 5];
    inserted.push(buffer_identity(&second));
    let (displaced, aged) = record_prediction(&mut predicted, 0, 1, second, bound);
    assert_eq!(displaced.as_ref(), Some(&vec![1, 2, 3]));
    assert!(aged.is_empty());
    returned.extend(displaced);
    returned.extend(aged);

    // Bound ageing: with bound 2 the third live entry ages the front one out.
    let third = vec![7; 6];
    inserted.push(buffer_identity(&third));
    let (displaced, aged) = record_prediction(&mut predicted, 0, 2, third, bound);
    assert!(displaced.is_none() && aged.is_empty());
    assert_eq!(predicted.len(), 2);

    let fourth = vec![9; 7];
    inserted.push(buffer_identity(&fourth));
    let (displaced, aged) = record_prediction(&mut predicted, 0, 3, fourth, bound);
    assert!(displaced.is_none());
    assert_eq!(
        aged.len(),
        1,
        "one entry ages out per insert past the bound"
    );
    returned.extend(displaced);
    returned.extend(aged);

    let fifth = vec![11; 8];
    inserted.push(buffer_identity(&fifth));
    let (displaced, aged) = record_prediction(&mut predicted, 0, 4, fifth, bound);
    assert!(displaced.is_none());
    assert_eq!(aged.len(), 1);
    returned.extend(displaced);
    returned.extend(aged);

    assert_eq!(
        returned,
        vec![vec![1, 2, 3], vec![1, 2, 3, 4, 5], vec![7; 6]],
        "the re-recorded buffer, then the aged-out buffers, oldest first"
    );

    // The recovered allocations are the three that left the store, in that order.
    let recovered: Vec<(*const i64, usize)> = returned.iter().map(buffer_identity).collect();
    assert_eq!(
        recovered,
        inserted[..3].to_vec(),
        "the returned rows are the allocations that left the store, not copies"
    );
    for (index, identity) in recovered.iter().enumerate() {
        assert!(
            !recovered[..index].contains(identity),
            "the buffer returned at position {index} was returned before"
        );
    }

    // Every entry still in the store holds the buffer that was inserted for it,
    // and that buffer was never returned.
    for (rank, sequence, rows) in &predicted {
        let identity = buffer_identity(rows);
        assert!(
            !recovered.contains(&identity),
            "entry ({rank}, {sequence}) is live but its buffer was returned"
        );
        assert_eq!(
            inserted[3..]
                .iter()
                .filter(|other| **other == identity)
                .count(),
            1,
            "entry ({rank}, {sequence}) holds a buffer that was not inserted once"
        );
    }
}
