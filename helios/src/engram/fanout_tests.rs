//! Advisory-warming fan-out over the rank-local stores of one Engram table.
//!
//! A rank-partitioned embedding table gives every local core its own `Shared`,
//! built in `crate::runtime_core::model`, so one serving step's advisory batch
//! has to reach each of those stores; a replicated table gives several bindings
//! one shared store, which a single batch already covers. `store_groups` reports
//! the stores a binding list needs and, for each store, the delivery ranks that
//! store serves, and `crate::server::runtime::prefetch_rows` runs one prediction
//! and one advisory batch per group.
//!
//! Each test drives a recording warmer over the groups `store_groups` returns, in
//! the order the serving fan-out visits them. A store from
//! `Shared::layout_fixture` holds no layer, and a layer needs the `PinnedHost`
//! staging a CUDA driver provides, so `Shared::prefetch` refuses every batch here
//! and no batch reaches a reader queue in these tests. They exercise
//! `store_groups` and a recording warmer, not `prefetch_rows` itself, so all they
//! establish is which stores a fan-out over these groups would visit: a fan-out
//! that named the right stores and queued nothing would still pass, and only a run
//! with a CUDA driver and a warm row cache can show a reader dequeueing a batch.

use super::*;

/// Catalog of one rank's partition of a single embedding table.
///
/// The partition number selects the plane file, offset and row count, so two
/// fixture stores never describe the same partition even though both are built
/// without a device.
fn partition_index(partition: usize) -> Index {
    Index {
        tables: vec![Table {
            layer: 1,
            planes: [256, 8].map(|row_bytes| Plane {
                file: partition,
                offset: partition as u64 * 4096,
                nbytes: (100 + partition) * row_bytes,
                row_bytes,
            }),
            heads: 24,
            rows: 100 + partition as u64,
            first_row: 0,
            geometry: RowGeometry::default(),
            weights: "weights".into(),
            scales: "scales".into(),
            hashes: "hashes".into(),
            event: "ready".into(),
            request: Some("requests".into()),
        }],
        token_map: Vec::new(),
        pad_token_id: 0,
    }
}

/// One advisory batch handed to one store.
///
/// `store` is the address of the `Shared` the batch was dispatched to, and it is
/// how the assertions tell four warmed partitions from one partition warmed four
/// times.
#[derive(Debug)]
struct Dispatch {
    store: *const Shared,
    table: usize,
    sequence: u64,
    rows: Vec<i64>,
}

/// One prediction left for one delivery rank of one store.
#[derive(Debug)]
struct Prediction {
    store: *const Shared,
    rank: usize,
    sequence: u64,
    rows: Vec<i64>,
}

/// The advisory batches one store received, keyed by that store's address.
///
/// Splitting the dispatch log per store lets the assertions report a store that
/// was never warmed, a store warmed twice, and a batch sent to a store no binding
/// holds as three distinct failures.
#[derive(Debug)]
struct StoreDispatches {
    store: *const Shared,
    /// One `(table, sequence, rows)` entry per batch that store received, in
    /// dispatch order.
    batches: Vec<(usize, u64, Vec<i64>)>,
}

/// The batches `log` holds for `store`, or `None` when the store was never warmed.
fn batches_for<'a>(
    log: &'a [StoreDispatches],
    store: &Arc<Shared>,
) -> Option<&'a [(usize, u64, Vec<i64>)]> {
    log.iter()
        .find(|entry| entry.store == Arc::as_ptr(store))
        .map(|entry| entry.batches.as_slice())
}

/// Records what a fan-out over store groups does, standing in for the loop in
/// `crate::server::runtime::prefetch_rows`.
///
/// Per group it predicts and warms that group's store once and leaves the
/// prediction for every delivery rank the group holds, because one store can
/// serve several ranks. The predicted rows are arguments rather than the output
/// of each store's `predict_rows`: this recorder measures the fan-out, not row
/// addressing, and a fixture catalog declares no n-gram geometry to address.
#[derive(Default)]
struct RecordingWarmer {
    dispatches: Vec<Dispatch>,
    predictions: Vec<Prediction>,
}

impl RecordingWarmer {
    fn warm(
        &mut self,
        groups: &[(&Arc<Shared>, Vec<usize>)],
        table: usize,
        sequence: u64,
        rows: &[i64],
    ) {
        for (store, ranks) in groups {
            let store = *store;
            // Advisory by contract: a store without a layer refuses the batch and
            // reports it, so the fan-out cannot use the result to decide anything.
            let _ = store.prefetch(table, rows);
            self.dispatches.push(Dispatch {
                store: Arc::as_ptr(store),
                table,
                sequence,
                rows: rows.to_vec(),
            });
            for rank in ranks {
                self.predictions.push(Prediction {
                    store: Arc::as_ptr(store),
                    rank: *rank,
                    sequence,
                    rows: rows.to_vec(),
                });
            }
        }
    }

    fn dispatches_to(&self, store: &Arc<Shared>) -> Vec<&Dispatch> {
        let address = Arc::as_ptr(store);
        self.dispatches
            .iter()
            .filter(|dispatch| dispatch.store == address)
            .collect()
    }

    fn dispatches_by_store(&self) -> Vec<StoreDispatches> {
        let mut log: Vec<StoreDispatches> = Vec::new();
        for dispatch in &self.dispatches {
            let batch = (dispatch.table, dispatch.sequence, dispatch.rows.clone());
            match log.iter_mut().find(|entry| entry.store == dispatch.store) {
                Some(entry) => entry.batches.push(batch),
                None => log.push(StoreDispatches {
                    store: dispatch.store,
                    batches: vec![batch],
                }),
            }
        }
        log
    }

    fn predictions_at(&self, store: &Arc<Shared>) -> Vec<&Prediction> {
        let address = Arc::as_ptr(store);
        self.predictions
            .iter()
            .filter(|prediction| prediction.store == address)
            .collect()
    }
}

#[test]
fn four_rank_partitioned_bindings_warm_four_stores() {
    // A rank-partitioned table gives each local core its own store, and every one
    // of those bindings carries delivery rank 0: the store holds one rank's rows
    // and that rank is the core's only delivery target. Four cores therefore
    // present four distinct stores that all report rank 0.
    let stores: Vec<Arc<Shared>> = (0..4)
        .map(|partition| Shared::layout_fixture(partition_index(partition), 1))
        .collect();
    let bindings: Vec<Binding> = stores
        .iter()
        .map(|shared| Binding {
            shared: shared.clone(),
            rank: 0,
            events: Vec::new(),
        })
        .collect();
    // Stated as a precondition of the observation below: a fixture store cannot
    // accept a real advisory batch, so this test judges the dispatch.
    assert!(
        stores.iter().all(|store| !store.prefetching()),
        "a fixture store claims to be warming, so the dispatch is not all this test can see"
    );

    let groups = store_groups(&bindings);

    let table = 0;
    let sequence = 11;
    let rows = vec![3, 5, 8];
    let mut warmer = RecordingWarmer::default();
    warmer.warm(&groups, table, sequence, &rows);

    // The claim under test comes first: each partition's store received this
    // step's batch once. One record per store that received anything, so a store
    // never warmed and a store warmed twice read as different failures.
    let log = warmer.dispatches_by_store();
    for (partition, store) in stores.iter().enumerate() {
        let Some(batches) = batches_for(&log, store) else {
            panic!(
                "partition {partition} received no advisory dispatch for table {table} at step {sequence}"
            );
        };
        assert_eq!(
            batches.len(),
            1,
            "partition {partition} received {} advisory batches for table {table} at step {sequence}: {batches:?}",
            batches.len()
        );
        assert_eq!(
            batches[0],
            (table, sequence, rows.clone()),
            "partition {partition} was dispatched table {}, step {}, rows {:?} instead of table {table}, step {sequence}, rows {rows:?}",
            batches[0].0,
            batches[0].1,
            batches[0].2
        );
        assert_eq!(
            warmer.predictions_at(store).len(),
            1,
            "partition {partition} recorded {} predictions, not one for its single rank",
            warmer.predictions_at(store).len()
        );
    }
    // Four partitions and four warmed stores: a batch sent to a store no binding
    // holds would make the log longer without matching any partition.
    assert_eq!(
        log.len(),
        stores.len(),
        "advisory batches landed on {} distinct stores, not on the four partitions",
        log.len()
    );
    for dispatch in &warmer.dispatches {
        assert!(
            stores
                .iter()
                .any(|store| Arc::as_ptr(store) == dispatch.store),
            "an advisory batch went to a store no binding holds"
        );
    }
    assert_eq!(
        warmer.dispatches.len(),
        4,
        "the fan-out made {} advisory dispatches for four partitions",
        warmer.dispatches.len()
    );

    // Groups describe the same four stores: one partition each, each carrying the
    // delivery rank 0 its binding uses.
    assert_eq!(
        groups.len(),
        4,
        "four rank partitions collapsed into {} store groups",
        groups.len()
    );
    for (partition, (store, ranks)) in groups.iter().enumerate() {
        assert!(
            Arc::ptr_eq(store, &stores[partition]),
            "group {partition} holds a store other than partition {partition}'s"
        );
        assert_eq!(
            *ranks,
            vec![0],
            "group {partition} delivers to ranks {ranks:?} instead of the rank 0 its binding carries"
        );
    }
}

#[test]
fn one_replicated_store_is_warmed_once_for_every_rank() {
    // A replicated table gives every rank the same store, so the bindings differ
    // only in the delivery rank they attach to it. Warming that store once and
    // recording the prediction for each of its ranks warms everything; warming it
    // once per binding would repeat the same disk work four times.
    let store = Shared::layout_fixture(partition_index(0), 4);
    let bindings: Vec<Binding> = (0..4)
        .map(|rank| Binding {
            shared: store.clone(),
            rank,
            events: vec![rank],
        })
        .collect();

    let groups = store_groups(&bindings);

    assert_eq!(
        groups.len(),
        1,
        "one shared store appeared as {} store groups",
        groups.len()
    );
    let (group_store, ranks) = &groups[0];
    assert!(
        Arc::ptr_eq(group_store, &store),
        "the single group holds a store other than the replicated one"
    );
    assert_eq!(
        *ranks,
        vec![0, 1, 2, 3],
        "the replicated store delivers to ranks {ranks:?} instead of every rank in binding order"
    );

    let table = 0;
    let sequence = 12;
    let rows = vec![21, 34];
    let mut warmer = RecordingWarmer::default();
    warmer.warm(&groups, table, sequence, &rows);

    assert_eq!(
        warmer.dispatches_to(&store).len(),
        1,
        "the replicated store was warmed {} times, not once",
        warmer.dispatches_to(&store).len()
    );
    assert_eq!(
        warmer.dispatches.len(),
        1,
        "the fan-out made {} advisory dispatches for one store",
        warmer.dispatches.len()
    );
    let recorded: Vec<usize> = warmer
        .predictions_at(&store)
        .iter()
        .map(|p| p.rank)
        .collect();
    assert_eq!(
        recorded,
        vec![0, 1, 2, 3],
        "the replicated store recorded predictions at ranks {recorded:?}, not at every delivery rank"
    );
    assert!(
        warmer
            .predictions
            .iter()
            .all(|prediction| prediction.rows == rows && prediction.sequence == sequence),
        "a rank's prediction does not carry this step's rows"
    );
}

#[test]
fn single_binding_keeps_the_single_store_behaviour() {
    // One binding and one store are what a single-rank run presents, and that run
    // must keep exactly one group, one rank and one advisory dispatch.
    let store = Shared::layout_fixture(partition_index(0), 1);
    let bindings = vec![Binding {
        shared: store.clone(),
        rank: 0,
        events: Vec::new(),
    }];

    let groups = store_groups(&bindings);

    assert_eq!(
        groups.len(),
        1,
        "one binding produced {} groups",
        groups.len()
    );
    assert!(Arc::ptr_eq(groups[0].0, &store));
    assert_eq!(groups[0].1, vec![0]);

    let mut warmer = RecordingWarmer::default();
    warmer.warm(&groups, 0, 13, &[55]);
    assert_eq!(warmer.dispatches.len(), 1);
    assert_eq!(warmer.predictions.len(), 1);
    assert_eq!(warmer.predictions[0].rank, 0);
}

#[test]
fn no_binding_yields_no_warm_dispatch() {
    // A run without Engram bindings has no store to group and no step to warm, so
    // the fan-out must stay inert rather than invent a store or index an empty
    // binding list.
    let bindings: Vec<Binding> = Vec::new();

    let groups = store_groups(&bindings);

    assert!(
        groups.is_empty(),
        "an empty binding list produced {} store groups",
        groups.len()
    );
    let mut warmer = RecordingWarmer::default();
    warmer.warm(&groups, 0, 14, &[1, 2]);
    assert!(
        warmer.dispatches.is_empty(),
        "an empty binding list warmed a store"
    );
    assert!(warmer.predictions.is_empty());
}
