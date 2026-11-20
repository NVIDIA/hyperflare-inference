//! The serving dashboard reads observation results; it never executes predictors.
use super::*;
use base64::Engine as _;
use ratatui::widgets::{Row, Table, Wrap};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

const FLASH_SECONDS: f64 = 6.0;
const BG_GRACE: (u8, u8, u8) = (24, 33, 52);
const BG_PROTECTED: (u8, u8, u8) = (34, 148, 174);
const BG_PROTECTED_DIM: (u8, u8, u8) = (21, 44, 58);
const BG_DYNAMIC: (u8, u8, u8) = (162, 116, 245);
const BG_DYNAMIC_BRIGHT: (u8, u8, u8) = (188, 138, 255);
const BG_PROMOTED: (u8, u8, u8) = (64, 255, 168);
const BG_EVICTED: (u8, u8, u8) = (255, 206, 86);

/// Display mode for the Parallax dashboard pane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParallaxMode {
    /// Live graphical expert residency matrix, hit-rate and swap charts, and commit ticker.
    #[default]
    Visual,
    /// Full accounting ledger and shadow predictor table.
    Ledger,
}

/// Matrix color lens for the expert placement map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlacementLens {
    /// Show protected HBM, dynamic HBM, Grace DRAM, and live promotion/eviction flashes.
    #[default]
    All,
    /// Mute static protected HBM so dynamic HBM slots, Grace pool, and live swaps stand out.
    Dynamic,
    /// Cumulative swap heatmap over recent movement transactions.
    Heatmap,
}

impl PlacementLens {
    pub fn next(self) -> Self {
        match self {
            Self::All => Self::Dynamic,
            Self::Dynamic => Self::Heatmap,
            Self::Heatmap => Self::All,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::All => "ALL",
            Self::Dynamic => "DYNAMIC",
            Self::Heatmap => "HEATMAP",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ParallaxPoint {
    time: f64,
    interval_hit_pct: f64,
    cumulative_hit_pct: f64,
    swaps_per_sec: f64,
}

#[derive(Clone, Debug)]
struct CommitSummary {
    generation: u64,
    promoted: Vec<(usize, usize)>,
    evicted: Vec<(usize, usize)>,
    staged_bytes: u64,
    copy_gpu_ns: u64,
    publication_host_ns: u64,
}

/// Stateful Parallax view holding rolling 120s hit-rate/swap history, live
/// promotion/eviction flashes, and interactive view/lens selection.
#[derive(Debug)]
pub struct ParallaxView {
    start: Instant,
    mode: ParallaxMode,
    lens: PlacementLens,
    points: VecDeque<ParallaxPoint>,
    last_time: Option<Instant>,
    last_hbm_routes: Option<u64>,
    last_cold_routes: Option<u64>,
    last_promoted_total: Option<u64>,
    last_generation: u64,
    promoted_flash: HashMap<(usize, usize), Instant>,
    evicted_flash: HashMap<(usize, usize), Instant>,
    expert_heat: HashMap<(usize, usize), u32>,
    layer_churn: HashMap<usize, u32>,
    last_commit: Option<CommitSummary>,
    latest_report: Value,
}

impl Default for ParallaxView {
    fn default() -> Self {
        Self::new(Instant::now())
    }
}

impl ParallaxView {
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            mode: ParallaxMode::Visual,
            lens: PlacementLens::All,
            points: VecDeque::with_capacity(256),
            last_time: None,
            last_hbm_routes: None,
            last_cold_routes: None,
            last_promoted_total: None,
            last_generation: 0,
            promoted_flash: HashMap::new(),
            evicted_flash: HashMap::new(),
            expert_heat: HashMap::new(),
            layer_churn: HashMap::new(),
            last_commit: None,
            latest_report: Value::Null,
        }
    }

    pub fn mode(&self) -> ParallaxMode {
        self.mode
    }

    pub fn lens(&self) -> PlacementLens {
        self.lens
    }

    pub fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            ParallaxMode::Visual => ParallaxMode::Ledger,
            ParallaxMode::Ledger => ParallaxMode::Visual,
        };
    }

    pub fn cycle_lens(&mut self) {
        self.lens = self.lens.next();
    }

    pub fn sample(&mut self, now: Instant, report: Value) {
        let elapsed = now.saturating_duration_since(self.start).as_secs_f64();
        let dt = self
            .last_time
            .map(|prev| now.saturating_duration_since(prev).as_secs_f64())
            .unwrap_or(0.0);

        let ledger = &report["actual_ledger"];
        let hbm_routes = ledger["hbm_routes"].as_u64();
        let cold_routes = ledger["cold_routes"].as_u64();
        let physical = &report["physical_residency"];
        let promoted_total = physical["promoted_experts_including_startup"].as_u64();

        let cumulative_hit_pct = match (hbm_routes, cold_routes) {
            (Some(hbm), Some(cold)) if hbm + cold > 0 => {
                (hbm as f64) * 100.0 / ((hbm + cold) as f64)
            }
            _ => self
                .points
                .back()
                .map(|p| p.cumulative_hit_pct)
                .unwrap_or(0.0),
        };

        let interval_hit_pct = match (
            hbm_routes,
            cold_routes,
            self.last_hbm_routes,
            self.last_cold_routes,
        ) {
            (Some(h), Some(c), Some(ph), Some(pc)) => {
                let dh = h.saturating_sub(ph);
                let dc = c.saturating_sub(pc);
                if dh + dc > 0 {
                    (dh as f64) * 100.0 / ((dh + dc) as f64)
                } else {
                    self.points
                        .back()
                        .map(|p| p.interval_hit_pct)
                        .unwrap_or(cumulative_hit_pct)
                }
            }
            _ => cumulative_hit_pct,
        };

        let swaps_per_sec = match (promoted_total, self.last_promoted_total) {
            (Some(cur), Some(prev)) if dt > 0.05 => (cur.saturating_sub(prev) as f64) / dt,
            _ => 0.0,
        };

        if hbm_routes.is_some() || promoted_total.is_some() {
            self.points.push_back(ParallaxPoint {
                time: elapsed,
                interval_hit_pct,
                cumulative_hit_pct,
                swaps_per_sec,
            });
            while self
                .points
                .front()
                .is_some_and(|p| elapsed - p.time > HISTORY)
            {
                self.points.pop_front();
            }
        }

        self.last_time = Some(now);
        if hbm_routes.is_some() {
            self.last_hbm_routes = hbm_routes;
            self.last_cold_routes = cold_routes;
        }
        if promoted_total.is_some() {
            self.last_promoted_total = promoted_total;
        }

        // Ingest movement transactions for flashes, heatmap, and ticker.
        self.ingest_events(now, physical);
        // Expire old flashes cleanly.
        self.promoted_flash
            .retain(|_, ts| now.saturating_duration_since(*ts).as_secs_f64() <= FLASH_SECONDS);
        self.evicted_flash
            .retain(|_, ts| now.saturating_duration_since(*ts).as_secs_f64() <= FLASH_SECONDS);

        self.latest_report = report;
    }

    fn ingest_events(&mut self, now: Instant, physical: &Value) {
        let Some(events) = physical["movement_events"].as_array() else {
            return;
        };
        let initial_seed = self.last_generation == 0;
        let max_gen = events
            .iter()
            .filter_map(|e| e["generation"].as_u64())
            .max()
            .unwrap_or(0);
        if max_gen < self.last_generation {
            // Server restarted with a new generation counter.
            self.last_generation = 0;
            self.expert_heat.clear();
            self.layer_churn.clear();
            self.promoted_flash.clear();
            self.evicted_flash.clear();
            self.last_commit = None;
        }

        for event in events {
            let Some(generation) = event["generation"].as_u64() else {
                continue;
            };
            if generation <= self.last_generation {
                continue;
            }
            let promoted = parse_expert_list(&event["promote"]);
            let evicted = parse_expert_list(&event["evict"]);
            for &(l, e) in &promoted {
                *self.expert_heat.entry((l, e)).or_insert(0) += 1;
                *self.layer_churn.entry(l).or_insert(0) += 1;
            }
            for &(l, e) in &evicted {
                *self.expert_heat.entry((l, e)).or_insert(0) += 1;
            }
            let should_flash = !initial_seed || max_gen.saturating_sub(generation) < 3;
            if should_flash {
                let flash_time = if initial_seed {
                    let age = (max_gen.saturating_sub(generation) as f64) * 1.5;
                    now.checked_sub(Duration::from_secs_f64(age)).unwrap_or(now)
                } else {
                    now
                };
                for &id in &promoted {
                    self.evicted_flash.remove(&id);
                    self.promoted_flash.insert(id, flash_time);
                }
                for &id in &evicted {
                    self.promoted_flash.remove(&id);
                    self.evicted_flash.insert(id, flash_time);
                }
            }
            self.last_commit = Some(CommitSummary {
                generation,
                promoted,
                evicted,
                staged_bytes: event["staged_bytes"].as_u64().unwrap_or(0),
                copy_gpu_ns: event["copy_gpu_ns"].as_u64().unwrap_or(0),
                publication_host_ns: event["publication_host_ns"].as_u64().unwrap_or(0),
            });
            self.last_generation = generation;
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect, now: Instant) {
        self.render_with_report(frame, area, now, &self.latest_report);
    }

    pub fn render_with_report(&self, frame: &mut Frame, area: Rect, now: Instant, report: &Value) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let has_placement = report["physical_residency"]["placement"].is_object();
        if self.mode == ParallaxMode::Visual && has_placement {
            render_visual(frame, area, now, report, self);
            return;
        }
        render(frame, area, report.clone());
    }
}

fn parse_expert_list(value: &Value) -> Vec<(usize, usize)> {
    value
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    Some((
                        item["layer"].as_u64()? as usize,
                        item["expert"].as_u64()? as usize,
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn decode_bitset(encoded: Option<&str>, expected_bits: usize) -> Option<Vec<u8>> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(encoded?)
        .ok()?;
    if raw.len() < expected_bits.div_ceil(8) {
        return None;
    }
    Some(raw)
}

#[inline]
fn bit_is_set(bits: &[u8], idx: usize) -> bool {
    bits.get(idx / 8)
        .is_some_and(|byte| (byte & (1u8 << (idx % 8))) != 0)
}

#[inline]
fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    let t = t.clamp(0.0, 1.0);
    ((a as f64) + ((b as f64) - (a as f64)) * t).round() as u8
}

#[inline]
fn lerp_rgb(a: (u8, u8, u8), b: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    (
        lerp_u8(a.0, b.0, t),
        lerp_u8(a.1, b.1, t),
        lerp_u8(a.2, b.2, t),
    )
}

fn heat_rgb(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let stops = [
        (18, 24, 38),
        (78, 48, 168),
        (196, 62, 158),
        (246, 154, 72),
        (255, 244, 178),
    ];
    let scaled = t * ((stops.len() - 1) as f64);
    let idx = (scaled.floor() as usize).min(stops.len() - 2);
    let frac = scaled - (idx as f64);
    lerp_rgb(stops[idx], stops[idx + 1], frac)
}

fn format_commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn render_visual(
    frame: &mut Frame,
    area: Rect,
    now: Instant,
    report: &Value,
    view: &ParallaxView,
) {
    let physical = &report["physical_residency"];
    let ledger = &report["actual_ledger"];
    let placement = &physical["placement"];
    let layers = placement["layers"].as_u64().unwrap_or(0) as usize;
    let experts_per_layer = placement["experts_per_layer"].as_u64().unwrap_or(0) as usize;
    let total_experts = layers.saturating_mul(experts_per_layer);
    let Some(hbm_bits) = decode_bitset(placement["hbm_bitset"].as_str(), total_experts) else {
        render(frame, area, report.clone());
        return;
    };
    let protected_bits = decode_bitset(placement["protected_bitset"].as_str(), total_experts);

    // Compact fallback for tiny terminals.
    if area.width < 60 || area.height < 14 {
        let hbm = physical["hbm_experts"].as_u64().unwrap_or(0);
        let grace = physical["grace_experts"].as_u64().unwrap_or(0);
        let generation = physical["generation"].as_u64().unwrap_or(0);
        let hit = hit_rate_pct(ledger)
            .map(|p| format!("{p:.2}%"))
            .unwrap_or_else(|| "—".into());
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    format!("PARALLAX · {} · gen {generation} · HBM {hit}", mode(report)),
                    Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
                )),
                Line::from(format!("HBM {hbm} / Grace {grace} · {layers}×{experts_per_layer}")),
                Line::from("Enlarge terminal (≥ 60×14) for live residency matrix · [v] Ledger")
                    .style(Style::default().fg(MUTED)),
            ]),
            area,
        );
        return;
    }

    let outer = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(10),
        Constraint::Length(2),
    ])
    .split(area);

    render_visual_header(
        frame,
        outer[0],
        report,
        &hbm_bits,
        protected_bits.as_deref(),
        total_experts,
    );

    if outer[1].width >= 96 {
        let split = Layout::horizontal([Constraint::Percentage(64), Constraint::Percentage(36)])
            .split(outer[1]);
        render_matrix_panel(
            frame,
            split[0],
            now,
            layers,
            experts_per_layer,
            &hbm_bits,
            protected_bits.as_deref(),
            view,
        );
        render_telemetry_column(frame, split[1], report, view);
    } else {
        let split = Layout::vertical([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(outer[1]);
        render_matrix_panel(
            frame,
            split[0],
            now,
            layers,
            experts_per_layer,
            &hbm_bits,
            protected_bits.as_deref(),
            view,
        );
        render_compact_telemetry(frame, split[1], report, view);
    }

    render_commit_ticker(frame, outer[2], view, physical);
}

fn hit_rate_pct(ledger: &Value) -> Option<f64> {
    let hbm = ledger["hbm_routes"].as_u64()?;
    let cold = ledger["cold_routes"].as_u64()?;
    let total = hbm + cold;
    (total > 0).then(|| (hbm as f64) * 100.0 / (total as f64))
}

fn controller_badge(report: &Value) -> (String, Color) {
    if report["active_policy"]["failure"].is_string() {
        return ("FALLBACK · static".into(), GOLD);
    }
    match (
        report["active_policy"]["predictor"].as_str(),
        report["active_policy"]["policy"].as_str(),
    ) {
        (Some(pred), Some(pol)) => (format!("ACTIVE · {pred}/{pol}"), GREEN),
        (None, Some(pol)) => (format!("ACTIVE · {pol}"), GREEN),
        _ if report["physical_residency"].is_object() => ("STATIC · fixed".into(), PURPLE),
        _ => ("SHADOW".into(), MUTED),
    }
}

fn render_visual_header(
    frame: &mut Frame,
    area: Rect,
    report: &Value,
    hbm_bits: &[u8],
    protected_bits: Option<&[u8]>,
    total_experts: usize,
) {
    let physical = &report["physical_residency"];
    let ledger = &report["actual_ledger"];
    let generation = physical["generation"].as_u64().unwrap_or(0);
    let hbm = physical["hbm_experts"].as_u64().unwrap_or(0);
    let grace = physical["grace_experts"].as_u64().unwrap_or(0);
    let (badge, badge_color) = controller_badge(report);

    let hit_pct = hit_rate_pct(ledger);
    let hit_color = match hit_pct {
        Some(p) if p >= 95.0 => GREEN,
        Some(p) if p >= 88.0 => CYAN,
        Some(_) => GOLD,
        None => MUTED,
    };
    let hit_str = hit_pct
        .map(|p| format!("{p:.2}%"))
        .unwrap_or_else(|| "—".into());

    let line1 = Line::from(vec![
        Span::styled(
            " PARALLAX ",
            Style::default()
                .fg(BACKGROUND)
                .bg(CYAN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            format!(" {badge} "),
            Style::default()
                .fg(BACKGROUND)
                .bg(badge_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  GEN {}", format_commas(generation)), Style::default().fg(TEXT)),
        Span::styled("  ·  HBM HIT ", Style::default().fg(MUTED)),
        Span::styled(hit_str, Style::default().fg(hit_color).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("  ·  {} HBM / {} Grace", format_commas(hbm), format_commas(grace)),
            Style::default().fg(MUTED),
        ),
    ]);

    // Line 2: visual HBM / Grace capacity bar.
    let mut protected_count = 0usize;
    let mut dynamic_count = 0usize;
    if total_experts > 0 {
        for idx in 0..total_experts {
            if bit_is_set(hbm_bits, idx) {
                if protected_bits.is_some_and(|p| bit_is_set(p, idx)) {
                    protected_count += 1;
                } else {
                    dynamic_count += 1;
                }
            }
        }
    }
    let grace_count = total_experts.saturating_sub(protected_count + dynamic_count);
    let bar_width = (area.width as usize).clamp(20, 44).min(area.width.saturating_sub(36) as usize);
    let mut bar_spans = vec![Span::styled(" ", Style::default())];
    if bar_width >= 10 && total_experts > 0 {
        let prot_cols = ((protected_count * bar_width) + total_experts / 2) / total_experts;
        let dyn_cols = ((dynamic_count * bar_width) + total_experts / 2) / total_experts;
        let prot_cols = prot_cols.min(bar_width);
        let dyn_cols = dyn_cols.min(bar_width - prot_cols);
        let grace_cols = bar_width.saturating_sub(prot_cols + dyn_cols);
        bar_spans.push(Span::styled(
            "█".repeat(prot_cols),
            Style::default().fg(Color::Rgb(BG_PROTECTED.0, BG_PROTECTED.1, BG_PROTECTED.2)),
        ));
        bar_spans.push(Span::styled(
            "█".repeat(dyn_cols),
            Style::default().fg(Color::Rgb(BG_DYNAMIC.0, BG_DYNAMIC.1, BG_DYNAMIC.2)),
        ));
        bar_spans.push(Span::styled(
            "░".repeat(grace_cols),
            Style::default().fg(MUTED),
        ));
    }
    let hbm_gib = physical["hbm_pool_bytes"].as_u64().unwrap_or(0) as f64 / (1024.0 * 1024.0 * 1024.0);
    let cal_bw = report["hardware"]["transfer_calibration"]["host_to_device"]["bandwidth_bytes_per_ns"].as_f64();
    let bw_text = cal_bw
        .map(|b| format!(" · C2C {b:.1} GB/s"))
        .unwrap_or_default();
    let hbm_pct = if total_experts > 0 {
        ((protected_count + dynamic_count) as f64) * 100.0 / (total_experts as f64)
    } else {
        0.0
    };
    bar_spans.push(Span::styled(
        format!(
            "  {hbm_pct:.1}% HBM ({} prot · {} dyn · {} grace · {hbm_gib:.1} GiB{bw_text})",
            format_commas(protected_count as u64),
            format_commas(dynamic_count as u64),
            format_commas(grace_count as u64),
        ),
        Style::default().fg(MUTED),
    ));

    frame.render_widget(Paragraph::new(vec![line1, Line::from(bar_spans)]), area);
}

#[allow(clippy::too_many_arguments)]
fn render_matrix_panel(
    frame: &mut Frame,
    area: Rect,
    now: Instant,
    layers: usize,
    experts_per_layer: usize,
    hbm_bits: &[u8],
    protected_bits: Option<&[u8]>,
    view: &ParallaxView,
) {
    let title = format!(
        " EXPERT RESIDENCY MATRIX ({layers}×{experts_per_layer}) · LENS: {} ",
        view.lens.label()
    );
    let controls = Line::from(vec![
        Span::styled(" [v] ", Style::default().fg(CYAN).add_modifier(Modifier::BOLD)),
        Span::styled("Ledger ", Style::default().fg(MUTED)),
        Span::styled("[f] ", Style::default().fg(CYAN).add_modifier(Modifier::BOLD)),
        Span::styled("Lens ", Style::default().fg(MUTED)),
    ])
    .right_aligned();
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(MUTED))
        .title(title)
        .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD))
        .title_bottom(controls);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width < 16 || inner.height < 4 || layers == 0 || experts_per_layer == 0 {
        return;
    }

    // Reserve 2 bottom lines inside the block: expert axis + color legend.
    let matrix_rows = inner.height.saturating_sub(2) as usize;
    let label_cols = 4usize; // "L00 "
    let churn_cols = 3usize; // " ▓ "
    let plot_cols = (inner.width as usize).saturating_sub(label_cols + churn_cols);
    if matrix_rows == 0 || plot_cols == 0 {
        return;
    }

    // Half-block packing: 2 vertical sub-rows per terminal row.
    let half_rows = (matrix_rows * 2).min(layers);
    let used_rows = half_rows.div_ceil(2);
    let max_heat = view.expert_heat.values().copied().max().unwrap_or(1).max(1) as f64;
    let max_layer_churn = view.layer_churn.values().copied().max().unwrap_or(1).max(1) as f64;

    let mut lines = Vec::with_capacity(inner.height as usize);
    let sample_bucket = |l_start: usize, l_end: usize, e_start: usize, e_end: usize| -> (u8, u8, u8) {
        let mut prot = 0usize;
        let mut dyn_hbm = 0usize;
        let mut total = 0usize;
        let mut best_promo = 0.0f64;
        let mut best_evict = 0.0f64;
        let mut bucket_heat = 0u32;

        for l in l_start..l_end {
            let row_base = l * experts_per_layer;
            for e in e_start..e_end {
                total += 1;
                let idx = row_base + e;
                if bit_is_set(hbm_bits, idx) {
                    if protected_bits.is_some_and(|p| bit_is_set(p, idx)) {
                        prot += 1;
                    } else {
                        dyn_hbm += 1;
                    }
                }
                if let Some(ts) = view.promoted_flash.get(&(l, e)) {
                    let age = now.saturating_duration_since(*ts).as_secs_f64();
                    if age <= FLASH_SECONDS {
                        best_promo = best_promo.max(1.0 - age / FLASH_SECONDS);
                    }
                }
                if let Some(ts) = view.evicted_flash.get(&(l, e)) {
                    let age = now.saturating_duration_since(*ts).as_secs_f64();
                    if age <= FLASH_SECONDS {
                        best_evict = best_evict.max(1.0 - age / FLASH_SECONDS);
                    }
                }
                if view.lens == PlacementLens::Heatmap {
                    if let Some(&h) = view.expert_heat.get(&(l, e)) {
                        bucket_heat = bucket_heat.max(h);
                    }
                }
            }
        }

        if view.lens == PlacementLens::Heatmap {
            let base = if bucket_heat > 0 {
                heat_rgb((bucket_heat as f64 / max_heat).sqrt())
            } else {
                (18, 24, 38)
            };
            return if best_promo > 0.15 {
                lerp_rgb(base, BG_PROMOTED, best_promo * 0.85)
            } else if best_evict > 0.15 {
                lerp_rgb(base, BG_EVICTED, best_evict * 0.75)
            } else {
                base
            };
        }

        let total_f = total.max(1) as f64;
        let p_frac = (prot as f64) / total_f;
        let d_frac = (dyn_hbm as f64) / total_f;
        let g_frac = (1.0 - p_frac - d_frac).max(0.0);

        let (prot_rgb, dyn_rgb, grace_rgb) = match view.lens {
            PlacementLens::All => (BG_PROTECTED, BG_DYNAMIC, BG_GRACE),
            PlacementLens::Dynamic => (BG_PROTECTED_DIM, BG_DYNAMIC_BRIGHT, (16, 22, 34)),
            PlacementLens::Heatmap => unreachable!(),
        };

        let base = (
            (prot_rgb.0 as f64 * p_frac + dyn_rgb.0 as f64 * d_frac + grace_rgb.0 as f64 * g_frac)
                .round() as u8,
            (prot_rgb.1 as f64 * p_frac + dyn_rgb.1 as f64 * d_frac + grace_rgb.1 as f64 * g_frac)
                .round() as u8,
            (prot_rgb.2 as f64 * p_frac + dyn_rgb.2 as f64 * d_frac + grace_rgb.2 as f64 * g_frac)
                .round() as u8,
        );

        if best_promo > 0.05 {
            lerp_rgb(base, BG_PROMOTED, 0.35 + 0.65 * best_promo)
        } else if best_evict > 0.05 {
            lerp_rgb(base, BG_EVICTED, 0.35 + 0.65 * best_evict)
        } else {
            base
        }
    };

    for row in 0..used_rows {
        let top_hr = row * 2;
        let bot_hr = top_hr + 1;
        let top_l0 = top_hr * layers / half_rows;
        let top_l1 = ((top_hr + 1) * layers / half_rows).max(top_l0 + 1).min(layers);
        let has_bot = bot_hr < half_rows;
        let (bot_l0, bot_l1) = if has_bot {
            let b0 = bot_hr * layers / half_rows;
            let b1 = ((bot_hr + 1) * layers / half_rows).max(b0 + 1).min(layers);
            (b0, b1)
        } else {
            (0, 0)
        };

        let row_flashing = (top_l0..if has_bot { bot_l1 } else { top_l1 }).any(|l| {
            view.promoted_flash.keys().any(|&(fl, _)| fl == l)
                || view.evicted_flash.keys().any(|&(fl, _)| fl == l)
        });
        let label_style = if row_flashing {
            Style::default().fg(GREEN).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(MUTED)
        };
        let mut spans = Vec::with_capacity(plot_cols + 3);
        let label = if row % 2 == 0 || row + 1 == used_rows {
            format!("L{top_l0:02} ")
        } else {
            "    ".into()
        };
        spans.push(Span::styled(label, label_style));

        for col in 0..plot_cols {
            let e0 = col * experts_per_layer / plot_cols;
            let e1 = ((col + 1) * experts_per_layer / plot_cols)
                .max(e0 + 1)
                .min(experts_per_layer);
            let top = sample_bucket(top_l0, top_l1, e0, e1);
            let bot = if has_bot {
                sample_bucket(bot_l0, bot_l1, e0, e1)
            } else {
                (13, 18, 28)
            };
            spans.push(Span::styled(
                "▀",
                Style::default()
                    .fg(Color::Rgb(top.0, top.1, top.2))
                    .bg(Color::Rgb(bot.0, bot.1, bot.2)),
            ));
        }

        // Per-layer churn pulse strip on right gutter.
        let end_l = if has_bot { bot_l1 } else { top_l1 };
        let churn: u32 = (top_l0..end_l)
            .filter_map(|l| view.layer_churn.get(&l))
            .copied()
            .sum();
        let (glyph, churn_color) = if churn == 0 {
            (" · ", Color::Rgb(54, 68, 88))
        } else {
            let ratio = (churn as f64) / max_layer_churn;
            if ratio >= 0.75 {
                (" █ ", GOLD)
            } else if ratio >= 0.45 {
                (" ▓ ", GREEN)
            } else if ratio >= 0.20 {
                (" ▒ ", CYAN)
            } else {
                (" ░ ", PURPLE)
            }
        };
        spans.push(Span::styled(glyph, Style::default().fg(churn_color)));
        lines.push(Line::from(spans));
    }

    // Pad vertical space between matrix and bottom axis if terminal is extra tall.
    while lines.len() < matrix_rows {
        lines.push(Line::default());
    }

    // Bottom expert index axis.
    let mut axis = vec![' '; label_cols + plot_cols];
    let ticks = 6usize.min(plot_cols / 10).max(2);
    for t in 0..=ticks {
        let expert = if t == ticks {
            experts_per_layer.saturating_sub(1)
        } else {
            (t * experts_per_layer) / ticks
        };
        let col = if t == ticks {
            plot_cols.saturating_sub(1)
        } else {
            (t * plot_cols) / ticks
        };
        let txt = expert.to_string();
        let pos = label_cols + col.min(plot_cols.saturating_sub(txt.len()));
        for (i, ch) in txt.chars().enumerate() {
            if pos + i < axis.len() {
                axis[pos + i] = ch;
            }
        }
    }
    lines.push(Line::from(Span::styled(
        axis.into_iter().collect::<String>(),
        Style::default().fg(MUTED),
    )));

    // Legend line.
    let legend = if view.lens == PlacementLens::Heatmap {
        Line::from(vec![
            Span::raw("    "),
            Span::styled("▀ ", Style::default().fg(Color::Rgb(78, 48, 168))),
            Span::styled("Low churn  ", Style::default().fg(MUTED)),
            Span::styled("▀ ", Style::default().fg(Color::Rgb(196, 62, 158))),
            Span::styled("Warm  ", Style::default().fg(MUTED)),
            Span::styled("▀ ", Style::default().fg(Color::Rgb(246, 154, 72))),
            Span::styled("Hot  ", Style::default().fg(MUTED)),
            Span::styled("▀ ", Style::default().fg(Color::Rgb(255, 244, 178))),
            Span::styled("Peak  ", Style::default().fg(MUTED)),
            Span::styled("█", Style::default().fg(GOLD)),
            Span::styled(" Layer churn", Style::default().fg(MUTED)),
        ])
    } else {
        Line::from(vec![
            Span::raw("    "),
            Span::styled(
                "▀ ",
                Style::default().fg(Color::Rgb(BG_PROTECTED.0, BG_PROTECTED.1, BG_PROTECTED.2)),
            ),
            Span::styled("Protected  ", Style::default().fg(MUTED)),
            Span::styled(
                "▀ ",
                Style::default().fg(Color::Rgb(BG_DYNAMIC.0, BG_DYNAMIC.1, BG_DYNAMIC.2)),
            ),
            Span::styled("Dynamic HBM  ", Style::default().fg(MUTED)),
            Span::styled(
                "▀ ",
                Style::default().fg(Color::Rgb(BG_GRACE.0, BG_GRACE.1, BG_GRACE.2)),
            ),
            Span::styled("Grace  ", Style::default().fg(MUTED)),
            Span::styled(
                "▀ ",
                Style::default().fg(Color::Rgb(BG_PROMOTED.0, BG_PROMOTED.1, BG_PROMOTED.2)),
            ),
            Span::styled("Promoted↑  ", Style::default().fg(GREEN)),
            Span::styled(
                "▀ ",
                Style::default().fg(Color::Rgb(BG_EVICTED.0, BG_EVICTED.1, BG_EVICTED.2)),
            ),
            Span::styled("Evicted↓", Style::default().fg(GOLD)),
        ])
    };
    lines.push(legend);

    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_telemetry_column(frame: &mut Frame, area: Rect, report: &Value, view: &ParallaxView) {
    let rows = Layout::vertical([
        Constraint::Percentage(38),
        Constraint::Percentage(30),
        Constraint::Min(7),
    ])
    .split(area);

    render_hit_rate_chart(frame, rows[0], report, view);
    render_swap_velocity_chart(frame, rows[1], view);
    render_ledger_card(frame, rows[2], report);
}

fn render_compact_telemetry(frame: &mut Frame, area: Rect, report: &Value, view: &ParallaxView) {
    let cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    render_hit_rate_chart(frame, cols[0], report, view);
    render_ledger_card(frame, cols[1], report);
}

fn render_hit_rate_chart(frame: &mut Frame, area: Rect, report: &Value, view: &ParallaxView) {
    let now_t = view.points.back().map_or(0.0, |p| p.time);
    let live_data: Vec<(f64, f64)> = view
        .points
        .iter()
        .map(|p| (p.time - now_t, p.interval_hit_pct))
        .collect();
    let cum_data: Vec<(f64, f64)> = view
        .points
        .iter()
        .map(|p| (p.time - now_t, p.cumulative_hit_pct))
        .collect();

    let min_val = live_data
        .iter()
        .chain(cum_data.iter())
        .map(|&(_, y)| y)
        .filter(|y| *y > 0.0)
        .fold(100.0f64, f64::min);
    let y_min = ((min_val - 3.0).floor() / 5.0).floor() * 5.0;
    let y_min = y_min.clamp(0.0, 90.0);

    let live_now = view
        .points
        .back()
        .map(|p| format!("{:.1}%", p.interval_hit_pct))
        .or_else(|| hit_rate_pct(&report["actual_ledger"]).map(|p| format!("{p:.1}%")))
        .unwrap_or_else(|| "—".into());
    let avg_now = hit_rate_pct(&report["actual_ledger"])
        .map(|p| format!("{p:.1}%"))
        .unwrap_or_else(|| "—".into());
    let title = Line::from(vec![
        Span::styled(
            " HBM HIT  ",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{live_now} live"),
            Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" · ", Style::default().fg(MUTED)),
        Span::styled(format!("{avg_now} avg "), Style::default().fg(CYAN)),
    ]);

    let datasets = vec![
        Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(CYAN))
            .data(&cum_data),
        Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(GREEN))
            .data(&live_data),
    ];

    frame.render_widget(
        Chart::new(datasets)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(MUTED))
                    .title(title),
            )
            .style(Style::default().bg(BACKGROUND))
            .x_axis(
                Axis::default()
                    .bounds([-HISTORY, 0.0])
                    .labels(["−120s", "now"])
                    .style(Style::default().fg(MUTED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([y_min, 100.0])
                    .labels([format!("{y_min:.0}%"), "100%".into()])
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}

fn render_swap_velocity_chart(frame: &mut Frame, area: Rect, view: &ParallaxView) {
    let now_t = view.points.back().map_or(0.0, |p| p.time);
    let swap_data: Vec<(f64, f64)> = view
        .points
        .iter()
        .map(|p| (p.time - now_t, p.swaps_per_sec))
        .collect();
    let peak = swap_data
        .iter()
        .map(|&(_, y)| y)
        .fold(0.0f64, f64::max);
    let cur = view.points.back().map_or(0.0, |p| p.swaps_per_sec);
    let title = format!(" PROMOTIONS  {cur:.0} exp/s ");

    let datasets = vec![
        Dataset::default()
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(GOLD))
            .data(&swap_data),
    ];

    frame.render_widget(
        Chart::new(datasets)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(MUTED))
                    .title(title)
                    .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
            )
            .style(Style::default().bg(BACKGROUND))
            .x_axis(
                Axis::default()
                    .bounds([-HISTORY, 0.0])
                    .labels(["−120s", "now"])
                    .style(Style::default().fg(MUTED)),
            )
            .y_axis(
                Axis::default()
                    .bounds([0.0, peak.max(10.0) * 1.15])
                    .labels(["0".into(), format!("{:.0}", peak.max(10.0))])
                    .style(Style::default().fg(MUTED)),
            ),
        area,
    );
}

fn render_ledger_card(frame: &mut Frame, area: Rect, report: &Value) {
    let ledger = &report["actual_ledger"];
    let physical = &report["physical_residency"];
    let tx = ledger["timed_copy_transactions"].as_u64().unwrap_or(0);
    let copy_ms = ledger["copy_gpu_ns"].as_f64().unwrap_or(0.0) / 1e6;
    let avg_copy = if tx > 0 {
        format!("{:.2} ms/tx ({copy_ms:.0} ms)", copy_ms / (tx as f64))
    } else {
        "—".into()
    };
    let pub_ms = ledger["publication_host_ns"].as_f64().unwrap_or(0.0) / 1e6;
    let gate_ms = ledger["submission_gate_closed_ns"].as_f64().unwrap_or(0.0) / 1e6;
    let avoided = ledger["cold_expert_touches_avoided_vs_static"]
        .as_u64()
        .unwrap_or(0);
    let evict_touches = ledger["eviction_expert_touches_vs_static"]
        .as_u64()
        .unwrap_or(0);
    let used = ledger["promotions_used"].as_u64().unwrap_or(0);
    let unused = ledger["promotions_unused_before_eviction"]
        .as_u64()
        .unwrap_or(0);
    let pending = ledger["promotions_awaiting_use"].as_u64().unwrap_or(0);
    let balance = ledger["estimated_balance_ns"].as_f64().map(|v| v / 1e6);
    let (bal_str, bal_color) = match balance {
        Some(v) if v >= 0.0 => (format!("+{v:.1} ms"), GREEN),
        Some(v) => (format!("{v:.1} ms"), GOLD),
        None => ("learning".into(), MUTED),
    };
    let frames = ledger["observed_frames"].as_u64().unwrap_or(0);
    let rollbacks = physical["rollbacks"].as_u64().unwrap_or(0);

    let kv = |k: &'static str, v: String, color: Color| {
        Line::from(vec![
            Span::styled(format!(" {k:<12}"), Style::default().fg(MUTED)),
            Span::styled(v, Style::default().fg(color)),
        ])
    };
    let lines = vec![
        kv("GPU copy", avg_copy, TEXT),
        kv("Pub / gate", format!("{pub_ms:.1} / {gate_ms:.1} ms"), TEXT),
        kv(
            "Cold avoid",
            format!("+{} / −{}", format_commas(avoided), format_commas(evict_touches)),
            GREEN,
        ),
        kv("Promos", format!("{used} used · {unused} un · {pending} live"), CYAN),
        kv("Est. net", bal_str, bal_color),
        kv(
            "Observed",
            format!("{} frames · {rollbacks} rollbacks", format_commas(frames)),
            MUTED,
        ),
    ];

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(MUTED))
                .title(" LIVE LEDGER ")
                .title_style(Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
        ),
        area,
    );
}

fn render_commit_ticker(frame: &mut Frame, area: Rect, view: &ParallaxView, physical: &Value) {
    let Some(commit) = &view.last_commit else {
        let rollbacks = physical["rollbacks"].as_u64().unwrap_or(0);
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    Span::styled(" COMMIT ", Style::default().fg(BACKGROUND).bg(MUTED)),
                    Span::styled(
                        format!(
                            " Standing by · initial residency active · awaiting cold-route demand · {rollbacks} rollbacks"
                        ),
                        Style::default().fg(MUTED),
                    ),
                ]),
                Line::from(Span::styled(
                    " Tip: press [f] to switch between ALL / DYNAMIC / HEATMAP lenses, or [v] for the full accounting table.",
                    Style::default().fg(MUTED),
                )),
            ]),
            area,
        );
        return;
    };

    let mib = (commit.staged_bytes as f64) / (1024.0 * 1024.0);
    let gpu_ms = (commit.copy_gpu_ns as f64) / 1e6;
    let pub_ms = (commit.publication_host_ns as f64) / 1e6;

    let line1 = Line::from(vec![
        Span::styled(
            format!(" COMMIT #{} ", commit.generation),
            Style::default()
                .fg(BACKGROUND)
                .bg(GREEN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  +{}↑", commit.promoted.len()),
            Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" -{}↓", commit.evicted.len()),
            Style::default().fg(GOLD).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  ·  {mib:.1} MiB  ·  {gpu_ms:.2} ms GPU copy  ·  {pub_ms:.2} ms pub"),
            Style::default().fg(TEXT),
        ),
    ]);

    let max_items = ((area.width as usize).saturating_sub(14) / 10).clamp(2, 10);
    let mut line2_spans = vec![Span::styled(" ↑ ", Style::default().fg(GREEN).add_modifier(Modifier::BOLD))];
    for &(l, e) in commit.promoted.iter().take(max_items) {
        line2_spans.push(Span::styled(format!("L{l:02}:E{e:<3} "), Style::default().fg(GREEN)));
    }
    if commit.promoted.len() > max_items {
        line2_spans.push(Span::styled(
            format!("+{} ", commit.promoted.len() - max_items),
            Style::default().fg(MUTED),
        ));
    }
    if !commit.evicted.is_empty() {
        line2_spans.push(Span::styled("  ↓ ", Style::default().fg(GOLD).add_modifier(Modifier::BOLD)));
        let ev_items = (max_items / 2).max(2);
        for &(l, e) in commit.evicted.iter().take(ev_items) {
            line2_spans.push(Span::styled(format!("L{l:02}:E{e:<3} "), Style::default().fg(GOLD)));
        }
        if commit.evicted.len() > ev_items {
            line2_spans.push(Span::styled(
                format!("+{}", commit.evicted.len() - ev_items),
                Style::default().fg(MUTED),
            ));
        }
    }

    frame.render_widget(Paragraph::new(vec![line1, Line::from(line2_spans)]), area);
}

fn number(row: &Value, key: &str, scale: f64) -> String {
    row[key]
        .as_f64()
        .map(|v| format!("{:.2}", v * scale))
        .unwrap_or_else(|| "—".into())
}
fn mode(report: &Value) -> &'static str {
    if report["active_policy"].is_object() {
        "ACTIVE + SHADOW"
    } else if report["physical_residency"].is_object() {
        "STATIC + SHADOW"
    } else {
        "SHADOW"
    }
}
fn physical_status(report: &Value) -> String {
    let state = &report["physical_residency"];
    if let Some(error) = state["error"].as_str() {
        return format!("Physical storage: {error}");
    }
    if state.is_object() {
        let control = if report["active_policy"]["failure"].is_string() {
            "static fallback"
        } else {
            report["active_policy"]["predictor"]
                .as_str()
                .or_else(|| report["active_policy"]["policy"].as_str())
                .unwrap_or("initial placement")
        };
        format!(
            "Actual weights: {} HBM / {} Grace experts · {control} · generation {}",
            state["hbm_experts"], state["grace_experts"], state["generation"]
        )
    } else {
        "No physical expert movement. Placement costs are modeled, not serving speedups.".into()
    }
}
pub(super) fn render(frame: &mut Frame, area: Rect, report: Value) {
    if report["actual_ledger"].is_object() {
        render_actual(frame, area, report);
        return;
    }
    if report["evaluation"]["version"] == 2 {
        render_ledger(frame, area, report);
        return;
    }
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(6),
        Constraint::Length(6),
    ])
    .split(area);
    let enabled = report.get("mode").is_some();
    let title = if enabled {
        format!(
            "PARALLAX · {} · {} routing frames · in-process predictors",
            mode(&report),
            report["routing_frames"]
        )
    } else {
        "PARALLAX · disabled".into()
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(title, Style::default().fg(CYAN))),
            Line::from(physical_status(&report)),
        ]),
        rows[0],
    );
    if !enabled {
        frame.render_widget(
            Paragraph::new("Enable with --parallax-routing --parallax-profile PATH.\nProviders are compiled into Flare and selected by the runtime profile.")
                .wrap(Wrap { trim: false }),
            rows[1],
        );
        return;
    }
    if let Some(error) = report["error"].as_str() {
        frame.render_widget(
            Paragraph::new(error)
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(GOLD)),
            rows[1],
        );
        return;
    }
    let entries = report["contenders"]
        .as_object()
        .map(|r| {
            r.iter()
                .map(|(name, v)| {
                    Row::new(vec![
                        name.clone(),
                        v["prediction_records"].to_string(),
                        number(v, "forecast_precision", 100.),
                        number(v, "demand_coverage", 100.),
                        number(v, "prediction_compute_p50_ns", 1e-6),
                        number(v, "usable_lead_p50_ns", 1e-6),
                        number(v, "hbm_hit_rate", 100.),
                        number(v, "estimated_net_route_cost_saved_ns", 1e-6),
                        format!("{}/{}", v["resident_experts"], v["cold_experts"]),
                        format!("{}/{}", v["promotion_count"], v["eviction_count"]),
                    ])
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let widths = [
        Constraint::Min(18),
        Constraint::Length(6),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(8),
        Constraint::Length(11),
        Constraint::Length(11),
        Constraint::Length(11),
    ];
    frame.render_widget(
        Table::new(entries, widths).header(
            Row::new([
                "Predictor / policy",
                "N",
                "P@8 %",
                "Coverage %",
                "Pred ms",
                "Lead ms",
                "HBM %*",
                "Net ms*",
                "Hot/cold*",
                "Move/evict*",
            ])
            .style(Style::default().fg(CYAN)),
        ),
        rows[1],
    );
    let workers = report["workers"]
        .as_object()
        .map(|r| {
            r.iter()
                .map(|(name, c)| {
                    format!(
                        "{name}: {} busy drops, {} failures, {} timeouts",
                        c["dropped_busy"], c["failures"], c["timeouts"]
                    )
                })
                .collect::<Vec<_>>()
                .join(" · ")
        })
        .unwrap_or_default();
    let calibration = &report["hardware"]["transfer_calibration"];
    let costs = if calibration.is_null() {
        "Transfer and cold-execution costs supplied by profile; not calibrated here.".into()
    } else if calibration["failure"].is_string() {
        "Transfer calibration unavailable; observing without placement recommendations.".into()
    } else {
        format!(
            "Pinned copies: H→D {} GB/s · D→H {} GB/s · cold-execution penalty supplied by profile.",
            number(&calibration["host_to_device"], "bandwidth_bytes_per_ns", 1.),
            number(&calibration["device_to_host"], "bandwidth_bytes_per_ns", 1.)
        )
    };
    frame.render_widget(
        Paragraph::new(format!(
            "{}\n{}\nPrediction window: {} frames · residency: lifetime · predictors continue after trace retention fills.\n{}\n* Contender placements and additive costs are simulated. Frames are correlated.",
            report["hardware"]["identity"].as_str().unwrap_or(""),
            workers,
            report["evaluation_window_frames"],
            costs
        ))
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(MUTED)),
        rows[2],
    );
}

fn render_actual(frame: &mut Frame, area: Rect, report: Value) {
    let ledger = &report["actual_ledger"];
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(10),
        Constraint::Min(4),
        Constraint::Length(5),
    ])
    .split(area);
    let controller = match (
        report["active_policy"]["predictor"].as_str(),
        report["active_policy"]["policy"].as_str(),
    ) {
        (Some(predictor), Some(policy)) => format!("{predictor} · {policy}"),
        (None, Some(policy)) => format!("reactive · {policy}"),
        _ => "static · fixed initial placement".into(),
    };
    let visual_hint = if report["physical_residency"]["placement"].is_object() {
        "  [v] Residency Map"
    } else {
        ""
    };
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                format!(
                    "PARALLAX · {} · ACTUAL RESIDENCY LEDGER · {} frames{visual_hint}",
                    mode(&report),
                    ledger["observed_frames"]
                ),
                Style::default().fg(CYAN),
            )),
            Line::from(physical_status(&report)),
            Line::from(format!("Controller: {controller}")),
        ]),
        areas[0],
    );
    let ms = |key| {
        ledger[key]
            .as_f64()
            .map(|v| format!("{:+.2} ms", v / 1e6))
            .unwrap_or("learning".into())
    };
    let (hbm_rate, hbm_scope) = match (
        ledger["hbm_routes"].as_u64(),
        ledger["cold_routes"].as_u64(),
    ) {
        (Some(hbm), Some(cold)) if hbm + cold > 0 => (
            format!("{:.2}%", (hbm as f64) * 100.0 / ((hbm + cold) as f64)),
            format!("{hbm} HBM / {cold} cold routes"),
        ),
        _ => ("—".into(), "Awaiting observed routes".into()),
    };
    let rows = vec![
        Row::new(vec!["HBM route hit rate".into(), hbm_rate, hbm_scope]),
        Row::new(vec![
            "Copy GPU time".into(),
            ms("copy_gpu_ns"),
            format!(
                "{} timed transactions; startup excluded",
                ledger["timed_copy_transactions"]
            ),
        ]),
        Row::new(vec![
            "Publication CPU time".into(),
            ms("publication_host_ns"),
            "Actual mapping calls".into(),
        ]),
        Row::new(vec![
            "Ready → publication wait".into(),
            ms("ready_wait_ns"),
            "Lower bound from first ready poll; not GPU idle time".into(),
        ]),
        Row::new(vec![
            "Submission gate closed".into(),
            ms("submission_gate_closed_ns"),
            "Includes reader drain; not GPU idle time".into(),
        ]),
        Row::new(vec![
            "Cold touches avoided/added".into(),
            format!(
                "{}/{}",
                ledger["cold_expert_touches_avoided_vs_static"],
                ledger["eviction_expert_touches_vs_static"]
            ),
            "Against initial placement; one per expert/graph".into(),
        ]),
        Row::new(vec![
            "Promotions used/unused".into(),
            format!(
                "{}/{}",
                ledger["promotions_used"], ledger["promotions_unused_before_eviction"]
            ),
            format!("{} still awaiting use", ledger["promotions_awaiting_use"]),
        ]),
        Row::new(vec![
            "Execution saving (estimate)".into(),
            ms("estimated_execution_saved_ns"),
            format!(
                "Prior timing fit on {}/{} frames",
                ledger["modeled_frames"], ledger["observed_frames"]
            ),
        ]),
        Row::new(vec![
            "Accounting balance (est.)".into(),
            ms("estimated_balance_ns"),
            "Saving minus full copy + publication charge".into(),
        ]),
    ];
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(28),
                Constraint::Length(18),
                Constraint::Min(12),
            ],
        )
        .header(Row::new(["Measurement", "Value", "Scope"]).style(Style::default().fg(CYAN))),
        areas[1],
    );
    let workers = report["workers"]
        .as_object()
        .map(|workers| {
            workers
                .iter()
                .map(|(name, w)| {
                    let row = &report["contenders"][format!("{name}/static")];
                    Row::new(vec![
                        name.clone(),
                        number(row, "demand_coverage", 100.),
                        number(row, "usable_lead_p50_ns", 1e-6),
                        number(row, "prediction_compute_p50_ns", 1e-6),
                        w["dropped_busy"].to_string(),
                    ])
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    frame.render_widget(
        Table::new(
            workers,
            [
                Constraint::Min(16),
                Constraint::Length(12),
                Constraint::Length(12),
                Constraint::Length(12),
                Constraint::Length(8),
            ],
        )
        .header(
            Row::new(["Shadow predictor", "Coverage %", "Lead ms", "Compute ms", "Drops"])
                .style(Style::default().fg(CYAN)),
        ),
        areas[2],
    );
    let error = ledger["error"]
        .as_str()
        .or(report["error"].as_str())
        .unwrap_or("Ledger complete for observed frames.");
    frame.render_widget(
        Paragraph::new(format!(
            "{error}\n{} frames overlapped movement; excluded from cost fitting.\nCost model: matching graph/batch/rows; cold and total distinct weight bytes.\nBalance is observational, not measured serving improvement.\nCopy interference and predictor CPU overhead are not attributed.",
            ledger["movement_overlap_frames"]
        ))
        .style(Style::default().fg(MUTED))
        .wrap(Wrap { trim: false }),
        areas[3],
    );
}

fn render_ledger(frame: &mut Frame, area: Rect, report: Value) {
    use ratatui::widgets::Cell;
    let provider_count = report["workers"].as_object().map_or(0, |w| w.len());
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(7),
        Constraint::Length(provider_count as u16 + 4),
    ])
    .split(area);
    let copies = if report["hardware"]["transfer_calibration"]["host_to_device"].is_object() {
        "measured"
    } else {
        "assumed"
    };
    let penalty = report["hardware"]["cold_cost_ns"].as_f64().unwrap_or(0.) / 1000.;
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                format!(
                    "PARALLAX · {} · {} routing frames",
                    mode(&report),
                    report["routing_frames"]
                ),
                Style::default().fg(CYAN),
            )),
            Line::from(format!(
                "Copies {copies}; cold penalty assumed: {penalty:.3} µs/use. Contenders remain shadow."
            )),
            Line::from(physical_status(&report)),
        ]),
        areas[0],
    );
    if let Some(error) = report["error"].as_str() {
        frame.render_widget(
            Paragraph::new(error)
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(GOLD)),
            areas[1],
        );
        return;
    }
    let entries = report["contenders"]
        .as_object()
        .map(|rows| {
            rows.iter()
                .map(|(name, row)| {
                    let balance = row["modeled_balance_ns"].as_f64().unwrap_or(0.) / 1e6;
                    Row::new(vec![
                        Cell::from(name.clone()),
                        Cell::from(number(row, "cold_execution_saved_ns", 1e-6)),
                        Cell::from(number(row, "eviction_execution_loss_ns", 1e-6)),
                        Cell::from(number(row, "modeled_movement_cost_ns", 1e-6)),
                        Cell::from(format!("{balance:+.2}")).style(Style::default().fg(
                            if balance > 0. {
                                GREEN
                            } else if balance < 0. {
                                GOLD
                            } else {
                                MUTED
                            },
                        )),
                        Cell::from(format!(
                            "{}/{}",
                            row["promotion_count"], row["eviction_count"]
                        )),
                        Cell::from(number(row, "hbm_hit_rate", 100.)),
                    ])
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    frame.render_widget(
        Table::new(
            entries,
            [
                Constraint::Min(18),
                Constraint::Length(9),
                Constraint::Length(9),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(8),
                Constraint::Length(8),
            ],
        )
        .header(
            Row::new([
                "Predictor / policy",
                "Saved ms",
                "Evict ms",
                "Copy ms",
                "Balance*",
                "Moves",
                "HBM hit%",
            ])
            .style(Style::default().fg(CYAN)),
        ),
        areas[1],
    );
    let mut lines = vec![format!(
        "Quality: top {}/layer · recent {} frames. Ledger: server lifetime.",
        report["evaluation"]["top_k_per_layer"], report["evaluation_window_frames"]
    )];
    if let Some(workers) = report["workers"].as_object() {
        for (name, worker) in workers {
            let row = &report["contenders"][format!("{name}/static")];
            lines.push(format!(
                "{name}: covered {}% · ahead {} ms · predict {} ms · {} drops",
                number(row, "demand_coverage", 100.),
                number(row, "usable_lead_p50_ns", 1e-6),
                number(row, "prediction_compute_p50_ns", 1e-6),
                worker["dropped_busy"]
            ));
        }
    }
    lines.push(
        report["hardware"]["identity"]
            .as_str()
            .unwrap_or("Unidentified cost model")
            .into(),
    );
    lines.push("Moves = promotions/evictions. Saved/lost accesses are relative to static.".into());
    lines.push(
        "* Balance = Saved − Evict − Copy; excludes contention. Not throughput gains.".into(),
    );
    frame.render_widget(
        Paragraph::new(lines.join("\n")).style(Style::default().fg(MUTED)),
        areas[2],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::parallax::encode_expert_bitset;

    #[test]
    fn actual_ledger_never_displays_the_synthetic_shadow_balance() {
        use ratatui::{Terminal, backend::TestBackend};
        let report = serde_json::json!({"actual_ledger":{"observed_frames":10,"modeled_frames":0,
            "copy_gpu_ns":1000000,"publication_host_ns":2000000,"submission_gate_closed_ns":3000000},
            "physical_residency":{"hbm_experts":8,"grace_experts":4,"generation":2},
            "active_policy":{"predictor":"ewma","policy":"predictor_rank"},"workers":{},
            "contenders":{"ewma/economic":{"modeled_balance_ns":99999999999.}}});
        for width in [80, 120] {
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            terminal
                .draw(|frame| render(frame, frame.area(), report.clone()))
                .unwrap();
            let text = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect::<String>();
            for label in [
                "ACTUAL RESIDENCY LEDGER",
                "Copy GPU time",
                "learning",
                "predictor_rank",
            ] {
                assert!(text.contains(label), "{label}");
            }
            assert!(!text.contains("99999"));
            assert!(!text.contains("cold penalty assumed"));
        }
    }

    #[test]
    fn ledger_labels_and_signed_balances_fit_normal_terminal_widths() {
        use ratatui::{Terminal, backend::TestBackend};
        let report = serde_json::json!({"evaluation":{"version":2,"top_k_per_layer":32},"routing_frames":1000,
            "evaluation_window_frames":100,"hardware":{"cold_cost_ns":1000},"workers":{"ewma":{"dropped_busy":0}},
            "contenders":{"ewma/economic":{"cold_execution_saved_ns":10000000.,"eviction_execution_loss_ns":2000000.,
                "modeled_movement_cost_ns":1000000,"modeled_balance_ns":7000000.,"promotion_count":3,"eviction_count":3,"hbm_hit_rate":0.6}}});
        for width in [80, 120] {
            let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
            terminal
                .draw(|frame| render(frame, frame.area(), report.clone()))
                .unwrap();
            let text = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect::<String>();
            for label in [
                "Saved ms",
                "Evict ms",
                "Copy ms",
                "Balance*",
                "+7.00",
                "cold penalty assumed",
            ] {
                assert!(text.contains(label), "missing {label} at width {width}");
            }
            assert!(!text.contains("Net ms"));
        }
    }

    #[test]
    fn physical_storage_is_distinct_from_shadow_ledger() {
        let mut report =
            serde_json::json!({"physical_residency":{"hbm_experts":8,"grace_experts":4,"generation":12}});
        assert_eq!(mode(&report), "STATIC + SHADOW");
        assert!(physical_status(&report).contains("8 HBM / 4 Grace"));
        report["active_policy"] = serde_json::json!({"predictor":"ewma","failure":null});
        assert_eq!(mode(&report), "ACTIVE + SHADOW");
        assert!(physical_status(&report).contains("ewma"));
        report["active_policy"]["failure"] = serde_json::json!("provider failed");
        assert!(physical_status(&report).contains("static fallback"));
    }

    #[test]
    fn reactive_decayed_lru_displays_live_hit_rate_and_controller_identity() {
        use ratatui::{Terminal, backend::TestBackend};
        let report = serde_json::json!({"actual_ledger":{"observed_frames":42,"modeled_frames":40,
            "hbm_routes":950,"cold_routes":50,"copy_gpu_ns":1000000,"publication_host_ns":500000,
            "ready_wait_ns":2000000,"submission_gate_closed_ns":2500000,"timed_copy_transactions":5},
            "physical_residency":{"hbm_experts":11680,"grace_experts":3680,"generation":5},
            "active_policy":{"predictor":null,"policy":"decayed_miss_lru","failure":null},
            "workers":{"ewma":{"dropped_busy":0}},"contenders":{}});
        for width in [80, 120] {
            let mut terminal = Terminal::new(TestBackend::new(width, 28)).unwrap();
            terminal
                .draw(|frame| render(frame, frame.area(), report.clone()))
                .unwrap();
            let text = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect::<String>();
            for label in [
                "HBM route hit rate",
                "95.00%",
                "reactive · decayed_miss_lru",
                "Shadow predictor",
                "42 frames",
            ] {
                assert!(text.contains(label), "missing {label} at width {width}");
            }
            assert!(!text.contains("· initial placement"));
            assert!(!text.contains("Controller: static"));
        }
    }

    #[test]
    fn visual_residency_matrix_renders_half_blocks_charts_and_toggles_cleanly() {
        use ratatui::{Terminal, backend::TestBackend};
        let layers = 40;
        let experts = 384;
        let hbm_bitset = encode_expert_bitset(
            layers,
            experts,
            (0..layers).flat_map(|l| (0..292).map(move |e| (l, e))),
        );
        let protected_bitset = encode_expert_bitset(
            layers,
            experts,
            (0..layers).flat_map(|l| (0..196).map(move |e| (l, e))),
        );
        let start = Instant::now();
        let mut view = ParallaxView::new(start);
        for step in 1..=6u64 {
            let report = serde_json::json!({
                "actual_ledger": {
                    "observed_frames": step * 20,
                    "modeled_frames": step * 18,
                    "hbm_routes": step * 9900,
                    "cold_routes": step * 100,
                    "copy_gpu_ns": step * 1_500_000,
                    "publication_host_ns": step * 750_000,
                    "submission_gate_closed_ns": step * 2_000_000,
                    "timed_copy_transactions": step,
                    "cold_expert_touches_avoided_vs_static": step * 1000,
                    "eviction_expert_touches_vs_static": step * 50,
                    "promotions_used": step * 20,
                    "promotions_unused_before_eviction": 1,
                    "promotions_awaiting_use": 4,
                    "estimated_balance_ns": (step as f64) * 12_000_000.0,
                },
                "physical_residency": {
                    "hbm_experts": 11680,
                    "grace_experts": 3680,
                    "hbm_pool_bytes": 219_591_475_200u64,
                    "generation": step,
                    "promoted_experts_including_startup": step * 24,
                    "rollbacks": 0,
                    "movement_events": [{
                        "generation": step,
                        "promote": [{"layer": 2, "expert": 33}, {"layer": 19, "expert": 191}],
                        "evict": [{"layer": 2, "expert": 353}],
                        "staged_bytes": 451_215_360u64,
                        "copy_gpu_ns": 1_520_000u64,
                        "publication_host_ns": 750_000u64,
                    }],
                    "placement": {
                        "layers": layers,
                        "experts_per_layer": experts,
                        "hbm_bitset": hbm_bitset,
                        "protected_bitset": protected_bitset,
                    },
                },
                "active_policy": {"predictor": null, "policy": "decayed_miss_lru", "failure": null},
                "hardware": {"transfer_calibration": {"host_to_device": {"bandwidth_bytes_per_ns": 400.3}}},
                "workers": {},
                "contenders": {},
            });
            view.sample(start + Duration::from_secs(step), report);
        }

        let now = start + Duration::from_secs(6);
        for (w, h) in [(140, 36), (100, 28), (80, 24), (40, 12), (1, 1), (0, 0)] {
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal
                .draw(|frame| view.render(frame, frame.area(), now))
                .unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect();
            assert!(!text.contains("NaN"), "NaN at {w}x{h}");
            assert!(!text.contains("inf"), "inf at {w}x{h}");
            if w >= 100 && h >= 28 {
                assert!(text.contains("EXPERT RESIDENCY MATRIX (40×384)"), "{text}");
                assert!(text.contains("99.00%"));
                assert!(text.contains("COMMIT #6"));
                assert!(text.contains("L02:E33"));
                assert!(text.contains('▀'));
                assert!(text.chars().any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)));
            }
        }

        // Cycle lenses and toggle to ledger.
        view.cycle_lens();
        assert_eq!(view.lens(), PlacementLens::Dynamic);
        view.cycle_lens();
        assert_eq!(view.lens(), PlacementLens::Heatmap);
        let mut terminal = Terminal::new(TestBackend::new(140, 36)).unwrap();
        terminal
            .draw(|frame| view.render(frame, frame.area(), now))
            .unwrap();
        let heat_text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(heat_text.contains("LENS: HEATMAP"));

        view.toggle_mode();
        assert_eq!(view.mode(), ParallaxMode::Ledger);
        terminal
            .draw(|frame| view.render(frame, frame.area(), now))
            .unwrap();
        let ledger_text: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(ledger_text.contains("ACTUAL RESIDENCY LEDGER"));
        assert!(ledger_text.contains("[v] Residency Map"));
    }
}
