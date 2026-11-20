/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Parallax presentation uses the same terminal toolkit and palette as Flare serving.

use super::{Failure, Shared, View, read, sync_watch_state};
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use flare_helios::server::view::{
    BACKGROUND, CYAN, GOLD, GREEN, MUTED, PURPLE, TEXT, crossterm, ratatui,
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Row, Table, Wrap},
};
use serde_json::Value;
use std::{
    io,
    sync::atomic::Ordering,
    thread,
    time::{Duration, Instant},
};

struct Screen;
impl Drop for Screen {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stderr(), LeaveAlternateScreen, Show);
    }
}

fn refresh(view: &Shared) {
    sync_watch_state(view);
    let (output, done, has_report) = {
        let state = view.lock().unwrap();
        (state.output.clone(), state.done, state.report.is_some())
    };
    if done && has_report {
        return;
    }
    let capture = output.join("capture");
    let candidates = if done {
        [
            capture.join("comparison.json"),
            capture.join("live-comparison.json"),
            capture.join("leaderboard.json"),
            output.join("comparison.json"),
            output.join("live-comparison.json"),
            output.join("leaderboard.json"),
        ]
    } else {
        [
            capture.join("live-comparison.json"),
            capture.join("leaderboard.json"),
            capture.join("comparison.json"),
            output.join("live-comparison.json"),
            output.join("leaderboard.json"),
            output.join("comparison.json"),
        ]
    };
    for path in candidates {
        if let Ok(report) = read(&path) {
            let mut state = view.lock().unwrap();
            if !(state.done && state.report.is_some()) {
                state.report = Some(report);
            }
            break;
        }
    }
}

fn value(row: &Value, key: &str, scale: f64, suffix: &str) -> String {
    row[key]
        .as_f64()
        .map(|v| format!("{:.2}{suffix}", v * scale))
        .unwrap_or_else(|| "—".into())
}

fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .title(title)
        .border_style(Style::default().fg(MUTED))
}

fn draw(frame: &mut Frame<'_>, view: &View, pane: usize, elapsed: Duration, scroll: u16) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND).fg(TEXT)),
        area,
    );
    if area.width < 78 || area.height < 21 {
        frame.render_widget(
            Paragraph::new(format!(
                "FLARE · PARALLAX\n{}\nResize to at least 78 × 21.\nq: close viewer\n{}",
                view.phase,
                view.error.as_deref().unwrap_or("")
            ))
            .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(4),
        Constraint::Min(9),
        Constraint::Length(4),
        Constraint::Length(2),
    ])
    .split(area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    " FLARE / PARALLAX ",
                    Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
                ),
                Span::styled(" SHADOW ", Style::default().fg(PURPLE)),
                Span::raw(format!(
                    "  {:.0}s  ·  {}",
                    elapsed.as_secs_f64(),
                    view.phase
                )),
            ]),
            Line::from(format!(
                " {}",
                if view.model.is_empty() {
                    "Waiting for comparison metadata"
                } else {
                    &view.model
                }
            )),
        ]),
        areas[0],
    );
    let frames = view
        .report
        .as_ref()
        .map(|r| r["routing_frames"].to_string())
        .unwrap_or_else(|| "0".into());
    frame.render_widget(
        Paragraph::new(format!(
            "Frames evaluated: {frames}    Responses: {}    Physical moves: 0\nMode: {}",
            view.responses.len(),
            view.gpu
        ))
        .block(panel(" Session ")),
        areas[1],
    );
    if let Some(error) = view
        .error
        .as_deref()
        .or_else(|| view.report.as_ref().and_then(|r| r["error"].as_str()))
    {
        frame.render_widget(
            Paragraph::new(error)
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(GOLD))
                .block(panel(" Capture failed · no completed result ")),
            areas[2],
        );
    } else if pane == 2 {
        let text = view
            .responses
            .iter()
            .enumerate()
            .map(|(i, s)| format!("Response {}\n{s}\n", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        frame.render_widget(
            Paragraph::new(if text.is_empty() {
                "Waiting for generated responses".into()
            } else {
                text
            })
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
            .block(panel(
                " Generated responses · ↑ ↓ scroll · saved in responses.json ",
            )),
            areas[2],
        );
    } else if let Some(report) = &view.report {
        let rows = report["contenders"]
            .as_object()
            .map(|r| {
                r.iter()
                    .map(|(name, row)| {
                        let cells = if pane == 0 {
                            vec![
                                name.clone(),
                                row["prediction_records"].to_string(),
                                value(row, "forecast_precision", 100., "%"),
                                value(row, "demand_coverage", 100., "%"),
                                value(row, "usable_lead_p50_ns", 1e-6, ""),
                                value(row, "hbm_hit_rate", 100., "%"),
                                row["promotion_count"].to_string(),
                                value(
                                    row,
                                    if report["evaluation"]["version"] == 2 {
                                        "modeled_balance_ns"
                                    } else {
                                        "estimated_net_route_cost_saved_ns"
                                    },
                                    1e-6,
                                    "",
                                ),
                            ]
                        } else {
                            vec![
                                name.clone(),
                                value(row, "prediction_compute_p50_ns", 1e-6, ""),
                                value(row, "prediction_compute_p95_ns", 1e-6, ""),
                                value(row, "prediction_queue_p50_ns", 1e-6, ""),
                                value(row, "prediction_publication_p50_ns", 1e-6, ""),
                                value(row, "actionable_rate", 100., "%"),
                                row["late_predictions"].to_string(),
                                row["expired_predictions"].to_string(),
                            ]
                        };
                        Row::new(cells).style(Style::default().fg(TEXT))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let header = if pane == 0 {
            vec![
                "Predictor / policy",
                "N",
                if report["evaluation"]["version"] == 2 {
                    "P/layer"
                } else {
                    "P@8"
                },
                "Coverage",
                "Lead ms",
                "HBM*",
                "Moves*",
                if report["evaluation"]["version"] == 2 {
                    "Balance*"
                } else {
                    "Legacy ms*"
                },
            ]
        } else {
            vec![
                "Predictor / policy",
                "P50 ms",
                "P95 ms",
                "Queue ms",
                "Recv ms",
                "Actionable",
                "Late",
                "Expired",
            ]
        };
        let widths = [
            Constraint::Min(19),
            Constraint::Length(6),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(7),
            Constraint::Length(10),
        ];
        frame.render_widget(
            Table::new(rows, widths)
                .column_spacing(1)
                .header(Row::new(header).style(Style::default().fg(CYAN)))
                .block(panel(if pane == 0 {
                    " Prediction and modeled residency "
                } else {
                    " Predictor timing · measured wall latency "
                })),
            areas[2],
        );
    } else {
        let log = std::fs::read_to_string(view.output.join("server.log")).unwrap_or_default();
        let tail = log
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        frame.render_widget(
            Paragraph::new(format!("{}\n{tail}", view.phase))
                .wrap(Wrap { trim: false })
                .block(panel(" Waiting for comparison report ")),
            areas[2],
        );
    }
    let profile = view
        .report
        .as_ref()
        .and_then(|r| r["hardware"]["identity"].as_str())
        .unwrap_or("Cost model will be shown when a comparison report is available");
    let counts = view
        .report
        .as_ref()
        .and_then(|r| r["workers"].as_object())
        .map(|workers| {
            workers
                .iter()
                .map(|(name, w)| {
                    format!(
                        "{name}: {} busy drops / {} failures / {} timeouts",
                        w["dropped_busy"], w["failures"], w["timeouts"]
                    )
                })
                .collect::<Vec<_>>()
                .join("   ")
        })
        .unwrap_or_default();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                "* Virtual placement; additive route-cost estimates, not serving speedups.",
                Style::default().fg(GOLD),
            )),
            Line::from(profile),
            Line::from(counts),
        ])
        .wrap(Wrap { trim: false }),
        areas[3],
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                if view.done {
                    " Tab / ← →: comparison · timing · responses    q / Enter: close"
                } else {
                    " Tab / ← →: comparison · timing · responses    q: close viewer"
                },
                Style::default().fg(if view.done { GREEN } else { MUTED }),
            )),
            Line::from(format!(" {}", view.output.display())),
        ]),
        areas[4],
    );
}

pub fn run(view: &Shared) -> Result<(), Failure> {
    let _screen = Screen;
    enable_raw_mode()?;
    execute!(io::stderr(), EnterAlternateScreen, Hide)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stderr()))?;
    let start = Instant::now();
    let mut pane = 0;
    let mut scroll = 0u16;
    loop {
        refresh(view);
        {
            let state = view.lock().unwrap();
            terminal.draw(|frame| draw(frame, &state, pane, start.elapsed(), scroll))?;
            if crate::STOP.load(Ordering::Relaxed) {
                break;
            }
        }
        if event::poll(Duration::from_millis(200))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Up if pane == 2 => scroll = scroll.saturating_sub(1),
                KeyCode::Down if pane == 2 => scroll = scroll.saturating_add(1),
                KeyCode::Tab | KeyCode::Right => pane = (pane + 1) % 3,
                KeyCode::Left => pane = (pane + 2) % 3,
                KeyCode::Char('q') | KeyCode::Esc | KeyCode::Enter => break,
                KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => break,
                _ => {}
            }
        }
    }
    Ok(())
}

pub fn plain(view: &Shared) -> Result<(), Failure> {
    let mut previous = String::new();
    let mut last_frames = None;
    loop {
        refresh(view);
        {
            let state = view.lock().unwrap();
            if state.phase != previous {
                eprintln!("Parallax: {}", state.phase);
                previous = state.phase.clone();
            }
            let frames = state
                .report
                .as_ref()
                .and_then(|report| report["routing_frames"].as_u64());
            if frames != last_frames
                && let Some(count) = frames
            {
                eprintln!("Parallax: {count} routing frames evaluated");
                last_frames = Some(count);
            }
            if state.done || crate::STOP.load(Ordering::Relaxed) {
                if let Some(report) = &state.report {
                    println!("{}", serde_json::to_string_pretty(report)?);
                }
                if let Some(error) = &state.error {
                    eprintln!("{error}");
                }
                break;
            }
        }
        thread::sleep(Duration::from_millis(200));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    #[test]
    fn dashboard_keeps_failure_and_no_movement_scope_visible() {
        let mut terminal = Terminal::new(TestBackend::new(110, 28)).unwrap();
        let state = View {
            phase: "Incomplete".into(),
            error: Some("routing retention gap at sequence 353".into()),
            ..View::default()
        };
        terminal
            .draw(|f| draw(f, &state, 0, Duration::from_secs(1), 0))
            .unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(text.contains("routing retention gap"));
        assert!(text.contains("Physical moves: 0"));
        assert!(text.contains("not serving speedups"));
        let mut small = Terminal::new(TestBackend::new(60, 15)).unwrap();
        small
            .draw(|f| draw(f, &state, 0, Duration::ZERO, 0))
            .unwrap();
        assert!(
            small
                .backend()
                .buffer()
                .content
                .iter()
                .map(|c| c.symbol())
                .collect::<String>()
                .contains("Resize")
        );
    }

    #[test]
    fn refresh_reads_live_comparison_during_serving_and_preserves_final_report() {
        use serde_json::json;
        use std::sync::{Arc, Mutex};

        let root = std::env::temp_dir().join(format!(
            "flare-parallax-tui-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let capture = root.join("capture");
        std::fs::create_dir_all(&capture).unwrap();
        let live = json!({
            "routing_frames": 42,
            "evaluation": {"version": 2},
            "hardware": {"identity": "synthetic-gb300"},
            "workers": {"ewma": {"dropped_busy": 0, "failures": 0, "timeouts": 0}},
            "contenders": {
                "ewma/economic": {
                    "prediction_records": 40,
                    "forecast_precision": 0.5,
                    "demand_coverage": 0.75,
                    "usable_lead_p50_ns": 12_000_000.0,
                    "hbm_hit_rate": 0.8,
                    "promotion_count": 3,
                    "modeled_balance_ns": 4_500_000.0
                }
            }
        });
        std::fs::write(
            capture.join("live-comparison.json"),
            serde_json::to_vec(&live).unwrap(),
        )
        .unwrap();

        let view = Arc::new(Mutex::new(View {
            phase: "Serving".into(),
            output: root.clone(),
            ..View::default()
        }));
        refresh(&view);
        assert_eq!(
            view.lock().unwrap().report.as_ref().unwrap()["routing_frames"],
            json!(42)
        );

        // Render with the live report to confirm the TUI displays periodic live updates.
        let mut terminal = Terminal::new(TestBackend::new(110, 28)).unwrap();
        {
            let state = view.lock().unwrap();
            terminal
                .draw(|f| draw(f, &state, 0, Duration::from_secs(2), 0))
                .unwrap();
        }
        let text = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<String>();
        assert!(text.contains("Frames evaluated: 42"));
        assert!(text.contains("ewma/economic"));

        // While still running, live-comparison.json takes precedence over comparison.json.
        let final_report = json!({"routing_frames": 100, "contenders": {}});
        std::fs::write(
            capture.join("comparison.json"),
            serde_json::to_vec(&final_report).unwrap(),
        )
        .unwrap();
        refresh(&view);
        assert_eq!(
            view.lock().unwrap().report.as_ref().unwrap()["routing_frames"],
            json!(42)
        );

        // Once finished with an explicit report (for example verified replay), refresh preserves it.
        {
            let mut state = view.lock().unwrap();
            state.done = true;
            state.report = Some(json!({"routing_frames": 999}));
        }
        refresh(&view);
        assert_eq!(
            view.lock().unwrap().report.as_ref().unwrap()["routing_frames"],
            json!(999)
        );

        // When finished without an in-memory report, refresh falls back to comparison.json.
        view.lock().unwrap().report = None;
        refresh(&view);
        assert_eq!(
            view.lock().unwrap().report.as_ref().unwrap()["routing_frames"],
            json!(100)
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
