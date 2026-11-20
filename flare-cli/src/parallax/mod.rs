/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Terminal viewer for external Parallax shadow evaluation artifacts.
//! Predictors, shadow simulation, and replay run outside Helios; this module
//! reads saved or live JSON reports and renders them using Flare's terminal UI.

mod tui;

use serde_json::Value;
use std::{
    fs,
    io::IsTerminal,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;
type Shared = Arc<Mutex<View>>;

#[derive(Default)]
struct View {
    phase: String,
    model: String,
    gpu: String,
    output: PathBuf,
    responses: Vec<String>,
    report: Option<Value>,
    error: Option<String>,
    done: bool,
}

fn read(path: &Path) -> Result<Value, Failure> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn normalize_report(mut report: Value) -> Value {
    let Some(object) = report.as_object_mut() else {
        return report;
    };
    if object.contains_key("routing_frames") {
        return report;
    }
    let Some(contenders) = object.get("contenders").and_then(Value::as_object) else {
        return report;
    };
    if contenders.is_empty()
        || !contenders
            .values()
            .all(|entry| entry.get("summary").and_then(Value::as_object).is_some())
    {
        return report;
    }
    let mut flattened = serde_json::Map::new();
    let mut routing_frames = None;
    for (name, entry) in contenders {
        let summary = entry["summary"].clone();
        if routing_frames.is_none() {
            routing_frames = summary["routing_frames"].as_u64();
        }
        flattened.insert(name.clone(), summary);
    }
    if let Some(frames) = routing_frames {
        object.insert("routing_frames".into(), Value::from(frames));
        object.insert("contenders".into(), Value::Object(flattened));
    }
    report
}

fn validate_report(report: &Value) -> Result<(), Failure> {
    let Some(object) = report.as_object() else {
        return Err("Parallax comparison report must be a JSON object".into());
    };
    let has_contenders =
        object.get("routing_frames").and_then(Value::as_u64).is_some()
            && object.get("contenders").and_then(Value::as_object).is_some();
    let has_error = object.get("error").and_then(Value::as_str).is_some();
    if !has_contenders && !has_error {
        return Err(
            "Parallax comparison report must contain routing_frames/contenders or an error".into(),
        );
    }
    Ok(())
}

fn report_model_label(report: &Value) -> String {
    report["model"]["identity"]
        .as_str()
        .or_else(|| report["model"]["architecture"].as_str())
        .or_else(|| report["workload"].as_str())
        .unwrap_or("External Parallax comparison")
        .to_owned()
}

fn session_root_for(path: &Path) -> PathBuf {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    if dir.file_name().is_some_and(|name| name == "capture") {
        dir.parent().unwrap_or(&dir).to_path_buf()
    } else {
        dir
    }
}

fn load_report(path: &Path) -> Result<(PathBuf, Value), Failure> {
    let report_path = if path.is_dir() {
        let capture = path.join("capture");
        let candidates = [
            capture.join("comparison.json"),
            path.join("comparison.json"),
            capture.join("live-comparison.json"),
            path.join("live-comparison.json"),
            capture.join("leaderboard.json"),
            path.join("leaderboard.json"),
        ];
        candidates
            .into_iter()
            .find(|candidate| candidate.is_file())
            .ok_or_else(|| {
                format!(
                    "no Parallax comparison report found in {}",
                    path.display()
                )
            })?
    } else {
        path.to_path_buf()
    };
    let report = normalize_report(read(&report_path)?);
    validate_report(&report)?;
    Ok((session_root_for(&report_path), report))
}

fn load_responses(output: &Path) -> Vec<String> {
    let candidates = [
        output.join("responses.json"),
        output
            .parent()
            .map(|parent| parent.join("responses.json"))
            .unwrap_or_else(|| output.join("responses.json")),
    ];
    for path in candidates {
        let Ok(Value::Array(items)) = read(&path) else {
            continue;
        };
        return items
            .into_iter()
            .filter_map(|item| {
                item["choices"][0]["text"]
                    .as_str()
                    .or_else(|| item["choices"][0]["message"]["content"].as_str())
                    .map(ToOwned::to_owned)
            })
            .collect();
    }
    Vec::new()
}

fn sync_watch_state(view: &Shared) {
    let (output, done) = {
        let state = view.lock().unwrap();
        (state.output.clone(), state.done)
    };
    let responses = load_responses(&output);
    let demo = read(&output.join("demo.json")).ok();
    let mut state = view.lock().unwrap();
    if !responses.is_empty() {
        state.responses = responses;
    }
    if state.model.is_empty()
        && let Some(report) = &state.report
    {
        state.model = report_model_label(report);
    }
    if !done && let Some(summary) = demo {
        if let Some(error) = summary["error"].as_str() {
            state.error = Some(error.to_owned());
        }
        let complete = summary["complete"].as_bool().unwrap_or(false);
        state.phase = if complete {
            "Complete · external session finished".into()
        } else {
            "Incomplete · evidence retained".into()
        };
        state.done = true;
    }
}

pub fn run(args: crate::cli::ParallaxArgs) -> Result<(), Failure> {
    if let Some(path) = args.report {
        let (output, report) = load_report(&path)?;
        let responses = load_responses(&output);
        if !args.plain && std::io::stderr().is_terminal() {
            let error = report["error"].as_str().map(ToOwned::to_owned);
            let view = Arc::new(Mutex::new(View {
                phase: if error.is_some() {
                    "Saved report · capture incomplete".into()
                } else {
                    "Saved report · offline inspection".into()
                },
                model: report_model_label(&report),
                gpu: "Offline · no GPU".into(),
                output,
                responses,
                report: Some(report),
                error,
                done: true,
            }));
            tui::run(&view)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        return Ok(());
    }

    let watch = args.watch.ok_or("either --report or --watch is required")?;
    if !watch.is_dir() {
        return Err(format!("watch path is not a directory: {}", watch.display()).into());
    }
    // SAFETY: the handler only stores to the same atomic stop flag as normal serving.
    unsafe {
        libc::signal(libc::SIGINT, crate::stop as *const () as usize);
        libc::signal(libc::SIGTERM, crate::stop as *const () as usize);
    }
    let view = Arc::new(Mutex::new(View {
        phase: "Watching external Parallax session".into(),
        gpu: "External viewer · no GPU".into(),
        output: watch,
        ..View::default()
    }));
    if !args.plain && std::io::stderr().is_terminal() {
        tui::run(&view)
    } else {
        tui::plain(&view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "flare-parallax-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn load_report_reads_file_and_capture_directory_and_rejects_invalid_json() {
        let root = temp_dir("load-report");
        let capture = root.join("capture");
        fs::create_dir_all(&capture).unwrap();
        let report = json!({
            "routing_frames": 12,
            "model": {"identity": "synthetic;package=abc"},
            "contenders": {"recent/static": {"prediction_records": 10}}
        });
        let comparison = capture.join("comparison.json");
        fs::write(&comparison, serde_json::to_vec(&report).unwrap()).unwrap();

        let (resolved_from_file, loaded_from_file) = load_report(&comparison).unwrap();
        assert_eq!(resolved_from_file, root);
        assert_eq!(loaded_from_file["routing_frames"], json!(12));

        let (resolved_from_dir, loaded_from_dir) = load_report(&root).unwrap();
        assert_eq!(resolved_from_dir, root);
        assert_eq!(loaded_from_dir["routing_frames"], json!(12));

        let invalid = root.join("invalid.json");
        fs::write(&invalid, b"{}").unwrap();
        assert!(load_report(&invalid).is_err());

        let replay_output = root.join("offline-replay.json");
        fs::write(
            &replay_output,
            serde_json::to_vec(&json!({
                "workload": "SYNTHETIC-block-locality-test-only",
                "contenders": {
                    "ewma-economic": {
                        "summary": {
                            "routing_frames": 64,
                            "prediction_records": 63,
                            "hbm_hit_rate": 0.75
                        }
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let (_, loaded_replay) = load_report(&replay_output).unwrap();
        assert_eq!(loaded_replay["routing_frames"], json!(64));
        assert_eq!(
            loaded_replay["contenders"]["ewma-economic"]["prediction_records"],
            json!(63)
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn load_responses_extracts_generated_texts_from_session_directory() {
        let root = temp_dir("responses");
        fs::write(
            root.join("responses.json"),
            serde_json::to_vec(&json!([
                {"choices": [{"text": "first answer"}]},
                {"choices": [{"message": {"content": "second answer"}}]}
            ]))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            load_responses(&root),
            vec!["first answer".to_string(), "second answer".to_string()]
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sync_watch_state_tracks_responses_and_demo_completion() {
        let root = temp_dir("watch-sync");
        fs::write(
            root.join("responses.json"),
            serde_json::to_vec(&json!([{"choices": [{"text": "hello"}]}])).unwrap(),
        )
        .unwrap();
        let view = Arc::new(Mutex::new(View {
            phase: "Watching".into(),
            output: root.clone(),
            ..View::default()
        }));

        sync_watch_state(&view);
        assert_eq!(view.lock().unwrap().responses, vec!["hello".to_string()]);
        assert!(!view.lock().unwrap().done);

        fs::write(
            root.join("demo.json"),
            serde_json::to_vec(&json!({"complete": false, "error": "collector timed out"})).unwrap(),
        )
        .unwrap();
        sync_watch_state(&view);
        let state = view.lock().unwrap();
        assert!(state.done);
        assert_eq!(state.error.as_deref(), Some("collector timed out"));
        drop(state);

        let _ = fs::remove_dir_all(&root);
    }
}
