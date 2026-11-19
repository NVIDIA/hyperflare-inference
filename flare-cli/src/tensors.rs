/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Execute fixed tensor requests without loading a Python interpreter.

use flare_helios::runtime_core::{
    AuthorizedCore, NativeResources,
    tensor_contract::PreparedTensorContract,
    tensors::{TensorRequest, TensorRunner},
};
use serde_json::json;
use std::{fs, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn read_inputs(path: &Path, contract: &PreparedTensorContract) -> Result<Vec<Vec<u8>>> {
    let request: TensorRequest = serde_json::from_slice(&fs::read(path)?)?;
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    request
        .bind(contract)?
        .into_iter()
        .zip(contract.inputs())
        .map(|(input, view)| {
            let file = root.join(&input.file);
            let expected = view.nbytes()?;
            if fs::metadata(&file)?.len() != expected as u64 {
                return Err(format!(
                    "input {:?} file length differs from signed contract",
                    input.name
                )
                .into());
            }
            let bytes = fs::read(file)?;
            if bytes.len() != expected {
                return Err(
                    format!("input {:?} file changed size while reading", input.name).into(),
                );
            }
            Ok(bytes)
        })
        .collect()
}

pub fn run(args: &crate::cli::Run, authorized: AuthorizedCore) -> Result<()> {
    let path = args
        .tensor_request
        .as_ref()
        .ok_or("missing tensor request")?;
    let destination = args
        .tensor_output
        .as_ref()
        .ok_or("missing tensor output directory")?;
    let contract = authorized.tensor_contract()?;
    let inputs = read_inputs(path, &contract)?;
    if destination.exists() {
        return Err("tensor output directory already exists".into());
    }
    let digest = authorized
        .envelope_digest()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let loading = Instant::now();
    // SAFETY: The CLI authenticated and accepted this core before loading. The
    // runner owns its context and resources through confirmed stream drain.
    let owner = unsafe { NativeResources::load(authorized, 0)? };
    let mut runner = TensorRunner::new(owner)?;
    let load_wall_ms = loading.elapsed().as_secs_f64() * 1000.0;
    let correctness_timing = runner.run(&inputs)?;
    let downloading = Instant::now();
    let outputs = runner.outputs()?;
    let output_download_wall_ms = downloading.elapsed().as_secs_f64() * 1000.0;
    for _ in 0..args.tensor_warmup {
        runner.run(&inputs)?;
    }
    let timings = (0..args.tensor_repetitions)
        .map(|_| runner.run(&inputs))
        .collect::<flare_helios::Result<Vec<_>>>()?;
    let repeated_outputs = runner.outputs()?;
    #[cfg(target_os = "linux")]
    let process_libraries = {
        let maps = fs::read_to_string("/proc/self/maps")?;
        let python_loaded = maps.contains("libpython");
        let torch_loaded = maps.contains("libtorch");
        if python_loaded || torch_loaded {
            return Err("native tensor execution process loaded Python or PyTorch".into());
        }
        json!({"checked":true,"python_loaded":python_loaded,"torch_loaded":torch_loaded})
    };
    #[cfg(not(target_os = "linux"))]
    let process_libraries = json!({"checked":false});
    runner.close()?;
    fs::create_dir_all(destination)?;
    let mut descriptors = Vec::new();
    for ((view, bytes), repeat) in contract.outputs().iter().zip(outputs).zip(repeated_outputs) {
        let file = format!("{}.bin", view.name);
        let repeat_file = format!("{}.repeat.bin", view.name);
        fs::write(destination.join(&file), bytes)?;
        fs::write(destination.join(&repeat_file), repeat)?;
        descriptors.push(json!({"name":view.name,"dtype":view.dtype,"shape":view.shape,"file":file,"repeat_file":repeat_file}));
    }
    let report = json!({
        "executor":"flare-helios-exported-cuda-graphs",
        "core_envelope_sha256":digest,
        "graphs":contract.graphs(),
        "outputs":descriptors,
        "load_wall_ms":load_wall_ms,
        "output_download_wall_ms":output_download_wall_ms,
        "correctness_timing":correctness_timing,
        "warmup_repetitions":args.tensor_warmup,
        "timing":{
            "gpu_ms":timings.iter().map(|sample| sample.replay_gpu_ns as f64 / 1e6).collect::<Vec<_>>(),
            "wall_ms":timings.iter().map(|sample| sample.replay_wall_ns as f64 / 1e6).collect::<Vec<_>>(),
            "input_upload_wall_ms":timings.iter().map(|sample| sample.input_upload_wall_ns as f64 / 1e6).collect::<Vec<_>>(),
            "graph_gpu_ms":timings.iter().map(|sample| sample.graph_gpu_ns.iter().map(|ns| *ns as f64 / 1e6).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "scope":"all declared graphs, with device-resident inputs; excludes input upload, output download, file IO and core loading"
        },
        "process_libraries":process_libraries,
        "scope":"single fixed tensor request; no HTTP frontend or concurrent request scheduling"
    });
    fs::write(
        destination.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "Wrote {} tensors and {} native replay measurements to {}",
        contract.outputs().len(),
        timings.len(),
        destination.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flare_helios::runtime_core::wire;

    fn contract() -> PreparedTensorContract {
        let binding = |name: &str, offset| wire::TensorBinding {
            name: name.into(),
            reference: Some(wire::Reference {
                allocation: "storage".into(),
                offset,
            }),
            dtype: wire::TensorDtype::I32 as i32,
            shape: vec![2],
        };
        PreparedTensorContract::parse(&wire::TensorContract {
            version: 1,
            graphs: vec!["run".into()],
            inputs: vec![binding("first", 0), binding("second", 8)],
            outputs: vec![binding("result", 16)],
        })
        .unwrap()
    }

    #[test]
    fn reads_named_files_relative_to_request_in_signed_order() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("a.bin"), [1u8; 8]).unwrap();
        fs::write(root.path().join("b.bin"), [2u8; 8]).unwrap();
        let request = root.path().join("request.json");
        fs::write(
            &request,
            r#"{"inputs":[{"name":"second","file":"b.bin"},{"name":"first","file":"a.bin"}]}"#,
        )
        .unwrap();
        assert_eq!(
            read_inputs(&request, &contract()).unwrap(),
            vec![vec![1; 8], vec![2; 8]]
        );
        for nbytes in [7, 9] {
            fs::write(root.path().join("a.bin"), vec![1u8; nbytes]).unwrap();
            let error = read_inputs(&request, &contract()).unwrap_err().to_string();
            assert!(error.contains("file length differs from signed contract"));
        }
    }

    #[test]
    fn rejects_request_metadata_before_reading_input_files() {
        let root = tempfile::tempdir().unwrap();
        let request = root.path().join("request.json");
        for data in [
            r#"{"graphs":["run"],"inputs":[]}"#,
            r#"{"inputs":[{"name":"first","file":"absent","allocation":"weights"}]}"#,
            r#"{"inputs":[{"name":"first","file":"absent"},{"name":"unknown","file":"absent"}]}"#,
        ] {
            fs::write(&request, data).unwrap();
            let error = read_inputs(&request, &contract()).unwrap_err().to_string();
            assert!(
                error.contains("unknown field") || error.contains("missing signed input"),
                "{error}"
            );
        }
    }
}
