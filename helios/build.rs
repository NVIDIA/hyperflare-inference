/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

fn main() {
    println!("cargo:rerun-if-changed=native/kv_codec");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-ffp-contract=off")
        .flag_if_supported("-fno-fast-math")
        .file("native/kv_codec/cpu.cpp")
        .compile("flare_kv_cpu");
    println!("cargo:rustc-check-cfg=cfg(kv_codec_cuda)");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NVCC");
    let cuda = std::env::var_os("CUDA_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/usr/local/cuda".into());
    let compiler = std::env::var_os("NVCC")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let bundled = cuda.join("bin/nvcc");
            if bundled.is_file() {
                bundled
            } else {
                "nvcc".into()
            }
        });
    let targets = std::process::Command::new(&compiler)
        .arg("--list-gpu-code")
        .output()
        .ok()
        .filter(|p| p.status.success())
        .map(|p| String::from_utf8_lossy(&p.stdout).into_owned())
        .unwrap_or_default();
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && ["sm_120", "sm_121"]
            .iter()
            .all(|sm| targets.split_whitespace().any(|s| s == *sm))
    {
        let object =
            std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("kv_gpu.o");
        let status = std::process::Command::new(compiler)
            .args([
                "-std=c++17",
                "-O3",
                "--fmad=false",
                "-gencode=arch=compute_120,code=sm_120",
                "-gencode=arch=compute_121,code=sm_121",
                "-Xcompiler=-fPIC",
                "-c",
                "native/kv_codec/gpu.cu",
                "-o",
            ])
            .arg(&object)
            .status()
            .expect("start CUDA compiler for KV compression");
        assert!(status.success(), "compile KV codec for SM120 and SM121");
        cc::Build::new()
            .cpp(true)
            .object(object)
            .compile("flare_kv_gpu");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-cfg=kv_codec_cuda");
        println!("cargo:rustc-link-search=native={}/lib64", cuda.display());
        println!(
            "cargo:rustc-link-search=native={}/lib64/stubs",
            cuda.display()
        );
        println!("cargo:rustc-link-lib=cuda");
    }
    #[cfg(feature = "runtime-core")]
    {
        println!("cargo:rerun-if-changed=proto/runtime_core.proto");
        println!("cargo:rerun-if-changed=proto/core.proto");
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        // The execution schema is the single definition of
        // contracts, graph selection, frontend settings, and engine placement.
        let mut runtime = prost_build::Config::new();
        runtime
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .file_descriptor_set_path(out.join("runtime_core_descriptor.bin"))
            .field_attribute(".flare.runtime.RebuildSpeculation.vocabulary_parallel_sampling", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildSpeculation.mtp_only", "#[serde(default)]")
            .field_attribute(".flare.runtime.RebuildManifest.online_mxfp8_scope", "#[serde(default)]")
            .field_attribute(
                ".flare.runtime.ExecutionGraph.packed_token_capacity",
                "#[serde(default)]",
            )
            .compile_protos(&["proto/runtime_core.proto"], &["proto"])
            .expect("compile runtime core protobuf schema");
        // The payload schema imports the execution schema and refers to the generated types
        // instead of restating them.
        let mut core = prost_build::Config::new();
        core
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .extern_path(".flare.runtime", "crate::runtime_core::wire")
            .compile_protos(&["proto/core.proto"], &["proto"])
            .expect("compile payload core protobuf schema");
    }
    #[cfg(feature = "roce")]
    {
        println!("cargo:rerun-if-changed=native/roce_proxy.c");
        cc::Build::new()
            .file("native/roce_proxy.c")
            .std("gnu11")
            .compile("flare_roce_proxy");
        println!("cargo:rustc-link-lib=ibverbs");
        println!("cargo:rustc-link-lib=pthread");
    }
}
