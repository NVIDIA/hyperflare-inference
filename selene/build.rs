/*
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
    let proto = "proto/selene.proto";
    println!("cargo:rerun-if-changed={proto}");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&[proto], &["proto"])
        .expect("compile the selene pool protocol");
}
