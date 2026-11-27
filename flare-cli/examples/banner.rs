/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use flare_helios::startup::StartupProgress;

fn main() {
    let mut progress = StartupProgress::stderr();
    progress.finish_animation();
}
