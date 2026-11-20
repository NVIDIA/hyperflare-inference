# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Annotate the bundled GLM template, or fetch the pinned canonical source for the calibrated checkpoint."""

import hashlib
from pathlib import Path

from ignition.preparation.frontend import annotated_template_asset, downloaded_template_asset

ASSETS = Path(__file__).with_name("assets")
# SHA-256 of the chat template bundled with local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c4.
CALIBRATED_TEMPLATE_SHA256 = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"


def chat_template_asset(snapshot, *, cache_dir=None):
    """Return the annotated template for a GLM checkpoint snapshot.

    The calibrated checkpoint's bundled template is replaced by the canonical
    template it was derived from. Any other bytes must match the NVIDIA
    checkpoint's template, or annotation raises.
    """
    source = (Path(snapshot) / "chat_template.jinja").read_bytes()
    if hashlib.sha256(source).hexdigest() == CALIBRATED_TEMPLATE_SHA256:
        return downloaded_template_asset(ASSETS / "chat_template.zai.patch.json", cache_dir=cache_dir)
    return annotated_template_asset(source, ASSETS / "chat_template.patch.json")
