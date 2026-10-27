# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest

import dynamo.vllm.multimodal_utils.media_config as media_config
from dynamo.llm import MediaDecoder

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_vllm_runtime_includes_working_frontend_video_decoder(tmp_path: Path):
    decoder = MediaDecoder()
    assert hasattr(decoder, "enable_video")
    decoder.enable_video({"num_frames": 2, "strict": True})

    # MediaDecoder is configuration-only. Importing it proves the Rust extension's
    # external libav* dependencies resolve; decode with the same in-tree FFmpeg to
    # prove its allowlist includes the VP9 codec used by frontend decoding.
    fixture = Path(__file__).resolve().parents[6] / "lib/llm/tests/data/media/2p_10.mp4"
    output = tmp_path / "decoded.webm"
    ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE") or "ffmpeg"
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(fixture),
            "-frames:v",
            "2",
            "-an",
            "-c:v",
            "libvpx-vp9",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.stat().st_size > 0


class FakeDecoder:
    def enable_image(self, options):
        self.image_options = options

    def enable_video(self, options):
        self.video_options = options


class FakePreprocessor:
    def enable_video(self, model_type, config_json):
        self.video_config = (model_type, json.loads(config_json))


class FakeFetcher:
    def timeout_ms(self, value):
        self.timeout = value

    def allow_direct_ip(self, value):
        self.direct_ip = value

    def allow_direct_port(self, value):
        self.direct_port = value


def test_video_processor_selection_is_delegated_to_registry(tmp_path, monkeypatch):
    (tmp_path / "preprocessor_config.json").write_text(
        '{"patch_size": 16}', encoding="utf-8"
    )
    monkeypatch.setattr(media_config, "MediaDecoder", FakeDecoder)
    monkeypatch.setattr(media_config, "MediaPreprocessor", FakePreprocessor)
    monkeypatch.setattr(media_config, "MediaFetcher", FakeFetcher)

    decoder, processor, fetcher = media_config.create_frontend_media_config(
        True,
        video_preprocessing=True,
        model=str(tmp_path),
        model_type="qwen3_vl",
    )

    assert decoder.video_options["fps"] == 2.0
    assert processor.video_config == ("qwen3_vl", {"patch_size": 16})
    assert fetcher.timeout == 30000
