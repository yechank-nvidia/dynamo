# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

from dynamo.common.multimodal.processed_media import ProcessedField, ProcessedMedia
from dynamo.vllm.multimodal_utils.processed_media_adapter import _combine_fields
from dynamo.vllm.multimodal_utils.request_processor import (
    VllmMultimodalRequestProcessor,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _media(t: int) -> ProcessedMedia:
    patches = t * 4
    return ProcessedMedia(
        modality="video",
        fields={
            "pixel_values_videos": ProcessedField(
                np.zeros((patches, 6), dtype=np.float32),
                {"kind": "flat", "sizes_key": "patches_per_video"},
                False,
                True,
            ),
            "patches_per_video": ProcessedField(
                np.asarray([patches]), {"kind": "batched"}, True, False
            ),
            "video_grid_thw": ProcessedField(
                np.asarray([[t, 2, 2]]), {"kind": "batched"}, True, True
            ),
            "timestamps": ProcessedField(
                np.arange(t, dtype=np.float64)[None, :],
                {"kind": "batched"},
                True,
                True,
            ),
        },
        feature_token_counts=[t],
        original_sizes=[(2, 2)],
        content_hashes=[str(t)],
    )


def test_combines_layouts_without_model_specific_adapter_logic():
    inputs, configs = _combine_fields([_media(1), _media(2)])

    assert inputs["pixel_values_videos"].shape == (12, 6)
    assert inputs["video_grid_thw"].shape == (2, 3)
    assert [value.shape for value in inputs["timestamps"]] == [(1,), (2,)]
    assert "patches_per_video" not in inputs
    assert set(configs) == set(inputs)
    assert torch.equal(inputs["video_grid_thw"][1], torch.tensor([2, 2, 2]))


def test_single_item_reuses_numpy_storage_without_cat_or_stack():
    media = _media(2)

    inputs, _ = _combine_fields([media])

    pixel_values = media.fields["pixel_values_videos"].value
    grid = media.fields["video_grid_thw"].value
    timestamps = media.fields["timestamps"].value
    assert (
        inputs["pixel_values_videos"].data_ptr()
        == torch.from_numpy(pixel_values).data_ptr()
    )
    assert inputs["video_grid_thw"].data_ptr() == torch.from_numpy(grid).data_ptr()
    assert inputs["timestamps"].data_ptr() == torch.from_numpy(timestamps).data_ptr()


@pytest.mark.asyncio
async def test_processed_video_preserves_client_cache_identity():
    media_items = [_media(1), _media(2)]
    engine_client = MagicMock()
    engine_client.input_processor.get_tokenizer.return_value.encode.return_value = [90]
    engine_client.vllm_config.model_config.hf_config = SimpleNamespace(
        model_type="qwen3_vl",
        vision_config=SimpleNamespace(spatial_merge_size=2),
        vision_start_token_id=1,
        video_token_id=2,
        vision_end_token_id=3,
    )
    processor = SimpleNamespace(
        video_loader=SimpleNamespace(
            load_processed_media_batch=AsyncMock(return_value=media_items)
        ),
        engine_client=engine_client,
        use_unified_vision_chunk=False,
    )
    identities = []
    content_hashes = ["1".ljust(64, "0"), "2".ljust(64, "0")]
    for uuids, expected in [
        (["client-cache-a", "other-video"], ["client-cache-a", "other-video"]),
        (["client-cache-b", "other-video"], ["client-cache-b", "other-video"]),
        (None, content_hashes),
        ([None, None], content_hashes),
        (["client-cache-a", None], ["client-cache-a", content_hashes[1]]),
    ]:
        request = {
            "token_ids": [9, 1, 2, 3, 1, 2, 3, 8],
            "multi_modal_data": {"video_url": [{"Preprocessed": {}}] * 2},
        }
        if uuids is not None:
            request["multi_modal_uuids"] = {"video_url": uuids}
        result = await VllmMultimodalRequestProcessor.try_processed_media(
            processor, request, None
        )
        assert result["mm_hashes"] == {"video": expected}
        engine_client.input_processor.inject_into_mm_cache.assert_called_with(
            result["mm_hashes"], result["mm_kwargs"]
        )
        identities.append(result["mm_hashes"])

    assert identities[0] != identities[1]

    request["multi_modal_uuids"] = {"video_url": ["unaligned"]}
    engine_client.input_processor.inject_into_mm_cache.reset_mock()
    with pytest.raises(ValueError, match="counts must match"):
        await VllmMultimodalRequestProcessor.try_processed_media(
            processor, request, None
        )
    engine_client.input_processor.inject_into_mm_cache.assert_not_called()
