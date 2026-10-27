// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use dynamo_memory::nixl::{self, AgentConfig, NixlAgent, NixlDescriptor, RegisteredView};
use dynamo_memory::{MemoryDescriptor, StorageKind, SystemStorage};
use dynamo_runtime::config::{
    env_config::parse_or_default, environment_names::llm::DYN_MM_NIXL_PROGRESS_DELAY_US,
};
use flate2::{Compression, write::ZlibEncoder};
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Arc, LazyLock};

use super::decoders::DecodedMediaMetadata;

#[cfg(feature = "media-ffmpeg")]
const DEFAULT_MAX_PROCESSED_MEDIA_BYTES: usize = 256 * 1024 * 1024;

type SharedNixlStorage = Arc<nixl::NixlRegistered<Arc<dyn nixl::NixlMemory + Send + Sync>>>;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    UINT8,
    FLOAT32,
    INT64,
    FLOAT64,
}

// Common tensor metadata shared between decoded and RDMA descriptors
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MediaTensorInfo {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DataType,
    pub(crate) metadata: Option<DecodedMediaMetadata>,
}

// Decoded media data (image RGB, video frames pixels, ...)
#[derive(Debug)]
pub struct DecodedMediaData {
    pub(crate) data: SystemStorage,
    pub(crate) tensor_info: MediaTensorInfo,
    pub(crate) content_hash: Option<u64>,
}

// Decoded media data NIXL descriptor (sent to the next step in the pipeline / NATS)

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RdmaMediaDataDescriptor {
    // b64 agent metadata
    pub(crate) nixl_metadata: String,
    // tensor descriptor
    pub(crate) nixl_descriptor: NixlDescriptor,

    #[serde(flatten)]
    pub(crate) tensor_info: MediaTensorInfo,

    /// Canonical xxh3-64 key for decoded media. Image identity covers shape,
    /// dtype, and RGB bytes; video identity additionally covers decoded
    /// metadata that affects the model-visible token sequence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) content_hash: Option<String>,

    // reference to the actual data, kept alive while the rdma descriptor is alive
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<Arc<nixl::NixlRegistered<SystemStorage>>>,
}

#[derive(Debug)]
struct OwnedF32Storage {
    data: Vec<f32>,
}

impl MemoryDescriptor for OwnedF32Storage {
    fn addr(&self) -> usize {
        self.data.as_ptr() as usize
    }

    fn size(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::System
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl nixl::NixlCompatible for OwnedF32Storage {
    fn nixl_params(&self) -> (*const u8, usize, nixl::MemType, u64) {
        (
            self.data.as_ptr().cast::<u8>(),
            self.size(),
            nixl::MemType::Dram,
            0,
        )
    }
}

/// Model-ready named fields. The wire contract is independent of model and backend.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProcessedMediaDataDescriptor {
    pub modality: dynamo_multimodal::types::Modality,
    pub fields: BTreeMap<String, ProcessedFieldDescriptor>,
    pub feature_token_counts: Vec<usize>,
    pub original_sizes: Vec<(u32, u32)>,
    pub content_hashes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProcessedFieldDescriptor {
    pub layout: dynamo_multimodal::types::FieldLayout,
    pub keep_on_host: bool,
    pub forward: bool,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    #[serde(flatten)]
    pub storage: ProcessedFieldStorageDescriptor,
    #[serde(skip, default)]
    #[allow(dead_code)]
    pub(crate) source_storage: Option<SharedNixlStorage>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum ProcessedFieldStorageDescriptor {
    Rdma {
        nixl_metadata: String,
        nixl_descriptor: NixlDescriptor,
    },
    Inline {
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
}

impl RdmaMediaDataDescriptor {
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    fn local_payload(&self) -> Option<&[u8]> {
        use dynamo_memory::actions::Slice;
        let registered = self.source_storage.as_ref()?;
        let storage = registered.storage();
        // SAFETY: the descriptor keeps the registered SystemStorage alive and
        // request construction does not mutate it while this borrow exists.
        unsafe { storage.as_slice().ok() }
    }

    /// Canonical cache/routing key serialized on the descriptor.
    pub(crate) fn content_hash_key(&self) -> Option<&str> {
        self.content_hash.as_deref()
    }

    /// Numeric form used by MM-aware KV routing.
    #[cfg(feature = "mm-routing")]
    pub(crate) fn content_hash(&self) -> Option<u64> {
        self.content_hash_key()
            .and_then(|key| u64::from_str_radix(key, 16).ok())
    }

    /// Canonical identity for one frontend-decoded video:
    /// `(modality, shape, dtype, decoded metadata, sampled RGB bytes)`.
    /// Metadata is serialized from the typed object sent to the worker, so new
    /// model-visible fields automatically participate in the identity.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_content_hash(&self) -> Result<u64> {
        self.video_metadata()?;
        self.content_hash()
            .context("decoded video content hash is missing")
    }

    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_metadata(&self) -> Result<&super::decoders::VideoMetadata> {
        match self.tensor_info.metadata.as_ref() {
            Some(DecodedMediaMetadata::Video(metadata)) => Ok(metadata),
            Some(_) => anyhow::bail!("decoded media metadata is not video metadata"),
            None => anyhow::bail!("decoded video metadata is missing"),
        }
    }

    /// Validate the contiguous `[T, H, W, 3]` payload and return its video
    /// dimensions without copying or preprocessing pixels.
    #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
    pub(crate) fn video_dimensions(&self) -> Result<(usize, u32, u32)> {
        let bytes = self
            .local_payload()
            .ok_or_else(|| anyhow::anyhow!("decoded video has no local payload"))?;
        video_dimensions_from_parts(&self.tensor_info, bytes)
    }
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn hash_video_content(tensor_info: &MediaTensorInfo, bytes: &[u8]) -> Result<u64> {
    use xxhash_rust::xxh3::Xxh3;

    anyhow::ensure!(!bytes.is_empty(), "decoded video payload is empty");
    let metadata = tensor_info
        .metadata
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("decoded video metadata is missing"))?;
    let DecodedMediaMetadata::Video(metadata) = metadata else {
        anyhow::bail!("decoded media metadata is not video metadata");
    };
    let metadata_bytes = serde_json::to_vec(metadata)
        .map_err(|error| anyhow::anyhow!("failed to serialize video metadata: {error}"))?;

    let mut hasher = Xxh3::new();
    update_len_prefixed(&mut hasher, b"video");
    hasher.update(&(tensor_info.shape.len() as u64).to_le_bytes());
    for &dim in &tensor_info.shape {
        hasher.update(&(dim as u64).to_le_bytes());
    }
    let dtype_byte = match tensor_info.dtype {
        DataType::UINT8 => 0,
        DataType::FLOAT32 => 1,
        DataType::INT64 => 2,
        DataType::FLOAT64 => 3,
    };
    hasher.update(&[dtype_byte]);
    update_len_prefixed(&mut hasher, &metadata_bytes);
    update_len_prefixed(&mut hasher, bytes);
    Ok(hasher.digest())
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn video_dimensions_from_parts(
    tensor_info: &MediaTensorInfo,
    bytes: &[u8],
) -> Result<(usize, u32, u32)> {
    let [frames, height, width, channels] = tensor_info.shape.as_slice() else {
        anyhow::bail!(
            "decoded video shape must be [T, H, W, C], got {:?}",
            tensor_info.shape
        );
    };
    anyhow::ensure!(*frames > 0, "decoded video has no frames");
    anyhow::ensure!(
        *height > 0 && *width > 0,
        "decoded video dimensions are zero"
    );
    anyhow::ensure!(*channels == 3, "decoded video must contain RGB frames");
    anyhow::ensure!(
        tensor_info.dtype == DataType::UINT8,
        "decoded video dtype must be uint8"
    );

    let frame_len = height
        .checked_mul(*width)
        .and_then(|value| value.checked_mul(*channels))
        .ok_or_else(|| anyhow::anyhow!("decoded video frame size overflow"))?;
    let expected_len = frames
        .checked_mul(frame_len)
        .ok_or_else(|| anyhow::anyhow!("decoded video payload size overflow"))?;
    anyhow::ensure!(
        bytes.len() == expected_len,
        "decoded video payload has {} bytes, expected {}",
        bytes.len(),
        expected_len
    );
    let width = u32::try_from(*width).context("decoded video width exceeds u32")?;
    let height = u32::try_from(*height).context("decoded video height exceeds u32")?;

    Ok((*frames, width, height))
}

#[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
fn update_len_prefixed(hasher: &mut xxhash_rust::xxh3::Xxh3, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn canonical_content_hash(shape: &[usize], dtype: DataType, bytes: &[u8]) -> u64 {
    use xxhash_rust::xxh3::Xxh3;

    let mut hasher = Xxh3::new();
    // Rank and dimensions are fixed-width so distinct shapes cannot alias
    // after concatenation.
    hasher.update(&(shape.len() as u64).to_le_bytes());
    for &dim in shape {
        hasher.update(&(dim as u64).to_le_bytes());
    }
    // Widen this discriminant if DataType gains another variant.
    let dtype_byte: u8 = match dtype {
        DataType::UINT8 => 0,
        DataType::FLOAT32 => 1,
        DataType::INT64 => 2,
        DataType::FLOAT64 => 3,
    };
    hasher.update(&[dtype_byte]);
    hasher.update(bytes);
    hasher.digest()
}

fn content_hash_for_storage(
    tensor_info: &MediaTensorInfo,
    storage: &SystemStorage,
    _hash_video: bool,
) -> Option<u64> {
    use dynamo_memory::{MemoryDescriptor, actions::Slice};

    if storage.size() == 0 {
        return None;
    }

    // SAFETY: storage owns this stable buffer and is only read while the
    // descriptor is being built, before NIXL can access it concurrently.
    let bytes = unsafe { storage.as_slice().ok()? };
    match tensor_info.metadata.as_ref() {
        Some(DecodedMediaMetadata::Image(_)) => Some(canonical_content_hash(
            &tensor_info.shape,
            tensor_info.dtype,
            bytes,
        )),
        #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
        Some(DecodedMediaMetadata::Video(_)) if _hash_video => {
            match hash_video_content(tensor_info, bytes) {
                Ok(hash) => Some(hash),
                Err(error) => {
                    tracing::debug!(%error, "Skipping exact routing hash for decoded video");
                    None
                }
            }
        }
        #[cfg(all(feature = "mm-routing", feature = "media-ffmpeg"))]
        Some(DecodedMediaMetadata::Video(_)) => None,
        #[cfg(all(not(feature = "mm-routing"), feature = "media-ffmpeg"))]
        Some(DecodedMediaMetadata::Video(_)) => None,
        None => None,
    }
}

impl DecodedMediaData {
    /// Precompute the canonical media hash while still running on the decode
    /// thread. Images are always hashed; videos are hashed only when the
    /// request is eligible for exact MM routing.
    pub(crate) fn compute_content_hash(&mut self, hash_video: bool) {
        self.content_hash = content_hash_for_storage(&self.tensor_info, &self.data, hash_video);
    }

    pub fn into_rdma_descriptor(self, nixl_agent: &NixlAgent) -> Result<RdmaMediaDataDescriptor> {
        let source_storage = self.data;
        let content_hash = self.content_hash.map(|hash| format!("{hash:016x}"));
        let registered = nixl::register_with_nixl(source_storage, nixl_agent, None)
            .map_err(|_| anyhow::anyhow!("Failed to register storage with NIXL"))?;

        let nixl_descriptor = registered.descriptor();
        let nixl_metadata = get_nixl_metadata(nixl_agent, registered.storage())?;

        Ok(RdmaMediaDataDescriptor {
            nixl_metadata,
            nixl_descriptor,
            tensor_info: self.tensor_info,
            content_hash,
            // Keep registered storage alive
            source_storage: Some(Arc::new(registered)),
        })
    }

    #[cfg(feature = "media-ffmpeg")]
    pub(crate) fn preprocess_video(
        self,
        processor: &dyn dynamo_multimodal::registry::VideoProcessor,
    ) -> Result<dynamo_multimodal::processed::ProcessedMedia> {
        anyhow::ensure!(
            self.tensor_info.dtype == DataType::UINT8,
            "decoded video must be UINT8"
        );
        anyhow::ensure!(
            self.tensor_info.shape.len() == 4 && self.tensor_info.shape[3] == 3,
            "decoded video must have shape [T,H,W,3], got {:?}",
            self.tensor_info.shape
        );
        let timing = match self.tensor_info.metadata {
            Some(DecodedMediaMetadata::Video(metadata)) => dynamo_multimodal::VideoTiming {
                source_fps: metadata.source_fps,
                source_duration: metadata.source_duration,
                sampled_timestamps: metadata.sampled_timestamps,
            },
            _ => anyhow::bail!("decoded video timing metadata is required for preprocessing"),
        };
        let [frame_count, height, width, _] = self.tensor_info.shape.as_slice() else {
            unreachable!("shape validated above")
        };
        let frame_bytes = (*height)
            .checked_mul(*width)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| anyhow::anyhow!("decoded video frame size overflow"))?;
        use dynamo_memory::actions::Slice;
        let bytes = unsafe {
            self.data
                .as_slice()
                .map_err(|_| anyhow::anyhow!("decoded video storage is not host-readable"))?
        };
        anyhow::ensure!(
            bytes.len() == *frame_count * frame_bytes,
            "decoded video storage size mismatch"
        );
        let frames = bytes
            .chunks_exact(frame_bytes)
            .map(|data| dynamo_multimodal::types::RgbFrameRef {
                width: *width as u32,
                height: *height as u32,
                data,
            })
            .collect::<Vec<_>>();
        processor.process(&frames, &timing).map_err(Into::into)
    }
}

impl ProcessedMediaDataDescriptor {
    #[cfg(feature = "media-ffmpeg")]
    pub(crate) fn from_processed(
        processed: dynamo_multimodal::processed::ProcessedMedia,
        nixl_agent: &NixlAgent,
    ) -> Result<Self> {
        use dynamo_multimodal::processed::ProcessedValue;

        processed.validate()?;
        anyhow::ensure!(
            processed.feature_token_counts.len() == 1,
            "a processed-media descriptor must represent exactly one media item"
        );
        let total_bytes = processed
            .fields
            .values()
            .try_fold(0usize, |total, field| {
                let (count, element_size) = match &field.value {
                    ProcessedValue::F32Tensor { data, .. } => {
                        (data.len(), std::mem::size_of::<f32>())
                    }
                    ProcessedValue::I64Tensor { data, .. } => {
                        (data.len(), std::mem::size_of::<i64>())
                    }
                    ProcessedValue::F64Tensor { data, .. } => {
                        (data.len(), std::mem::size_of::<f64>())
                    }
                };
                count
                    .checked_mul(element_size)
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| anyhow::anyhow!("processed media byte size overflow"))?;
        anyhow::ensure!(
            total_bytes <= DEFAULT_MAX_PROCESSED_MEDIA_BYTES,
            "processed media is {total_bytes} bytes, exceeding the 256 MiB limit"
        );
        let mut hasher = blake3::Hasher::new();
        let mut fields = BTreeMap::new();
        for (name, field) in processed.fields {
            hasher.update(&(name.len() as u64).to_le_bytes());
            hasher.update(name.as_bytes());
            hash_layout(&mut hasher, &field.layout);
            hasher.update(&[u8::from(field.keep_on_host), u8::from(field.forward)]);
            let (dtype, shape, storage, source_storage) = match field.value {
                ProcessedValue::F32Tensor { data, shape } => {
                    hash_shape(&mut hasher, 0, &shape);
                    hasher.update(bytemuck::cast_slice(&data));
                    let (storage, source_storage) = f32_into_rdma_storage(data, nixl_agent)?;
                    (DataType::FLOAT32, shape, storage, Some(source_storage))
                }
                ProcessedValue::I64Tensor { data, shape } => {
                    hash_shape(&mut hasher, 1, &shape);
                    hasher.update(bytemuck::cast_slice(&data));
                    (
                        DataType::INT64,
                        shape,
                        ProcessedFieldStorageDescriptor::Inline {
                            data: encode_i64_le(data),
                        },
                        None,
                    )
                }
                ProcessedValue::F64Tensor { data, shape } => {
                    hash_shape(&mut hasher, 2, &shape);
                    hasher.update(bytemuck::cast_slice(&data));
                    (
                        DataType::FLOAT64,
                        shape,
                        ProcessedFieldStorageDescriptor::Inline {
                            data: encode_f64_le(data),
                        },
                        None,
                    )
                }
            };
            fields.insert(
                name,
                ProcessedFieldDescriptor {
                    layout: field.layout,
                    keep_on_host: field.keep_on_host,
                    forward: field.forward,
                    dtype,
                    shape,
                    storage,
                    source_storage,
                },
            );
        }
        Ok(Self {
            modality: processed.modality,
            fields,
            feature_token_counts: processed.feature_token_counts,
            original_sizes: processed.original_sizes,
            content_hashes: vec![hasher.finalize().to_hex().to_string()],
        })
    }
}

fn hash_shape(hasher: &mut blake3::Hasher, dtype: u8, shape: &[usize]) {
    hasher.update(&[dtype]);
    hasher.update(&(shape.len() as u64).to_le_bytes());
    for dim in shape {
        hasher.update(&(*dim as u64).to_le_bytes());
    }
}

fn hash_layout(hasher: &mut blake3::Hasher, layout: &dynamo_multimodal::types::FieldLayout) {
    use dynamo_multimodal::types::FieldLayout;
    match layout {
        FieldLayout::Batched => {
            hasher.update(&[0]);
        }
        FieldLayout::Flat { sizes_key } => {
            hasher.update(&[1]);
            hasher.update(&(sizes_key.len() as u64).to_le_bytes());
            hasher.update(sizes_key.as_bytes());
        }
        FieldLayout::Shared => {
            hasher.update(&[2]);
        }
    };
}

fn f32_into_rdma_storage(
    data: Vec<f32>,
    nixl_agent: &NixlAgent,
) -> Result<(ProcessedFieldStorageDescriptor, SharedNixlStorage)> {
    let storage: Arc<dyn nixl::NixlMemory + Send + Sync> = Arc::new(OwnedF32Storage { data });
    let registered = nixl::register_with_nixl(storage, nixl_agent, None)
        .map_err(|_| anyhow::anyhow!("Failed to register processed tensor with NIXL"))?;
    let nixl_descriptor = registered.descriptor();
    let nixl_metadata = get_nixl_metadata(nixl_agent, registered.storage())?;
    Ok((
        ProcessedFieldStorageDescriptor::Rdma {
            nixl_metadata,
            nixl_descriptor,
        },
        Arc::new(registered),
    ))
}

fn encode_i64_le(data: Vec<i64>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<i64>());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_f64_le(data: Vec<f64>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<f64>());
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

// convert Array{N}<u8> to DecodedMediaData
// TODO: Array1<f32> for audio

impl<D: Dimension> TryFrom<ArrayBase<OwnedRepr<u8>, D>> for DecodedMediaData {
    type Error = anyhow::Error;

    fn try_from(array: ArrayBase<OwnedRepr<u8>, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();

        let (data_vec, _) = array.into_raw_vec_and_offset();
        let mut storage = SystemStorage::new(data_vec.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(data_vec.as_ptr(), storage.as_mut_ptr(), data_vec.len());
        }

        Ok(Self {
            data: storage,
            tensor_info: MediaTensorInfo {
                shape,
                dtype: DataType::UINT8,
                metadata: None,
            },
            content_hash: None,
        })
    }
}

impl<D: Dimension> TryFrom<ArrayBase<OwnedRepr<f32>, D>> for DecodedMediaData {
    type Error = anyhow::Error;

    fn try_from(array: ArrayBase<OwnedRepr<f32>, D>) -> Result<Self, Self::Error> {
        let shape = array.shape().to_vec();
        let (data_vec, _) = array.into_raw_vec_and_offset();
        let byte_len = data_vec
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("preprocessed tensor byte size overflow"))?;
        let mut storage = SystemStorage::new(byte_len)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data_vec.as_ptr().cast::<u8>(),
                storage.as_mut_ptr(),
                byte_len,
            );
        }
        Ok(Self {
            data: storage,
            tensor_info: MediaTensorInfo {
                shape,
                dtype: DataType::FLOAT32,
                metadata: None,
            },
            content_hash: None,
        })
    }
}

// Get NIXL metadata for a descriptor
// Returns zlib-compressed, base64-encoded metadata in format: "b64:<compressed_base64>"
// This format matches what Python nixl_connect expects for RdmaMetadata.nixl_metadata
// TODO: pre-allocate a fixed NIXL-registered RAM pool so metadata can be cached on the target?
pub fn get_nixl_metadata<S: ?Sized>(agent: &NixlAgent, _storage: &S) -> Result<String> {
    // WAR: Until https://github.com/ai-dynamo/nixl/pull/970 is merged, can't use get_local_partial_md
    let nixl_md = agent.raw_agent().get_local_md()?;
    // let mut reg_desc_list = RegDescList::new(MemType::Dram)?;
    // reg_desc_list.add_storage_desc(storage)?;
    // let nixl_partial_md = agent.raw_agent().get_local_partial_md(&reg_desc_list, None)?;

    // Compress with zlib (level 6, matching Python's default)
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
    encoder.write_all(&nixl_md)?;
    let compressed = encoder.finish()?;

    let b64_encoded = general_purpose::STANDARD.encode(&compressed);
    Ok(format!("b64:{}", b64_encoded))
}

// Readers send a notification with each read, and the agent keeps every notification until
// someone takes it. The frontend frees buffers by reference count and never reads them, so
// without this drain each read leaks about 100 bytes.
pub fn drain_nixl_notifications(agent: &NixlAgent) {
    let result = nixl::NotificationMap::new()
        .and_then(|mut notifs| agent.get_notifications(&mut notifs, None));
    if let Err(error) = result {
        tracing::warn!(%error, "failed to drain media-loader NIXL notifications");
    }
}

// NIXL's default progress-thread delay of 0 is the `poll()` timeout of the UCX progress
// thread, so each idle agent spins a core. With a delay, the thread sleeps in `poll()` for
// at most this long and wakes early on UCX events. Keep it short: with UCX over TCP, some
// events do not wake `poll()` and wait for the timeout.
const DEFAULT_PROGRESS_THREAD_DELAY_US: u64 = 1_000;
// NIXL 1.4.1 and earlier cast the delay to `int` before they clamp it. The cast wraps
// modulo 2^32, so values of 2^31 or more give a negative, zero, or unrelated positive
// `poll()` timeout. NIXL main clamps first. 1 s is far below that limit.
const MAX_PROGRESS_THREAD_DELAY_US: u64 = 1_000_000;

static PROGRESS_THREAD_DELAY_US: LazyLock<u64> = LazyLock::new(|| {
    progress_thread_delay_or_default(parse_or_default(
        DYN_MM_NIXL_PROGRESS_DELAY_US,
        DEFAULT_PROGRESS_THREAD_DELAY_US,
    ))
});

fn progress_thread_delay_or_default(delay_us: u64) -> u64 {
    if delay_us <= MAX_PROGRESS_THREAD_DELAY_US {
        return delay_us;
    }
    tracing::warn!(
        env_var = DYN_MM_NIXL_PROGRESS_DELAY_US,
        value = delay_us,
        max = MAX_PROGRESS_THREAD_DELAY_US,
        "invalid environment variable, using default"
    );
    DEFAULT_PROGRESS_THREAD_DELAY_US
}

pub fn get_nixl_agent() -> Result<NixlAgent> {
    let name = format!("media-loader-{}", uuid::Uuid::new_v4());
    let config = AgentConfig {
        pthr_delay_us: *PROGRESS_THREAD_DELAY_US,
        ..AgentConfig::default()
    };
    let mut nixl_agent = NixlAgent::new_with_config(&name, &config)?;
    nixl_agent
        .add_backend("UCX")
        .context("add UCX backend to media-loader NIXL agent")?;
    Ok(nixl_agent)
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_PROGRESS_THREAD_DELAY_US, DataType, MAX_PROGRESS_THREAD_DELAY_US,
        canonical_content_hash, progress_thread_delay_or_default,
    };

    #[test]
    fn progress_thread_delay_out_of_range_uses_default() {
        assert_eq!(progress_thread_delay_or_default(0), 0);
        assert_eq!(
            progress_thread_delay_or_default(MAX_PROGRESS_THREAD_DELAY_US),
            MAX_PROGRESS_THREAD_DELAY_US
        );
        assert_eq!(
            progress_thread_delay_or_default(MAX_PROGRESS_THREAD_DELAY_US + 1),
            DEFAULT_PROGRESS_THREAD_DELAY_US
        );
    }

    #[test]
    fn canonical_content_hash_payload_is_stable() {
        let bytes = [0_u8, 1, 2, 3, 4, 5];
        let hash = canonical_content_hash(&[1, 2, 3], DataType::UINT8, &bytes);

        assert_eq!(hash, 0x7a9b_bcb1_1a89_8630);
        assert_eq!(format!("{hash:016x}"), "7a9bbcb11a898630");
    }
}
#[cfg(all(test, feature = "mm-routing", feature = "media-ffmpeg"))]
mod video_tests {
    use super::*;
    use crate::preprocessor::media::decoders::VideoMetadata;

    fn video_info(sampled_timestamps: Vec<f64>) -> MediaTensorInfo {
        MediaTensorInfo {
            shape: vec![2, 1, 2, 3],
            dtype: DataType::UINT8,
            metadata: Some(DecodedMediaMetadata::Video(VideoMetadata {
                source_fps: 24.0,
                source_duration: 10.0,
                sampled_timestamps,
            })),
        }
    }

    #[test]
    fn video_hash_covers_metadata_shape_and_rgb_bytes() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let expected = hash_video_content(&info, &bytes).unwrap();

        assert_eq!(hash_video_content(&info, &bytes).unwrap(), expected);

        let changed_metadata = video_info(vec![0.0, 5.1]);
        assert_ne!(
            hash_video_content(&changed_metadata, &bytes).unwrap(),
            expected
        );

        let mut changed_shape = info.clone();
        changed_shape.shape = vec![1, 2, 2, 3];
        assert_ne!(
            hash_video_content(&changed_shape, &bytes).unwrap(),
            expected
        );

        let mut changed_bytes = bytes;
        changed_bytes[0] = 42;
        assert_ne!(hash_video_content(&info, &changed_bytes).unwrap(), expected);
    }

    #[test]
    fn video_hash_is_precomputed_from_decoded_storage() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let mut storage = SystemStorage::new(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.as_mut_ptr(), bytes.len());
        }

        assert_eq!(
            content_hash_for_storage(&info, &storage, true),
            Some(hash_video_content(&info, &bytes).unwrap())
        );
    }

    #[test]
    fn video_hash_is_skipped_when_exact_routing_is_ineligible() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let mut storage = SystemStorage::new(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), storage.as_mut_ptr(), bytes.len());
        }

        assert_eq!(content_hash_for_storage(&info, &storage, false), None);
    }

    #[test]
    fn video_dimensions_validate_rgb_payload_layout() {
        let bytes = [0_u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let info = video_info(vec![0.0, 5.0]);
        let dimensions = video_dimensions_from_parts(&info, &bytes).unwrap();

        assert_eq!(dimensions, (2, 2, 1));

        let mut rgba = info.clone();
        rgba.shape[3] = 4;
        assert!(video_dimensions_from_parts(&rgba, &bytes).is_err());
        assert!(video_dimensions_from_parts(&info, &bytes[..11]).is_err());
    }
}

#[cfg(test)]
mod processed_tests {
    use super::*;

    #[test]
    fn processed_inline_field_has_backend_neutral_wire_shape() {
        let field = ProcessedFieldDescriptor {
            layout: dynamo_multimodal::types::FieldLayout::Batched,
            keep_on_host: true,
            forward: true,
            dtype: DataType::INT64,
            shape: vec![1, 3],
            storage: ProcessedFieldStorageDescriptor::Inline {
                data: encode_i64_le(vec![1, 2, 3]),
            },
            source_storage: None,
        };
        let value = serde_json::to_value(&field).unwrap();
        assert_eq!(value["storage"], "inline");
        assert_eq!(value["dtype"], "INT64");
        assert_eq!(value["layout"]["kind"], "batched");
        assert_eq!(value["shape"], serde_json::json!([1, 3]));
        assert!(value.get("format").is_none());

        let decoded: ProcessedFieldDescriptor = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.dtype, DataType::INT64);
        assert_eq!(decoded.shape, vec![1, 3]);
        let ProcessedFieldStorageDescriptor::Inline { data } = decoded.storage else {
            panic!("expected inline storage");
        };
        assert_eq!(data, encode_i64_le(vec![1, 2, 3]));
    }
}
