/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native tokenization and rendering over verified Hugging Face assets.

mod deepseek;
mod boundaries;
mod tml;
mod templates;
#[cfg(feature = "images")]
pub mod images;
mod json;
mod jinja;

mod vocabulary;

use std::fs;
use std::path::PathBuf;

use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::{Error, Failure, FrontendError, Result};

pub(crate) fn validate_chat_template(source: &str) -> Result<()> {
    templates::compile(source).map(|_| ()).map_err(|error| {
        Error::frontend(FrontendError::Template(Failure::caused_by(error.to_string(), error)))
    })
}

pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
    pub range: Option<(u64, usize)>,
}

impl Asset {
    pub(crate) fn read(&self) -> Result<Vec<u8>> {
        if !self.path.is_absolute() {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset path must be absolute: {}",
                    self.path.display()
                ),
            ))));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!("invalid expected SHA256 for {}", self.path.display()),
            ))));
        }
        let bytes = (|| -> std::io::Result<Vec<u8>> {
            if let Some((offset, size)) = self.range {
                use crate::os::FileReadAt;
                let file = fs::File::open(&self.path)?;
                if offset
                    .checked_add(size as u64)
                    .is_none_or(|end| end > file.metadata().map_or(0, |m| m.len()))
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "frontend asset exceeds core",
                    ));
                }
                let mut bytes = vec![0; size];
                file.read_exact_at(&mut bytes, offset)?;
                Ok(bytes)
            } else {
                fs::read(&self.path)
            }
        })()
        .map_err(|error| {
            Error::frontend(FrontendError::Asset(Failure::caused_by(
                format!("cannot read {}: {error}", self.path.display()),
                error,
            )))
        })?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&self.sha256) {
            return Err(Error::frontend(FrontendError::Asset(Failure::new(
                format!(
                    "frontend asset SHA256 mismatch for {}: expected {}, received {actual}",
                    self.path.display(),
                    self.sha256
                ),
            ))));
        }
        Ok(bytes)
    }
}

pub struct FrontendAssets {
    pub tokenizer: Asset,
    pub tokens: TokenizerSettings,
    pub chat_template: Option<Asset>,
}

#[cfg(any(test, feature = "test-tools"))]
#[path = "../../tests/support/frontend.rs"]
pub mod test_support;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSettings {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Option<TokenList>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenList {
    pub values: Vec<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontendSettings {
    #[serde(default)]
    pub prompt_encoder: PromptEncoder,
    #[serde(default)]
    pub image_placeholder_token_id: Option<u32>,
    pub tokenizer: TokenizerSettings,
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub template_encoding: TemplateEncoding,
    #[cfg(feature = "images")]
    pub image: Option<images::ImageSettings>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum PromptEncoder {
    #[default]
    Jinja = 0,
    DeepseekV4 = 1,
    DeepseekV41 = 2,
    /// TMLv0 message frames: role and content-kind control tokens around ordinary text.
    TmlV0 = 3,
}

impl PromptEncoder {
    /// Whether the encoder renders `tool_choice` and JSON response formats in its
    /// own syntax. Other encoders receive the shared instruction messages instead.
    pub fn renders_tool_choice(self) -> bool {
        matches!(self, Self::DeepseekV4 | Self::DeepseekV41)
    }

    /// Whether the encoder renders failed tool results in its own syntax. Other
    /// encoders receive them prefixed with a text part saying the tool failed.
    pub fn renders_tool_errors(self) -> bool {
        matches!(self, Self::TmlV0)
    }
}

impl From<PromptEncoder> for i32 {
    fn from(value: PromptEncoder) -> Self {
        value as i32
    }
}
impl TryFrom<i32> for PromptEncoder {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Jinja),
            1 => Ok(Self::DeepseekV4),
            2 => Ok(Self::DeepseekV41),
            3 => Ok(Self::TmlV0),
            _ => Err("unsupported prompt encoder"),
        }
    }
}

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "i32", into = "i32")]
pub enum TemplateEncoding {
    #[default]
    Text = 0,
    TokenSegments = 1,
}

impl From<TemplateEncoding> for i32 {
    fn from(value: TemplateEncoding) -> Self {
        value as i32
    }
}

impl TryFrom<i32> for TemplateEncoding {
    type Error = &'static str;
    fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Text),
            1 => Ok(Self::TokenSegments),
            _ => Err("unsupported chat template encoding"),
        }
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenSegment {
    text: String,
    allow_special: bool,
    #[serde(default)]
    image_index: Option<usize>,
}

fn order_segment_images(
    segments: &[TokenSegment],
    sources: &mut Vec<String>,
    hints: &mut [usize],
) -> Result<()> {
    let order: Vec<_> = segments.iter().filter_map(|segment| segment.image_index).collect();
    if order.is_empty() {
        return Ok(());
    }
    if order.len() != sources.len() {
        return Err(deepseek::invalid("image segment indices must include each image source exactly once"));
    }
    let mut positions = vec![None; sources.len()];
    for (position, &index) in order.iter().enumerate() {
        let Some(slot) = positions.get_mut(index) else {
            return Err(deepseek::invalid("image segment index exceeds the image source count"));
        };
        if slot.replace(position).is_some() {
            return Err(deepseek::invalid("image segment indices must not repeat an image source"));
        }
    }
    for hint in hints {
        *hint = positions[*hint].unwrap();
    }
    let mut original: Vec<_> = std::mem::take(sources).into_iter().map(Some).collect();
    *sources = order.into_iter().map(|index| original[index].take().unwrap()).collect();
    Ok(())
}

fn token_segments(text: &str, hints: &mut Vec<usize>) -> serde_json::Result<Vec<TokenSegment>> {
    if hints.is_empty() {
        return serde_json::from_str(text);
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawSegment<'a> {
        #[serde(borrow)]
        text: &'a serde_json::value::RawValue,
        allow_special: bool,
        #[serde(default)]
        image_index: Option<usize>,
    }
    let raw_segments: Vec<RawSegment<'_>> = serde_json::from_str(text)?;
    let mut segments = Vec::with_capacity(raw_segments.len());
    let mut positions = Vec::new();
    let mut byte_start = 0;
    for segment in raw_segments {
        let raw = segment.text.get();
        let start = raw.as_ptr() as usize - text.as_ptr() as usize;
        let decoded: String = serde_json::from_str(raw)?;
        for &hint in hints.iter() {
            if hint > start && hint < start + raw.len() {
                // The hint ends inside this JSON string, before its closing quote.
                let mut prefix = raw[..hint - start].to_owned();
                prefix.push('"');
                let prefix: String = serde_json::from_str(&prefix)?;
                positions.push(byte_start + prefix.len());
            }
        }
        byte_start += decoded.len();
        segments.push(TokenSegment {
            text: decoded,
            allow_special: segment.allow_special,
            image_index: segment.image_index,
        });
    }
    *hints = positions;
    Ok(segments)
}

pub struct RenderedPrompt {
    pub image_sources: Vec<String>,
    /// Indices in image_sources whose expanded spans end at requested checkpoints.
    pub image_checkpoint_hints: Vec<usize>,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub message_boundaries: Vec<usize>,
}

pub struct Frontend {
    prompt_encoder: PromptEncoder,
    image_placeholder_token_id: Option<u32>,
    tokenizer: Tokenizer,
    ordinary_tokenizer: std::sync::OnceLock<Tokenizer>,
    segmented_template: bool,
    template: Option<jinja::CompiledTemplate>,
    template_source: String,
    bos_token_id: Option<u32>,
    effort_levels: std::sync::OnceLock<Vec<String>>,
    vocabulary_bytes: std::sync::OnceLock<Result<vocabulary::VocabularyBytes>>,
    inline_system_messages: std::sync::OnceLock<bool>,
    defaults: Map<String, Json>,
}

impl Frontend {
    pub fn prompt_encoder(&self) -> PromptEncoder {
        self.prompt_encoder
    }
    pub fn with_template_encoding(mut self, encoding: TemplateEncoding) -> Self {
        if self.prompt_encoder == PromptEncoder::Jinja {
            self.segmented_template = matches!(encoding, TemplateEncoding::TokenSegments);
        }
        self
    }

    fn ordinary_from(tokenizer: &Tokenizer) -> Tokenizer {
        let mut tokenizer = tokenizer.clone();
        tokenizer.with_added_vocabulary(tokenizers::AddedVocabulary::new());
        tokenizer
    }

    fn ordinary(&self) -> &Tokenizer {
        self.ordinary_tokenizer.get_or_init(|| Self::ordinary_from(&self.tokenizer))
    }

    fn encode_splicing_untrusted_markers(
        &self,
        text: &str,
        untrusted: &[(usize, usize)],
    ) -> Result<(Vec<u32>, Vec<(usize, usize)>)> {
        let encoding = self.tokenizer.encode(text, false).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        let token_ids = encoding.get_ids().to_vec();
        let offsets = encoding.get_offsets().to_vec();
        if untrusted.is_empty() {
            return Ok((token_ids, offsets));
        }
        let added = self.tokenizer.get_added_tokens_decoder();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut token = 0usize;
        for &(run_start, run_end) in untrusted {
            while token < offsets.len() && offsets[token].1 <= run_start {
                token += 1;
            }
            if token == offsets.len() || offsets[token].0 >= run_end {
                continue;
            }
            let mut span_end = token;
            while span_end < offsets.len() && offsets[span_end].0 < run_end {
                span_end += 1;
            }
            if !token_ids[token..span_end]
                .iter()
                .any(|id| added.contains_key(id))
            {
                continue;
            }
            match spans.last_mut() {
                Some(previous) if token <= previous.1 => previous.1 = previous.1.max(span_end),
                _ => spans.push((token, span_end)),
            }
        }
        let mut spliced_ids = Vec::with_capacity(token_ids.len());
        let mut spliced_offsets = Vec::with_capacity(offsets.len());
        let mut cursor = 0usize;
        for (span_start, span_end) in spans {
            let byte_start = offsets[span_start].0;
            let byte_end = offsets[span_end - 1].1;
            if byte_end <= byte_start {
                continue;
            }
            spliced_ids.extend_from_slice(&token_ids[cursor..span_start]);
            spliced_offsets.extend_from_slice(&offsets[cursor..span_start]);
            cursor = span_end;
            let slice = text.get(byte_start..byte_end).unwrap_or("");
            let replacement = self.ordinary().encode(slice, false).map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })?;
            spliced_ids.extend_from_slice(replacement.get_ids());
            spliced_offsets.extend(
                replacement
                    .get_offsets()
                    .iter()
                    .map(|&(start, end)| (byte_start + start, byte_start + end)),
            );
        }
        spliced_ids.extend_from_slice(&token_ids[cursor..]);
        spliced_offsets.extend_from_slice(&offsets[cursor..]);
        Ok((spliced_ids, spliced_offsets))
    }
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Resolves advisory effort against the vocabulary accepted by the prepared template.
    pub fn resolve_effort<'a>(&self, candidates: &'a [&str]) -> Result<&'a str> {
        if self.prompt_encoder == PromptEncoder::TmlV0 {
            return candidates
                .iter()
                .copied()
                .find(|level| tml::effort_level(level))
                .ok_or_else(|| deepseek::invalid("unsupported reasoning effort"));
        }
        if self.prompt_encoder != PromptEncoder::Jinja {
            return candidates
                .iter()
                .copied()
                .find(|level| matches!(*level, "low" | "high" | "xhigh" | "max"))
                .ok_or_else(|| deepseek::invalid("unsupported reasoning effort"));
        }
        let accepted = self.effort_levels.get_or_init(|| {
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([{"role":"user","content":"Hello"}]),
            );
            context.insert("enable_thinking".into(), true.into());
            context.insert("thinking_mode".into(), "thinking".into());
            context.insert("add_generation_prompt".into(), true.into());
            context.insert("continue_final_message".into(), false.into());
            ["low", "medium", "high", "xhigh", "max"]
                .into_iter()
                .filter(|effort| {
                    context.insert("reasoning_effort".into(), (*effort).into());
                    self.template
                        .as_ref()
                        .unwrap()
                        .render(&context, &[])
                        .is_ok()
                })
                .map(str::to_owned)
                .collect()
        });
        candidates
            .iter()
            .copied()
            .find(|candidate| accepted.iter().any(|level| level == candidate))
            .ok_or_else(|| {
                Error::admission(
                    crate::AdmissionError::Feature,
                    "prepared chat template does not accept the requested reasoning effort",
                )
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn template_default(&self, name: &str) -> Option<&Json> {
        self.defaults.get(name)
    }

    pub fn normalize_inline_system_messages(&self, messages: &mut [Json]) -> Result<()> {
        if self.prompt_encoder != PromptEncoder::Jinja {
            return Ok(());
        }
        if !messages.iter().skip(1).any(|m| m["role"] == "system") {
            return Ok(());
        }
        let supported = self.inline_system_messages.get_or_init(|| {
            let marker = "FLARE_INLINE_SYSTEM_PROBE";
            let mut context = self.defaults.clone();
            context.insert(
                "messages".into(),
                serde_json::json!([
                    {"role":"system","content":"Instructions"},
                    {"role":"user","content":"Hello"},
                    {"role":"system","content":marker},
                    {"role":"user","content":"Continue"}
                ]),
            );
            context.insert("add_generation_prompt".into(), false.into());
            context.insert("continue_final_message".into(), false.into());
            self.template
                .as_ref()
                .unwrap()
                .render(&context, &[])
                .is_ok_and(|rendered| rendered.text.contains(marker))
        });
        if *supported {
            return Ok(());
        }
        for message in messages.iter_mut().skip(1) {
            if message["role"] != "system" {
                continue;
            }
            let invalid = || {
                Error::frontend(FrontendError::Input(Failure::new(
                    "inline system messages require text content",
                )))
            };
            let text = match &message["content"] {
                Json::String(text) => text.clone(),
                Json::Array(parts) => parts
                    .iter()
                    .map(|part| {
                        if part["type"] != "text" {
                            return Err(invalid());
                        }
                        part["text"].as_str().ok_or_else(invalid)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join("\n\n"),
                _ => return Err(invalid()),
            };
            message["role"] = "user".into();
            message["content"] = format!("<system>{text}</system>").into();
        }
        Ok(())
    }

    pub fn load(assets: FrontendAssets, template_defaults: Map<String, Json>) -> Result<Self> {
        Self::load_encoder(assets, template_defaults, PromptEncoder::Jinja, None)
    }

    pub fn load_with_settings(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        settings: &FrontendSettings,
    ) -> Result<Self> {
        if settings.prompt_encoder != PromptEncoder::Jinja && assets.chat_template.is_some() {
            return Err(deepseek::invalid(
                "official prompt encoding does not use a chat template asset",
            ));
        }
        if settings.prompt_encoder != PromptEncoder::Jinja
            && matches!(settings.template_encoding, TemplateEncoding::TokenSegments)
        {
            return Err(deepseek::invalid(
                "official prompt encoding requires text template encoding",
            ));
        }
        Self::load_encoder(
            assets,
            template_defaults,
            settings.prompt_encoder,
            settings.image_placeholder_token_id,
        )
        .map(|frontend| frontend.with_template_encoding(settings.template_encoding))
    }

    fn load_encoder(
        assets: FrontendAssets,
        template_defaults: Map<String, Json>,
        prompt_encoder: PromptEncoder,
        image_placeholder_token_id: Option<u32>,
    ) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_bytes(assets.tokenizer.read()?).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                format!("invalid tokenizer: {error}"),
                error,
            )))
        })?;
        tokenizer.with_truncation(None).map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })?;
        tokenizer.with_padding(None);
        boundaries::preserve_offsets(&mut tokenizer);
        if prompt_encoder != PromptEncoder::Jinja
            && image_placeholder_token_id.is_some_and(|id| tokenizer.id_to_token(id).is_none())
        {
            return Err(deepseek::invalid(
                "image placeholder token ID is absent from the checkpoint tokenizer",
            ));
        }
        let template_bytes = match assets.chat_template {
            Some(asset) if prompt_encoder == PromptEncoder::Jinja => asset.read()?,
            _ if prompt_encoder != PromptEncoder::Jinja => Vec::new(),
            _ => {
                return Err(deepseek::invalid(
                    "Jinja prompt encoding requires a chat template asset",
                ));
            }
        };
        let template_source = String::from_utf8(template_bytes).map_err(|error| {
            Error::frontend(FrontendError::Template(Failure::caused_by(
                format!("chat template is not UTF-8: {error}"),
                error,
            )))
        })?;
        let template = if prompt_encoder == PromptEncoder::Jinja {
            Some(templates::compile(&template_source).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(error.to_string(), error)))
            })?)
        } else { None };
        let bos_token_id = assets
            .tokens
            .bos_token
            .as_deref()
            .and_then(|token| tokenizer.token_to_id(token));
        let mut defaults = Map::new();
        for (name, value) in [
            ("bos_token", assets.tokens.bos_token),
            ("eos_token", assets.tokens.eos_token),
            ("unk_token", assets.tokens.unk_token),
            ("sep_token", assets.tokens.sep_token),
            ("pad_token", assets.tokens.pad_token),
            ("cls_token", assets.tokens.cls_token),
            ("mask_token", assets.tokens.mask_token),
        ] {
            if let Some(value) = value {
                defaults.insert(name.into(), Json::String(value));
            }
        }
        if let Some(tokens) = assets.tokens.additional_special_tokens {
            defaults.insert(
                "additional_special_tokens".into(),
                Json::Array(tokens.values.into_iter().map(Json::String).collect()),
            );
        }
        defaults.insert(
            "add_generation_prompt".into(),
            Json::Bool(prompt_encoder != PromptEncoder::Jinja),
        );
        defaults.insert("tools".into(), Json::Null);
        defaults.insert("documents".into(), Json::Null);
        defaults.extend(template_defaults);
        let frontend = Self {
            prompt_encoder,
            image_placeholder_token_id,
            bos_token_id,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            tokenizer,
            ordinary_tokenizer: std::sync::OnceLock::new(),
            segmented_template: false,
            template,
            template_source,
            defaults,
        };
        match prompt_encoder {
            PromptEncoder::Jinja => {}
            PromptEncoder::TmlV0 => tml::validate_defaults(&frontend)?,
            PromptEncoder::DeepseekV4 | PromptEncoder::DeepseekV41 => deepseek::validate_defaults(&frontend)?,
        }
        Ok(frontend)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })
    }

    pub fn render_chat(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        match self.prompt_encoder {
            PromptEncoder::Jinja => {}
            PromptEncoder::TmlV0 => return tml::render(self, messages, options),
            PromptEncoder::DeepseekV4 | PromptEncoder::DeepseekV41 => {
                return deepseek::render(self, messages, options);
            }
        }
        if messages.is_empty() {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "Cannot apply chat template to an empty conversation. Provide at least one message.",
            ))));
        }
        self.render_jinja(messages, options)
    }

    fn render_jinja(
        &self,
        messages: &[Json],
        options: &Map<String, Json>,
    ) -> Result<RenderedPrompt> {
        let mut context = self.defaults.clone();
        context.extend(options.clone());
        let continuation = match context.remove("continue_final_message") {
            None | Some(Json::Null | Json::Bool(false)) => None,
            Some(Json::Bool(true)) => Some("content".to_owned()),
            Some(Json::String(field)) if field.is_empty() => None,
            Some(Json::String(field)) => Some(field),
            _ => {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "continue_final_message must be a boolean or field name",
                ))));
            }
        };
        if continuation.is_some() && context.get("add_generation_prompt") == Some(&Json::Bool(true))
        {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead.",
            ))));
        }
        if let Some(tools) = context.get("tools").filter(|value| !value.is_null()) {
            if !tools
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Tools should either be a JSON schema, or a callable function with type hints and a docstring suitable for auto-conversion to a schema.",
                ))));
            }
        }
        if let Some(documents) = context.get("documents").filter(|value| !value.is_null()) {
            if !documents
                .as_array()
                .is_some_and(|values| values.iter().all(Json::is_object))
            {
                return Err(Error::frontend(FrontendError::Input(Failure::new(
                    "Documents should be a list of dicts with 'title' and 'text' keys!",
                ))));
            }
        }
        let mut messages = messages.to_vec();
        let mut image_sources = Vec::new();
        let mut image_checkpoint_hints = Vec::new();
        let mut checkpoint_hints = Vec::new();
        for (index, message) in messages.iter_mut().enumerate() {
            if !message.is_object() {
                return Err(deepseek::invalid("messages must contain objects"));
            }
            if let Some(Json::Array(hints)) =
                message.as_object_mut().unwrap().remove("_checkpoint_hints")
            {
                checkpoint_hints.extend(
                    hints
                        .iter()
                        .filter_map(|hint| hint.as_u64().map(|count| (index, count as usize))),
                );
            }
            if let Some(parts) = message.get_mut("content").and_then(Json::as_array_mut) {
                for (part_index, part) in parts.iter_mut().enumerate() {
                    let kind = part
                        .get("type")
                        .and_then(Json::as_str)
                        .ok_or_else(|| deepseek::invalid("message content part requires type"))?;
                    if kind == "text" && !part.get("text").is_some_and(Json::is_string) {
                        return Err(deepseek::invalid("text content must be a string"));
                    }
                    if kind == "image_url" {
                        let url = part["image_url"]["url"]
                            .as_str()
                            .ok_or_else(|| deepseek::invalid("image_url.url must be a string"))?;
                        if checkpoint_hints.contains(&(index, part_index + 1)) {
                            image_checkpoint_hints.push(image_sources.len());
                        }
                        let image_index = image_sources.len();
                        image_sources.push(url.to_owned());
                        *part = serde_json::json!({"type":"image", "image_index":image_index});
                    }
                }
            }
        }
        let final_text = continuation
            .map(|field| self.mark_continuation(&mut messages, &field))
            .transpose()?;
        context.insert("messages".into(), Json::Array(messages));
        let trusted: std::collections::HashSet<String> = self
            .defaults
            .keys()
            .filter(|name| !options.contains_key(*name) && name.as_str() != "messages")
            .cloned()
            .collect();
        let rendered = self
            .template
            .as_ref()
            .unwrap()
            .render_with_trust(&context, &checkpoint_hints, &trusted)
            .map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    error.message.clone(),
                    error,
                )))
            })?;
        let marks = rendered.marks;
        let runs = rendered.runs;
        let rendered = rendered.text;
        let mut hint_bytes = marks
            .iter()
            .filter_map(|mark| match mark {
                jinja::Mark::Hint(byte) => Some(*byte),
                _ => None,
            })
            .collect();
        let mut segments: Vec<TokenSegment> = if self.segmented_template {
            token_segments(&rendered, &mut hint_bytes).map_err(|error| {
                Error::frontend(FrontendError::Template(Failure::caused_by(
                    "chat template must render an array of text and allow_special segments",
                    error,
                )))
            })?
        } else {
            Vec::new()
        };
        order_segment_images(&segments, &mut image_sources, &mut image_checkpoint_hints)?;
        let mut segment_ends = Vec::with_capacity(segments.len() + 1);
        segment_ends.push(0usize);
        for segment in &segments {
            segment_ends.push(segment_ends.last().unwrap() + segment.text.len());
        }
        let mut byte_boundaries = Vec::new();
        for mark in marks {
            match mark {
                jinja::Mark::Hint(_) => {}
                jinja::Mark::Byte(byte) if !self.segmented_template && byte <= rendered.len() => {
                    byte_boundaries.push(byte);
                }
                jinja::Mark::Position(count) if self.segmented_template => {
                    byte_boundaries.push(*segment_ends.get(count).ok_or_else(|| {
                        deepseek::invalid("mark position exceeds the rendered segment count")
                    })?);
                }
                _ => {
                    return Err(deepseek::invalid(
                        "text templates require byte marks; segmented templates require segment-count marks",
                    ));
                }
            }
        }
        let mut text: String = if self.segmented_template {
            segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect()
        } else {
            rendered
        };
        if let Some(final_text) = final_text {
            let marker = "CONTINUE_FINAL_MESSAGE_TAG";
            let index = text.rfind(marker).filter(|_| text.contains(final_text.trim())).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but the final message does not appear in the chat after applying the chat template!"))))?;
            let preserved_space = text[index + marker.len()..].starts_with(' ');
            text.truncate(index);
            if !preserved_space {
                text.truncate(text.trim_end().len());
            }
        }
        let mut token_ids = Vec::new();
        let mut offsets = Vec::new();
        if self.segmented_template {
            let mut remaining = text.len();
            let mut byte_start = 0;
            for segment in &mut segments {
                segment.text.truncate(remaining.min(segment.text.len()));
                remaining -= segment.text.len();
                let tokenizer = if segment.allow_special {
                    &self.tokenizer
                } else {
                    self.ordinary()
                };
                let encoding = tokenizer
                    .encode(segment.text.as_str(), false)
                    .map_err(|error| {
                        Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                            error.to_string(),
                            error,
                        )))
                    })?;
                token_ids.extend_from_slice(encoding.get_ids());
                offsets.extend(
                    encoding
                        .get_offsets()
                        .iter()
                        .map(|&(start, end)| (byte_start + start, byte_start + end)),
                );
                byte_start += segment.text.len();
                if remaining == 0 {
                    break;
                }
            }
        } else {
            let mut cursor = 0;
            let mut untrusted: Vec<(usize, usize)> = Vec::new();
            for (end, trusted) in runs {
                let end = end.min(text.len());
                if !trusted && end > cursor {
                    if untrusted.last().is_some_and(|last| last.1 == cursor) {
                        untrusted.last_mut().unwrap().1 = end;
                    } else {
                        untrusted.push((cursor, end));
                    }
                }
                cursor = end;
            }
            let spliced = self.encode_splicing_untrusted_markers(&text, &untrusted)?;
            token_ids = spliced.0;
            offsets = spliced.1;
        }
        let message_boundaries =
            boundaries::token_ends(&byte_boundaries, &hint_bytes, &offsets, text.len())?;
        Ok(RenderedPrompt {
            message_boundaries,
            text,
            token_ids,
            image_sources,
            image_checkpoint_hints,
        })
    }

    fn mark_continuation(&self, messages: &mut [Json], field: &str) -> Result<String> {
        let message = messages.last_mut().ok_or_else(|| {
            Error::frontend(FrontendError::Input(Failure::new(
                "list index out of range",
            )))
        })?;
        let value = message.get_mut(field).filter(|value| !value.is_null()).ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new(format!("continue_final_message is set but the final message has no \"{field}\" to continue!")))))?;
        if !self.template_source.contains(field) {
            return Err(Error::frontend(FrontendError::Input(Failure::new(
                format!(
                    "continue_final_message is set to \"{field}\" but this is not an accepted field in the chat_template"
                ),
            ))));
        }
        let text = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .and_then(|value| value.as_str())
                .map(str::to_owned)
        } else {
            value.as_str().map(str::to_owned)
        };
        let original = text.ok_or_else(|| Error::frontend(FrontendError::Input(Failure::new("continue_final_message is set but we could not find any text to continue in the final message!"))))?;
        let target = if let Some(blocks) = value.as_array_mut() {
            blocks
                .iter_mut()
                .rev()
                .find_map(|block| block.get_mut("text"))
                .unwrap()
        } else {
            value
        };
        *target = Json::String(format!("{original}CONTINUE_FINAL_MESSAGE_TAG "));
        Ok(original)
    }

    #[cfg(feature = "server")]
    pub(crate) fn prepare_token_bytes(&self) -> Result<()> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map(|_| ()).map_err(Clone::clone)
    }

    pub fn token_bytes(&self, token: u32) -> Result<&[u8]> {
        self.vocabulary_bytes.get_or_init(|| vocabulary::VocabularyBytes::new(&self.tokenizer))
            .as_ref().map_err(Clone::clone)?.get(token)
    }

    pub fn incremental_decoder(&self, skip_special_tokens: bool) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: &self.tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
            finished: false,
        }
    }
}

pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    finished: bool,
}

impl IncrementalDecoder<'_> {
    pub fn push(&mut self, token_ids: &[u32]) -> Result<Option<String>> {
        if self.finished {
            return Err(Error::frontend(FrontendError::Tokenizer(Failure::new(
                "incremental decoder is finished",
            ))));
        }
        if token_ids.is_empty() {
            return Ok(None);
        }
        tokenizers::tokenizer::step_decode_stream(
            self.tokenizer,
            token_ids.to_vec(),
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| {
            Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                error.to_string(),
                error,
            )))
        })
    }

    #[cfg(feature = "server")]
    pub(crate) fn pending_text(&self) -> Result<String> {
        let text = self.tokenizer.decode(&self.ids, self.skip_special_tokens).map_err(|e| e.to_string())?;
        Ok(text.strip_prefix(&self.prefix).ok_or("decoder changed its emitted prefix")?.to_owned())
    }

    pub fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        let remainder = self
            .tokenizer
            .decode(&self.ids, self.skip_special_tokens)
            .map_err(|error| {
                Error::frontend(FrontendError::Tokenizer(Failure::boxed(
                    error.to_string(),
                    error,
                )))
            })?;
        let suffix = remainder
            .strip_prefix(&self.prefix)
            .ok_or_else(|| {
                Error::frontend(FrontendError::Tokenizer(Failure::new(
                    "incremental decoder final text changed its emitted prefix",
                )))
            })?
            .to_owned();
        self.finished = true;
        Ok(suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_is_shareable_between_native_request_workers() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<Frontend>();
    }

    fn role_frontend(guard: &str) -> Frontend {
        let source = format!(
            "{{% for message in messages %}}{guard}{{{{ message.role }}}}:{{{{ message.content }}}}\n{{% endfor %}}"
        );
        let template = Some(jinja::CompiledTemplate::compile(&source).unwrap());
        Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            ordinary_tokenizer: std::sync::OnceLock::new(),
            segmented_template: false,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template,
            template_source: source,
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            defaults: Map::new(),
        }
    }

    #[test]
    fn inline_system_roles_follow_template_capability() {
        let source = serde_json::json!([
            {"role":"system","content":"Initial instructions"},
            {"role":"user","content":"Hello"},
            {"role":"system","content":[
                {"type":"text","text":"Environment"},
                {"type":"text","text":"Use the repository"}
            ]},
            {"role":"assistant","content":"Ready"},
            {"role":"user","content":"Continue"}
        ]);
        let original = source.as_array().unwrap();
        for (guard, supported) in [
            ("", true),
            (
                "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
                false,
            ),
            (
                "{% if message.role == 'system' and not loop.first %}{% continue %}{% endif %}",
                false,
            ),
        ] {
            let frontend = role_frontend(guard);
            let mut messages = original.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(&messages[..2], &original[..2]);
            assert_eq!(&messages[3..], &original[3..]);
            if supported {
                assert_eq!(messages, *original);
            } else {
                assert_eq!(
                    messages[2],
                    serde_json::json!({
                        "role":"user","content":"<system>Environment\n\nUse the repository</system>"
                    })
                );
                frontend.render_chat(&messages, &Map::new()).unwrap();
            }
            let once = messages.clone();
            frontend
                .normalize_inline_system_messages(&mut messages)
                .unwrap();
            assert_eq!(messages, once);
        }
    }

    #[test]
    fn inline_system_conversion_preserves_text_and_rejects_nontext() {
        let frontend = role_frontend(
            "{% if message.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}",
        );
        let mut messages = vec![
            serde_json::json!({"role":"user","content":"Hello"}),
            serde_json::json!({"role":"system","content":"\nUse <tag> & café.\n"}),
        ];
        frontend
            .normalize_inline_system_messages(&mut messages)
            .unwrap();
        assert_eq!(
            messages[1]["content"],
            "<system>\nUse <tag> & café.\n</system>"
        );
        messages[1] = serde_json::json!({"role":"system","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]});
        assert!(
            frontend
                .normalize_inline_system_messages(&mut messages)
                .is_err()
        );
    }

    #[test]
    fn advisory_effort_uses_the_prepared_templates_vocabulary() {
        for (levels, expected) in [
            ("['low','medium','xhigh']", ["low", "medium", "xhigh"]),
            ("['low','high','max']", ["low", "high", "max"]),
        ] {
            let source = format!(
                "{{% if reasoning_effort not in {levels} %}}{{{{ raise_exception('unsupported effort') }}}}{{% endif %}}"
            );
            let template = Some(jinja::CompiledTemplate::compile(&source).unwrap());
            let frontend = Frontend {
                prompt_encoder: PromptEncoder::Jinja,
                image_placeholder_token_id: None,
                ordinary_tokenizer: std::sync::OnceLock::new(),
                segmented_template: false,
                tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
                template,
                template_source: source,
                bos_token_id: None,
                vocabulary_bytes: Default::default(),
                effort_levels: Default::default(),
                inline_system_messages: Default::default(),
                defaults: Map::new(),
            };
            for (candidates, expected) in [
                (&["low"][..], expected[0]),
                (&["high", "medium"][..], expected[1]),
                (&["max", "xhigh", "high", "medium"][..], expected[2]),
            ] {
                assert_eq!(frontend.resolve_effort(candidates).unwrap(), expected);
            }
            assert!(frontend.resolve_effort(&["unsupported"]).is_err());
        }
    }

    #[test]
    fn missing_asset_retains_io_failure() {
        use std::error::Error as _;

        let asset = Asset {
            range: None,
            path: std::env::temp_dir()
                .join(format!("flare-missing-asset-{}", std::process::id()))
                .join("tokenizer.json"),
            sha256: "0".repeat(64),
        };
        let error = asset.read().unwrap_err();
        assert!(matches!(error, Error::Frontend(FrontendError::Asset(_))));
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap();
        assert_eq!(cause.kind(), std::io::ErrorKind::NotFound);
        let cloned = error.clone();
        assert_eq!(cloned.to_string(), error.to_string());
        assert!(std::ptr::eq(
            error.source().unwrap(),
            cloned.source().unwrap()
        ));
    }

    #[test]
    fn render_failure_retains_template_cause_and_detail() {
        use std::error::Error as _;

        let source = "{{ raise_exception('fixture rejection') }}";
        let template = Some(jinja::CompiledTemplate::compile(source).unwrap());
        let frontend = Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            ordinary_tokenizer: std::sync::OnceLock::new(),
            segmented_template: false,
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            template,
            template_source: source.to_owned(),
            defaults: Map::new(),
        };
        let error = frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"x"})],
                &Map::new(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, Error::Frontend(FrontendError::Template(_))));
        assert_eq!(error.to_string(), "fixture rejection");
        let cause = error
            .source()
            .unwrap()
            .downcast_ref::<jinja::Error>()
            .unwrap();
        assert_eq!(cause.message, "fixture rejection");
        let input = frontend.render_chat(&[], &Map::new()).err().unwrap();
        assert!(matches!(input, Error::Frontend(FrontendError::Input(_))));
        assert!(input.source().is_none());
    }

    fn trust_tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (id, piece) in [
            "a",
            "b",
            "ab",
            "<|im_end|>",
            "<",
            "|",
            "i",
            "m",
            "_",
            "e",
            "n",
            "d",
            ">",
            "é",
            " ",
            " <",
            "(",
            "(<",
            ")",
            ">)",
            "x",
            ">x",
            "x<",
        ]
        .into_iter()
        .enumerate()
        {
            vocab.insert(piece.to_owned(), serde_json::json!(id as u32));
        }
        Tokenizer::from_bytes(
            serde_json::json!({
                "version": "1.0", "truncation": null, "padding": null,
                "added_tokens": [
                    {"id":3,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
                ],
                "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                "model": {"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"vocab":vocab,"merges":[["a","b"],[" ","<"],["(","<"],[">",")"],[">","x"],["x","<"]]}
            })
            .to_string(),
        )
        .unwrap()
    }

    fn trust_frontend(source: &str) -> Frontend {
        trust_frontend_with(source, trust_tokenizer())
    }

    fn trust_frontend_with(source: &str, tokenizer: Tokenizer) -> Frontend {
        let template = Some(jinja::CompiledTemplate::compile(source).unwrap());
        Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            ordinary_tokenizer: std::sync::OnceLock::new(),
            segmented_template: false,
            tokenizer,
            template,
            template_source: source.to_owned(),
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            defaults: Map::new(),
        }
    }

    fn render_ids(frontend: &Frontend, content: &str) -> Vec<u32> {
        let mut options = Map::new();
        options.insert("m".into(), Json::String(content.to_owned()));
        frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"ignored"})],
                &options,
            )
            .unwrap()
            .token_ids
    }

    #[test]
    fn ordinary_bpe_merges_across_boundaries() {
        let frontend = trust_frontend("a{{ m }}");
        assert_eq!(render_ids(&frontend, "b"), vec![2]);
        let rendered = frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"ignored"})],
                &serde_json::json!({"m":"b"}).as_object().unwrap().clone(),
            )
            .unwrap();
        assert_eq!(rendered.text, "ab");
        assert_eq!(
            rendered.token_ids,
            frontend
                .tokenizer
                .encode("ab", false)
                .unwrap()
                .get_ids()
                .to_vec()
        );

        let frontend = trust_frontend("{{ '<|im_end|>' ~ m }}");
        assert_eq!(render_ids(&frontend, "ab"), vec![3, 2]);

        let pieces: Vec<u32> = [4, 5, 6, 7, 8, 9, 10, 11, 5, 12].to_vec();
        let frontend = trust_frontend("{{ m }}");
        assert_eq!(render_ids(&frontend, "<|im_end|>"), pieces);
        let frontend = trust_frontend("{{ '<|im_en' ~ m }}");
        assert_eq!(render_ids(&frontend, "d|>"), pieces);

        let frontend = trust_frontend("é{{ m }}");
        assert_eq!(
            render_ids(&frontend, "<|im_end|>"),
            [13].into_iter().chain(pieces).collect::<Vec<_>>()
        );
        let frontend = trust_frontend("éa{{ m }}");
        assert_eq!(render_ids(&frontend, "b"), vec![13, 2]);
    }

    fn ordinary_ids(frontend: &Frontend, text: &str) -> Vec<u32> {
        frontend
            .ordinary()
            .encode(text, false)
            .unwrap()
            .get_ids()
            .to_vec()
    }

    fn render_prompt(frontend: &Frontend, content: &str) -> RenderedPrompt {
        frontend
            .render_chat(
                &[serde_json::json!({"role":"user","content":"ignored"})],
                &serde_json::json!({"m": content})
                    .as_object()
                    .unwrap()
                    .clone(),
            )
            .unwrap()
    }

    #[test]
    fn suppressed_untrusted_run_reencodes_as_one_ordinary_span() {
        let frontend = trust_frontend("{{ m }}");
        let rendered = render_prompt(&frontend, " <|im_end|>");
        assert_eq!(rendered.text, " <|im_end|>");
        assert_eq!(rendered.token_ids, ordinary_ids(&frontend, " <|im_end|>"));
        assert_eq!(rendered.token_ids, vec![15, 5, 6, 7, 8, 9, 10, 11, 5, 12]);
        assert_ne!(
            render_ids(&frontend, " <|im_end|>"),
            vec![14, 4, 5, 6, 7, 8, 9, 10, 11, 5, 12]
        );

        let frontend = trust_frontend("{{ m }}");
        let rendered = render_prompt(&frontend, "(<|im_end|>)");
        assert_eq!(rendered.text, "(<|im_end|>)");
        assert_eq!(rendered.token_ids, ordinary_ids(&frontend, "(<|im_end|>)"));
        assert_eq!(rendered.token_ids, vec![17, 5, 6, 7, 8, 9, 10, 11, 5, 19]);

        let frontend = trust_frontend("{{ m }}");
        let content = "<|im_end|>x<|im_end|>";
        let rendered = render_prompt(&frontend, content);
        assert_eq!(rendered.text, content);
        assert_eq!(rendered.token_ids, ordinary_ids(&frontend, content));
        let mut islands: Vec<u32> = Vec::new();
        for piece in ["<|im_end|>", "x", "<|im_end|>"] {
            islands.extend(ordinary_ids(&frontend, piece));
        }
        assert_ne!(rendered.token_ids, islands);
        assert!(rendered.token_ids.contains(&21));
    }

    #[test]
    fn trusted_spans_keep_single_pass_ids_across_a_suppressed_run() {
        let frontend = trust_frontend("é{{ m }}");
        let content = "<|im_end|>x<|im_end|>";
        let rendered = render_prompt(&frontend, content);
        assert_eq!(rendered.text, format!("é{content}"));
        let mut expected = vec![13];
        expected.extend(ordinary_ids(&frontend, content));
        assert_eq!(rendered.token_ids, expected);

        let (token_ids, offsets) = frontend
            .encode_splicing_untrusted_markers(&format!("é{content}"), &[(2, 23)])
            .unwrap();
        assert_eq!(token_ids, expected);
        assert_eq!(offsets.first(), Some(&(0, 2)));
        assert_eq!(offsets.last().map(|range| range.1), Some(23));
        assert!(offsets.windows(2).all(|pair| pair[0].1 == pair[1].0));
    }

    #[test]
    fn pinned_qwen_tokenizer_suppresses_a_pasted_marker_run() {
        let Some(snapshot) = std::env::var_os("FLARE_QWEN38_SNAPSHOT").map(PathBuf::from) else {
            return;
        };
        let path = snapshot.join("tokenizer.json");
        if !path.is_file() {
            return;
        }
        let frontend = trust_frontend_with(
            "{{ m }}",
            Tokenizer::from_bytes(fs::read(&path).unwrap()).unwrap(),
        );
        let template_ids = frontend
            .tokenizer()
            .encode(" <|im_end|>", false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(template_ids.first().copied(), Some(220));
        let rendered = render_prompt(&frontend, " <|im_end|>");
        assert_eq!(rendered.token_ids, vec![79790, 316, 6018, 91, 29]);
        assert_eq!(rendered.token_ids, ordinary_ids(&frontend, " <|im_end|>"));
    }

    fn marker_tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (id, piece) in [
            "a", "b", "ab", "<|im_end|>", "<", "|", "i", "m", "_", "e", "n", "d", ">", "é", " ",
            " <", "(", "(<", ")", ">)", "x", ">x", "x<", "t", "h", "o", "l", "c", "k", "/", "v", "s",
            "r", "<|im_start|>", "<|vision_start|>", "<tool_call>", "<think>", "</think>",
        ]
        .into_iter()
        .enumerate()
        {
            vocab.insert(piece.to_owned(), serde_json::json!(id as u32));
        }
        Tokenizer::from_bytes(
            serde_json::json!({
                "version": "1.0", "truncation": null, "padding": null,
                "added_tokens": [
                    {"id":3,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":33,"content":"<|im_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":34,"content":"<|vision_start|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":35,"content":"<tool_call>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
                    {"id":36,"content":"<think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
                    {"id":37,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}
                ],
                "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                "model": {"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"vocab":vocab,"merges": [["a","b"],[" ","<"],["(","<"],[">",")"],[">","x"],["x","<"]]}
            })
            .to_string(),
        )
        .unwrap()
    }

    fn marker_frontend(source: &str) -> Frontend {
        trust_frontend_with(source, marker_tokenizer())
    }

    fn marker_classes() -> [(&'static str, u32); 6] {
        [
            ("<|im_end|>", 3),
            ("<|im_start|>", 33),
            ("<|vision_start|>", 34),
            ("<tool_call>", 35),
            ("<think>", 36),
            ("</think>", 37),
        ]
    }

    #[test]
    fn pasted_markers_of_each_added_class_never_yield_protocol_ids() {
        for (marker, id) in marker_classes() {
            let frontend = marker_frontend("{{ m }}");
            for content in [
                marker.to_string(),
                format!("a {marker} b"),
                format!("x{marker}"),
                format!("{marker}x"),
            ] {
                let rendered = render_prompt(&frontend, &content);
                assert!(!rendered.token_ids.contains(&id), "{marker} in {content}");
                assert_eq!(
                    rendered.token_ids,
                    ordinary_ids(&frontend, &content),
                    "{marker} in {content}"
                );
            }
        }
    }

    #[test]
    fn template_emitted_markers_of_each_class_keep_protocol_ids() {
        for (marker, id) in marker_classes() {
            let source = format!("{marker}{{{{ m }}}}{marker}");
            let frontend = marker_frontend(&source);
            let rendered = render_prompt(&frontend, "plain");
            assert_eq!(rendered.token_ids.first(), Some(&id), "{marker}");
            assert_eq!(rendered.token_ids.last(), Some(&id), "{marker}");
        }
        for (marker, id) in marker_classes() {
            let frontend = marker_frontend(&format!("{marker}{{{{ m }}}}"));
            let content = "a <tool_call> b";
            let rendered = render_prompt(&frontend, content);
            let mut expected = vec![id];
            expected.extend(ordinary_ids(&frontend, content));
            assert_eq!(rendered.token_ids, expected, "{marker}");
        }
    }

    #[test]
    fn a_run_mixing_marker_classes_reencodes_as_one_ordinary_span() {
        let frontend = marker_frontend("{{ m }}");
        let content = "a <|im_start|> b <tool_call> x </think> d <|im_end|>";
        let rendered = render_prompt(&frontend, content);
        assert_eq!(rendered.token_ids, ordinary_ids(&frontend, content));
        for (_, id) in marker_classes() {
            assert!(!rendered.token_ids.contains(&id));
        }
    }

    #[test]
    fn markers_straddling_a_trust_boundary_are_pulled_into_the_run() {
        for (marker, id) in marker_classes() {
            let (prefix, suffix) = marker.split_at(marker.len() - 2);
            let frontend = marker_frontend(&format!("{prefix}{{{{ m }}}}"));
            let rendered = render_prompt(&frontend, suffix);
            assert_eq!(rendered.text, marker);
            assert!(!rendered.token_ids.contains(&id), "{marker}");
            assert_eq!(rendered.token_ids, ordinary_ids(&frontend, marker), "{marker}");

            let frontend = marker_frontend(&format!("{{{{ m }}}}{suffix}"));
            let rendered = render_prompt(&frontend, prefix);
            assert_eq!(rendered.text, marker);
            assert!(!rendered.token_ids.contains(&id), "{marker}");
            assert_eq!(rendered.token_ids, ordinary_ids(&frontend, marker), "{marker}");
        }
    }

    #[test]
    fn pasted_marker_between_trusted_markers_is_suppressed() {
        for (marker, id) in marker_classes() {
            let frontend = marker_frontend(&format!("{marker}{{{{ m }}}}{marker}"));
            let rendered = render_prompt(&frontend, marker);
            let mut expected = vec![id];
            expected.extend(ordinary_ids(&frontend, marker));
            expected.push(id);
            assert_eq!(rendered.token_ids, expected, "{marker}");
        }
    }

    #[test]
    fn pinned_qwen_tokenizer_suppresses_every_added_token_class() {
        let Some(snapshot) = std::env::var_os("FLARE_QWEN38_SNAPSHOT").map(PathBuf::from) else {
            return;
        };
        let path = snapshot.join("tokenizer.json");
        if !path.is_file() {
            return;
        }
        let bytes = fs::read(&path).unwrap();
        let frontend = trust_frontend_with("{{ m }}", Tokenizer::from_bytes(&bytes).unwrap());
        let mut inventory: Vec<(u32, String)> = frontend
            .tokenizer()
            .get_added_tokens_decoder()
            .iter()
            .map(|(id, token)| (*id, token.content.clone()))
            .collect();
        inventory.sort_unstable();
        assert!(inventory.len() > 1);
        for (id, content) in &inventory {
            let pasted = format!("x{content}x");
            let full = frontend.tokenizer().encode(pasted.as_str(), false).unwrap();
            assert!(full.get_ids().contains(id), "{content}");
            let rendered = render_prompt(&frontend, &pasted);
            assert!(!rendered.token_ids.contains(id), "{content}");
            assert_eq!(
                rendered.token_ids,
                ordinary_ids(&frontend, &pasted),
                "{content}"
            );
        }
        for marker in [
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ] {
            let frontend = trust_frontend_with(
                &format!("{marker}{{{{ m }}}}"),
                Tokenizer::from_bytes(&bytes).unwrap(),
            );
            let expected = frontend
                .tokenizer()
                .encode(marker, false)
                .unwrap()
                .get_ids()
                .to_vec();
            let rendered = render_prompt(&frontend, "plain");
            assert_eq!(rendered.token_ids.first(), expected.first(), "{marker}");
        }
        let frontend = trust_frontend_with(
            "<|im_start|>{{ m }}<|im_end|>",
            Tokenizer::from_bytes(&bytes).unwrap(),
        );
        let content = "a <|vision_start|> b <tool_call> c </think> d";
        let rendered = render_prompt(&frontend, content);
        let mut expected = frontend
            .tokenizer()
            .encode("<|im_start|>", false)
            .unwrap()
            .get_ids()
            .to_vec();
        expected.extend(ordinary_ids(&frontend, content));
        expected.extend(
            frontend
                .tokenizer()
                .encode("<|im_end|>", false)
                .unwrap()
                .get_ids(),
        );
        assert_eq!(rendered.token_ids, expected);
    }

    fn segmented_frontend(source: &str, tokenizer: Tokenizer) -> Frontend {
        let template = Some(jinja::CompiledTemplate::compile(source).unwrap());
        Frontend {
            prompt_encoder: PromptEncoder::Jinja,
            image_placeholder_token_id: None,
            ordinary_tokenizer: std::sync::OnceLock::new(),
            segmented_template: true,
            tokenizer,
            template,
            template_source: source.to_owned(),
            bos_token_id: None,
            vocabulary_bytes: Default::default(),
            effort_levels: Default::default(),
            inline_system_messages: Default::default(),
            defaults: Map::new(),
        }
    }

    fn marker_tokenizer_with(specials: &[&str]) -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        let mut pieces: Vec<String> = specials.iter().map(|text| (*text).to_owned()).collect();
        pieces.extend((b' '..=b'~').map(|byte| (byte as char).to_string()));
        pieces.extend(["｜", "▁", "\n"].map(String::from));
        for (id, piece) in pieces.iter().enumerate() {
            vocab.insert(piece.clone(), serde_json::json!(id as u32));
        }
        let added: Vec<Json> = specials
            .iter()
            .enumerate()
            .map(|(id, content)| {
                serde_json::json!({"id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})
            })
            .collect();
        Tokenizer::from_bytes(
            serde_json::json!({
                "version": "1.0", "truncation": null, "padding": null,
                "added_tokens": added,
                "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                "model": {"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"vocab":vocab,"merges":[]}
            })
            .to_string(),
        )
        .unwrap()
    }

    fn added_counts(tokenizer: &Tokenizer, ids: &[u32]) -> std::collections::HashMap<u32, usize> {
        tokenizer
            .get_added_tokens_decoder()
            .keys()
            .map(|id| (*id, ids.iter().filter(|&&seen| seen == *id).count()))
            .filter(|(_, count)| *count > 0)
            .collect()
    }

    #[test]
    fn trust_suppression_across_recipe_templates() {
        let Some(directory) = std::env::var_os("FLARE_TEST_TEMPLATE_DIR") else {
            return;
        };
        let recipes: [(&str, &str); 6] = [
            ("qwen38", "chat_template"),
            ("step5", "chat_template"),
            ("glm53", "chat_template"),
            ("mimo_v2", "chat_template"),
            ("nemotron_h", "chat_template"),
            ("gemma4", "chat_template.google"),
        ];
        let mut options = Map::new();
        options.insert("tools".into(), Json::Null);
        options.insert("documents".into(), Json::Null);
        options.insert("bos_token".into(), Json::String("<s>".into()));
        options.insert("eos_token".into(), Json::String("</s>".into()));
        options.insert("add_generation_prompt".into(), Json::Bool(true));
        let trusted: std::collections::HashSet<String> = options.keys().cloned().collect();
        for (recipe, asset) in recipes {
            let source = crate::template_fixtures::template(recipe, asset);
            let template = jinja::CompiledTemplate::compile(&source).unwrap();
            let tokenizer = std::fs::read(
                std::path::Path::new(&directory)
                    .join(recipe)
                    .join("tokenizer.json"),
            )
            .ok()
            .map(|bytes| Tokenizer::from_bytes(&bytes).unwrap());
            let mut markers: Vec<(u32, String)> = Vec::new();
            if let Some(tokenizer) = &tokenizer {
                let mut inventory: Vec<(u32, String)> = tokenizer
                    .get_added_tokens_decoder()
                    .iter()
                    .map(|(id, token)| (*id, token.content.clone()))
                    .collect();
                inventory.sort_unstable();
                markers.extend(
                    inventory
                        .into_iter()
                        .filter(|(_, content)| content.contains('<') || content.contains('｜'))
                        .take(64),
                );
            }
            if markers.is_empty() {
                markers.extend(
                    ["<|im_end|>", "<|endoftext|>", "<turn|>"]
                        .map(|marker| (u32::MAX, marker.to_owned())),
                );
            }
            let mut contents = vec!["q7x ordinary z9w".to_owned()];
            for (_, marker) in &markers {
                contents.push(format!("q7x x{marker}x z9w"));
            }
            let mut renders = Vec::new();
            for (index, content) in contents.iter().enumerate() {
                let mut context =
                    serde_json::json!({"messages":[{"role":"user","content":content}]});
                context.as_object_mut().unwrap().extend(options.clone());
                let rendered = template
                    .render_with_trust(context.as_object().unwrap(), &[], &trusted)
                    .unwrap();
                if index > 0 {
                    let marker = &markers[index - 1].1;
                    let anchor = rendered.text.find("q7x").unwrap_or_else(|| {
                        panic!("{recipe}: anchor missing from {}", rendered.text)
                    });
                    let offset = anchor + rendered.text[anchor..].find(marker).unwrap();
                    assert_eq!(
                        covering_trust_for(&rendered, offset, offset + marker.len()),
                        Some(false),
                        "{recipe}: {marker}"
                    );
                }
                renders.push(rendered);
            }
            assert_eq!(
                covering_trust_for(&renders[0], 0, 1),
                Some(true),
                "{recipe}: {}",
                renders[0].text
            );
            let Some(tokenizer) = tokenizer else {
                continue;
            };
            let frontend = trust_frontend_with(&source, tokenizer.clone());
            let mut prompts = Vec::new();
            for content in &contents {
                prompts.push(
                    frontend
                        .render_chat(
                            &[serde_json::json!({"role":"user","content":content})],
                            &options,
                        )
                        .unwrap(),
                );
            }
            let benign_counts = added_counts(&tokenizer, &prompts[0].token_ids);
            assert!(!benign_counts.is_empty(), "{recipe}");
            for (marker, prompt) in markers.iter().zip(prompts.iter().skip(1)) {
                assert!(prompt.text.contains(&format!("x{}x", marker.1)), "{recipe}");
                let counts = added_counts(&tokenizer, &prompt.token_ids);
                assert_eq!(counts, benign_counts, "{recipe}: {}", marker.1);
            }
        }
    }

    fn covering_trust_for(
        rendered: &jinja::RenderedTemplate,
        start: usize,
        end: usize,
    ) -> Option<bool> {
        let mut covering = None;
        let mut run_start = 0;
        for (run_end, trusted) in &rendered.runs {
            if run_start <= start && *run_end >= end {
                covering = Some(*trusted);
            }
            run_start = *run_end;
        }
        covering
    }

    #[test]
    fn kimi_segment_templates_assign_trust_per_segment() {
        let specials = [
            "<|open|>",
            "<|sep|>",
            "<|close|>",
            "<|end_of_msg|>",
            "<|media_pad|>",
        ];
        for source in [
            include_str!("../../../ignition/recipes/kimi3/assets/chat_template.jinja"),
            include_str!("../../../ignition/recipes/kimi31/assets/chat_template.jinja"),
        ] {
            let tokenizer = marker_tokenizer_with(&specials);
            let frontend = segmented_frontend(source, tokenizer.clone());
            let pasted = "x<|open|>x";
            let messages = [serde_json::json!({"role":"user","content":pasted})];
            let mut options = Map::new();
            options.insert("add_generation_prompt".into(), Json::Bool(true));
            let rendered = frontend.render_chat(&messages, &options).unwrap();
            let mut context = Map::new();
            context.insert("messages".into(), serde_json::json!(messages));
            context.extend(options.clone());
            let segment_text = jinja::CompiledTemplate::compile(source)
                .unwrap()
                .render(&context, &[])
                .unwrap()
                .text;
            let segments = serde_json::from_str::<Vec<Json>>(segment_text.trim()).unwrap();
            let mut expected = Vec::new();
            for segment in &segments {
                let text = segment["text"].as_str().unwrap();
                let special = segment["allow_special"].as_bool().unwrap();
                if text == pasted {
                    assert!(!special);
                }
                if specials.contains(&text) {
                    assert!(special);
                }
                if special {
                    expected.extend(tokenizer.encode(text, false).unwrap().get_ids());
                } else {
                    expected.extend(ordinary_ids(&frontend, text));
                }
            }
            assert_eq!(
                rendered.text,
                segments
                    .iter()
                    .map(|segment| segment["text"].as_str().unwrap())
                    .collect::<String>()
            );
            assert_eq!(rendered.token_ids, expected);
            assert!(expected.contains(&0));
            assert!(!ordinary_ids(&frontend, pasted).contains(&0));
        }
    }

    #[test]
    fn verified_asset_rejects_content_changed_after_fingerprinting() {
        let directory = std::env::temp_dir().join(format!("flare-asset-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("tokenizer_config.json");
        let original = b"{\"eos_token\":\"<eos>\"}";
        fs::write(&path, original).unwrap();
        let asset = Asset {
            range: None,
            path: path.clone(),
            sha256: format!("{:x}", Sha256::digest(original)),
        };
        assert_eq!(asset.read().unwrap(), original);
        fs::write(&path, b"{\"eos_token\":\"changed\"}").unwrap();
        assert!(matches!(
            asset.read().unwrap_err(),
            Error::Frontend(FrontendError::Asset(_))
        ));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn image_segment_indices_reorder_sources_and_hints_and_reject_ambiguity() {
        let mut hints = Vec::new();
        let segments = token_segments(r#"[{"text":"second","allow_special":true,"image_index":1},{"text":"first","allow_special":true,"image_index":0}]"#, &mut hints).unwrap();
        let mut sources = vec!["first".into(), "second".into()];
        let mut checkpoints = vec![0];
        order_segment_images(&segments, &mut sources, &mut checkpoints).unwrap();
        assert_eq!(sources, ["second", "first"]);
        assert_eq!(checkpoints, [1]);
        for indices in [vec![0,0], vec![0,2], vec![1]] {
            let segments = indices.into_iter().map(|index| TokenSegment {
                text: "image".into(), allow_special:true, image_index:Some(index),
            }).collect::<Vec<_>>();
            assert!(order_segment_images(&segments, &mut sources, &mut checkpoints).is_err());
        }
        let segments = token_segments(r#"[{"text":"legacy","allow_special":true}]"#, &mut hints).unwrap();
        order_segment_images(&segments, &mut sources, &mut checkpoints).unwrap();
        assert_eq!(sources, ["second", "first"]);
        assert_eq!(checkpoints, [1]);
    }

    #[test]
    fn segment_hints_survive_json_escapes_key_order_and_nested_serialization() {
        use jinja::{CompiledTemplate, Mark};
        use serde_json::json;
        let context = json!({"messages":[{"content":[{"text":"é\n\"\\😀"}]}]});
        for ascii in ["true", "false"] {
            for sort in ["true", "false"] {
                let source = format!(
                    concat!(
                        "{{% set history = messages[0].content[0].text %}}",
                        "{{% set segments = [{{'text': 'prefix' + history + 'tail', 'allow_special': false}},",
                        "{{'text': history | tojson(ensure_ascii={0}), 'allow_special': true}}] %}}",
                        "{{{{ segments | tojson(ensure_ascii={0}, sort_keys={1}, indent=2) }}}}",
                    ),
                    ascii, sort
                );
                let template = CompiledTemplate::compile(&source).unwrap();
                let plain = template.render(context.as_object().unwrap(), &[]).unwrap();
                let marked = template
                    .render(context.as_object().unwrap(), &[(0, 1)])
                    .unwrap();
                assert_eq!(plain.text, marked.text);
                let mut hints = marked
                    .marks
                    .into_iter()
                    .map(|mark| match mark {
                        Mark::Hint(byte) => byte,
                        _ => panic!("unexpected mark"),
                    })
                    .collect();
                let segments = token_segments(&marked.text, &mut hints).unwrap();
                assert_eq!(segments[0].text, "prefixé\n\"\\😀tail");
                assert!(!segments[0].allow_special);
                assert!(segments[1].allow_special);
                assert_eq!(
                    hints,
                    [
                        "prefixé\n\"\\😀".len(),
                        segments[0].text.len() + segments[1].text.len() - 1
                    ]
                );
            }
        }
        let template = CompiledTemplate::compile(
            "{{ {'text': messages[0].content[0].text, 'text': 'x'} | tojson }}",
        )
        .unwrap();
        let rendered = template
            .render(context.as_object().unwrap(), &[(0, 1)])
            .unwrap();
        assert_eq!(rendered.text, "{\"text\": \"x\"}");
        assert!(rendered.marks.is_empty());
    }
}
