/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The GLM MoE DSA text frontend against its checkpoint template rendered by Transformers.

#![cfg(feature = "server")]

#[path = "support/templates.rs"]
mod template_fixtures;

// The shared support module addresses these through the test crate's root.
use flare_helios::frontend;
use flare_helios::{model, scheduler};
#[path = "support/frontend.rs"]
mod support;

use flare_helios::{
    frontend::{Asset, Frontend, FrontendAssets},
    grammar::GrammarCompiler,
    model::{ModelConfig, SamplingParameters},
    output::{Event, Input, Termination},
    scheduler::SchedulerLimits,
    server::prepare_request,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[test]
#[ignore = "requires FLARE_GLM_DSA_TEST_SNAPSHOT, FLARE_TEST_TEMPLATE_DIR and FLARE_GLM_DSA_FRONTEND_REFERENCE from generate_glm_fixtures.py --recipe glm_dsa --text-only"]
fn checkpoint_template_tokenization_requests_and_message_boundaries_match_transformers() {
    // The container keeps the tokenizer and template in its metadata directory.
    let snapshot = PathBuf::from(std::env::var_os("FLARE_GLM_DSA_TEST_SNAPSHOT").unwrap());
    let asset = |name: &str| {
        let path = snapshot.join(name);
        Asset {
            range: None,
            sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
            path,
        }
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokens: support::tokens(&asset("tokenizer_config.json")).unwrap(),
            chat_template: Some({
                let path = template_fixtures::path("glm_dsa", "chat_template");
                let contents = fs::read_to_string(&path).unwrap();
                assert_eq!(
                    contents.replace("{%- mark -%}", ""),
                    fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap()
                );
                Asset {
                    range: None,
                    sha256: format!("{:x}", Sha256::digest(contents.as_bytes())),
                    path,
                }
            }),
        },
        json!({"add_generation_prompt":true})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let config = ModelConfig {
        completion_token_capacity: 8,
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 1048576,
        contract: support::contract("glm_dsa"),
        identity: [9; 32],
        model_id: "glm_dsa".into(),
        limits: SchedulerLimits {
            request_slots: 4,
            max_active: 4,
            max_sequence_tokens: 1048576,
            prefill_chunk_size: 4096,
            pages: 4096,
            snapshots: 16,
        },
        eos_token_ids: vec![154820, 154827, 154829],
        sampling_ranges: Default::default(),
        sampling_defaults: SamplingParameters::default(),
        max_new_tokens: 128,
        speculative_widths: vec![],
        speculative_fixed_width: None,
        speculative_token_capacity: 0,
        decode_weight_bytes: 0,
        history_drafts: false,
        speculative_proposal_source: Default::default(),
    };
    let delimiters =
        model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend).unwrap();
    assert_eq!(
        delimiters,
        [
            (154841, "<think>".to_owned()),
            (154842, "</think>".to_owned())
        ]
    );
    let compiler = GrammarCompiler::new(
        frontend.tokenizer(),
        154880,
        &config.eos_token_ids,
        &delimiters,
    )
    .unwrap();
    let reference: Value = serde_json::from_slice(
        &fs::read(std::env::var_os("FLARE_GLM_DSA_FRONTEND_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prepared = prepare_request(&frontend, None, &compiler, &config, case["body"].clone())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        model::admission::validate_prompt(&config, &prepared.prompt).unwrap();
        assert_eq!(
            prepared.rendered_prompt,
            case["rendered"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            serde_json::to_value(&prepared.prompt.tokens).unwrap(),
            case["token_ids"],
            "{name}"
        );
        assert_eq!(
            serde_json::to_value(&prepared.prompt.message_boundaries).unwrap(),
            case["message_boundaries"],
            "{name}"
        );
        let reasoning =
            prepared.output.content.start == flare_helios::output::tagged::OutputStart::Reasoning;
        assert_eq!(reasoning, case["reasoning"].as_bool().unwrap(), "{name}");
        assert!(!prepared.output.content.structured, "{name}");
        assert!(prepared.prompt.grammar.is_none(), "{name}");
        let mut output = flare_helios::output::OutputPipeline::new(prepared.output);
        let mut events = Vec::new();
        if reasoning {
            events.extend(output.push(Input::Text("analysis")).unwrap());
            events.extend(output.push(Input::Token(154842)).unwrap());
            events.extend(output.push(Input::Text("</think>")).unwrap());
        }
        events.extend(output.push(Input::Text("result")).unwrap());
        events.extend(output.finish(Termination::Eos).unwrap());
        assert_eq!(
            events.last(),
            Some(&Event::Content("result".into())),
            "{name}"
        );
        assert_eq!(
            events.contains(&Event::Reasoning("analysis".into())),
            reasoning,
            "{name}"
        );
        if name == "parallel_tool_results" {
            // The rendered assistant turn parses back into the calls that produced it.
            let text = prepared
                .rendered_prompt
                .split_once("<|assistant|>")
                .unwrap()
                .1
                .split_once("<|observation|>")
                .unwrap()
                .0;
            let tools = case["body"]["chat_template_kwargs"]["tools"]
                .as_array()
                .unwrap()
                .clone();
            let mut output =
                flare_helios::output::OutputPipeline::new(flare_helios::output::OutputConfig {
                    syntax: support::contract("glm_dsa").frontend,
                    control_tokens: vec![],
                    content: flare_helios::output::tagged::OutputConfig {
                        structured: false,
                        start: flare_helios::output::tagged::OutputStart::Content,
                        tools,
                    },
                });
            let mut events = Vec::new();
            for character in text.chars() {
                events.extend(output.push(Input::Text(&character.to_string())).unwrap());
            }
            events.extend(output.finish(Termination::Eos).unwrap());
            let arguments: Vec<Value> = events
                .iter()
                .filter_map(|event| match event {
                    Event::ToolArguments { text, .. } => Some(serde_json::from_str(text).unwrap()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                arguments,
                [
                    json!({"city":"Paris α", "count":2}),
                    json!({"city":"London", "count":1})
                ]
            );
        }
    }
    // The checkpoint has no vision encoder: image parts are rejected instead of rendered.
    let request = json!({"model":"glm_dsa", "messages":[{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}}]}]});
    assert!(prepare_request(&frontend, None, &compiler, &config, request).is_err());
    let structured = prepare_request(
        &frontend,
        None,
        &compiler,
        &config,
        json!({"model":"glm_dsa", "messages":[{"role":"user", "content":"JSON please"}], "response_format":{"type":"json_object"}}),
    )
    .unwrap();
    assert!(
        structured.output.content.start == flare_helios::output::tagged::OutputStart::Reasoning
    );
    assert!(structured.output.content.structured);
    assert!(structured.prompt.grammar.is_some());
}
