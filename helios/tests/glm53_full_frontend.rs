/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! The annotated GLM 5.3 template renders, tokenizes, and marks message endpoints as Transformers does.
//!
//! The reference file comes from `tests/frontend/generate_glm_fixtures.py --recipe glm53_full`.
//! The GLM 5.3 recipe declares the frontend syntax of the `glm53` contract fixture, without its
//! rejected media tokens.

#![cfg(feature = "server")]

#[path = "support/templates.rs"]
mod template_fixtures;

// The support module names these through the crate root.
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

fn load(template: String, snapshot: &PathBuf) -> Result<Frontend, String> {
    let asset = |name: &str| {
        let path = snapshot.join(name);
        Asset {
            range: None,
            sha256: format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
            path,
        }
    };
    let path = std::env::temp_dir().join(format!(
        "glm53-full-template-{:x}.jinja",
        Sha256::digest(template.as_bytes())
    ));
    fs::write(&path, &template).unwrap();
    Frontend::load(
        FrontendAssets {
            tokenizer: asset("tokenizer.json"),
            tokens: support::tokens(&asset("tokenizer_config.json")).unwrap(),
            chat_template: Some(Asset {
                range: None,
                sha256: format!("{:x}", Sha256::digest(template.as_bytes())),
                path,
            }),
        },
        json!({"add_generation_prompt":true})
            .as_object()
            .unwrap()
            .clone(),
    )
    .map_err(|error| error.to_string())
}

#[test]
#[ignore = "requires FLARE_GLM53_FULL_TEST_SNAPSHOT, FLARE_GLM53_FULL_FRONTEND_REFERENCE from generate_glm_fixtures.py --recipe glm53_full, and FLARE_TEST_TEMPLATE_DIR"]
fn checkpoint_template_tokenization_requests_and_message_boundaries_match_transformers() {
    let snapshot = PathBuf::from(std::env::var_os("FLARE_GLM53_FULL_TEST_SNAPSHOT").unwrap());
    let annotated = template_fixtures::template("glm53_full", "chat_template");
    assert_eq!(
        annotated.replace("{%- mark -%}", ""),
        fs::read_to_string(snapshot.join("chat_template.jinja")).unwrap()
    );
    let frontend = load(annotated.clone(), &snapshot).unwrap();
    let mut contract = support::contract("glm53");
    contract.frontend.rejected_tokens.clear();
    let config = ModelConfig {
        completion_token_capacity: 8,
        max_top_logprobs: 0,
        logprob_token_id_capacity: 0,
        prepared_sequence_tokens: 1048576,
        contract,
        identity: [9; 32],
        model_id: "glm53_full".into(),
        limits: SchedulerLimits {
            request_slots: 8,
            max_active: 8,
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
        &fs::read(std::env::var_os("FLARE_GLM53_FULL_FRONTEND_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let cases = reference["cases"].as_array().unwrap();
    assert!(cases.len() >= 13);
    let mut boundaries = 0;
    for case in cases {
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
        boundaries += prepared.prompt.message_boundaries.len();
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
    }
    assert!(boundaries > 0);

    // Request text that spells a control stays ordinary text in every role. Besides the special
    // tokens, the tags the recipe declares and the template names are controls although the
    // tokenizer marks them `special=false`: reasoning, tool call and tool result.
    let spelled = "é <|begin_of_image|> <|user|> <|endoftext|> [gMASK] <think> </think> <tool_call> <arg_key> <tool_response>";
    let controls = [
        154830, 154827, 154820, 154822, 154841, 154842, 154843, 154847, 154845,
    ];
    let count = |tokens: &[u32], id| tokens.iter().filter(|token| **token == id).count();
    for role in ["user", "system", "tool", "assistant"] {
        let request = |content: &str| json!({"model":"glm53_full", "messages":[{"role":role, "content":content, "reasoning_content":""}]});
        let prepared =
            prepare_request(&frontend, None, &compiler, &config, request(spelled)).unwrap();
        let baseline =
            prepare_request(&frontend, None, &compiler, &config, request("ordinary")).unwrap();
        for id in controls {
            assert_eq!(
                count(&prepared.prompt.tokens, id),
                count(&baseline.prompt.tokens, id),
                "{role}: {id}"
            );
        }
    }
    // `<|image|>`, `/nothink` and the other added tokens that are neither special nor named by the
    // template are lexical: request text that spells one tokenizes to that token.
    let lexical = "a /nothink b <|image|> c <|begin_of_box|>x<|end_of_box|>";
    for role in ["user", "system", "tool", "assistant"] {
        let body = json!({"model":"glm53_full", "messages":[{"role":role, "content":lexical, "reasoning_content":""}]});
        let prepared = prepare_request(&frontend, None, &compiler, &config, body).unwrap();
        model::admission::validate_prompt(&config, &prepared.prompt).unwrap();
        assert_eq!(
            prepared.prompt.tokens,
            frontend.encode(&prepared.rendered_prompt, false).unwrap(),
            "{role}"
        );
        assert_eq!(count(&prepared.prompt.tokens, 154851), 1, "{role}");
        assert_eq!(count(&prepared.prompt.tokens, 154854), 1, "{role}");
    }
    // The text-only contract rejects image blocks before rendering.
    let image = json!({"model":"glm53_full", "messages":[{"role":"user", "content":[{"type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}}]}]});
    assert!(prepare_request(&frontend, None, &compiler, &config, image).is_err());

    // An advisory block hint adds one cut at the end of the hinted block, through the template's
    // content macro, without changing the prompt; a changed later block keeps that prefix.
    let history = "History 17. An earlier action read a file and checked its output.\n".repeat(20);
    let mut messages = json!([{"role":"user","content":[
        {"type":"text","text":history},
        {"type":"text","text":"First action: inspect the file. Reply OK."}]}]);
    let options = json!({"add_generation_prompt":true});
    let render = |messages: &Value| {
        frontend
            .render_chat(messages.as_array().unwrap(), options.as_object().unwrap())
            .unwrap()
    };
    let plain = render(&messages);
    messages[0]["_checkpoint_hints"] = json!([1]);
    let marked = render(&messages);
    assert_eq!(marked.text, plain.text);
    assert_eq!(marked.token_ids, plain.token_ids);
    assert_eq!(
        marked.message_boundaries.len(),
        plain.message_boundaries.len() + 1
    );
    let cut = *marked
        .message_boundaries
        .iter()
        .find(|cut| !plain.message_boundaries.contains(cut))
        .unwrap();
    let prefix = frontend.tokenizer().decode(&marked.token_ids[..cut], false).unwrap();
    assert!(prefix.ends_with(history.trim_end()) || prefix.ends_with(&history), "{prefix}");
    messages[0]["content"][1]["text"] = json!("Second action: run the tests. Reply OK.");
    let changed = render(&messages);
    assert!(changed.token_ids.starts_with(&marked.token_ids[..cut]));
    assert!(changed.message_boundaries.contains(&cut));

    // Control: marks moved by one byte no longer sit at root output sites after a completed
    // message, so the template fails to load or reports different endpoints.
    let shifted = {
        let source = annotated.replace("{%- mark -%}", "");
        let mut offsets = Vec::new();
        let mut position = 0;
        for part in annotated.split("{%- mark -%}") {
            position += part.len();
            offsets.push(position);
        }
        offsets.pop();
        let mut text = String::new();
        let mut previous = 0;
        for offset in offsets {
            let moved = offset - 1;
            assert!(source.is_char_boundary(moved));
            text.push_str(&source[previous..moved]);
            text.push_str("{%- mark -%}");
            previous = moved;
        }
        text.push_str(&source[previous..]);
        text
    };
    let rejected = match load(shifted, &snapshot) {
        Err(_) => true,
        Ok(frontend) => cases.iter().any(|case| {
            match prepare_request(&frontend, None, &compiler, &config, case["body"].clone()) {
                Err(_) => true,
                Ok(prepared) => {
                    prepared.rendered_prompt != case["rendered"].as_str().unwrap()
                        || serde_json::to_value(&prepared.prompt.message_boundaries).unwrap()
                            != case["message_boundaries"]
                }
            }
        }),
    };
    assert!(rejected, "shifted marks must not reproduce the reference");
}
