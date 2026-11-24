/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! ChatML reasoning delimiters and function/parameter-tag tool calls.

pub use crate::output::tagged::OutputConfig;
use crate::{
    Result,
    output::tagged::{RawArguments, TaggedOutput, ToolDialect, strict_nonstring_raw},
};
use serde_json::Value;

pub struct FunctionTags<const WRAPPING_NEWLINES: bool>;
impl<const WRAPPING_NEWLINES: bool> ToolDialect for FunctionTags<WRAPPING_NEWLINES> {
    const NAME: &'static str = "function-tag";
    const TRIM_NEWLINES: bool = WRAPPING_NEWLINES;
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";
    const FUNCTION: Option<(&'static str, &'static str)> = Some(("<function=", "</function>"));
    const TOOL_ENDS_REASONING: bool = true;
    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        Ok(vec![parse_tool(body, tools, WRAPPING_NEWLINES)?])
    }
    fn parse_candidate(body: &str, tools: &[Value]) -> Result<Option<Vec<(String, String)>>> {
        Ok(parse_tool_candidate(body, tools, WRAPPING_NEWLINES)?.map(|call| vec![call]))
    }
    fn starts_call(text: &str) -> Option<bool> {
        // Calls continue with `<function=NAME>` and a parameter or closing tag; quoted
        // or inline tags followed by prose remain text.
        let text = text.strip_prefix(Self::OPEN).map_or(text, str::trim_start);
        let Some(header) = text.strip_prefix("<function=") else {
            return if "<function=".starts_with(text) {
                None
            } else {
                Some(false)
            };
        };
        let end = header.find(|c: char| c == '>' || c == '<' || c.is_whitespace())?;
        if end == 0 {
            return Some(false);
        }
        let Some(body) = header[end..].strip_prefix('>') else {
            return Some(false);
        };
        let body = body.trim_start();
        if body.is_empty() {
            return None;
        }
        if parameter_start(body).is_some() || body.starts_with("</function>") {
            Some(true)
        } else if possible_parameter_start(body) || "</function>".starts_with(body) {
            None
        } else {
            Some(false)
        }
    }
}

/// Whether incomplete `text` can still become a parameter opener accepted by
/// `parameter_start`.
fn possible_parameter_start(text: &str) -> bool {
    let Some(text) = text.strip_prefix('<') else {
        return false;
    };
    let text = text.trim_start();
    if "parameter".starts_with(text) {
        return true;
    }
    let Some(text) = text.strip_prefix("parameter") else {
        return false;
    };
    let text = text.trim_start();
    text.is_empty() || text.starts_with('=')
}

pub type ChatMlOutput = TaggedOutput<FunctionTags<true>>;
pub type CompactChatMlOutput = TaggedOutput<FunctionTags<false>>;

fn parse_tool(body: &str, tools: &[Value], wrapped_parameters: bool) -> Result<(String, String)> {
    parse_tool_candidate(body, tools, wrapped_parameters)?
        .ok_or_else(|| "incomplete JSON parameter".into())
}

fn parse_tool_candidate(
    body: &str,
    tools: &[Value],
    wrapped_parameters: bool,
) -> Result<Option<(String, String)>> {
    let body = body
        .trim()
        .strip_prefix("<function=")
        .ok_or("tool call requires a function tag")?;
    let (name, mut rest) = body.split_once('>').ok_or("unterminated function name")?;
    if name.is_empty() {
        return Err("tool function name is empty".into());
    }
    let function = tools
        .iter()
        .filter_map(|tool| tool.get("function"))
        .find(|function| function.get("name").and_then(Value::as_str) == Some(name));
    let strict = function.is_some_and(|f| f["strict"].as_bool() == Some(true));
    let properties = function
        .and_then(|function| function.get("parameters"))
        .and_then(|parameters| parameters.get("properties"))
        .and_then(Value::as_object);
    let mut arguments = RawArguments::new();
    loop {
        rest = rest.trim_start();
        if let Some(tail) = rest.strip_prefix("</function>") {
            if !tail.trim().is_empty() {
                return Err("unexpected content after tool function".into());
            }
            break;
        }
        let (parameter, tail) =
            parameter_start(rest).ok_or("tool function requires parameter or closing tag")?;
        if parameter.is_empty() || arguments.contains_key(parameter) {
            return Err("tool parameter names must be nonempty and unique".into());
        }
        let schema = properties.and_then(|properties| properties.get(parameter));
        let Ok(container) = super::tagged::container_parameter(tail, schema.filter(|_| strict))
        else {
            return Ok(None);
        };
        if let Some((value, remaining)) = container
            && let Some(next) = parameter_end(remaining).or_else(|| {
                (remaining.starts_with("</function>") || parameter_start(remaining).is_some())
                    .then_some(remaining)
            })
        {
            arguments.insert(parameter.into(), value);
            rest = next;
            continue;
        }
        let (end, consumed) = tail
            .match_indices('<')
            .find_map(|(at, _)| {
                let suffix = &tail[at..];
                if let Some(rest) = parameter_end(suffix) {
                    Some((at, suffix.len() - rest.len()))
                } else if parameter_start(suffix).is_some() || suffix.starts_with("</function>") {
                    Some((at, 0))
                } else {
                    None
                }
            })
            .ok_or("unterminated tool parameter value")?;
        let (value, tail) = (&tail[..end], &tail[end + consumed..]);
        let value = if wrapped_parameters {
            let value = value.strip_prefix('\n').unwrap_or(value);
            value.strip_suffix('\n').unwrap_or(value)
        } else {
            value
        };
        if let Some(raw_json) = strict_nonstring_raw(value, schema.filter(|_| strict)) {
            arguments.insert(parameter.into(), raw_json);
        } else {
            arguments.insert_value(parameter.into(), &parameter_value(value, schema)?)?;
        }
        rest = tail;
    }
    Ok(Some((name.into(), arguments.to_json_string()?)))
}

fn parameter_start(text: &str) -> Option<(&str, &str)> {
    text.strip_prefix('<')?
        .trim_start()
        .strip_prefix("parameter")?
        .trim_start()
        .strip_prefix('=')?
        .trim_start()
        .split_once('>')
}

fn parameter_end(text: &str) -> Option<&str> {
    text.strip_prefix('<')?
        .trim_start()
        .strip_prefix('/')?
        .trim_start()
        .strip_prefix("parameter")?
        .trim_start()
        .strip_prefix('>')
}

pub(super) fn scalar_parameter_value(raw: &str, schema: Option<&Value>) -> Value {
    let mut kinds = Vec::new();
    parameter_types(schema, &mut kinds);
    let parsed = serde_json::from_str::<Value>(raw);
    let mut value = None;
    for kind in [
        "null", "integer", "number", "boolean", "object", "array", "string",
    ] {
        if !kinds.contains(&kind) {
            continue;
        }
        value = match kind {
            "null" if raw.eq_ignore_ascii_case("null") => Some(Value::Null),
            "integer" => integer_value(raw),
            "number" => parsed
                .as_ref()
                .ok()
                .filter(|value| value.is_number())
                .cloned()
                .or_else(|| integer_value(raw))
                .or_else(|| {
                    raw.trim()
                        .parse::<f64>()
                        .ok()
                        .and_then(serde_json::Number::from_f64)
                        .map(Value::Number)
                }),
            "boolean" => match raw.trim().to_ascii_lowercase().as_str() {
                "true" | "1" => Some(Value::Bool(true)),
                "false" | "0" => Some(Value::Bool(false)),
                _ => None,
            },
            "object" | "array" => parsed.as_ref().ok().cloned(),
            "string" => Some(Value::String(raw.into())),
            _ => None,
        };
        if value.is_some() {
            break;
        }
    }
    // Tool schemas guide tag-to-JSON conversion; non-strict calls retain invalid values.
    value
        .or_else(|| parsed.ok())
        .unwrap_or_else(|| Value::String(raw.into()))
}

pub(super) fn parameter_value(raw: &str, schema: Option<&Value>) -> Result<Value> {
    let mut value = scalar_parameter_value(raw, schema);
    if let Some(schema) = schema {
        match &mut value {
            Value::Object(fields) => {
                for (key, value) in fields {
                    if let Some(property) = schema.get("properties").and_then(|p| p.get(key)) {
                        let raw = match value {
                            Value::String(text) => text.clone(),
                            _ => value.to_string(),
                        };
                        *value = parameter_value(&raw, Some(property))?;
                    }
                }
            }
            Value::Array(items) => {
                if let Some(item_schema) = schema.get("items").filter(|v| v.is_object()) {
                    for value in items {
                        let raw = match value {
                            Value::String(text) => text.clone(),
                            _ => value.to_string(),
                        };
                        *value = parameter_value(&raw, Some(item_schema))?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(value)
}

fn integer_value(raw: &str) -> Option<Value> {
    let raw = raw.trim();
    raw.parse::<i64>()
        .ok()
        .map(Value::from)
        .or_else(|| raw.parse::<u64>().ok().map(Value::from))
}

pub(super) fn parameter_types<'a>(schema: Option<&'a Value>, kinds: &mut Vec<&'a str>) {
    let start = kinds.len();
    if let Some(schema) = schema {
        match schema.get("type") {
            Some(Value::String(kind)) => kinds.push(kind),
            Some(Value::Array(types)) => kinds.extend(types.iter().filter_map(Value::as_str)),
            _ => {}
        }
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            kinds.extend(values.iter().map(|value| match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            }));
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(branches) = schema.get(key).and_then(Value::as_array) {
                for branch in branches {
                    parameter_types(Some(branch), kinds);
                }
            }
        }
    }
    if kinds.len() == start {
        kinds.push("string");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{Input, ModelOutput, Termination};
    use crate::server::output::Accumulator;
    use serde_json::json;
    struct Harness {
        parser: ChatMlOutput,
        output: Accumulator,
    }
    impl Harness {
        fn new(config: OutputConfig, id: String) -> Self {
            Self {
                parser: ChatMlOutput::new(config),
                output: Accumulator::new(id),
            }
        }
        fn push(&mut self, text: &str) -> Result<Vec<Value>> {
            self.output.apply(self.parser.push(Input::Text(text))?)
        }
        fn finish(&mut self) -> Result<Vec<Value>> {
            let result = self.output.apply(self.parser.finish(Termination::Eos)?)?;
            self.output.finish()?;
            Ok(result)
        }
        fn message(&self) -> Value {
            self.output.message()
        }
        fn has_tools(&self) -> bool {
            self.output.has_tools()
        }
    }

    fn config() -> OutputConfig {
        OutputConfig {
            structured: false,
            start: crate::output::tagged::OutputStart::Reasoning,
            tools: vec![
                json!({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"city":{"type":"string"},"n":{"type":"integer"},"data":{"type":"object"}}}}}),
            ],
        }
    }
    fn sample() -> &'static str {
        "café 世界\n</think>\n\nSearching.\n\n<tool_call>\n<function=lookup>\n<parameter=city>\n Paris \n</parameter>\n<parameter=n>\n3\n</parameter>\n<parameter=data>\n{\"flag\":true,\"nothing\":null}\n</parameter>\n</function>\n</tool_call>"
    }
    #[test]
    fn every_utf8_split_preserves_reasoning_tools_and_parameter_types() {
        let mut whole = Harness::new(config(), "x".into());
        whole.push(sample()).unwrap();
        whole.finish().unwrap();
        let expected = whole.message();
        assert_eq!(expected["reasoning_content"], "café 世界");
        assert_eq!(expected["content"], "Searching.");
        let args: Value = serde_json::from_str(
            expected["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            args,
            json!({"city":" Paris ","n":3,"data":{"flag":true,"nothing":null}})
        );
        for split in sample().char_indices().map(|(at, _)| at) {
            let mut parser = Harness::new(config(), "x".into());
            parser.push(&sample()[..split]).unwrap();
            parser.push(&sample()[split..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(parser.message(), expected, "split {split}");
        }
        let mut parser = Harness::new(config(), "x".into());
        for ch in sample().chars() {
            parser.push(&ch.to_string()).unwrap();
        }
        parser.finish().unwrap();
        assert_eq!(parser.message(), expected);
    }
    #[test]
    fn incomplete_empty_and_duplicate_calls_fail() {
        let mut parser = Harness::new(
            OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Content,
                ..config()
            },
            "x".into(),
        );
        parser.push("<tool_call><function=lookup>").unwrap();
        assert!(parser.finish().is_err());
        assert!(parse_tool("<function=></function>", &config().tools, true).is_err());
        assert!(
            parse_tool(
                "<function=lookup><parameter=n>1</parameter><parameter=n>2</parameter></function>",
                &config().tools,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn qwen_function_boundaries_match_vllm_recovery_cases_at_every_split() {
        for text in [
            "<function=lookup><parameter=city>Paris</parameter></function>",
            "<tool_call><function=lookup><parameter=city>Paris</function></tool_call>",
            "<tool_call><function=lookup><parameter=city>Paris<parameter=n>3</function></tool_call>",
            "<tool_call><function=lookup>< parameter = city>Paris< / parameter ></function></tool_call>",
            "<tool_call><function=lookup><parameter=city>Paris</parameter></function><tool_call><function=lookup><parameter=city>Paris</parameter></function></tool_call>",
        ] {
            for split in text.char_indices().map(|(index, _)| index) {
                let mut parser = Harness::new(
                    OutputConfig {
                        start: crate::output::tagged::OutputStart::Content,
                        ..config()
                    },
                    "quirk".into(),
                );
                parser.push(&text[..split]).unwrap();
                parser.push(&text[split..]).unwrap();
                parser.finish().unwrap();
                let message = parser.message();
                let calls = message["tool_calls"].as_array().unwrap();
                assert_eq!(
                    calls.len(),
                    text.matches("<function=").count(),
                    "{text} at {split}"
                );
                for call in calls {
                    let args: Value =
                        serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
                            .unwrap();
                    assert_eq!(args["city"], "Paris");
                    if text.contains("<parameter=n>") {
                        assert_eq!(args["n"], 3);
                    }
                }
            }
        }
    }

    #[test]
    fn qwen_tools_end_reasoning_and_duplicate_think_markers_do_not_leak() {
        for text in [
            "<think>Look up Paris.<tool_call><function=lookup><parameter=city>Paris</parameter></function></tool_call>",
            "Look up Paris.</think></think><tool_call><function=lookup><parameter=city>Paris</parameter></function></tool_call>",
        ] {
            for split in text.char_indices().map(|(index, _)| index) {
                let mut parser = Harness::new(config(), "implicit".into());
                parser.push(&text[..split]).unwrap();
                parser.push(&text[split..]).unwrap();
                parser.finish().unwrap();
                let message = parser.message();
                assert_eq!(
                    message["reasoning_content"], "Look up Paris.",
                    "{text} at {split}"
                );
                assert!(message["content"].is_null() || message["content"] == "");
                assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
            }
        }
    }

    #[test]
    fn qwen_parameter_conversion_prioritizes_types_and_preserves_invalid_values() {
        for (raw, schema, expected) in [
            (
                "3",
                json!({"anyOf":[{"type":"string"},{"type":"integer"}]}),
                json!(3),
            ),
            ("null", json!({"type":["string","null"]}), Value::Null),
            ("TRUE", json!({"type":"boolean"}), json!(true)),
            ("1", json!({"type":"boolean"}), json!(true)),
            ("0003", json!({"type":"integer"}), json!(3)),
            ("wrong", json!({"type":"integer"}), json!("wrong")),
            ("inf", json!({"type":"number"}), json!("inf")),
            ("3", json!({"enum":[1,2,3]}), json!(3)),
            (
                "  exact \\n",
                json!({"type":"string"}),
                json!("  exact \\n"),
            ),
            (
                r#"{"count":"3"}"#,
                json!({"type":"object","properties":{"count":{"type":"integer"}}}),
                json!({"count":3}),
            ),
            ("25000", json!({"type":"number"}), json!(25000)),
            ("1.5", json!({"type":"number"}), json!(1.5)),
            ("2.0", json!({"type":"number"}), json!(2.0)),
            ("1e3", json!({"type":"number"}), json!(1000.0)),
            ("0003", json!({"type":"number"}), json!(3)),
            (
                r#"{"n":25000}"#,
                json!({"type":"object","properties":{"n":{"type":"number"}}}),
                json!({"n":25000}),
            ),
            (
                r#"[1,2.5]"#,
                json!({"type":"array","items":{"type":"number"}}),
                json!([1, 2.5]),
            ),
        ] {
            assert_eq!(
                parameter_value(raw, Some(&schema)).unwrap(),
                expected,
                "{raw} / {schema}"
            );
        }
    }

    #[test]
    fn integral_number_arguments_serialize_without_fraction() {
        let mut parser = Harness::new(
            OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Content,
                tools: vec![
                    json!({"type":"function","function":{"name":"shell","parameters":{"type":"object","properties":{"command":{"type":"string"},"yield_time_ms":{"type":"number"}}}}}),
                ],
            },
            "number".into(),
        );
        parser
            .push("<tool_call>\n<function=shell>\n<parameter=command>\nsleep 25\n</parameter>\n<parameter=yield_time_ms>\n25000\n</parameter>\n</function>\n</tool_call>")
            .unwrap();
        parser.finish().unwrap();
        let arguments = parser.message()["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(
            arguments.contains(r#""yield_time_ms":25000"#),
            "{arguments}"
        );
        assert!(!arguments.contains("25000.0"), "{arguments}");
    }

    #[test]
    fn compact_envelopes_preserve_newlines_and_indentation() {
        let text = "\nreason\n</think>\n\nContent\n<tool_call><function=lookup><parameter=city>\n    café\n\n</parameter></function></tool_call>";
        for split in text.char_indices().map(|(at, _)| at) {
            let mut parser = CompactChatMlOutput::new(config());
            let mut output = Accumulator::new("compact".into());
            output
                .apply(parser.push(Input::Text(&text[..split])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[split..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "\nreason\n", "split {split}");
            assert_eq!(message["content"], "\n\nContent\n", "split {split}");
            let args: Value = serde_json::from_str(
                message["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(args["city"], "\n    café\n\n", "split {split}");
        }
    }
    #[test]
    fn prose_tool_tag_mentions_remain_text_with_registered_tools() {
        let call = "<tool_call>\n<function=lookup>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        for (start, text, reasoning, content, calls) in [
            (
                crate::output::tagged::OutputStart::Content,
                "The parser chokes on a literal `<tool_call>` tag in normal text. END".to_owned(),
                None,
                "The parser chokes on a literal `<tool_call>` tag in normal text. END",
                0,
            ),
            (
                crate::output::tagged::OutputStart::Content,
                "Write `<function=lookup>` or <tool_call>\nliterally, or `<tool_call><function=shell>` inline.".to_owned(),
                None,
                "Write `<function=lookup>` or <tool_call>\nliterally, or `<tool_call><function=shell>` inline.",
                0,
            ),
            (
                crate::output::tagged::OutputStart::Content,
                format!("Quote `<tool_call>` first.\n\n{call}"),
                None,
                "Quote `<tool_call>` first.",
                1,
            ),
            (
                crate::output::tagged::OutputStart::Reasoning,
                format!("Mention `<tool_call>` here.\n</think>\n\nThen `<function=x>`.\n\n{call}"),
                Some("Mention `<tool_call>` here."),
                "Then `<function=x>`.",
                1,
            ),
        ] {
            for split in text.char_indices().map(|(at, _)| at) {
                let mut parser = Harness::new(OutputConfig { start, ..config() }, "x".into());
                parser.push(&text[..split]).unwrap();
                parser.push(&text[split..]).unwrap();
                parser.finish().unwrap();
                let message = parser.message();
                assert_eq!(message["content"], content, "{text:?} at {split}");
                assert_eq!(
                    message.get("reasoning_content").and_then(Value::as_str),
                    reasoning,
                    "{text:?} at {split}"
                );
                assert_eq!(
                    message["tool_calls"].as_array().map_or(0, Vec::len),
                    calls,
                    "{text:?} at {split}"
                );
            }
        }
    }
    #[test]
    fn parameterized_tag_after_function_mention_remains_prose() {
        let prose = "Use <function=weather> <parameterized> literally.";
        for split in prose.char_indices().map(|(at, _)| at).chain([prose.len()]) {
            let mut parser = Harness::new(
                OutputConfig {
                    start: crate::output::tagged::OutputStart::Content,
                    ..config()
                },
                "prose".into(),
            );
            parser.push(&prose[..split]).unwrap();
            parser.push(&prose[split..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(parser.message()["content"], prose, "split {split}");
            assert!(!parser.has_tools(), "split {split}");
        }

        let call = "<function=lookup> <parameter=city>Paris</parameter></function>";
        let mut parser = Harness::new(
            OutputConfig {
                start: crate::output::tagged::OutputStart::Content,
                ..config()
            },
            "call".into(),
        );
        parser.push(call).unwrap();
        parser.finish().unwrap();
        let arguments: Value = serde_json::from_str(
            parser.message()["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(arguments, json!({"city":"Paris"}));
    }
    #[test]
    fn qwen_call_detection_keeps_unknown_prefixes_distinct_from_prose() {
        type Dialect = FunctionTags<true>;
        for (text, expected) in [
            ("<function=>", Some(false)),
            ("<function=weather extra>", Some(false)),
            ("<function=weather> <parameter name=q>", Some(false)),
            ("<function=weather> <parameterized>", Some(false)),
            ("<function=weather> <3", Some(false)),
            ("<tool_call><3", Some(false)),
            ("<function=weather> <param", None),
            ("<function=weather> <parameter=q", None),
            ("<function=weather> <parameter=bad literally.", None),
            ("<function=weather> <parameter=q>", Some(true)),
            ("<function=weather> < parameter =q>", Some(true)),
            ("<function=weather> </function>", Some(true)),
        ] {
            assert_eq!(
                <Dialect as ToolDialect>::starts_call(text),
                expected,
                "{text}"
            );
        }

        for prose in [
            "<function=>",
            "<function=weather extra>",
            "<function=weather> <parameter name=q>",
            "<function=weather> <parameterized>",
            "<function=weather> <3",
            "<tool_call><3",
        ] {
            let mut parser = ChatMlOutput::new(OutputConfig {
                start: crate::output::tagged::OutputStart::Content,
                ..config()
            });
            let mut events = parser.push(Input::Text(prose)).unwrap();
            events.extend(parser.finish(Termination::OutputLimit).unwrap());
            assert_eq!(events, [crate::output::Event::Content(prose.into())]);
        }

        for partial in [
            "<function=lookup>",
            "<tool_call>",
            "<function=lookup><param",
            "<function=lookup><parameter=q",
            "<function=lookup><parameter=bad literally.",
            "<function=lookup><parameter=city>Paris",
        ] {
            let make = || {
                ChatMlOutput::new(OutputConfig {
                    start: crate::output::tagged::OutputStart::Content,
                    ..config()
                })
            };
            let mut parser = make();
            assert!(parser.push(Input::Text(partial)).unwrap().is_empty());
            assert!(parser.finish(Termination::OutputLimit).unwrap().is_empty());

            let mut parser = make();
            assert!(parser.push(Input::Text(partial)).unwrap().is_empty());
            assert!(parser.finish(Termination::Eos).is_err());
        }
    }
    #[test]
    fn compact_calls_directly_after_text_still_parse() {
        let text = "Checking.<tool_call><function=lookup><parameter=city>Paris</parameter></function></tool_call>";
        for split in text.char_indices().map(|(at, _)| at) {
            let mut parser = CompactChatMlOutput::new(OutputConfig {
                start: crate::output::tagged::OutputStart::Content,
                ..config()
            });
            let mut output = Accumulator::new("compact".into());
            for part in [&text[..split], &text[split..]] {
                output
                    .apply(parser.push(Input::Text(part)).unwrap())
                    .unwrap();
            }
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["content"], "Checking.", "split {split}");
            assert_eq!(message["tool_calls"][0]["function"]["name"], "lookup");
        }
    }
    #[test]
    fn tag_like_text_is_preserved_without_registered_tools() {
        let mut parser = Harness::new(
            OutputConfig {
                structured: false,
                start: crate::output::tagged::OutputStart::Content,
                tools: vec![],
            },
            "x".into(),
        );
        parser.push("Use <tool_call> as an example.\n").unwrap();
        parser.finish().unwrap();
        assert_eq!(
            parser.message()["content"],
            "Use <tool_call> as an example.\n"
        );
    }
    #[test]
    fn canonical_checkpoint_template_tool_record_preserves_nested_arguments() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/frontend/qwen38.json"
        )))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == "tool_calls")
            .unwrap();
        let rendered = case["rendered"].as_str().unwrap();
        let assistant = rendered.rsplit_once("<|im_start|>assistant\n").unwrap().1;
        let end = assistant.find("</tool_call>").unwrap() + "</tool_call>".len();
        let mut parser = Harness::new(
            OutputConfig {
                start: crate::output::tagged::OutputStart::Content,
                tools: case["options"]["tools"].as_array().unwrap().clone(),
                structured: false,
            },
            "canonical".into(),
        );
        for character in assistant[..end].chars() {
            parser.push(&character.to_string()).unwrap();
        }
        parser.finish().unwrap();
        let message = parser.message();
        assert_eq!(message["reasoning_content"], "Need two sources.");
        let arguments: Value = serde_json::from_str(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        let expected = &case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message.get("tool_calls").is_some())
            .unwrap()["tool_calls"][0]["function"]["arguments"];
        assert_eq!(&arguments, expected);
    }
    #[test]
    fn structured_content_preserves_literal_reasoning_and_tool_tags() {
        let text = r#"{"literal":"<think>café 世界</think><tool_call><function=lookup></function></tool_call>"}"#;
        for split in text.char_indices().map(|(index, _)| index) {
            let mut parser = Harness::new(
                OutputConfig {
                    structured: true,
                    start: crate::output::tagged::OutputStart::Content,
                    ..config()
                },
                "json".into(),
            );
            parser.push(&text[..split]).unwrap();
            parser.push(&text[split..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(parser.message()["content"], text);
            assert!(parser.message().get("reasoning_content").is_none());
            assert!(!parser.has_tools());
        }
    }
    #[test]
    fn structured_answers_follow_reasoning_and_tools_without_parsing_literal_tags() {
        let answer = r#"{"literal":"<think>世界</think><tool_call>literal</tool_call>"}"#;
        let text = format!(
            "Check sources.</think>\n\n<tool_call><function=lookup><parameter=query>alpha</parameter></function></tool_call>\n{answer}"
        );
        for split in text.char_indices().map(|(index, _)| index) {
            let mut parser = Harness::new(
                OutputConfig {
                    structured: true,
                    start: crate::output::tagged::OutputStart::Reasoning,
                    ..config()
                },
                "structured".into(),
            );
            parser.push(&text[..split]).unwrap();
            parser.push(&text[split..]).unwrap();
            parser.finish().unwrap();
            let message = parser.message();
            assert_eq!(message["reasoning_content"], "Check sources.");
            assert_eq!(message["content"].as_str().unwrap().trim(), answer);
            assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
            assert_eq!(message["tool_calls"][0]["function"]["name"], "lookup");
        }
    }
    #[test]
    fn terminal_cause_preserves_incomplete_call_errors_and_never_fabricates_calls() {
        for cause in [Termination::Eos, Termination::OutputLimit] {
            let mut parser = ChatMlOutput::new(OutputConfig {
                start: crate::output::tagged::OutputStart::Content,
                ..config()
            });
            assert!(
                parser
                    .push(Input::Text("<tool_call><function=lookup>"))
                    .unwrap()
                    .is_empty()
            );
            if cause == Termination::Eos {
                assert!(parser.finish(cause).is_err());
            } else {
                assert!(parser.finish(cause).unwrap().is_empty());
            }
        }
        for cause in [
            Termination::Cancelled,
            Termination::ExecutionError,
            Termination::UserStop,
        ] {
            let mut parser = ChatMlOutput::new(OutputConfig {
                start: crate::output::tagged::OutputStart::Content,
                ..config()
            });
            parser
                .push(Input::Text("<tool_call><function=lookup>"))
                .unwrap();
            assert!(parser.finish(cause).unwrap().is_empty());
        }
    }
}
