//! `coli serve [port] [warm-up prompt...]` — an OpenAI-compatible HTTP inference
//! server for the streaming GLM-5.2 engine.
//!
//! Endpoints:
//!   - `GET  /health`, `GET /`         liveness + model id
//!   - `GET  /v1/models`               list the one served model
//!   - `POST /v1/completions`          text prompt → completion
//!   - `POST /v1/chat/completions`     chat messages → reply
//!
//! Both completion endpoints honor `"stream": true` and reply with Server-Sent
//! Events (the OpenAI chunk protocol, terminated by `data: [DONE]`), so tokens
//! appear live — which matters at ~1 tok/s. There is no external HTTP or JSON
//! dependency: a minimal HTTP/1.1 server on `std::net`, requests parsed with
//! `colibri-json`, responses hand-emitted.
//!
//! Concurrency: one generation at a time. A single GPU streaming a 744B model can
//! only run one forward pass anyway, so connections are served sequentially — no
//! shared-state locking, no half-interleaved KV caches. A `read` timeout keeps a
//! silent client from wedging the accept loop.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;
use std::time::Duration;

use colibri_engine::{ExpertCache, KvCache, Model, ShardsExpertProvider};
use colibri_json::Json;
use colibri_tokenizer::Tokenizer;

/// Default listen port; overridden by a positional arg or `COLI_PORT`.
const DEFAULT_PORT: u16 = 8080;
/// Default and hard cap on generated tokens per request.
const DEFAULT_MAX_TOKENS: usize = 128;
const MAX_TOKENS_CAP: usize = 2048;
/// Tokens generated per warm-up prompt (enough to route a spread of experts).
const WARMUP_TOKENS: usize = 8;

type Provider<'a> = ExpertCache<ShardsExpertProvider<'a>>;

/// `coli serve <snap> [port] [warm-up prompt...]`. `snap` is injected by the CLI
/// dispatcher (position 2). An optional pure-integer next arg is the port; any
/// remaining args are one warm-up prompt. `COLI_PORT` / `COLI_WARMUP` (the latter
/// `|`-separated for several prompts) are the env equivalents.
pub fn cmd_serve(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: coli serve <snapshot-dir> [port] [warm-up prompt...]");
            return ExitCode::from(2);
        }
    };

    // Port: a leading bare integer arg, else COLI_PORT, else the default.
    let mut rest = &args[3.min(args.len())..];
    let port = match rest.first().and_then(|s| s.parse::<u16>().ok()) {
        Some(p) => {
            rest = &rest[1..];
            p
        }
        None => std::env::var("COLI_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_PORT),
    };

    // Warm-up prompts: the remaining positional args as one prompt, plus each
    // `|`-separated entry of COLI_WARMUP.
    let mut warmups: Vec<String> = Vec::new();
    if !rest.is_empty() {
        warmups.push(rest.join(" "));
    }
    if let Ok(w) = std::env::var("COLI_WARMUP") {
        warmups.extend(w.split('|').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
    }

    // ---- load model + tokenizer -------------------------------------------
    let model = match colibri_engine::load_model(&snap) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli serve: load model: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok_path = format!("{snap}/tokenizer.json");
    let tok = match Tokenizer::load(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("coli serve: load tokenizer ({tok_path}): {e}");
            return ExitCode::FAILURE;
        }
    };
    let base = ShardsExpertProvider::new(&model.shards, &model.cfg, model.ebits as u32);
    let provider = ExpertCache::new(base, crate::ram_budget());
    let model_id = model_id_from(&snap);

    // ---- warm-up ----------------------------------------------------------
    for (i, w) in warmups.iter().enumerate() {
        let ids = tok.encode(w);
        if ids.is_empty() {
            continue;
        }
        eprintln!("[serve] warm-up {}/{}: {} tokens", i + 1, warmups.len(), ids.len());
        let mut kv = mk_kv(&model, ids.len() + WARMUP_TOKENS);
        if let Err(e) = colibri_engine::generate_greedy(&model, &mut kv, &provider, &ids, WARMUP_TOKENS) {
            eprintln!("[serve] warm-up failed: {e}");
        }
    }
    if !warmups.is_empty() {
        let s = provider.stats();
        eprintln!("[serve] warm-up done: {} experts resident ({:.1} GB)", s.resident, s.bytes as f64 / (1u64 << 30) as f64);
    }

    // ---- listen -----------------------------------------------------------
    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("coli serve: bind {addr}: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[serve] OpenAI-compatible server on http://{addr}  (model: {model_id})");
    println!("[serve]   POST /v1/chat/completions   POST /v1/completions   GET /v1/models   GET /health");

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => handle(stream, &model, &provider, &tok, &model_id),
            Err(e) => eprintln!("[serve] accept: {e}"),
        }
    }
    ExitCode::SUCCESS
}

fn mk_kv(model: &Model, max_t: usize) -> KvCache {
    KvCache::new(
        model.cfg.n_layers as usize,
        model.cfg.kv_lora as usize,
        model.cfg.qk_rope as usize,
        max_t,
    )
}

/// Derive a display model id from the snapshot path (the HF repo dir name, or the
/// leaf directory).
fn model_id_from(snap: &str) -> String {
    let trimmed = snap.trim_end_matches('/');
    // HF cache layout: .../models--org--name/snapshots/<hash>
    if let Some(pos) = trimmed.find("models--") {
        let seg = &trimmed[pos + "models--".len()..];
        let name = seg.split('/').next().unwrap_or(seg).replace("--", "/");
        if !name.is_empty() {
            return name;
        }
    }
    trimmed.rsplit('/').next().unwrap_or("glm-5.2").to_string()
}

// ---- request handling -----------------------------------------------------

struct Request {
    method: String,
    path: String,
    body: String,
}

fn handle(mut stream: TcpStream, model: &Model, provider: &Provider, tok: &Tokenizer, model_id: &str) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_nodelay(true);
    let mut reader = match stream.try_clone() {
        Ok(s) => BufReader::new(s),
        Err(_) => return,
    };
    let req = match read_request(&mut reader) {
        Some(r) => r,
        None => return, // malformed / timed out / client closed
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/health") => {
            send_json(&mut stream, 200, &format!("{{\"status\":\"ok\",\"model\":{}}}", jstr(model_id)));
        }
        ("GET", "/v1/models") => {
            let body = format!(
                "{{\"object\":\"list\",\"data\":[{{\"id\":{},\"object\":\"model\",\"owned_by\":\"colibri\"}}]}}",
                jstr(model_id)
            );
            send_json(&mut stream, 200, &body);
        }
        ("POST", "/v1/completions") => complete(&mut stream, model, provider, tok, model_id, &req.body, false),
        ("POST", "/v1/chat/completions") => complete(&mut stream, model, provider, tok, model_id, &req.body, true),
        ("OPTIONS", _) => send_json(&mut stream, 204, ""),
        _ => send_json(&mut stream, 404, "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}"),
    }
}

/// Read an HTTP/1.1 request: request line, headers, and a `Content-Length` body.
/// Generic over the reader so it can be exercised with an in-memory buffer.
fn read_request<R: BufRead>(reader: &mut R) -> Option<Request> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 {
            break;
        }
        let t = h.trim_end();
        if t.is_empty() {
            break; // end of headers
        }
        if let Some((k, v)) = t.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some(Request { method, path, body: String::from_utf8_lossy(&body).into_owned() })
}

/// Shared handler for `/v1/completions` (chat=false) and `/v1/chat/completions`
/// (chat=true): parse the request, build the prompt token ids, then either stream
/// SSE chunks or return one JSON object.
fn complete(
    stream: &mut TcpStream,
    model: &Model,
    provider: &Provider,
    tok: &Tokenizer,
    model_id: &str,
    body: &str,
    chat: bool,
) {
    let req = match Json::parse(body) {
        Some(j) => j,
        None => {
            send_json(stream, 400, "{\"error\":{\"message\":\"invalid JSON body\",\"type\":\"invalid_request_error\"}}");
            return;
        }
    };
    let obj = match req.as_object() {
        Some(o) => o,
        None => {
            send_json(stream, 400, "{\"error\":{\"message\":\"body must be a JSON object\",\"type\":\"invalid_request_error\"}}");
            return;
        }
    };

    let max_tokens = obj
        .get("max_tokens")
        .and_then(|v| v.as_i64())
        .map(|n| (n.max(1) as usize).min(MAX_TOKENS_CAP))
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let stream_mode = obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // Build the prompt token ids.
    let ids = if chat {
        let msgs = obj.get("messages").and_then(|v| v.as_array());
        match msgs {
            Some(m) => build_chat_prompt(tok, m),
            None => {
                send_json(stream, 400, "{\"error\":{\"message\":\"missing 'messages'\",\"type\":\"invalid_request_error\"}}");
                return;
            }
        }
    } else {
        match obj.get("prompt").and_then(|v| v.as_str()) {
            Some(p) => tok.encode(p),
            None => {
                send_json(stream, 400, "{\"error\":{\"message\":\"missing 'prompt' (string)\",\"type\":\"invalid_request_error\"}}");
                return;
            }
        }
    };
    if ids.is_empty() {
        send_json(stream, 400, "{\"error\":{\"message\":\"empty prompt\",\"type\":\"invalid_request_error\"}}");
        return;
    }

    let object = if chat { "chat.completion" } else { "text_completion" };
    let id = format!("cmpl-{}", ids.len().wrapping_mul(2654435761) ^ max_tokens);
    let mut kv = mk_kv(model, ids.len() + max_tokens);

    if stream_mode {
        stream_completion(stream, model, provider, tok, &ids, max_tokens, &id, model_id, object, chat, &mut kv);
    } else {
        block_completion(stream, model, provider, tok, &ids, max_tokens, &id, model_id, object, chat, &mut kv);
    }
}

/// Official GLM-5.2 chat template (byte-matches `chat_template.jinja`, mirrored
/// from the C reference): `[gMASK]<sop>` then `<|role|>{content}` per message with
/// **no** separators, ending with `<|assistant|><think></think>` — the empty think
/// block disables reasoning so the model answers directly. The control tokens
/// (`<|user|>`, `<|assistant|>`, …) are added-vocab entries, so encoding the
/// assembled string resolves them to their ids exactly as the C engine does.
fn build_chat_prompt(tok: &Tokenizer, messages: &[Json]) -> Vec<i32> {
    let mut s = String::from("[gMASK]<sop>");
    for m in messages {
        let o = match m.as_object() {
            Some(o) => o,
            None => continue,
        };
        let role = o.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = o.get("content").and_then(|v| v.as_str()).unwrap_or("");
        s.push_str(&format!("<|{role}|>{content}"));
    }
    s.push_str("<|assistant|><think></think>");
    tok.encode(&s)
}

/// Non-streaming: generate everything, then send one JSON object.
#[allow(clippy::too_many_arguments)]
fn block_completion(
    stream: &mut TcpStream,
    model: &Model,
    provider: &Provider,
    tok: &Tokenizer,
    prompt: &[i32],
    max_tokens: usize,
    id: &str,
    model_id: &str,
    object: &str,
    chat: bool,
    kv: &mut KvCache,
) {
    let seq = match colibri_engine::generate_greedy(model, kv, provider, prompt, max_tokens) {
        Ok(s) => s,
        Err(e) => {
            send_json(stream, 500, &format!("{{\"error\":{{\"message\":{},\"type\":\"internal_error\"}}}}", jstr(&e.to_string())));
            return;
        }
    };
    let cont = &seq[prompt.len()..];
    let text = tok.decode(cont);
    let finish = if cont.last().is_some_and(|t| model.cfg.stop_ids.contains(t)) { "stop" } else { "length" };
    let choice = if chat {
        format!("{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":{}}}", jstr(&text), jstr(finish))
    } else {
        format!("{{\"index\":0,\"text\":{},\"finish_reason\":{}}}", jstr(&text), jstr(finish))
    };
    let usage = format!(
        "{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}",
        prompt.len(),
        cont.len(),
        prompt.len() + cont.len()
    );
    let body = format!(
        "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{}],\"usage\":{}}}",
        jstr(id),
        jstr(object),
        jstr(model_id),
        choice,
        usage
    );
    send_json(stream, 200, &body);
}

/// Streaming: emit an SSE chunk per token (the OpenAI delta protocol). Aborts
/// generation if the client disconnects (a chunk write fails).
#[allow(clippy::too_many_arguments)]
fn stream_completion(
    stream: &mut TcpStream,
    model: &Model,
    provider: &Provider,
    tok: &Tokenizer,
    prompt: &[i32],
    max_tokens: usize,
    id: &str,
    model_id: &str,
    object: &str,
    chat: bool,
    kv: &mut KvCache,
) {
    let chunk_obj = if chat { "chat.completion.chunk" } else { "text_completion" };
    // SSE response headers.
    let headers = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream\r\n\
        Cache-Control: no-cache\r\n\
        Connection: close\r\n\
        Access-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(headers.as_bytes()).is_err() {
        return;
    }

    // Decode the growing continuation each step and emit the *new* suffix, so
    // multi-byte tokens never split a UTF-8 boundary mid-chunk.
    let mut out_ids: Vec<i32> = Vec::with_capacity(max_tokens);
    let mut sent = String::new();
    let mut finish = "length";

    let _ = colibri_engine::generate_stream(model, kv, provider, prompt, max_tokens, |t| {
        if model.cfg.stop_ids.contains(&t) {
            finish = "stop";
            return true; // deliver nothing for the stop token; loop ends after this
        }
        out_ids.push(t);
        let full = tok.decode(&out_ids);
        if full.len() <= sent.len() {
            return true; // no new complete text yet
        }
        let delta = &full[sent.len()..];
        let payload = if chat {
            format!("{{\"role\":\"assistant\",\"content\":{}}}", jstr(delta))
        } else {
            jstr(delta) // /v1/completions puts the string directly in "text"
        };
        let field = if chat { "delta" } else { "text" };
        let chunk = format!(
            "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{{\"index\":0,\"{}\":{},\"finish_reason\":null}}]}}",
            jstr(id),
            jstr(chunk_obj),
            jstr(model_id),
            field,
            payload
        );
        let ok = write_sse(stream, &chunk);
        if ok {
            sent = full;
        }
        ok // false → client gone → stop generating
    });

    // Terminal chunk carrying finish_reason, then the OpenAI [DONE] sentinel.
    let last = if chat {
        format!(
            "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":{}}}]}}",
            jstr(id), jstr(object_chunk(chat, object)), jstr(model_id), jstr(finish)
        )
    } else {
        format!(
            "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":{}}}]}}",
            jstr(id), jstr(object_chunk(chat, object)), jstr(model_id), jstr(finish)
        )
    };
    let _ = write_sse(stream, &last);
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}

fn object_chunk(chat: bool, object: &str) -> &str {
    if chat {
        "chat.completion.chunk"
    } else {
        object
    }
}

fn write_sse(stream: &mut TcpStream, data: &str) -> bool {
    stream.write_all(b"data: ").is_ok()
        && stream.write_all(data.as_bytes()).is_ok()
        && stream.write_all(b"\n\n").is_ok()
        && stream.flush().is_ok()
}

/// Send a fixed JSON (or empty) body with CORS + `Content-Length`. Status 200/204/
/// 4xx/5xx.
fn send_json(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: *\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

/// Minimal JSON string literal: wraps in quotes and escapes per RFC 8259.
fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn jstr_escapes() {
        assert_eq!(jstr("hi"), "\"hi\"");
        assert_eq!(jstr("a\"b"), "\"a\\\"b\"");
        assert_eq!(jstr("a\\b"), "\"a\\\\b\"");
        assert_eq!(jstr("line\nbreak"), "\"line\\nbreak\"");
        assert_eq!(jstr("tab\tx"), "\"tab\\tx\"");
        assert_eq!(jstr("\u{0007}"), "\"\\u0007\""); // bell → 
        assert_eq!(jstr("café ☕"), "\"café ☕\""); // multibyte passes through
    }

    #[test]
    fn model_id_from_hf_cache_path() {
        assert_eq!(
            model_id_from("/root/.cache/huggingface/hub/models--mateogrgic--GLM-5.2-colibri-int4-with-int8-mtp/snapshots/abc123"),
            "mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp"
        );
        assert_eq!(model_id_from("/data/glm52-int4/"), "glm52-int4");
        assert_eq!(model_id_from("/model"), "model");
    }

    #[test]
    fn read_request_parses_post_with_body() {
        let raw = "POST /v1/chat/completions HTTP/1.1\r\n\
                   Host: localhost\r\n\
                   Content-Type: application/json\r\n\
                   Content-Length: 13\r\n\r\n\
                   {\"a\":\"hello\"}";
        let mut r = Cursor::new(raw.as_bytes());
        let req = read_request(&mut r).unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions");
        assert_eq!(req.body, "{\"a\":\"hello\"}");
    }

    #[test]
    fn read_request_get_no_body() {
        let raw = "GET /health HTTP/1.1\r\nHost: x\r\n\r\n";
        let mut r = Cursor::new(raw.as_bytes());
        let req = read_request(&mut r).unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/health");
        assert!(req.body.is_empty());
    }

    #[test]
    fn read_request_content_length_case_insensitive() {
        let raw = "POST /v1/completions HTTP/1.1\r\ncontent-length: 3\r\n\r\nabc";
        let mut r = Cursor::new(raw.as_bytes());
        let req = read_request(&mut r).unwrap();
        assert_eq!(req.body, "abc");
    }

    #[test]
    fn read_request_empty_is_none() {
        let mut r = Cursor::new(&b""[..]);
        assert!(read_request(&mut r).is_none());
    }
}
