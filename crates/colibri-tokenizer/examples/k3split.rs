//! Debug/fuzz helper: read one JSON string per line on stdin, print its Kimi-K3
//! pre-tokenizer split as a JSON array of strings, one line per input.
//!
//! Used by `scripts/fuzz_k3_pretok.py` to diff the Rust split against the reference
//! `regex` implementation over generated text. Not part of the library.
use std::io::{self, BufRead, Write};

fn unescape(s: &str) -> String {
    // Inputs are emitted by the fuzz script as \u{...}-free JSON strings.
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '\\' && i + 1 < b.len() {
            match b[i + 1] {
                'n' => { out.push('\n'); i += 2 }
                'r' => { out.push('\r'); i += 2 }
                't' => { out.push('\t'); i += 2 }
                '"' => { out.push('"'); i += 2 }
                '\\' => { out.push('\\'); i += 2 }
                'u' => {
                    let hex: String = b[i + 2..(i + 6).min(b.len())].iter().collect();
                    let cp = u32::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                    i += 6;
                }
                c => { out.push(c); i += 2 }
            }
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out
}

fn esc(s: &str) -> String {
    let mut o = String::from("\"");
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

fn main() {
    let stdin = io::stdin();
    let mut w = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let inner = line.trim();
        let inner = inner.strip_prefix('"').unwrap_or(inner);
        let inner = inner.strip_suffix('"').unwrap_or(inner);
        let text = unescape(inner);
        let pieces = colibri_tokenizer::k3_pretokenize(&text);
        let joined: Vec<String> = pieces.iter().map(|p| esc(p)).collect();
        writeln!(w, "[{}]", joined.join(",")).unwrap();
    }
}
