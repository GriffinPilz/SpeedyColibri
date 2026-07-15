//! Minimal JSON parser — a faithful Rust port of `c/json.h`.
//!
//! It exists for exactly two consumers in the engine:
//!   - the header of safetensors files (one big object `name -> {dtype, shape, data_offsets}`);
//!   - `config.json` / `ref.json` (reading hyperparameters and prompt ids).
//!
//! Like the C original it is deliberately small but it *does* handle the pieces
//! those inputs actually use, including `\uXXXX` escapes with surrogate pairs.
//! It is intentionally lenient in the same places the C parser is (bare
//! `true`/`false`/`null` are matched by first byte, numbers via a float parse),
//! so that byte-identical inputs produce the same tree.

use std::collections::BTreeMap;

/// A parsed JSON value. Mirrors the `jtype`/`jval` union from the C header, but
/// as an idiomatic Rust enum.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Object. We keep insertion order in a `Vec` (safetensors headers are
    /// iterated in order by `st_init`) *and* a name→index map for O(1) `get`.
    Obj(JsonObj),
}

/// Insertion-ordered JSON object.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JsonObj {
    keys: Vec<String>,
    vals: Vec<Json>,
    index: BTreeMap<String, usize>,
}

impl JsonObj {
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, key: String, val: Json) {
        // Last write wins, matching `hm_put`/`json_get`'s first-hit-after-overwrite.
        if let Some(&i) = self.index.get(&key) {
            self.vals[i] = val;
        } else {
            self.index.insert(key.clone(), self.keys.len());
            self.keys.push(key);
            self.vals.push(val);
        }
    }

    /// Look up a key. `None` if absent.
    pub fn get(&self, key: &str) -> Option<&Json> {
        self.index.get(key).map(|&i| &self.vals[i])
    }

    /// Number of members.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Iterate `(key, value)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Json)> {
        self.keys
            .iter()
            .map(|s| s.as_str())
            .zip(self.vals.iter())
    }

    /// Keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.keys.iter().map(|s| s.as_str())
    }
}

impl Json {
    /// Parse a JSON document. Returns `None` only if the input is empty; like the
    /// C parser it is otherwise best-effort and never errors on trailing junk.
    pub fn parse(text: &str) -> Option<Json> {
        let mut p = Parser {
            b: text.as_bytes(),
            i: 0,
        };
        if p.b.is_empty() {
            return None;
        }
        Some(p.parse_val())
    }

    /// `json_get`: member of an object by key, or `None` for non-objects.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(o) => o.get(key),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// Numbers truncate toward zero, matching the C `(int)v->num` casts.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Num(n) => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&JsonObj> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    #[inline]
    fn cur(&self) -> u8 {
        // The C parser dereferences a NUL-terminated string; past the end we
        // return 0, which every call site treats as "stop".
        *self.b.get(self.i).unwrap_or(&0)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn parse_val(&mut self) -> Json {
        self.ws();
        match self.cur() {
            b'"' => Json::Str(self.parse_str_raw()),
            b'{' => self.parse_obj(),
            b'[' => self.parse_arr(),
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => Json::Num(self.parse_num()),
        }
    }

    fn parse_obj(&mut self) -> Json {
        self.i += 1; // consume '{'
        let mut obj = JsonObj::new();
        self.ws();
        if self.cur() == b'}' {
            self.i += 1;
            return Json::Obj(obj);
        }
        loop {
            self.ws();
            let key = self.parse_str_raw();
            self.ws();
            if self.cur() == b':' {
                self.i += 1;
            }
            let val = self.parse_val();
            obj.push(key, val);
            self.ws();
            match self.cur() {
                b',' => {
                    self.i += 1;
                    continue;
                }
                b'}' => {
                    self.i += 1;
                    break;
                }
                _ => break,
            }
        }
        Json::Obj(obj)
    }

    fn parse_arr(&mut self) -> Json {
        self.i += 1; // consume '['
        let mut arr = Vec::new();
        self.ws();
        if self.cur() == b']' {
            self.i += 1;
            return Json::Arr(arr);
        }
        loop {
            let val = self.parse_val();
            arr.push(val);
            self.ws();
            match self.cur() {
                b',' => {
                    self.i += 1;
                    continue;
                }
                b']' => {
                    self.i += 1;
                    break;
                }
                _ => break,
            }
        }
        Json::Arr(arr)
    }

    /// Assumes the current byte is `"`. Decodes the standard escapes plus
    /// `\uXXXX` (with surrogate pairs) to UTF-8, exactly like `j_parse_str_raw`.
    fn parse_str_raw(&mut self) -> String {
        if self.cur() == b'"' {
            self.i += 1;
        }
        let mut out: Vec<u8> = Vec::new();
        while self.i < self.b.len() && self.b[self.i] != b'"' {
            let c = self.b[self.i];
            self.i += 1;
            if c == b'\\' && self.i < self.b.len() {
                let e = self.b[self.i];
                self.i += 1;
                match e {
                    b'n' => out.push(b'\n'),
                    b't' => out.push(b'\t'),
                    b'r' => out.push(b'\r'),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0c),
                    b'/' => out.push(b'/'),
                    b'\\' => out.push(b'\\'),
                    b'"' => out.push(b'"'),
                    b'u' => {
                        let mut cp = self.hex4();
                        if (0xD800..=0xDBFF).contains(&cp)
                            && self.peek(0) == b'\\'
                            && self.peek(1) == b'u'
                        {
                            // consume "\u" then the low surrogate's 4 hex digits
                            let save = self.i;
                            self.i += 2;
                            let lo = self.hex4();
                            if (0xDC00..=0xDFFF).contains(&lo) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                self.i = save; // not a valid pair; leave it
                            }
                        }
                        push_utf8(&mut out, cp);
                    }
                    other => out.push(other),
                }
            } else {
                out.push(c);
            }
        }
        if self.cur() == b'"' {
            self.i += 1;
        }
        // The bytes we emit are valid UTF-8 for well-formed input; fall back
        // lossily rather than panicking on malformed model metadata.
        String::from_utf8(out).unwrap_or_else(|e| {
            String::from_utf8_lossy(e.as_bytes()).into_owned()
        })
    }

    /// Read 4 hex digits at the cursor (advancing past them) into a codepoint,
    /// mirroring the C `strtoul(..., 16)` over a 4-char window.
    fn hex4(&mut self) -> u32 {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.cur();
            let d = match c {
                b'0'..=b'9' => (c - b'0') as u32,
                b'a'..=b'f' => (c - b'a' + 10) as u32,
                b'A'..=b'F' => (c - b'A' + 10) as u32,
                _ => break,
            };
            v = v * 16 + d;
            self.i += 1;
        }
        v
    }

    #[inline]
    fn peek(&self, ahead: usize) -> u8 {
        *self.b.get(self.i + ahead).unwrap_or(&0)
    }

    /// Parse a number the way C's `strtod` does at the cursor, advancing past it.
    fn parse_num(&mut self) -> f64 {
        let start = self.i;
        let mut end = self.i;
        let b = self.b;
        // Accept an optional sign, digits, one dot, and an exponent — the shape
        // strtod consumes for the numbers that appear in these files.
        if end < b.len() && (b[end] == b'+' || b[end] == b'-') {
            end += 1;
        }
        while end < b.len() && (b[end].is_ascii_digit() || b[end] == b'.') {
            end += 1;
        }
        if end < b.len() && (b[end] == b'e' || b[end] == b'E') {
            end += 1;
            if end < b.len() && (b[end] == b'+' || b[end] == b'-') {
                end += 1;
            }
            while end < b.len() && b[end].is_ascii_digit() {
                end += 1;
            }
        }
        self.i = end;
        std::str::from_utf8(&b[start..end])
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0)
    }
}

fn push_utf8(out: &mut Vec<u8>, cp: u32) {
    if cp < 0x80 {
        out.push(cp as u8);
    } else if cp < 0x800 {
        out.push(0xC0 | (cp >> 6) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
    } else if cp < 0x10000 {
        out.push(0xE0 | (cp >> 12) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
    } else {
        out.push(0xF0 | (cp >> 18) as u8);
        out.push(0x80 | ((cp >> 12) & 0x3F) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives() {
        assert_eq!(Json::parse("true"), Some(Json::Bool(true)));
        assert_eq!(Json::parse("false"), Some(Json::Bool(false)));
        assert_eq!(Json::parse("null"), Some(Json::Null));
        assert_eq!(Json::parse("  42 "), Some(Json::Num(42.0)));
        assert_eq!(Json::parse("-1.5e3"), Some(Json::Num(-1500.0)));
        assert_eq!(Json::parse(r#""hi""#), Some(Json::Str("hi".into())));
        assert_eq!(Json::parse(""), None);
    }

    #[test]
    fn object_and_get() {
        let j = Json::parse(r#"{"a": 1, "b": [2, 3], "c": {"d": "x"}}"#).unwrap();
        assert_eq!(j.get("a").and_then(Json::as_i64), Some(1));
        assert_eq!(j.get("b").unwrap().as_array().unwrap().len(), 2);
        assert_eq!(
            j.get("c").and_then(|c| c.get("d")).and_then(Json::as_str),
            Some("x")
        );
        assert!(j.get("missing").is_none());
    }

    #[test]
    fn escapes_and_unicode() {
        let j = Json::parse(r#""a\nb\t\"c\"""#).unwrap();
        assert_eq!(j.as_str(), Some("a\nb\t\"c\""));
        // U+00E9 é and an astral codepoint via a surrogate pair (U+1F600).
        let j = Json::parse(r#""é 😀""#).unwrap();
        assert_eq!(j.as_str(), Some("é 😀"));
    }

    #[test]
    fn safetensors_header_shape() {
        // The exact shape st_init walks: name -> {dtype, shape, data_offsets}.
        let hdr = r#"{"__metadata__":{"x":"y"},
            "model.embed":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,64]}}"#;
        let j = Json::parse(hdr).unwrap();
        let t = j.get("model.embed").unwrap();
        assert_eq!(t.get("dtype").and_then(Json::as_str), Some("BF16"));
        let shape = t.get("shape").unwrap().as_array().unwrap();
        let numel: i64 = shape.iter().map(|s| s.as_i64().unwrap()).product();
        assert_eq!(numel, 32);
        let off = t.get("data_offsets").unwrap().as_array().unwrap();
        assert_eq!((off[0].as_i64().unwrap(), off[1].as_i64().unwrap()), (0, 64));
    }

    #[test]
    fn duplicate_key_last_wins() {
        let j = Json::parse(r#"{"k":1,"k":2}"#).unwrap();
        assert_eq!(j.get("k").and_then(Json::as_i64), Some(2));
        assert_eq!(j.as_object().unwrap().len(), 1);
    }
}
