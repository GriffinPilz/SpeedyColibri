//! GLM-5.2 tokenizer in pure Rust — a faithful port of `c/tok.h`.
//!
//! Byte-level BPE (cl100k / tiktoken style), replicating `tokenizer.json`:
//!   - `model.type = BPE`, `ignore_merges = true`, `byte_fallback = false`
//!   - pre-tokenizer: regex Split (cl100k pattern) + ByteLevel (`add_prefix_space=false`)
//!   - merges ranked by list order; `\p{L}`/`\p{N}`/`\s` from [`unicode_tables`]
//!   - added tokens (special and non-special) are atomic in encode/decode
//!
//! ```no_run
//! # use colibri_tokenizer::Tokenizer;
//! let t = Tokenizer::load("tokenizer.json").unwrap();
//! let ids = t.encode("ciao!");
//! let text = t.decode(&ids);
//! ```

mod unicode_tables;

use colibri_json::Json;
use std::collections::HashMap;
use std::path::Path;
use unicode_tables::{is_l, is_n, is_s};

/// An added token (special or not): matched atomically, emitted literally.
#[derive(Debug, Clone)]
struct Special {
    bytes: Vec<u8>,
    id: i32,
}

/// A loaded tokenizer.
pub struct Tokenizer {
    /// byte-level string -> id
    vocab: HashMap<Vec<u8>, i32>,
    /// "left\0right" -> merge rank
    merges: HashMap<Vec<u8>, i32>,
    /// id -> byte-level string (vocab) or literal content (added tokens)
    id2str: Vec<Option<Vec<u8>>>,
    /// id -> is this an added token (output verbatim)?
    id_added: Vec<bool>,
    /// added tokens, sorted by content length descending (longest match first)
    specials: Vec<Special>,
    /// byte -> its ByteLevel UTF-8 mapping (1–2 bytes)
    byte2str: Vec<Vec<u8>>,
    /// codepoint (< 1024) -> original byte, or -1
    cp2byte: [i16; 1024],
}

impl Tokenizer {
    /// Load and parse a `tokenizer.json`.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Tokenizer> {
        let text = std::fs::read_to_string(path)?;
        let root = Json::parse(&text).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "empty tokenizer.json")
        })?;
        Tokenizer::from_json(&root).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tokenizer.json: missing model.vocab/merges",
            )
        })
    }

    /// Build from an already-parsed `tokenizer.json` root.
    pub fn from_json(root: &Json) -> Option<Tokenizer> {
        let model = root.get("model")?;
        let vocab_j = model.get("vocab")?.as_object()?;
        let merges_j = model.get("merges")?.as_array()?;
        let added_j = root.get("added_tokens").and_then(Json::as_array);

        // Byte<->unicode ByteLevel map (GPT-2).
        let (byte2str, cp2byte) = build_bytemap();

        // Size id2str from the max id across vocab and added tokens.
        let mut maxid = 0i32;
        for (_, v) in vocab_j.iter() {
            if let Some(id) = v.as_i64() {
                maxid = maxid.max(id as i32);
            }
        }
        if let Some(added) = added_j {
            for a in added {
                if let Some(id) = a.get("id").and_then(Json::as_i64) {
                    maxid = maxid.max(id as i32);
                }
            }
        }
        let n_ids = (maxid + 1).max(0) as usize;
        let mut id2str: Vec<Option<Vec<u8>>> = vec![None; n_ids];
        let mut id_added = vec![false; n_ids];

        // vocab: byte-level string -> id
        let mut vocab: HashMap<Vec<u8>, i32> = HashMap::with_capacity(vocab_j.len());
        for (k, v) in vocab_j.iter() {
            if let Some(id) = v.as_i64() {
                let id = id as i32;
                let kb = k.as_bytes().to_vec();
                if (id as usize) < id2str.len() {
                    id2str[id as usize] = Some(kb.clone());
                }
                vocab.insert(kb, id);
            }
        }

        // merges: "left\0right" -> rank (list order). Each entry is either a
        // 2-element array (C form) or a "left right" string (newer form).
        let mut merges: HashMap<Vec<u8>, i32> = HashMap::with_capacity(merges_j.len());
        for (rank, pr) in merges_j.iter().enumerate() {
            let (l, r) = match pr {
                Json::Arr(a) if a.len() >= 2 => (
                    a[0].as_str().unwrap_or("").as_bytes(),
                    a[1].as_str().unwrap_or("").as_bytes(),
                ),
                Json::Str(s) => match s.split_once(' ') {
                    Some((l, r)) => (l.as_bytes(), r.as_bytes()),
                    None => continue,
                },
                _ => continue,
            };
            merges.insert(merge_key(l, r), rank as i32);
        }

        // added tokens: atomic, literal output.
        let mut specials = Vec::new();
        if let Some(added) = added_j {
            for a in added {
                let content = a.get("content").and_then(Json::as_str).unwrap_or("");
                let id = a.get("id").and_then(Json::as_i64).unwrap_or(-1) as i32;
                if id < 0 {
                    continue;
                }
                let bytes = content.as_bytes().to_vec();
                if (id as usize) < id2str.len() {
                    id2str[id as usize] = Some(bytes.clone());
                    id_added[id as usize] = true;
                }
                specials.push(Special { bytes, id });
            }
            // longest content first
            specials.sort_by(|a, b| b.bytes.len().cmp(&a.bytes.len()));
        }

        Some(Tokenizer {
            vocab,
            merges,
            id2str,
            id_added,
            specials,
            byte2str,
            cp2byte,
        })
    }

    /// Total id space (max id + 1).
    pub fn n_ids(&self) -> usize {
        self.id2str.len()
    }

    /// The id of an added token given its content (e.g. `"<|endoftext|>"`), or
    /// `None`. Port of `tok_id_of`.
    pub fn id_of(&self, content: &str) -> Option<i32> {
        let cb = content.as_bytes();
        self.specials
            .iter()
            .find(|s| s.bytes == cb)
            .map(|s| s.id)
    }

    /// Encode text to token ids. Splits on added tokens (longest match), then
    /// pre-tokenizes and BPE-encodes each free-text span. Port of `tok_encode`.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let p = text.as_bytes();
        let len = p.len();
        let mut out = Vec::new();
        let mut i = 0;
        while i < len {
            // earliest position >= i where any special matches (longest, since
            // specials are sorted by length descending)
            let mut hit: Option<(usize, usize, i32)> = None;
            'scan: for j in i..len {
                for sp in &self.specials {
                    let sl = sp.bytes.len();
                    if sl > 0 && j + sl <= len && &p[j..j + sl] == sp.bytes.as_slice() {
                        hit = Some((j, sl, sp.id));
                        break 'scan;
                    }
                }
            }
            let chunk_end = hit.map(|(pos, _, _)| pos).unwrap_or(len);
            if chunk_end > i {
                self.pretok_chunk(p, i, chunk_end, &mut out);
            }
            match hit {
                Some((pos, sl, id)) => {
                    out.push(id);
                    i = pos + sl;
                }
                None => break,
            }
        }
        out
    }

    /// Decode ids back to bytes. Added tokens emit their content literally;
    /// normal tokens go through the byte-level inverse map. Port of `tok_decode`.
    pub fn decode_bytes(&self, ids: &[i32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if id < 0 || id as usize >= self.id2str.len() {
                continue;
            }
            let s = match &self.id2str[id as usize] {
                Some(s) => s,
                None => continue,
            };
            if self.id_added[id as usize] {
                out.extend_from_slice(s);
                continue;
            }
            let mut j = 0;
            while j < s.len() {
                let (cp, k) = u8_next(s, j);
                j += k;
                if cp < 1024 && self.cp2byte[cp as usize] >= 0 {
                    out.push(self.cp2byte[cp as usize] as u8);
                }
            }
        }
        out
    }

    /// Decode ids to a `String` (lossily, if the byte stream is mid-codepoint).
    pub fn decode(&self, ids: &[i32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    // ---- internals ---------------------------------------------------------

    /// BPE on the raw byte span `p[a..b)`, appending ids to `out`.
    fn bpe_piece(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>) {
        // byte-level string: concat of byte2str for each input byte.
        let mut s: Vec<u8> = Vec::with_capacity(2 * (b - a));
        for &bb in &p[a..b] {
            s.extend_from_slice(&self.byte2str[bb as usize]);
        }
        // ignore_merges: whole piece is a token -> emit directly.
        if let Some(&id) = self.vocab.get(s.as_slice()) {
            out.push(id);
            return;
        }
        // initial symbols = codepoints of the byte-level string.
        let mut soff: Vec<usize> = Vec::new();
        let mut slen: Vec<usize> = Vec::new();
        {
            let mut i = 0;
            while i < s.len() {
                let (_, k) = u8_next(&s, i);
                soff.push(i);
                slen.push(k);
                i += k;
            }
        }
        // merge loop: repeatedly apply the lowest-rank adjacent merge.
        loop {
            let mut best = i32::MAX;
            let mut bp: isize = -1;
            for i in 0..soff.len().saturating_sub(1) {
                let key = merge_key(
                    &s[soff[i]..soff[i] + slen[i]],
                    &s[soff[i + 1]..soff[i + 1] + slen[i + 1]],
                );
                if let Some(&rk) = self.merges.get(&key) {
                    if rk < best {
                        best = rk;
                        bp = i as isize;
                    }
                }
            }
            if bp < 0 {
                break;
            }
            let bp = bp as usize;
            // fuse bp and bp+1 (contiguous in s)
            slen[bp] = soff[bp + 1] + slen[bp + 1] - soff[bp];
            soff.remove(bp + 1);
            slen.remove(bp + 1);
        }
        for i in 0..soff.len() {
            if let Some(&id) = self.vocab.get(&s[soff[i]..soff[i] + slen[i]]) {
                out.push(id);
            }
        }
    }

    /// Pre-tokenizer over `p[a..b)`: decode codepoints, apply the cl100k
    /// alternatives in order, and BPE each piece. Port of `pretok_chunk`.
    fn pretok_chunk(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>) {
        if b <= a {
            return;
        }
        // codepoints and their byte offsets within p; off[n] == b.
        let mut cp: Vec<u32> = Vec::new();
        let mut off: Vec<usize> = Vec::new();
        {
            let mut i = a;
            while i < b {
                let (c, k) = u8_next(p, i);
                off.push(i);
                cp.push(c);
                i += k;
            }
            off.push(b);
        }
        let n = cp.len();
        let is_nl = |c: u32| c == b'\r' as u32 || c == b'\n' as u32;
        let low = |c: u32| {
            if (b'A' as u32..=b'Z' as u32).contains(&c) {
                c + 32
            } else {
                c
            }
        };

        let mut i = 0usize;
        while i < n {
            let start = i;
            let c = cp[i];

            // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if c == b'\'' as u32 && i + 1 < n {
                let d = low(cp[i + 1]);
                if i + 2 < n {
                    let d2 = low(cp[i + 2]);
                    if (d == b'r' as u32 && d2 == b'e' as u32)
                        || (d == b'v' as u32 && d2 == b'e' as u32)
                        || (d == b'l' as u32 && d2 == b'l' as u32)
                    {
                        i += 3;
                        self.bpe_piece(p, off[start], off[i], out);
                        continue;
                    }
                }
                if d == b's' as u32 || d == b't' as u32 || d == b'm' as u32 || d == b'd' as u32 {
                    i += 2;
                    self.bpe_piece(p, off[start], off[i], out);
                    continue;
                }
            }

            // 2) [^\r\n\p{L}\p{N}]? \p{L}+
            {
                let mut j: isize = i as isize;
                if !is_l(c) && !is_nl(c) && !is_n(c) {
                    if (j as usize) + 1 < n && is_l(cp[(j as usize) + 1]) {
                        j += 1;
                    } else {
                        j = -1;
                    }
                }
                if j >= 0 {
                    let ju = j as usize;
                    if is_l(cp[ju]) {
                        let mut k = ju;
                        while k < n && is_l(cp[k]) {
                            k += 1;
                        }
                        i = k;
                        self.bpe_piece(p, off[start], off[i], out);
                        continue;
                    }
                }
            }

            // 3) \p{N}{1,3}
            if is_n(c) {
                let mut j = i;
                let mut k = 0;
                while j < n && is_n(cp[j]) && k < 3 {
                    j += 1;
                    k += 1;
                }
                i = j;
                self.bpe_piece(p, off[start], off[i], out);
                continue;
            }

            // 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*'
            {
                let mut j = i;
                if c == b' ' as u32
                    && j + 1 < n
                    && !is_s(cp[j + 1])
                    && !is_l(cp[j + 1])
                    && !is_n(cp[j + 1])
                {
                    j += 1;
                }
                if j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
                    while j < n && !is_s(cp[j]) && !is_l(cp[j]) && !is_n(cp[j]) {
                        j += 1;
                    }
                    while j < n && is_nl(cp[j]) {
                        j += 1;
                    }
                    i = j;
                    self.bpe_piece(p, off[start], off[i], out);
                    continue;
                }
            }

            // 5) \s*[\r\n]+  -> whitespace run up to the last contiguous newline
            {
                let mut r = i;
                while r < n && is_s(cp[r]) {
                    r += 1;
                }
                if r > i {
                    let mut last: isize = -1;
                    for j in i..r {
                        if is_nl(cp[j]) {
                            last = j as isize;
                        }
                    }
                    if last >= 0 {
                        i = last as usize + 1;
                        self.bpe_piece(p, off[start], off[i], out);
                        continue;
                    }
                    // 6) \s+(?!\S): if followed by non-space, leave the last ws;
                    // else take it all.
                    let mut end = if r < n { r - 1 } else { r };
                    if end <= i {
                        end = i + 1; // \s+ minimum 1 (fallback alt 7)
                    }
                    i = end;
                    self.bpe_piece(p, off[start], off[i], out);
                    continue;
                }
            }

            // safety net: shouldn't happen
            i += 1;
            self.bpe_piece(p, off[start], off[i], out);
        }
    }
}

/// merge-map key: `left` bytes, a NUL separator, then `right` bytes.
fn merge_key(l: &[u8], r: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(l.len() + 1 + r.len());
    k.extend_from_slice(l);
    k.push(0);
    k.extend_from_slice(r);
    k
}

/// Decode one UTF-8 codepoint at `s[i]`, returning `(codepoint, byte_len)`.
/// Invalid bytes are passed through as a single-byte codepoint. Port of `u8_next`.
fn u8_next(s: &[u8], i: usize) -> (u32, usize) {
    let c = s[i];
    if c < 0x80 {
        (c as u32, 1)
    } else if c >> 5 == 0x6 && i + 1 < s.len() {
        (((c as u32 & 0x1F) << 6) | (s[i + 1] as u32 & 0x3F), 2)
    } else if c >> 4 == 0xE && i + 2 < s.len() {
        (
            ((c as u32 & 0x0F) << 12)
                | ((s[i + 1] as u32 & 0x3F) << 6)
                | (s[i + 2] as u32 & 0x3F),
            3,
        )
    } else if c >> 3 == 0x1E && i + 3 < s.len() {
        (
            ((c as u32 & 0x07) << 18)
                | ((s[i + 1] as u32 & 0x3F) << 12)
                | ((s[i + 2] as u32 & 0x3F) << 6)
                | (s[i + 3] as u32 & 0x3F),
            4,
        )
    } else {
        (c as u32, 1)
    }
}

/// Encode a codepoint to UTF-8 bytes (1–4). Port of `u8_put`.
fn u8_put(cp: u32) -> Vec<u8> {
    if cp < 0x80 {
        vec![cp as u8]
    } else if cp < 0x800 {
        vec![0xC0 | (cp >> 6) as u8, 0x80 | (cp & 0x3F) as u8]
    } else if cp < 0x10000 {
        vec![
            0xE0 | (cp >> 12) as u8,
            0x80 | ((cp >> 6) & 0x3F) as u8,
            0x80 | (cp & 0x3F) as u8,
        ]
    } else {
        vec![
            0xF0 | (cp >> 18) as u8,
            0x80 | ((cp >> 12) & 0x3F) as u8,
            0x80 | ((cp >> 6) & 0x3F) as u8,
            0x80 | (cp & 0x3F) as u8,
        ]
    }
}

/// GPT-2 / ByteLevel byte<->unicode map. Port of `tk_build_bytemap`.
fn build_bytemap() -> (Vec<Vec<u8>>, [i16; 1024]) {
    let mut cp2byte = [-1i16; 1024];
    let mut isdir = [false; 256];
    for b in 33..=126 {
        isdir[b] = true;
    }
    for b in 161..=172 {
        isdir[b] = true;
    }
    for b in 174..=255 {
        isdir[b] = true;
    }
    let mut byte2str: Vec<Vec<u8>> = Vec::with_capacity(256);
    let mut n = 0u32;
    for b in 0..256usize {
        let cp = if isdir[b] {
            b as u32
        } else {
            let v = 256 + n;
            n += 1;
            v
        };
        byte2str.push(u8_put(cp));
        if cp < 1024 {
            cp2byte[cp as usize] = b as i16;
        }
    }
    (byte2str, cp2byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(json: &str) -> Tokenizer {
        Tokenizer::from_json(&Json::parse(json).unwrap()).unwrap()
    }

    #[test]
    fn whole_token_ignore_merges() {
        // "ab" is directly in the vocab -> emitted as one id (ignore_merges).
        let t = tk(r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}"#);
        assert_eq!(t.encode("ab"), vec![2]);
        assert_eq!(t.decode(&[2]), "ab");
    }

    #[test]
    fn bpe_merge_path() {
        // "aab" is not a whole token; symbols a,a,b; merge a+b -> ab; result a, ab.
        let t = tk(r#"{"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}"#);
        assert_eq!(t.encode("aab"), vec![0, 2]);
        assert_eq!(t.decode(&[0, 2]), "aab");
    }

    #[test]
    fn added_tokens_are_atomic() {
        let t = tk(
            r#"{"model":{"vocab":{"a":0,"b":1},"merges":[]},
                "added_tokens":[{"id":99,"content":"<|x|>"}]}"#,
        );
        assert_eq!(t.encode("a<|x|>b"), vec![0, 99, 1]);
        assert_eq!(t.id_of("<|x|>"), Some(99));
        assert_eq!(t.decode(&[0, 99, 1]), "a<|x|>b");
    }

    #[test]
    fn byte_level_roundtrip_for_space() {
        // Space (0x20) is a non-direct byte -> maps to a 2-byte codepoint. If its
        // byte-level single-char token is in the vocab, encode/decode round-trips.
        // The ByteLevel form of ' ' is U+0120 ("Ġ").
        let t = tk(r#"{"model":{"vocab":{"a":0,"Ġ":1},"merges":[]}}"#);
        // " a" pretokenizes to piece " a"? No: leading space + letter -> alt 2's
        // optional prefix isn't a space case; here " " then "a" split. Regardless,
        // the byte-level of ' ' must decode back to a space.
        let ids = t.encode("a");
        assert_eq!(t.decode(&ids), "a");
        assert_eq!(t.decode(&[1]), " ");
    }

    #[test]
    fn contractions_split() {
        // "don't" -> "don" + "'t" under alt 1. With a vocab covering both, we get
        // two ids; without merges each falls back to whatever's in vocab.
        let t = tk(
            r#"{"model":{"vocab":{"don":0,"'t":1,"d":2,"o":3,"n":4,"'":5,"t":6},"merges":[]}}"#,
        );
        // "don" is a whole token, "'t" is a whole token.
        assert_eq!(t.encode("don't"), vec![0, 1]);
    }
}
