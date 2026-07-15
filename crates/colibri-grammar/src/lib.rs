//! Byte-level GBNF grammar engine and JSON-schema→GBNF compiler.
//!
//! Ports `c/grammar.h` (the grammar automaton that drives grammar-forced
//! speculative drafts — issue #48) and `c/schema_gbnf.h` (JSON-schema → GBNF).
//!
//! # Status: SKELETON
//!
//! The public surface is laid out to match how the engine uses grammars:
//!   - parse a GBNF subset (literals, char classes, `| ( ) ? * +`, comments);
//!   - walk the automaton one byte at a time, reporting the set of legal next
//!     bytes, and — crucially for drafting — whether exactly *one* byte is legal
//!     (a forced span that can be injected as a pre-accepted draft with ~1.0
//!     acceptance).
//!
//! The parsing and stepping bodies are TODO; they are the next milestone after
//! the CPU forward pass. See PORTING.md.

use colibri_json::Json;

/// A compiled GBNF grammar.
#[derive(Debug, Default)]
pub struct Grammar {
    // TODO(port c/grammar.h): rule table, char-class sets, stacks. The C engine
    // represents rules as sequences of elements over a rule id space.
    _private: (),
}

/// The result of asking the grammar what may come next at the current state.
#[derive(Debug, Clone)]
pub struct NextBytes {
    /// The set of byte values the grammar admits next (256-bit mask).
    pub allowed: [bool; 256],
    /// If the grammar admits exactly one byte, that byte (a forced span step).
    pub forced: Option<u8>,
    /// Whether the grammar can legally terminate here.
    pub can_end: bool,
}

impl Grammar {
    /// Parse a GBNF grammar string. Port of the grammar loader in `c/grammar.h`.
    pub fn parse(_gbnf: &str) -> Result<Grammar, GrammarError> {
        // TODO(port): tokenize and build the rule automaton.
        Err(GrammarError::NotImplemented)
    }

    /// Compile a JSON schema to a GBNF grammar, then parse it. Port of
    /// `c/schema_gbnf.h`.
    pub fn from_json_schema(_schema: &Json) -> Result<Grammar, GrammarError> {
        // TODO(port): emit GBNF for the schema, then Grammar::parse it.
        Err(GrammarError::NotImplemented)
    }

    /// Given the current parse stack, report the legal next bytes. Port of the
    /// per-byte stepping in `c/grammar.h`.
    pub fn next_bytes(&self) -> NextBytes {
        // TODO(port): walk the automaton.
        NextBytes {
            allowed: [false; 256],
            forced: None,
            can_end: true,
        }
    }

    /// Advance the grammar state by one accepted byte. Port of the accept step.
    pub fn accept(&mut self, _byte: u8) -> Result<(), GrammarError> {
        // TODO(port): push/pop the parse stack.
        Err(GrammarError::NotImplemented)
    }
}

/// Grammar parse / step errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarError {
    NotImplemented,
    Syntax(String),
    Rejected(u8),
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarError::NotImplemented => write!(f, "grammar: not yet implemented"),
            GrammarError::Syntax(s) => write!(f, "grammar syntax error: {s}"),
            GrammarError::Rejected(b) => write!(f, "grammar rejected byte 0x{b:02x}"),
        }
    }
}

impl std::error::Error for GrammarError {}
