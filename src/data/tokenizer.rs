use std::collections::HashMap;
use tokenizers::Tokenizer as HfTokenizer;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn encode_single(&self, c: char) -> Option<usize>;
    fn decode_single(&self, idx: usize) -> Option<char>;
    fn vocab_size(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct CharTokenizer {
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
    pub vocab_size: usize,
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&idx| self.idx_to_char.get(idx))
            .collect()
    }

    fn encode_single(&self, c: char) -> Option<usize> {
        self.char_to_idx.get(&c).copied()
    }

    fn decode_single(&self, idx: usize) -> Option<char> {
        self.idx_to_char.get(idx).copied()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let vocab_size = chars.len();
        let idx_to_char = chars.clone();
        let char_to_idx: HashMap<char, usize> =
            chars.into_iter().enumerate().map(|(i, c)| (c, i)).collect();

        CharTokenizer {
            char_to_idx,
            idx_to_char,
            vocab_size,
        }
    }
}

#[derive(Clone)]
pub struct HfTokenizerWrapper {
    tokenizer: HfTokenizer,
    vocab_size: usize,
}

impl HfTokenizerWrapper {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer =
            HfTokenizer::from_file(path).map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        let vocab_size = tokenizer.get_vocab_size(true);
        Ok(HfTokenizerWrapper {
            tokenizer,
            vocab_size,
        })
    }
}

impl Tokenizer for HfTokenizerWrapper {
    fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().iter().map(|&id| id as usize).collect())
            .unwrap_or_default()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        self.tokenizer.decode(&ids, false).unwrap_or_default()
    }

    fn encode_single(&self, c: char) -> Option<usize> {
        let s = c.to_string();
        self.tokenizer
            .encode(s.as_str(), false)
            .ok()
            .and_then(|encoding| encoding.get_ids().first().map(|&id| id as usize))
    }

    fn decode_single(&self, idx: usize) -> Option<char> {
        let decoded = self.tokenizer.decode(&[idx as u32], false).ok()?;
        decoded.chars().next()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[derive(Clone)]
pub enum TokenizerType {
    Char(CharTokenizer),
    HuggingFace(HfTokenizerWrapper),
}

impl Tokenizer for TokenizerType {
    fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            TokenizerType::Char(t) => t.encode(text),
            TokenizerType::HuggingFace(t) => t.encode(text),
        }
    }

    fn decode(&self, tokens: &[usize]) -> String {
        match self {
            TokenizerType::Char(t) => t.decode(tokens),
            TokenizerType::HuggingFace(t) => t.decode(tokens),
        }
    }

    fn encode_single(&self, c: char) -> Option<usize> {
        match self {
            TokenizerType::Char(t) => t.encode_single(c),
            TokenizerType::HuggingFace(t) => t.encode_single(c),
        }
    }

    fn decode_single(&self, idx: usize) -> Option<char> {
        match self {
            TokenizerType::Char(t) => t.decode_single(idx),
            TokenizerType::HuggingFace(t) => t.decode_single(idx),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            TokenizerType::Char(t) => t.vocab_size(),
            TokenizerType::HuggingFace(t) => t.vocab_size(),
        }
    }
}
