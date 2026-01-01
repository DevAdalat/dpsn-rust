use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CharTokenizer {
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: Vec<char>,
    pub vocab_size: usize,
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

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&idx| self.idx_to_char.get(idx))
            .collect()
    }

    pub fn encode_single(&self, c: char) -> Option<usize> {
        self.char_to_idx.get(&c).copied()
    }

    pub fn decode_single(&self, idx: usize) -> Option<char> {
        self.idx_to_char.get(idx).copied()
    }
}
