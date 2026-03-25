use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
}

impl Tokenizer {
    /// Build vocabulary from text. Characters are sorted for deterministic ordering.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort();

        let char_to_idx: HashMap<char, usize> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        Tokenizer {
            char_to_idx,
            idx_to_char: chars,
        }
    }

    /// Encode a string into a sequence of token indices.
    ///
    /// # Panics
    /// Panics if the string contains a character not in the vocabulary.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| {
                *self.char_to_idx.get(&c).unwrap_or_else(|| {
                    panic!("character {:?} not in vocabulary", c)
                })
            })
            .collect()
    }

    /// Decode a sequence of token indices back into a string.
    ///
    /// # Panics
    /// Panics if any index is out of bounds.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                *self.idx_to_char.get(id).unwrap_or_else(|| {
                    panic!("index {} out of vocabulary range (size {})", id, self.idx_to_char.len())
                })
            })
            .collect()
    }

    /// Return the number of unique characters in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.idx_to_char.len()
    }

    /// Save vocabulary to a file (one char per line, as unicode codepoints).
    pub fn save_vocab(&self, path: &str) -> std::io::Result<()> {
        let content: String = self.idx_to_char.iter()
            .map(|c| format!("{}", *c as u32))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path, content)
    }

    /// Load vocabulary from a file.
    pub fn load_vocab(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let chars: Vec<char> = content.lines()
            .filter(|l| !l.is_empty())
            .map(|l| char::from_u32(l.parse::<u32>().unwrap()).unwrap())
            .collect();
        let char_to_idx: HashMap<char, usize> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        Ok(Tokenizer { char_to_idx, idx_to_char: chars })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let text = "hello world";
        let tok = Tokenizer::from_text(text);
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size() {
        let tok = Tokenizer::from_text("aabbc");
        assert_eq!(tok.vocab_size(), 3); // 'a', 'b', 'c'
    }

    #[test]
    fn test_sorted_determinism() {
        let tok = Tokenizer::from_text("zab");
        assert_eq!(tok.idx_to_char, vec!['a', 'b', 'z']);
        assert_eq!(tok.encode("a"), vec![0]);
        assert_eq!(tok.encode("z"), vec![2]);
    }

    #[test]
    fn test_empty_text() {
        let tok = Tokenizer::from_text("");
        assert_eq!(tok.vocab_size(), 0);
        assert_eq!(tok.encode(""), Vec::<usize>::new());
        assert_eq!(tok.decode(&[]), String::new());
    }

    #[test]
    #[should_panic(expected = "not in vocabulary")]
    fn test_encode_unknown_char() {
        let tok = Tokenizer::from_text("ab");
        tok.encode("c");
    }

    #[test]
    #[should_panic(expected = "out of vocabulary range")]
    fn test_decode_out_of_bounds() {
        let tok = Tokenizer::from_text("ab");
        tok.decode(&[99]);
    }
}
