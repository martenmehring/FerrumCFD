use std::path::{Path, PathBuf};

use crate::{MeshError, Result};

#[derive(Clone, Debug)]
pub struct Token {
    pub value: String,
    pub line: usize,
}

pub fn tokenize(content: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    for (line_index, line) in content.lines().enumerate() {
        let mut current = String::new();
        let mut chars = line.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '/' && chars.peek() == Some(&'/') {
                break;
            }

            if ch == '"' {
                current.push(ch);
                for quoted in chars.by_ref() {
                    current.push(quoted);
                    if quoted == '"' {
                        break;
                    }
                }
                continue;
            }

            if ch.is_whitespace() {
                push_token(&mut tokens, &mut current, line_index + 1);
                continue;
            }

            if matches!(ch, '{' | '}' | '(' | ')' | '[' | ']' | ';') {
                push_token(&mut tokens, &mut current, line_index + 1);
                tokens.push(Token {
                    value: ch.to_string(),
                    line: line_index + 1,
                });
                continue;
            }

            current.push(ch);
        }
        push_token(&mut tokens, &mut current, line_index + 1);
    }
    tokens
}

fn push_token(tokens: &mut Vec<Token>, current: &mut String, line: usize) {
    if current.is_empty() {
        return;
    }

    tokens.push(Token {
        value: current.trim_matches('"').to_string(),
        line,
    });
    current.clear();
}

pub struct TokenCursor {
    path: PathBuf,
    tokens: Vec<Token>,
    index: usize,
}

impl TokenCursor {
    pub fn new(path: &Path, tokens: Vec<Token>) -> Self {
        Self {
            path: path.to_path_buf(),
            tokens,
            index: 0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn peek(&self) -> Option<&str> {
        self.tokens
            .get(self.index)
            .map(|token| token.value.as_str())
    }

    pub fn peek_next(&self) -> Option<&str> {
        self.tokens
            .get(self.index + 1)
            .map(|token| token.value.as_str())
    }

    pub fn peek_is(&self, expected: &str) -> Result<bool> {
        Ok(self.peek_required()? == expected)
    }

    pub fn expect(&mut self, expected: &str) -> Result<()> {
        let token = self.next_token()?;
        if token.value == expected {
            Ok(())
        } else {
            Err(MeshError::Parse {
                line: token.line,
                message: format!("expected '{expected}' but found '{}'", token.value),
            })
        }
    }

    pub fn expect_optional(&mut self, expected: &str) -> Result<()> {
        if self.peek() == Some(expected) {
            self.next_required()?;
        }
        Ok(())
    }

    pub fn next_required(&mut self) -> Result<String> {
        Ok(self.next_token()?.value)
    }

    pub fn read_value_until_semicolon(&mut self) -> Result<Vec<String>> {
        let mut values = Vec::new();
        let mut depth = 0usize;

        while let Some(token) = self.peek() {
            if token == ";" && depth == 0 {
                self.next_required()?;
                break;
            }
            if token == "}" && depth == 0 {
                break;
            }

            let token = self.next_required()?;
            match token.as_str() {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" if depth > 0 => depth -= 1,
                _ => {}
            }
            values.push(token);
        }

        Ok(values)
    }

    pub fn skip_braced_block(&mut self) -> Result<()> {
        self.expect("{")?;
        let mut depth = 1;
        while depth > 0 {
            let token = self.next_required()?;
            match token.as_str() {
                "{" => depth += 1,
                "}" => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    pub fn skip_value_or_block(&mut self) -> Result<()> {
        if self.peek() == Some("{") {
            self.skip_braced_block()?;
            return Ok(());
        }

        let mut depth = 0usize;
        while let Some(token) = self.peek() {
            if token == ";" && depth == 0 {
                self.next_required()?;
                break;
            }
            if token == "}" && depth == 0 {
                break;
            }

            let token = self.next_required()?;
            match token.as_str() {
                "{" | "(" | "[" => depth += 1,
                "}" | ")" | "]" if depth > 0 => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    fn peek_required(&self) -> Result<&str> {
        self.tokens
            .get(self.index)
            .map(|token| token.value.as_str())
            .ok_or_else(|| {
                MeshError::InvalidInput(format!("unexpected EOF in {}", self.path.display()))
            })
    }

    fn next_token(&mut self) -> Result<Token> {
        let token = self.tokens.get(self.index).cloned().ok_or_else(|| {
            MeshError::InvalidInput(format!("unexpected EOF in {}", self.path.display()))
        })?;
        self.index += 1;
        Ok(token)
    }
}
