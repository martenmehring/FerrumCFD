use std::path::{Path, PathBuf};

use crate::{MeshError, Result};

pub const MAX_DICTIONARY_NESTING: usize = 128;

#[derive(Clone, Debug)]
pub struct Token {
    pub value: String,
    pub line: usize,
}

pub fn tokenize(content: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut in_block_comment = false;
    for (line_index, line) in content.lines().enumerate() {
        let mut current = String::new();
        let mut chars = line.chars().peekable();
        let mut inline_paren_depth = 0usize;

        while let Some(ch) = chars.next() {
            if in_block_comment {
                if ch == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    in_block_comment = false;
                }
                continue;
            }

            if ch == '/' && chars.peek() == Some(&'*') {
                push_token(&mut tokens, &mut current, line_index + 1);
                chars.next();
                in_block_comment = true;
                continue;
            }

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
                inline_paren_depth = 0;
                continue;
            }

            if ch == '(' && !current.is_empty() {
                inline_paren_depth += 1;
                current.push(ch);
                continue;
            }

            if ch == ')' && inline_paren_depth > 0 {
                inline_paren_depth -= 1;
                current.push(ch);
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

pub mod streaming {
    use std::io::BufRead;
    use std::path::{Path, PathBuf};

    use crate::{MeshError, Result};

    pub const MAX_DICTIONARY_NESTING: usize = 128;
    pub const MAX_TOKEN_BYTES: usize = 1024 * 1024;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum TokenProvenance {
        Ordinary,
        Quoted,
        Structural,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct Token {
        pub value: String,
        pub line: usize,
        pub provenance: TokenProvenance,
    }

    #[derive(Clone)]
    struct Failure {
        line: usize,
        message: String,
    }

    pub struct TokenSource<R: BufRead> {
        path: PathBuf,
        reader: R,
        declared: usize,
        physical: usize,
        committed: usize,
        line: usize,
        decoded: Option<(char, usize)>,
        lookahead: Option<(Token, usize)>,
        eof_commit: Option<usize>,
        failure: Option<Failure>,
        stack: [char; MAX_DICTIONARY_NESTING],
        depth: usize,
    }

    impl<R: BufRead> TokenSource<R> {
        pub fn new(path: &Path, reader: R, exact_total_bytes: usize) -> Result<Self> {
            let mut owned = PathBuf::new();
            owned
                .try_reserve(path.as_os_str().len())
                .map_err(|_| MeshError::Parse {
                    line: 1,
                    message: "dictionary path allocation failed".to_owned(),
                })?;
            owned.push(path);
            Ok(Self {
                path: owned,
                reader,
                declared: exact_total_bytes,
                physical: 0,
                committed: 0,
                line: 1,
                decoded: None,
                lookahead: None,
                eof_commit: None,
                failure: None,
                stack: ['\0'; MAX_DICTIONARY_NESTING],
                depth: 0,
            })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
        pub fn declared_length(&self) -> usize {
            self.declared
        }
        pub fn physical_bytes_read(&self) -> usize {
            self.physical
        }
        pub fn remaining(&self) -> usize {
            self.declared
                .checked_sub(self.committed)
                .map_or(0, std::convert::identity)
        }

        pub fn peek(&mut self) -> Result<Option<&Token>> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            if self.lookahead.is_none() && self.eof_commit.is_none() {
                match self.lex() {
                    Ok(Some((token, end))) => self.lookahead = Some((token, end)),
                    Ok(None) => self.eof_commit = Some(self.physical),
                    Err((line, message)) => return Err(self.latch(line, message)),
                }
            }
            Ok(self.lookahead.as_ref().map(|entry| &entry.0))
        }

        #[allow(clippy::should_implement_trait)]
        pub fn next(&mut self) -> Result<Option<Token>> {
            self.peek()?;
            if let Some((token, end)) = self.lookahead.take() {
                self.committed = end;
                Ok(Some(token))
            } else {
                if let Some(end) = self.eof_commit {
                    self.committed = end;
                }
                Ok(None)
            }
        }

        pub fn next_required(&mut self) -> Result<Token> {
            match self.next()? {
                Some(token) => Ok(token),
                None => Err(self.latch(self.line, "unexpected end of dictionary")),
            }
        }

        fn sticky(&self) -> MeshError {
            let failure = self.failure.as_ref();
            match failure {
                Some(value) => MeshError::Parse {
                    line: value.line,
                    message: Self::copy_message(&value.message),
                },
                None => MeshError::Parse {
                    line: self.line,
                    message: "dictionary lexer failed".to_owned(),
                },
            }
        }

        fn latch(&mut self, line: usize, detail: &str) -> MeshError {
            if self.failure.is_none() {
                let path = self.path.to_str().unwrap_or("<non-UTF-8 dictionary path>");
                let capacity = path
                    .len()
                    .checked_add(2)
                    .and_then(|length| length.checked_add(detail.len()));
                let mut message = String::new();
                if capacity
                    .and_then(|length| message.try_reserve(length).ok())
                    .is_some()
                {
                    message.push_str(path);
                    message.push_str(": ");
                    message.push_str(detail);
                } else {
                    message.push_str("dictionary error allocation failed");
                }
                self.failure = Some(Failure { line, message });
            }
            self.sticky()
        }

        fn copy_message(message: &str) -> String {
            let mut copy = String::new();
            if copy.try_reserve(message.len()).is_ok() {
                copy.push_str(message);
            } else {
                copy.push_str("dictionary error allocation failed");
            }
            copy
        }

        fn byte(&mut self) -> std::result::Result<Option<u8>, (usize, &'static str)> {
            let bytes = self
                .reader
                .fill_buf()
                .map_err(|_| (self.line, "dictionary input read failed"))?;
            if bytes.is_empty() {
                if self.physical == self.declared {
                    return Ok(None);
                }
                return Err((
                    self.line,
                    "dictionary input ended before its declared length",
                ));
            }
            let value = bytes[0];
            self.reader.consume(1);
            self.physical = self
                .physical
                .checked_add(1)
                .ok_or((self.line, "dictionary byte counter overflow"))?;
            if self.physical > self.declared {
                return Err((self.line, "dictionary input exceeds its declared length"));
            }
            Ok(Some(value))
        }

        fn decode(&mut self) -> std::result::Result<Option<(char, usize)>, (usize, &'static str)> {
            if let Some(value) = self.decoded.take() {
                return Ok(Some(value));
            }
            let first = match self.byte()? {
                Some(value) => value,
                None => return Ok(None),
            };
            let width = if first < 0x80 {
                1
            } else if first & 0xe0 == 0xc0 {
                2
            } else if first & 0xf0 == 0xe0 {
                3
            } else if first & 0xf8 == 0xf0 {
                4
            } else {
                return Err((self.line, "invalid UTF-8 in dictionary"));
            };
            let mut data = [0u8; 4];
            data[0] = first;
            for slot in data.iter_mut().take(width).skip(1) {
                *slot = self
                    .byte()?
                    .ok_or((self.line, "truncated UTF-8 in dictionary"))?;
                if *slot & 0xc0 != 0x80 {
                    return Err((self.line, "invalid UTF-8 in dictionary"));
                }
            }
            let text = std::str::from_utf8(&data[..width])
                .map_err(|_| (self.line, "invalid UTF-8 in dictionary"))?;
            match text.chars().next() {
                Some(ch) => Ok(Some((ch, width))),
                None => Err((self.line, "invalid UTF-8 in dictionary")),
            }
        }

        fn take_char(
            &mut self,
        ) -> std::result::Result<Option<(char, usize)>, (usize, &'static str)> {
            let value = self.decode()?;
            if matches!(value, Some(('\n', _))) {
                self.line = self
                    .line
                    .checked_add(1)
                    .ok_or((self.line, "dictionary line counter overflow"))?;
            }
            Ok(value)
        }

        fn put_char(
            &mut self,
            value: (char, usize),
        ) -> std::result::Result<(), (usize, &'static str)> {
            if value.0 == '\n' {
                self.line = self
                    .line
                    .checked_sub(1)
                    .ok_or((self.line, "dictionary line counter underflow"))?;
            }
            self.decoded = Some(value);
            Ok(())
        }

        fn push_value(
            value: &mut String,
            ch: char,
            width: usize,
            line: usize,
        ) -> std::result::Result<(), (usize, &'static str)> {
            let wanted = value
                .len()
                .checked_add(width)
                .ok_or((line, "dictionary token length overflow"))?;
            if wanted > MAX_TOKEN_BYTES {
                return Err((line, "dictionary token byte limit exceeded"));
            }
            value
                .try_reserve(width)
                .map_err(|_| (line, "dictionary token allocation failed"))?;
            value.push(ch);
            Ok(())
        }

        fn structural(ch: char) -> bool {
            matches!(ch, '{' | '}' | '(' | ')' | '[' | ']' | ';')
        }
        fn matching(open: char, close: char) -> bool {
            matches!((open, close), ('{', '}') | ('(', ')') | ('[', ']'))
        }

        fn lex(&mut self) -> std::result::Result<Option<(Token, usize)>, (usize, &'static str)> {
            loop {
                let (ch, width) = match self.take_char()? {
                    Some(value) => value,
                    None => {
                        if self.depth != 0 {
                            return Err((self.line, "unclosed dictionary delimiter"));
                        }
                        return Ok(None);
                    }
                };
                let start = if ch == '\n' {
                    self.line
                        .checked_sub(1)
                        .ok_or((self.line, "dictionary line counter underflow"))?
                } else {
                    self.line
                };
                if ch.is_whitespace() {
                    continue;
                }
                if ch == '/' {
                    match self.take_char()? {
                        Some(('/', _)) => {
                            while let Some((next, _)) = self.take_char()? {
                                if next == '\n' {
                                    break;
                                }
                            }
                            continue;
                        }
                        Some(('*', _)) => {
                            let mut star = false;
                            loop {
                                match self.take_char()? {
                                    Some(('/', _)) if star => break,
                                    Some(('*', _)) => star = true,
                                    Some(_) => star = false,
                                    None => return Err((self.line, "unclosed block comment")),
                                }
                            }
                            continue;
                        }
                        Some(next) => self.put_char(next)?,
                        None => {}
                    }
                }
                if ch == '"' {
                    let mut value = String::new();
                    loop {
                        match self.take_char()? {
                            Some(('"', _)) => break,
                            Some((next, bytes)) => {
                                Self::push_value(&mut value, next, bytes, start)?
                            }
                            None => return Err((self.line, "unclosed quoted token")),
                        }
                    }
                    return Ok(Some((
                        Token {
                            value,
                            line: start,
                            provenance: TokenProvenance::Quoted,
                        },
                        self.physical,
                    )));
                }
                if Self::structural(ch) {
                    let mut token_end = self.physical;
                    if matches!(ch, '}' | ')' | ']') {
                        let top = self
                            .depth
                            .checked_sub(1)
                            .ok_or((start, "dictionary nesting counter underflow"))?;
                        if !Self::matching(self.stack[top], ch) {
                            return Err((start, "mismatched dictionary delimiter"));
                        }
                        self.depth = self
                            .depth
                            .checked_sub(1)
                            .ok_or((start, "dictionary nesting counter underflow"))?;
                    } else if ch != ';' {
                        if self.depth == MAX_DICTIONARY_NESTING {
                            return Err((start, "dictionary nesting limit exceeded"));
                        }
                        self.stack[self.depth] = ch;
                        self.depth = self
                            .depth
                            .checked_add(1)
                            .ok_or((start, "dictionary nesting counter overflow"))?;
                        match self.take_char()? {
                            Some(next) => {
                                token_end = self
                                    .physical
                                    .checked_sub(next.1)
                                    .ok_or((start, "dictionary byte counter underflow"))?;
                                self.put_char(next)?;
                            }
                            None => return Err((self.line, "unclosed dictionary delimiter")),
                        }
                    }
                    let mut value = String::new();
                    Self::push_value(&mut value, ch, width, start)?;
                    return Ok(Some((
                        Token {
                            value,
                            line: start,
                            provenance: TokenProvenance::Structural,
                        },
                        token_end,
                    )));
                }
                let mut value = String::new();
                Self::push_value(&mut value, ch, width, start)?;
                let mut function_stack = ['\0'; MAX_DICTIONARY_NESTING];
                let mut function_depth = 0usize;
                let mut token_end = self.physical;
                #[allow(clippy::while_let_loop)]
                loop {
                    let next = match self.take_char()? {
                        Some(value) => value,
                        None => break,
                    };
                    if next.0.is_whitespace() && function_depth == 0 {
                        token_end = self
                            .physical
                            .checked_sub(next.1)
                            .ok_or((start, "dictionary byte counter underflow"))?;
                        break;
                    }
                    if next.0 == '/' {
                        let slash_start = self
                            .physical
                            .checked_sub(next.1)
                            .ok_or((start, "dictionary byte counter underflow"))?;
                        match self.take_char()? {
                            Some(('/', _)) => {
                                while let Some((comment, _)) = self.take_char()? {
                                    if comment == '\n' {
                                        break;
                                    }
                                }
                                if function_depth == 0 {
                                    token_end = slash_start;
                                    break;
                                }
                                continue;
                            }
                            Some(('*', _)) => {
                                let mut star = false;
                                loop {
                                    match self.take_char()? {
                                        Some(('/', _)) if star => break,
                                        Some(('*', _)) => star = true,
                                        Some(_) => star = false,
                                        None => return Err((self.line, "unclosed block comment")),
                                    }
                                }
                                if function_depth == 0 {
                                    token_end = slash_start;
                                    break;
                                }
                                continue;
                            }
                            Some(after) => {
                                Self::push_value(&mut value, '/', 1, start)?;
                                self.put_char(after)?;
                                continue;
                            }
                            None => {
                                Self::push_value(&mut value, '/', 1, start)?;
                                break;
                            }
                        }
                    }
                    if next.0 == '"' && function_depth != 0 {
                        Self::push_value(&mut value, next.0, next.1, start)?;
                        loop {
                            match self.take_char()? {
                                Some((quoted, bytes)) => {
                                    Self::push_value(&mut value, quoted, bytes, start)?;
                                    if quoted == '"' {
                                        break;
                                    }
                                }
                                None => return Err((self.line, "unclosed quoted token")),
                            }
                        }
                        token_end = self.physical;
                        continue;
                    }
                    if matches!(next.0, '(' | '[' | '{') {
                        if self
                            .depth
                            .checked_add(function_depth)
                            .ok_or((start, "dictionary nesting overflow"))?
                            == MAX_DICTIONARY_NESTING
                        {
                            return Err((start, "dictionary nesting limit exceeded"));
                        }
                        function_stack[function_depth] = next.0;
                        function_depth = function_depth
                            .checked_add(1)
                            .ok_or((start, "dictionary nesting overflow"))?;
                        Self::push_value(&mut value, next.0, next.1, start)?;
                        continue;
                    }
                    if matches!(next.0, ')' | ']' | '}') && function_depth > 0 {
                        let top = function_depth
                            .checked_sub(1)
                            .ok_or((start, "dictionary nesting counter underflow"))?;
                        if !Self::matching(function_stack[top], next.0) {
                            return Err((start, "mismatched function delimiter"));
                        }
                        function_depth = top;
                        Self::push_value(&mut value, next.0, next.1, start)?;
                        continue;
                    }
                    if Self::structural(next.0) && function_depth == 0 {
                        token_end = self
                            .physical
                            .checked_sub(next.1)
                            .ok_or((start, "dictionary byte counter underflow"))?;
                        self.put_char(next)?;
                        break;
                    }
                    Self::push_value(&mut value, next.0, next.1, start)?;
                    token_end = self.physical;
                }
                if function_depth != 0 {
                    return Err((self.line, "unclosed function delimiter"));
                }
                return Ok(Some((
                    Token {
                        value,
                        line: start,
                        provenance: TokenProvenance::Ordinary,
                    },
                    token_end,
                )));
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::io::{self, BufReader, Cursor, Read};
        use std::path::Path;

        use crate::MeshError;

        use super::{MAX_DICTIONARY_NESTING, MAX_TOKEN_BYTES, TokenProvenance, TokenSource};

        fn source(data: &[u8]) -> TokenSource<BufReader<Cursor<Vec<u8>>>> {
            TokenSource::new(
                Path::new("fixture"),
                BufReader::with_capacity(1, Cursor::new(data.to_vec())),
                data.len(),
            )
            .unwrap()
        }

        fn values(data: &[u8]) -> Vec<(String, TokenProvenance)> {
            let mut lexer = source(data);
            let mut result = Vec::new();
            while let Some(token) = lexer.next().unwrap() {
                result.push((token.value, token.provenance));
            }
            result
        }

        #[test]
        fn physical_count_includes_extra_probe() {
            let mut lexer = TokenSource::new(Path::new("x"), Cursor::new(b"ab"), 1).unwrap();
            assert!(lexer.peek().is_err());
            assert_eq!(lexer.physical_bytes_read(), 2);

            for data in [b"( x )".as_slice(), b"(x)".as_slice()] {
                let mut lexer = source(data);
                assert_eq!(lexer.next().unwrap().unwrap().value, "(");
                assert_eq!(lexer.remaining(), data.len() - 1);
            }
        }

        #[test]
        fn terminal_errors_are_sticky() {
            let mut lexer = source(b"]");
            let first = lexer.peek().unwrap_err().to_string();
            assert_eq!(lexer.next().unwrap_err().to_string(), first);
            assert_eq!(lexer.next_required().unwrap_err().to_string(), first);
        }

        #[test]
        fn peek_none_is_noncommitting() {
            let mut lexer = source(b" \n");
            assert!(lexer.peek().unwrap().is_none());
            assert_eq!(lexer.remaining(), 2);
            assert!(lexer.next().unwrap().is_none());
            assert_eq!(lexer.remaining(), 0);
        }

        #[test]
        fn unicode_trivia_belongs_to_following_token() {
            let mut lexer = source("a\u{2003}b".as_bytes());
            assert_eq!(lexer.next().unwrap().unwrap().value, "a");
            assert_eq!(lexer.remaining(), 4);
            let before = lexer.remaining();
            assert_eq!(lexer.peek().unwrap().unwrap().value, "b");
            assert_eq!(lexer.remaining(), before);
            assert_eq!(lexer.next().unwrap().unwrap().value, "b");
        }

        #[test]
        fn comments_inside_function_tokens() {
            assert_eq!(values(b"grad(/*x*/U)")[0].0, "grad(U)");
        }

        #[test]
        fn adjacent_comments_belong_to_following_token() {
            let mut lexer = source(b"a/*x*///y\nb");
            assert_eq!(lexer.next().unwrap().unwrap().value, "a");
            assert_eq!(lexer.remaining(), 10);
            assert_eq!(lexer.peek().unwrap().unwrap().value, "b");
            assert_eq!(lexer.next().unwrap().unwrap().line, 2);
        }

        #[test]
        fn quoted_and_slash_data_preserve_provenance() {
            let tokens = values(br#""a//b/*c*/" a/b"#);
            assert_eq!(tokens[0], ("a//b/*c*/".to_owned(), TokenProvenance::Quoted));
            assert_eq!(tokens[1], ("a/b".to_owned(), TokenProvenance::Ordinary));
            assert_eq!(values(br#"f("a//b/*c*/")"#)[0].0, r#"f("a//b/*c*/")"#);
        }

        #[test]
        fn one_byte_utf8_chunks_decode() {
            let tokens = values("α \"β\" // γ\nδ /* ε */ ζ".as_bytes());
            assert_eq!(
                tokens.iter().map(|v| v.0.as_str()).collect::<Vec<_>>(),
                vec!["α", "β", "δ", "ζ"]
            );
        }

        #[test]
        fn four_invalid_utf8_states_are_sticky() {
            for data in [
                b"a\xff".as_slice(),
                b"\"\xff".as_slice(),
                b"//\xff".as_slice(),
                b"/*\xff".as_slice(),
            ] {
                let mut lexer = source(data);
                let first = lexer.peek().unwrap_err().to_string();
                assert_eq!(lexer.peek().unwrap_err().to_string(), first);
            }
        }

        struct Failing {
            data: Vec<u8>,
            at: usize,
            position: usize,
        }
        impl Read for Failing {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                if self.position == self.at {
                    return Err(io::Error::other("injected"));
                }
                if self.position == self.data.len() {
                    return Ok(0);
                }
                output[0] = self.data[self.position];
                self.position += 1;
                Ok(1)
            }
        }

        #[test]
        fn six_io_failures_are_sticky_and_line_aware() {
            for (data, at, expected_line) in [
                (b"word".as_slice(), 0, 1),
                (b"\nword".as_slice(), 3, 2),
                ("\né".as_bytes(), 2, 2),
                (b"\n\"x".as_slice(), 2, 2),
                (b"\n//x".as_slice(), 3, 2),
                (b"\n/*x".as_slice(), 3, 2),
            ] {
                let reader = BufReader::with_capacity(
                    1,
                    Failing {
                        data: data.to_vec(),
                        at,
                        position: 0,
                    },
                );
                let mut lexer = TokenSource::new(Path::new("io"), reader, data.len()).unwrap();
                let first = lexer.peek().unwrap_err();
                assert!(matches!(
                    first,
                    MeshError::Parse {
                        line,
                        ..
                    } if line == expected_line
                ));
                let first = first.to_string();
                assert_eq!(lexer.next().unwrap_err().to_string(), first);
            }
        }

        #[test]
        fn declared_length_mismatches_fail_at_current_line() {
            let mut extra = TokenSource::new(Path::new("length"), Cursor::new(b"a\n"), 1).unwrap();
            assert!(matches!(
                extra.peek().unwrap_err(),
                MeshError::Parse { line: 1, .. }
            ));

            let mut early = TokenSource::new(Path::new("length"), Cursor::new(b"a\n"), 3).unwrap();
            early.next().unwrap();
            assert!(matches!(
                early.peek().unwrap_err(),
                MeshError::Parse { line: 2, .. }
            ));
        }

        #[test]
        fn source_and_function_delimiters_share_depth_budget() {
            let mut okay = "{".repeat(MAX_DICTIONARY_NESTING - 1);
            okay.push_str("f(x)");
            okay.push_str(&"}".repeat(MAX_DICTIONARY_NESTING - 1));
            let mut lexer = source(okay.as_bytes());
            while lexer.next().unwrap().is_some() {}
            let too_deep = format!("{}f(x)", "{".repeat(MAX_DICTIONARY_NESTING));
            assert!(source(too_deep.as_bytes()).peek().is_ok());
            let mut lexer = source(too_deep.as_bytes());
            for _ in 0..MAX_DICTIONARY_NESTING {
                lexer.next().unwrap();
            }
            assert!(lexer.peek().is_err());

            let combined = format!("f({})", "[".repeat(MAX_DICTIONARY_NESTING));
            assert!(source(combined.as_bytes()).peek().is_err());
            assert!(source(b"f(x]").peek().is_err());
            assert!(source(b"{").peek().is_err());
        }

        #[test]
        fn token_byte_cap_is_exact() {
            assert_eq!(
                source(&vec![b'a'; MAX_TOKEN_BYTES])
                    .next()
                    .unwrap()
                    .unwrap()
                    .value
                    .len(),
                MAX_TOKEN_BYTES
            );
            assert!(source(&vec![b'a'; MAX_TOKEN_BYTES + 1]).peek().is_err());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tokenize;

    #[test]
    fn keeps_function_style_dictionary_keys_together() {
        let tokens = tokenize("grad(U) Gauss linear;");
        let values = tokens
            .iter()
            .map(|token| token.value.as_str())
            .collect::<Vec<_>>();

        assert_eq!(values, vec!["grad(U)", "Gauss", "linear", ";"]);
    }

    #[test]
    fn keeps_parenthesized_values_as_lists() {
        let tokens = tokenize("internalField uniform (0 0 0);");
        let values = tokens
            .iter()
            .map(|token| token.value.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            values,
            vec!["internalField", "uniform", "(", "0", "0", "0", ")", ";"]
        );
    }

    #[test]
    fn skips_openfoam_multiline_block_comments() {
        let tokens = tokenize(
            "/* OpenFOAM\n   generated banner */\nFoamFile { class volVectorField; } /* tail */",
        );
        let values = tokens
            .iter()
            .map(|token| token.value.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            values,
            vec!["FoamFile", "{", "class", "volVectorField", ";", "}"]
        );
    }
}
