pub mod streaming {
    use std::io::BufRead;
    use std::path::{Path, PathBuf};

    use crate::{MeshError, Result};

    pub const MAX_DICTIONARY_NESTING: usize = 128;
    pub const MAX_TOKEN_BYTES: usize = 1024 * 1024;
    pub const MAX_DICTIONARY_TOKENS: usize = 1_000_000;
    pub const MAX_DICTIONARY_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

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

    fn unsupported_directive(token: &Token) -> bool {
        token.provenance == TokenProvenance::Ordinary && token.value.starts_with('#')
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
        #[cfg(test)]
        inject_token_reservation_failure: bool,
        #[cfg(test)]
        inject_batch_reservation_failure: bool,
        #[cfg(test)]
        inject_diagnostic_reservation_failure: bool,
    }

    impl<R: BufRead> TokenSource<R> {
        #[cfg(test)]
        fn inject_physical_overflow(&mut self) {
            self.physical = usize::MAX;
        }

        #[cfg(test)]
        fn inject_commit_bound_violation(&mut self) {
            self.lookahead.as_mut().expect("lookahead required").1 = self.declared + 1;
        }

        #[cfg(test)]
        fn inject_source_token_reservation_failure(&mut self) {
            self.inject_token_reservation_failure = true;
        }

        #[cfg(test)]
        fn inject_batch_reservation_failure(&mut self) {
            self.inject_batch_reservation_failure = true;
        }

        #[cfg(test)]
        fn inject_source_diagnostic_reservation_failure(&mut self) {
            self.inject_diagnostic_reservation_failure = true;
        }

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
                #[cfg(test)]
                inject_token_reservation_failure: false,
                #[cfg(test)]
                inject_batch_reservation_failure: false,
                #[cfg(test)]
                inject_diagnostic_reservation_failure: false,
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

        pub(crate) fn checked_remaining_bytes(&mut self) -> Result<usize> {
            self.declared
                .checked_sub(self.committed)
                .ok_or_else(|| self.latch(self.line, "dictionary remaining-byte counter underflow"))
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
    }

    pub struct TokenBatch {
        path: PathBuf,
        tokens: Vec<Token>,
        eof_line: usize,
        payload_bytes: usize,
    }

    impl TokenBatch {
        pub fn path(&self) -> &Path {
            &self.path
        }
        pub fn tokens(&self) -> &[Token] {
            &self.tokens
        }
        pub fn len(&self) -> usize {
            self.tokens.len()
        }
        pub fn is_empty(&self) -> bool {
            self.tokens.is_empty()
        }
        pub fn eof_line(&self) -> usize {
            self.eof_line
        }
        pub fn payload_bytes(&self) -> usize {
            self.payload_bytes
        }
        pub fn into_cursor(self) -> TokenCursor {
            TokenCursor {
                path: self.path,
                tokens: self.tokens.into_iter(),
                eof_line: self.eof_line,
                failure: None,
                #[cfg(test)]
                inject_diagnostic_reservation_failure: false,
            }
        }
    }

    pub fn tokenize(path: &Path, content: &str) -> Result<TokenBatch> {
        tokenize_reader(
            path,
            std::io::Cursor::new(content.as_bytes()),
            content.len(),
        )
    }

    pub fn tokenize_reader<R: BufRead>(
        path: &Path,
        reader: R,
        exact_total_bytes: usize,
    ) -> Result<TokenBatch> {
        TokenSource::new(path, reader, exact_total_bytes)?.into_batch()
    }

    pub struct TokenCursor {
        path: PathBuf,
        tokens: std::vec::IntoIter<Token>,
        eof_line: usize,
        failure: Option<Failure>,
        #[cfg(test)]
        inject_diagnostic_reservation_failure: bool,
    }

    impl TokenCursor {
        #[cfg(test)]
        fn inject_cursor_diagnostic_reservation_failure(&mut self) {
            self.inject_diagnostic_reservation_failure = true;
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
        pub fn eof_line(&self) -> usize {
            self.eof_line
        }

        #[allow(dead_code)]
        pub(crate) fn remaining_tokens(&mut self) -> Result<&[Token]> {
            if self.failure.is_some() {
                Err(self.sticky())
            } else {
                Ok(self.tokens.as_slice())
            }
        }
        #[allow(dead_code)]
        pub(crate) fn reject_current(&mut self, detail: &'static str) -> Result<()> {
            self.reject_current_as(detail)
        }
        #[allow(dead_code)]
        pub(crate) fn reject_current_as<T>(&mut self, detail: &'static str) -> Result<T> {
            self.reject_at_as(0, detail)
        }
        pub(crate) fn reject_line_as<T>(&mut self, line: usize, detail: &'static str) -> Result<T> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            Err(self.latch(line, detail))
        }
        #[allow(dead_code)]
        pub(crate) fn reject_at_as<T>(&mut self, offset: usize, detail: &'static str) -> Result<T> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            let line = self
                .tokens
                .as_slice()
                .get(offset)
                .map_or(self.eof_line, |token| token.line);
            Err(self.latch(line, detail))
        }
        #[allow(dead_code)]
        pub(crate) fn try_reserve_current_value(&mut self, additional: usize) -> Result<()> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            let Some(token) = self.tokens.as_mut_slice().first_mut() else {
                return Err(self.latch(self.eof_line, "unexpected end of dictionary"));
            };
            let line = token.line;
            token
                .value
                .try_reserve(additional)
                .map_err(|_| self.latch(line, "dictionary token allocation failed"))
        }

        pub fn peek(&mut self) -> Result<Option<&Token>> {
            if self.failure.is_some() {
                Err(self.sticky())
            } else {
                Ok(self.tokens.as_slice().first())
            }
        }
        pub fn peek_next(&mut self) -> Result<Option<&Token>> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            Ok(self.tokens.as_slice().get(1))
        }
        #[allow(clippy::should_implement_trait)]
        pub fn next(&mut self) -> Result<Option<Token>> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            Ok(self.tokens.next())
        }
        pub fn next_required(&mut self) -> Result<Token> {
            match self.next()? {
                Some(token) if unsupported_directive(&token) => {
                    Err(self.latch_token(token.line, "unsupported dictionary directive"))
                }
                Some(token) => Ok(token),
                None => Err(self.latch(self.eof_line, "unexpected end of dictionary")),
            }
        }

        pub fn expect(&mut self, expected: &str) -> Result<()> {
            let token = self.next_required()?;
            let provenance = if Self::is_syntax(expected) {
                TokenProvenance::Structural
            } else {
                TokenProvenance::Ordinary
            };
            if token.value == expected && token.provenance == provenance {
                return Ok(());
            }
            Err(self.latch_token(token.line, "unexpected dictionary token"))
        }
        pub fn expect_optional(&mut self, expected: &str) -> Result<bool> {
            let provenance = if Self::is_syntax(expected) {
                TokenProvenance::Structural
            } else {
                TokenProvenance::Ordinary
            };
            let matches = self
                .peek()?
                .is_some_and(|token| token.value == expected && token.provenance == provenance);
            if matches {
                self.next_required()?;
            }
            Ok(matches)
        }
        pub fn expect_keyword(&mut self, expected: &str) -> Result<()> {
            let token = self.next_required()?;
            if token.value == expected && token.provenance == TokenProvenance::Ordinary {
                return Ok(());
            }
            Err(self.latch_token(token.line, "unexpected dictionary token"))
        }
        pub fn expect_optional_keyword(&mut self, expected: &str) -> Result<bool> {
            let matches = self.peek()?.is_some_and(|token| {
                token.value == expected && token.provenance == TokenProvenance::Ordinary
            });
            if matches {
                self.next_required()?;
            }
            Ok(matches)
        }

        pub fn read_value_until_semicolon(&mut self) -> Result<Vec<String>> {
            self.read_strict_value()
        }
        pub fn read_strict_value(&mut self) -> Result<Vec<String>> {
            self.read_strict_value_impl(false)
        }
        pub(crate) fn read_provenance_preserving_bare_entry(&mut self) -> Result<Vec<String>> {
            if self.peek()?.is_some_and(|t| Self::structural(t, ";")) {
                self.next_required()?;
                Ok(Vec::new())
            } else {
                self.read_strict_value_impl(true)
            }
        }
        fn read_strict_value_impl(&mut self, preserve_quoted: bool) -> Result<Vec<String>> {
            if self
                .peek()?
                .is_some_and(|t| Self::structural(t, ";") || Self::closer(t))
            {
                let eof_line = self.eof_line;
                let line = self.peek()?.map_or(eof_line, |t| t.line);
                return Err(self.latch(line, "dictionary value is missing"));
            }
            let mut values = Vec::new();
            let mut payload = 0usize;
            let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
            let mut depth = 0usize;
            loop {
                let mut token = self.next_required()?;
                if depth == 0 && Self::structural(&token, ";") {
                    return Ok(values);
                }
                if depth == 0 && Self::closer(&token) {
                    return Err(self.latch(token.line, "dictionary value is missing a semicolon"));
                }
                Self::track_delimiter(&token, &mut stack, &mut depth)
                    .map_err(|detail| self.latch(token.line, detail))?;
                if preserve_quoted && token.provenance == TokenProvenance::Quoted {
                    let represented_len = token.value.len().checked_add(2).ok_or_else(|| {
                        self.latch(token.line, "dictionary token length overflow")
                    })?;
                    if represented_len > MAX_TOKEN_BYTES {
                        return Err(self.latch(token.line, "dictionary token byte limit exceeded"));
                    }
                    token.value.try_reserve(2).map_err(|_| {
                        self.latch(token.line, "dictionary quoted value allocation failed")
                    })?;
                    token.value.insert(0, '"');
                    token.value.push('"');
                }
                payload = payload
                    .checked_add(token.value.len())
                    .ok_or_else(|| self.latch(token.line, "dictionary payload length overflow"))?;
                if payload > MAX_DICTIONARY_PAYLOAD_BYTES {
                    return Err(self.latch(token.line, "dictionary payload byte limit exceeded"));
                }
                values
                    .try_reserve(1)
                    .map_err(|_| self.latch(token.line, "dictionary value allocation failed"))?;
                values.push(token.value);
            }
        }
        pub fn read_bare_entry(&mut self) -> Result<Vec<String>> {
            if self.peek()?.is_some_and(|t| Self::structural(t, ";")) {
                self.next_required()?;
                Ok(Vec::new())
            } else {
                self.read_strict_value()
            }
        }
        pub fn skip_typed_balanced(&mut self) -> Result<()> {
            let first = self.next_required()?;
            if !Self::opener(&first) {
                return Err(self.latch(first.line, "expected dictionary delimiter"));
            }
            let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
            let mut depth = 0usize;
            Self::track_delimiter(&first, &mut stack, &mut depth)
                .map_err(|d| self.latch(first.line, d))?;
            while depth != 0 {
                let token = self.next_required()?;
                Self::track_delimiter(&token, &mut stack, &mut depth)
                    .map_err(|d| self.latch(token.line, d))?;
            }
            self.expect_optional(";")?;
            Ok(())
        }
        pub fn skip_braced_block(&mut self) -> Result<()> {
            if !self.peek()?.is_some_and(|t| Self::structural(t, "{")) {
                let eof_line = self.eof_line;
                let line = self.peek()?.map_or(eof_line, |t| t.line);
                return Err(self.latch(line, "expected dictionary block"));
            }
            self.skip_typed_balanced()
        }
        pub fn skip_value_or_block(&mut self) -> Result<()> {
            if self.peek()?.is_some_and(|t| Self::structural(t, "{")) {
                return self.skip_typed_balanced();
            }
            if self
                .peek()?
                .is_some_and(|t| Self::structural(t, ";") || Self::closer(t))
            {
                let eof_line = self.eof_line;
                let line = self.peek()?.map_or(eof_line, |t| t.line);
                return Err(self.latch(line, "dictionary value is missing"));
            }
            let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
            let mut depth = 0usize;
            loop {
                let token = self.next_required()?;
                if depth == 0 && Self::structural(&token, ";") {
                    return Ok(());
                }
                if depth == 0 && Self::closer(&token) {
                    return Err(self.latch(token.line, "dictionary value is missing a semicolon"));
                }
                Self::track_delimiter(&token, &mut stack, &mut depth)
                    .map_err(|d| self.latch(token.line, d))?;
            }
        }
        pub(crate) fn skip_exact_value_or_block(&mut self) -> Result<()> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            let plan = {
                let tokens = self.tokens.as_slice();
                (|| -> std::result::Result<(usize, bool), (usize, &'static str)> {
                    let first = tokens.first().ok_or((0, "dictionary value is missing"))?;
                    if Self::structural(first, ";") || Self::closer(first) {
                        return Err((0, "dictionary value is missing"));
                    }
                    let braced = Self::structural(first, "{");
                    let end = if Self::opener(first) {
                        let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
                        let mut depth = 0usize;
                        let mut result = None;
                        for (index, token) in tokens.iter().enumerate() {
                            if let Err(detail) =
                                Self::track_delimiter(token, &mut stack, &mut depth)
                            {
                                result = Some(Err((index, detail)));
                                break;
                            }
                            if depth == 0 {
                                result = Some(
                                    index
                                        .checked_add(1)
                                        .ok_or((index, "dictionary index overflow")),
                                );
                                break;
                            }
                        }
                        result.unwrap_or(Err((tokens.len(), "unterminated dictionary group")))?
                    } else {
                        1
                    };
                    let has_terminator = tokens
                        .get(end)
                        .is_some_and(|token| Self::structural(token, ";"));
                    if !braced && !has_terminator {
                        return Err((end, "dictionary value is missing a semicolon"));
                    }
                    Ok((end, has_terminator))
                })()
            };
            let (end, has_terminator) = match plan {
                Ok(value) => value,
                Err((offset, detail)) => return self.reject_at_as(offset, detail),
            };
            for _ in 0..end {
                self.next_required()?;
            }
            if has_terminator {
                self.next_required()?;
            }
            Ok(())
        }
        fn is_syntax(value: &str) -> bool {
            matches!(value, "{" | "}" | "(" | ")" | "[" | "]" | ";")
        }
        fn structural(token: &Token, value: &str) -> bool {
            token.provenance == TokenProvenance::Structural && token.value == value
        }
        fn opener(token: &Token) -> bool {
            token.provenance == TokenProvenance::Structural
                && matches!(token.value.as_str(), "{" | "(" | "[")
        }
        fn closer(token: &Token) -> bool {
            token.provenance == TokenProvenance::Structural
                && matches!(token.value.as_str(), "}" | ")" | "]")
        }
        fn track_delimiter(
            token: &Token,
            stack: &mut [char; MAX_DICTIONARY_NESTING],
            depth: &mut usize,
        ) -> std::result::Result<(), &'static str> {
            if token.provenance != TokenProvenance::Structural {
                return Ok(());
            }
            let ch = match token.value.as_str() {
                "{" => '{',
                "(" => '(',
                "[" => '[',
                "}" => '}',
                ")" => ')',
                "]" => ']',
                _ => return Ok(()),
            };
            if matches!(ch, '{' | '(' | '[') {
                if *depth == MAX_DICTIONARY_NESTING {
                    return Err("dictionary nesting limit exceeded");
                }
                stack[*depth] = ch;
                *depth = depth
                    .checked_add(1)
                    .ok_or("dictionary nesting counter overflow")?;
            } else {
                let top = depth
                    .checked_sub(1)
                    .ok_or("unexpected dictionary closing delimiter")?;
                if !TokenSource::<std::io::Empty>::matching(stack[top], ch) {
                    return Err("mismatched dictionary delimiter");
                }
                *depth = top;
            }
            Ok(())
        }
        fn sticky(&self) -> MeshError {
            match &self.failure {
                Some(f) => MeshError::Parse {
                    line: f.line,
                    message: TokenSource::<std::io::Empty>::copy_message(&f.message),
                },
                None => MeshError::Parse {
                    line: self.eof_line,
                    message: String::new(),
                },
            }
        }
        fn latch_token(&mut self, line: usize, detail: &str) -> MeshError {
            self.latch(line, detail)
        }
        fn latch(&mut self, line: usize, detail: &str) -> MeshError {
            if self.failure.is_none() {
                #[allow(clippy::manual_unwrap_or)]
                let path = match self.path.to_str() {
                    Some(value) => value,
                    None => "<non-UTF-8 dictionary path>",
                };
                let capacity = path
                    .len()
                    .checked_add(2)
                    .and_then(|n| n.checked_add(detail.len()));
                let mut message = String::new();
                #[cfg(test)]
                let requested = if self.inject_diagnostic_reservation_failure {
                    Some(usize::MAX)
                } else {
                    capacity
                };
                #[cfg(not(test))]
                let requested = capacity;
                if requested
                    .and_then(|n| message.try_reserve(n).ok())
                    .is_some()
                {
                    message.push_str(path);
                    message.push_str(": ");
                    message.push_str(detail);
                }
                self.failure = Some(Failure { line, message });
            }
            self.sticky()
        }
    }

    impl<R: BufRead> TokenSource<R> {
        #[allow(clippy::should_implement_trait)]
        pub fn next(&mut self) -> Result<Option<Token>> {
            self.peek()?;
            if let Some((_, end)) = self.lookahead.as_ref() {
                self.validate_commit(*end)?;
            }
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
                Some(token) if unsupported_directive(&token) => {
                    Err(self.latch(token.line, "unsupported dictionary directive"))
                }
                Some(token) => Ok(token),
                None => Err(self.latch(self.line, "unexpected end of dictionary")),
            }
        }

        pub(crate) fn expect_optional(&mut self, expected: &str) -> Result<bool> {
            let provenance = if Self::is_syntax(expected) {
                TokenProvenance::Structural
            } else {
                TokenProvenance::Ordinary
            };
            let matches = self
                .peek()?
                .is_some_and(|token| token.value == expected && token.provenance == provenance);
            if matches {
                self.next_required()?;
            }
            Ok(matches)
        }

        pub(crate) fn reject_current_as<T>(&mut self, detail: &'static str) -> Result<T> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            let eof_line = self.line;
            let line = self.peek()?.map_or(eof_line, |token| token.line);
            Err(self.latch(line, detail))
        }

        pub(crate) fn reject_line_as<T>(&mut self, line: usize, detail: &'static str) -> Result<T> {
            if self.failure.is_some() {
                return Err(self.sticky());
            }
            Err(self.latch(line, detail))
        }

        pub(crate) fn discard_exact_value_or_block(&mut self) -> Result<()> {
            let first = self.next_required()?;
            if Self::token_is(&first, ";") || Self::token_closer(&first) {
                return self.reject_line_as(first.line, "dictionary value is missing");
            }

            if Self::token_opener(&first) {
                let braced = Self::token_is(&first, "{");
                let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
                let mut depth = 0usize;
                Self::track_token_delimiter(&first, &mut stack, &mut depth)
                    .map_err(|detail| self.latch(first.line, detail))?;
                while depth != 0 {
                    let token = self.next_required()?;
                    Self::track_token_delimiter(&token, &mut stack, &mut depth)
                        .map_err(|detail| self.latch(token.line, detail))?;
                }
                if braced {
                    self.expect_optional(";")?;
                    return Ok(());
                }
                if !self.peek()?.is_some_and(|token| Self::token_is(token, ";")) {
                    return self.reject_current_as("dictionary value is missing a semicolon");
                }
                self.next_required()?;
                return Ok(());
            }

            if first.provenance == TokenProvenance::Structural {
                return self.reject_line_as(first.line, "dictionary value is missing");
            }
            if !self.peek()?.is_some_and(|token| Self::token_is(token, ";")) {
                return self.reject_current_as("dictionary value is missing a semicolon");
            }
            self.next_required()?;
            Ok(())
        }

        pub(crate) fn discard_semicolon_terminated_value_or_block(&mut self) -> Result<()> {
            let first = self.next_required()?;
            if Self::token_is(&first, ";") || Self::token_closer(&first) {
                return self.reject_line_as(first.line, "dictionary value is missing");
            }

            if Self::token_is(&first, "{") {
                let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
                let mut depth = 0usize;
                Self::track_token_delimiter(&first, &mut stack, &mut depth)
                    .map_err(|detail| self.latch(first.line, detail))?;
                while depth != 0 {
                    let token = self.next_required()?;
                    Self::track_token_delimiter(&token, &mut stack, &mut depth)
                        .map_err(|detail| self.latch(token.line, detail))?;
                }
                self.expect_optional(";")?;
                return Ok(());
            }

            let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
            let mut depth = 0usize;
            Self::track_token_delimiter(&first, &mut stack, &mut depth)
                .map_err(|detail| self.latch(first.line, detail))?;
            loop {
                let token = self.next_required()?;
                if depth == 0 && Self::token_is(&token, ";") {
                    return Ok(());
                }
                if depth == 0 && Self::token_closer(&token) {
                    return self
                        .reject_line_as(token.line, "dictionary value is missing a semicolon");
                }
                Self::track_token_delimiter(&token, &mut stack, &mut depth)
                    .map_err(|detail| self.latch(token.line, detail))?;
            }
        }

        fn is_syntax(value: &str) -> bool {
            matches!(value, "{" | "}" | "(" | ")" | "[" | "]" | ";")
        }

        fn token_is(token: &Token, value: &str) -> bool {
            token.provenance == TokenProvenance::Structural && token.value == value
        }

        fn token_opener(token: &Token) -> bool {
            token.provenance == TokenProvenance::Structural
                && matches!(token.value.as_str(), "{" | "(" | "[")
        }

        fn token_closer(token: &Token) -> bool {
            token.provenance == TokenProvenance::Structural
                && matches!(token.value.as_str(), "}" | ")" | "]")
        }

        fn track_token_delimiter(
            token: &Token,
            stack: &mut [char; MAX_DICTIONARY_NESTING],
            depth: &mut usize,
        ) -> std::result::Result<(), &'static str> {
            if Self::token_opener(token) {
                if *depth == MAX_DICTIONARY_NESTING {
                    return Err("dictionary nesting limit exceeded");
                }
                stack[*depth] = token.value.as_bytes()[0] as char;
                *depth = (*depth)
                    .checked_add(1)
                    .ok_or("dictionary nesting counter overflow")?;
            } else if Self::token_closer(token) {
                let top = (*depth)
                    .checked_sub(1)
                    .ok_or("unexpected dictionary closing delimiter")?;
                if !Self::matching(stack[top], token.value.as_bytes()[0] as char) {
                    return Err("mismatched dictionary delimiter");
                }
                *depth = top;
            }
            Ok(())
        }

        pub fn into_batch(mut self) -> Result<TokenBatch> {
            let mut tokens = Vec::new();
            let mut payload_bytes = 0usize;
            while let Some(token) = self.next()? {
                Self::checked_token_count(tokens.len(), 1)
                    .map_err(|detail| self.latch(token.line, detail))?;
                payload_bytes = Self::checked_payload_bytes(payload_bytes, token.value.len())
                    .map_err(|detail| self.latch(token.line, detail))?;
                #[cfg(test)]
                let additional = if self.inject_batch_reservation_failure {
                    usize::MAX
                } else {
                    1
                };
                #[cfg(not(test))]
                let additional = 1;
                tokens
                    .try_reserve(additional)
                    .map_err(|_| self.latch(token.line, "dictionary token allocation failed"))?;
                tokens.push(token);
            }
            Ok(TokenBatch {
                path: self.path,
                tokens,
                eof_line: self.line,
                payload_bytes,
            })
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
                    message: String::new(),
                },
            }
        }

        fn latch(&mut self, line: usize, detail: &str) -> MeshError {
            if self.failure.is_none() {
                #[allow(clippy::manual_unwrap_or)]
                let path = match self.path.to_str() {
                    Some(value) => value,
                    None => "<non-UTF-8 dictionary path>",
                };
                let capacity = path
                    .len()
                    .checked_add(2)
                    .and_then(|length| length.checked_add(detail.len()));
                let mut message = String::new();
                #[cfg(test)]
                let requested = if self.inject_diagnostic_reservation_failure {
                    Some(usize::MAX)
                } else {
                    capacity
                };
                #[cfg(not(test))]
                let requested = capacity;
                if requested
                    .and_then(|length| message.try_reserve(length).ok())
                    .is_some()
                {
                    message.push_str(path);
                    message.push_str(": ");
                    message.push_str(detail);
                }
                self.failure = Some(Failure { line, message });
            }
            self.sticky()
        }

        fn copy_message(message: &str) -> String {
            let mut copy = String::new();
            if copy.try_reserve(message.len()).is_ok() {
                copy.push_str(message);
            }
            copy
        }

        fn checked_token_count(
            current: usize,
            additional: usize,
        ) -> std::result::Result<usize, &'static str> {
            let count = current
                .checked_add(additional)
                .ok_or("dictionary token count overflow")?;
            if count > MAX_DICTIONARY_TOKENS {
                return Err("dictionary token count limit exceeded");
            }
            Ok(count)
        }

        fn checked_payload_bytes(
            current: usize,
            additional: usize,
        ) -> std::result::Result<usize, &'static str> {
            let bytes = current
                .checked_add(additional)
                .ok_or("dictionary payload length overflow")?;
            if bytes > MAX_DICTIONARY_PAYLOAD_BYTES {
                return Err("dictionary payload byte limit exceeded");
            }
            Ok(bytes)
        }

        fn validate_commit(&mut self, end: usize) -> Result<()> {
            if end > self.declared {
                return Err(self.latch(self.line, "dictionary commit exceeds declared length"));
            }
            Ok(())
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
            let physical = self
                .physical
                .checked_add(1)
                .ok_or((self.line, "dictionary byte counter overflow"))?;
            self.reader.consume(1);
            self.physical = physical;
            if physical > self.declared {
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
            &mut self,
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
            #[cfg(test)]
            let additional = if self.inject_token_reservation_failure {
                usize::MAX
            } else {
                width
            };
            #[cfg(not(test))]
            let additional = width;
            value
                .try_reserve(additional)
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
                                self.push_value(&mut value, next, bytes, self.line)?
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
                    self.push_value(&mut value, ch, width, start)?;
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
                self.push_value(&mut value, ch, width, start)?;
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
                                self.push_value(&mut value, '/', 1, self.line)?;
                                self.put_char(after)?;
                                continue;
                            }
                            None => {
                                self.push_value(&mut value, '/', 1, self.line)?;
                                break;
                            }
                        }
                    }
                    if next.0 == '"' && function_depth != 0 {
                        self.push_value(&mut value, next.0, next.1, self.line)?;
                        loop {
                            match self.take_char()? {
                                Some((quoted, bytes)) => {
                                    self.push_value(&mut value, quoted, bytes, self.line)?;
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
                            .ok_or((self.line, "dictionary nesting overflow"))?
                            == MAX_DICTIONARY_NESTING
                        {
                            return Err((self.line, "dictionary nesting limit exceeded"));
                        }
                        function_stack[function_depth] = next.0;
                        function_depth = function_depth
                            .checked_add(1)
                            .ok_or((self.line, "dictionary nesting overflow"))?;
                        self.push_value(&mut value, next.0, next.1, self.line)?;
                        continue;
                    }
                    if matches!(next.0, ')' | ']' | '}') && function_depth > 0 {
                        let top = function_depth
                            .checked_sub(1)
                            .ok_or((self.line, "dictionary nesting counter underflow"))?;
                        if !Self::matching(function_stack[top], next.0) {
                            return Err((self.line, "mismatched function delimiter"));
                        }
                        function_depth = top;
                        self.push_value(&mut value, next.0, next.1, self.line)?;
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
                    self.push_value(&mut value, next.0, next.1, self.line)?;
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

        fn assert_parse(error: &MeshError, expected_line: usize, expected_message: &str) {
            match error {
                MeshError::Parse { line, message } => {
                    assert_eq!(*line, expected_line);
                    assert_eq!(message, expected_message);
                }
                _ => panic!("expected parse error"),
            }
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

        #[test]
        fn token_batch_preserves_path_and_eof_line() {
            let batch = super::tokenize(Path::new("batch"), "a;\n").unwrap();
            assert_eq!(batch.path(), Path::new("batch"));
            assert_eq!(batch.eof_line(), 2);
        }

        #[test]
        fn tokenize_helpers_use_incremental_source() {
            let batch =
                super::tokenize_reader(Path::new("reader"), source(b"a;").reader, 2).unwrap();
            assert_eq!(batch.tokens().len(), 2);
        }

        #[test]
        fn cursor_peek_next_and_expect_are_provenance_safe() {
            let mut cursor = super::tokenize(Path::new("cursor"), "key;")
                .unwrap()
                .into_cursor();
            assert_eq!(cursor.peek_next().unwrap().unwrap().value, ";");
            cursor.expect("key").unwrap();
        }

        #[test]
        fn quoted_semicolon_remains_data() {
            let mut cursor = super::tokenize(Path::new("quoted"), "\";\";")
                .unwrap()
                .into_cursor();
            assert_eq!(cursor.read_strict_value().unwrap(), vec![";"]);
        }

        #[test]
        fn provenance_safe_keywords() {
            let mut cursor = super::tokenize(Path::new("keyword"), "\"FoamFile\";")
                .unwrap()
                .into_cursor();
            assert!(cursor.expect_keyword("FoamFile").is_err());

            let mut cursor = super::tokenize(Path::new("keyword-semicolon"), ";")
                .unwrap()
                .into_cursor();
            assert!(cursor.expect_keyword(";").is_err());
            let mut cursor = super::tokenize(Path::new("optional-keyword-semicolon"), ";")
                .unwrap()
                .into_cursor();
            assert!(!cursor.expect_optional_keyword(";").unwrap());
            assert_eq!(cursor.next_required().unwrap().value, ";");
        }

        #[test]
        fn strict_value_and_bare_entry_are_distinct() {
            assert!(
                super::tokenize(Path::new("strict"), ";")
                    .unwrap()
                    .into_cursor()
                    .read_strict_value()
                    .is_err()
            );
            assert!(
                super::tokenize(Path::new("bare"), ";")
                    .unwrap()
                    .into_cursor()
                    .read_bare_entry()
                    .unwrap()
                    .is_empty()
            );

            let exact_payload = "a".repeat(MAX_TOKEN_BYTES - 2);
            let exact_fixture = format!("\"{exact_payload}\";");
            let exact = super::tokenize(Path::new("quoted-exact"), &exact_fixture)
                .unwrap()
                .into_cursor()
                .read_provenance_preserving_bare_entry()
                .unwrap();
            assert_eq!(exact.len(), 1);
            assert_eq!(exact[0].len(), MAX_TOKEN_BYTES);
            assert!(exact[0].starts_with('"'));
            assert!(exact[0].ends_with('"'));

            let over_payload = "a".repeat(MAX_TOKEN_BYTES - 1);
            let over_fixture = format!("\"{over_payload}\";");
            let mut over = super::tokenize(Path::new("quoted-over"), &over_fixture)
                .unwrap()
                .into_cursor();
            let first = over.read_provenance_preserving_bare_entry().unwrap_err();
            assert_parse(
                &first,
                1,
                "quoted-over: dictionary token byte limit exceeded",
            );
            assert_eq!(over.next().unwrap_err().to_string(), first.to_string());
        }

        #[test]
        fn missing_semicolon_is_sticky() {
            let mut cursor = super::tokenize(Path::new("missing"), "value")
                .unwrap()
                .into_cursor();
            let first = cursor.read_strict_value().unwrap_err().to_string();
            assert_eq!(cursor.next().unwrap_err().to_string(), first);
        }

        #[test]
        fn unsupported_directives_are_line_aware_and_sticky() {
            let mut streaming = source(b"\n#include \"initialConditions\"\n");
            let first = streaming.next_required().unwrap_err();
            assert_parse(&first, 2, "fixture: unsupported dictionary directive");
            assert_eq!(
                streaming.next_required().unwrap_err().to_string(),
                first.to_string()
            );

            let mut cursor = super::tokenize(Path::new("directive"), "\n#includeFunc residuals\n")
                .unwrap()
                .into_cursor();
            let first = cursor.next_required().unwrap_err();
            assert_parse(&first, 2, "directive: unsupported dictionary directive");
            assert_eq!(
                cursor.next_required().unwrap_err().to_string(),
                first.to_string()
            );
        }

        #[test]
        fn typed_balanced_discard_validates_mixed_delimiters() {
            let mut cursor = super::tokenize(Path::new("typed"), "{([])};")
                .unwrap()
                .into_cursor();
            cursor.skip_typed_balanced().unwrap();
            assert!(cursor.next().unwrap().is_none());
        }

        #[test]
        fn streaming_semicolon_terminated_discard_is_typed_and_sticky() {
            for input in [
                "uniform 0; next;",
                "(a [b]) tail; next;",
                "alpha \";\" beta; next;",
                "{ nested (a [b]); } next;",
            ] {
                let mut streaming = source(input.as_bytes());
                streaming
                    .discard_semicolon_terminated_value_or_block()
                    .unwrap();
                assert_eq!(streaming.next_required().unwrap().value, "next");
                assert_eq!(streaming.next_required().unwrap().value, ";");
                assert!(streaming.next().unwrap().is_none());
            }

            for (input, detail) in [
                ("uniform 0 }", "dictionary nesting counter underflow"),
                ("(a];", "mismatched dictionary delimiter"),
                ("uniform\n#include bad;", "unsupported dictionary directive"),
            ] {
                let mut streaming = source(input.as_bytes());
                let first = streaming
                    .discard_semicolon_terminated_value_or_block()
                    .unwrap_err();
                assert!(first.to_string().contains(detail), "{first}");
                assert_eq!(
                    streaming.next_required().unwrap_err().to_string(),
                    first.to_string()
                );
                assert_eq!(first.to_string().matches("fixture").count(), 1);
            }
        }

        #[test]
        fn optional_semicolon_requires_structural_provenance() {
            let mut cursor = super::tokenize(Path::new("optional"), "\";\"")
                .unwrap()
                .into_cursor();
            assert!(!cursor.expect_optional(";").unwrap());
        }

        #[test]
        fn value_or_block_skip_collects_nothing() {
            let mut cursor = super::tokenize(Path::new("skip"), "{ value; };")
                .unwrap()
                .into_cursor();
            cursor.skip_value_or_block().unwrap();
            assert!(cursor.next().unwrap().is_none());

            for value in ["(a) tail; next;", "[a] tail; next;"] {
                let mut cursor = super::tokenize(Path::new("skip-value"), value)
                    .unwrap()
                    .into_cursor();
                cursor.skip_value_or_block().unwrap();
                assert_eq!(cursor.next_required().unwrap().value, "next");
            }

            for value in ["scalar; next;", "(a [b]); next;", "[a (b)]; next;"] {
                let mut cursor = super::tokenize(Path::new("skip-exact"), value)
                    .unwrap()
                    .into_cursor();
                cursor.skip_exact_value_or_block().unwrap();
                assert_eq!(cursor.next_required().unwrap().value, "next");
            }
            for value in ["{ value; } next;", "{ value; }; next;"] {
                let mut cursor = super::tokenize(Path::new("skip-block"), value)
                    .unwrap()
                    .into_cursor();
                cursor.skip_exact_value_or_block().unwrap();
                assert_eq!(cursor.next_required().unwrap().value, "next");
            }
            for value in ["scalar next;", "(a) next;", "[a] next;"] {
                let mut cursor = super::tokenize(Path::new("skip-reject"), value)
                    .unwrap()
                    .into_cursor();
                let before = cursor.tokens.as_slice().len();
                let error = cursor.skip_exact_value_or_block().unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("dictionary value is missing a semicolon")
                );
                assert_eq!(cursor.tokens.as_slice().len(), before);
            }
        }

        #[test]
        fn cursor_terminal_errors_are_sticky() {
            let mut current = super::tokenize(Path::new("terminal-current"), "\nfirst\nsecond\n\n")
                .unwrap()
                .into_cursor();
            assert_eq!(current.remaining_tokens().unwrap().len(), 2);
            assert_eq!(current.remaining_tokens().unwrap()[0].value, "first");
            assert_eq!(current.remaining_tokens().unwrap()[1].value, "second");
            let rejected = current
                .reject_current("planned field rejection")
                .unwrap_err();
            assert_parse(&rejected, 2, "terminal-current: planned field rejection");
            assert_eq!(
                rejected.to_string(),
                "line 2: terminal-current: planned field rejection"
            );
            let replay = current.remaining_tokens().unwrap_err();
            assert_parse(&replay, 2, "terminal-current: planned field rejection");
            assert_eq!(replay.to_string(), rejected.to_string());
            assert_eq!(current.tokens.as_slice().len(), 2);
            assert_eq!(current.tokens.as_slice()[0].value, "first");
            assert_eq!(current.tokens.as_slice()[1].value, "second");

            let mut explicit = super::tokenize(Path::new("terminal-explicit"), "first\nsecond")
                .unwrap()
                .into_cursor();
            let explicit_error = explicit
                .reject_line_as::<()>(2, "planned exact-line rejection")
                .unwrap_err();
            assert_parse(
                &explicit_error,
                2,
                "terminal-explicit: planned exact-line rejection",
            );
            assert_eq!(
                explicit.next().unwrap_err().to_string(),
                explicit_error.to_string()
            );

            let mut eof_reserve = super::tokenize(Path::new("terminal-eof-reserve"), "\n\n\n")
                .unwrap()
                .into_cursor();
            assert_eq!(eof_reserve.remaining_tokens().unwrap().len(), 0);
            let missing_reserve = eof_reserve.try_reserve_current_value(1).unwrap_err();
            assert_parse(
                &missing_reserve,
                4,
                "terminal-eof-reserve: unexpected end of dictionary",
            );
            assert_eq!(
                missing_reserve.to_string(),
                "line 4: terminal-eof-reserve: unexpected end of dictionary"
            );
            assert_eq!(
                missing_reserve
                    .to_string()
                    .matches("terminal-eof-reserve")
                    .count(),
                1
            );
            assert_eq!(eof_reserve.tokens.as_slice().len(), 0);
            assert_eq!(
                eof_reserve
                    .try_reserve_current_value(1)
                    .unwrap_err()
                    .to_string(),
                missing_reserve.to_string()
            );
            assert_eq!(
                eof_reserve.next().unwrap_err().to_string(),
                missing_reserve.to_string()
            );
            assert_eq!(eof_reserve.tokens.as_slice().len(), 0);

            let mut eof = super::tokenize(Path::new("terminal-eof"), "")
                .unwrap()
                .into_cursor();
            assert!(eof.remaining_tokens().unwrap().is_empty());
            let missing = eof.next_required().unwrap_err();
            assert_parse(&missing, 1, "terminal-eof: unexpected end of dictionary");
            assert_eq!(
                missing.to_string(),
                "line 1: terminal-eof: unexpected end of dictionary"
            );
            assert_eq!(eof.peek().unwrap_err().to_string(), missing.to_string());
            assert_eq!(eof.tokens.as_slice().len(), 0);

            let mut eof_reject = super::tokenize(Path::new("terminal-eof-reject"), "\n\n\n")
                .unwrap()
                .into_cursor();
            assert!(eof_reject.remaining_tokens().unwrap().is_empty());
            let rejected_eof = eof_reject
                .reject_current("planned EOF rejection")
                .unwrap_err();
            assert_parse(
                &rejected_eof,
                4,
                "terminal-eof-reject: planned EOF rejection",
            );
            assert_eq!(
                rejected_eof.to_string(),
                "line 4: terminal-eof-reject: planned EOF rejection"
            );
            assert_eq!(
                rejected_eof
                    .to_string()
                    .matches("terminal-eof-reject")
                    .count(),
                1
            );
            assert_eq!(
                eof_reject.next().unwrap_err().to_string(),
                rejected_eof.to_string()
            );
            assert_eq!(eof_reject.tokens.as_slice().len(), 0);

            let mut overflow =
                super::tokenize(Path::new("terminal-capacity"), "\nfirst\nsecond\n\n")
                    .unwrap()
                    .into_cursor();
            assert_eq!(overflow.remaining_tokens().unwrap().len(), 2);
            let capacity = overflow.try_reserve_current_value(usize::MAX).unwrap_err();
            assert_parse(
                &capacity,
                2,
                "terminal-capacity: dictionary token allocation failed",
            );
            assert_eq!(
                capacity.to_string(),
                "line 2: terminal-capacity: dictionary token allocation failed"
            );
            assert_eq!(capacity.to_string().matches("terminal-capacity").count(), 1);
            let reserve_replay = overflow.try_reserve_current_value(1).unwrap_err();
            assert_parse(
                &reserve_replay,
                2,
                "terminal-capacity: dictionary token allocation failed",
            );
            assert_eq!(reserve_replay.to_string(), capacity.to_string());
            let replay = overflow.next().unwrap_err();
            assert_parse(
                &replay,
                2,
                "terminal-capacity: dictionary token allocation failed",
            );
            assert_eq!(replay.to_string(), capacity.to_string());
            assert_eq!(overflow.tokens.as_slice().len(), 2);
            assert_eq!(overflow.tokens.as_slice()[0].value, "first");
            assert_eq!(overflow.tokens.as_slice()[1].value, "second");
        }

        #[test]
        fn batch_caps_fail_before_growth() {
            let token_fixture = ";".repeat(super::MAX_DICTIONARY_TOKENS + 1);
            let token_error = match super::tokenize(Path::new("token-cap"), &token_fixture) {
                Ok(_) => panic!("token cap fixture unexpectedly succeeded"),
                Err(error) => error.to_string(),
            };
            assert!(token_error.contains("token count limit exceeded"));

            let token_len = super::MAX_TOKEN_BYTES - 1;
            let token_count = super::MAX_DICTIONARY_PAYLOAD_BYTES / token_len + 1;
            let mut payload_fixture = String::new();
            payload_fixture
                .try_reserve(token_count * (token_len + 1))
                .unwrap();
            for _ in 0..token_count {
                payload_fixture.push_str(&"a".repeat(token_len));
                payload_fixture.push(' ');
            }
            let payload_error = match super::tokenize(Path::new("payload-cap"), &payload_fixture) {
                Ok(_) => panic!("payload cap fixture unexpectedly succeeded"),
                Err(error) => error.to_string(),
            };
            assert!(payload_error.contains("payload byte limit exceeded"));
        }

        #[test]
        fn multiline_token_cap_reports_offending_line() {
            let mut exact = Vec::with_capacity(MAX_TOKEN_BYTES + 3);
            exact.push(b'"');
            exact.extend(vec![b'a'; MAX_TOKEN_BYTES - 1]);
            exact.push(b'\n');
            exact.push(b'"');
            assert_eq!(
                source(&exact).next().unwrap().unwrap().value.len(),
                MAX_TOKEN_BYTES
            );

            exact.insert(exact.len() - 1, b'b');
            let mut lexer = source(&exact);
            let first = lexer.peek().unwrap_err();
            assert_parse(&first, 2, "fixture: dictionary token byte limit exceeded");
            assert_eq!(lexer.next().unwrap_err().to_string(), first.to_string());
            assert!(lexer.lookahead.is_none());
        }

        #[test]
        fn token_count_limit_math_is_exact() {
            assert_eq!(
                TokenSource::<io::Empty>::checked_token_count(super::MAX_DICTIONARY_TOKENS - 1, 1),
                Ok(super::MAX_DICTIONARY_TOKENS)
            );
            assert_eq!(
                TokenSource::<io::Empty>::checked_token_count(super::MAX_DICTIONARY_TOKENS, 1),
                Err("dictionary token count limit exceeded")
            );
        }

        #[test]
        fn aggregate_payload_limit_math_is_exact() {
            assert_eq!(
                TokenSource::<io::Empty>::checked_payload_bytes(
                    super::MAX_DICTIONARY_PAYLOAD_BYTES - 1,
                    1
                ),
                Ok(super::MAX_DICTIONARY_PAYLOAD_BYTES)
            );
            assert_eq!(
                TokenSource::<io::Empty>::checked_payload_bytes(
                    super::MAX_DICTIONARY_PAYLOAD_BYTES,
                    1
                ),
                Err("dictionary payload byte limit exceeded")
            );
        }

        #[test]
        fn combined_cursor_nesting_limit_is_exact() {
            let balanced = format!("{}{};", "(".repeat(128), ")".repeat(128));
            super::tokenize(Path::new("depth"), &balanced)
                .unwrap()
                .into_cursor()
                .read_strict_value()
                .unwrap();

            let tokens = (0..129)
                .map(|_| super::Token {
                    value: "(".to_owned(),
                    line: 7,
                    provenance: TokenProvenance::Structural,
                })
                .collect::<Vec<_>>();
            let mut cursor = super::TokenBatch {
                path: Path::new("depth").to_path_buf(),
                tokens,
                eof_line: 7,
                payload_bytes: 129,
            }
            .into_cursor();
            let first = cursor.read_strict_value().unwrap_err();
            assert_parse(&first, 7, "depth: dictionary nesting limit exceeded");
            assert_eq!(cursor.next().unwrap_err().to_string(), first.to_string());
            assert_eq!(cursor.tokens.as_slice().len(), 0);

            let mut mismatch = source(b"f(\n]");
            let first = mismatch.peek().unwrap_err();
            assert_parse(&first, 2, "fixture: mismatched function delimiter");
            assert_eq!(mismatch.next().unwrap_err().to_string(), first.to_string());
            assert!(mismatch.lookahead.is_none());

            let mut nested = Vec::from(&b"f(\n"[..]);
            nested.extend(std::iter::repeat_n(b'(', 128));
            let mut nested = source(&nested);
            let first = nested.peek().unwrap_err();
            assert_parse(&first, 2, "fixture: dictionary nesting limit exceeded");
            assert_eq!(nested.next().unwrap_err().to_string(), first.to_string());
            assert!(nested.lookahead.is_none());
        }

        #[test]
        fn injected_physical_byte_overflow_is_source_sticky() {
            let mut lexer = source(b"x");
            lexer.inject_physical_overflow();
            let first = lexer.peek().unwrap_err();
            assert_parse(&first, 1, "fixture: dictionary byte counter overflow");
            assert_eq!(lexer.next().unwrap_err().to_string(), first.to_string());
            assert!(lexer.lookahead.is_none());
        }

        #[test]
        fn injected_commit_bound_violation_is_source_sticky() {
            let mut lexer = source(b"x");
            lexer.peek().unwrap();
            lexer.inject_commit_bound_violation();
            let first = lexer.next().unwrap_err();
            assert_parse(
                &first,
                1,
                "fixture: dictionary commit exceeds declared length",
            );
            assert_eq!(lexer.next().unwrap_err().to_string(), first.to_string());
            assert!(lexer.lookahead.is_some());
            assert_eq!(lexer.committed, 0);
        }

        #[test]
        fn injected_source_token_reservation_failure_is_source_sticky() {
            let mut lexer = source(b"x");
            lexer.inject_source_token_reservation_failure();
            let first = lexer.peek().unwrap_err();
            assert_parse(&first, 1, "fixture: dictionary token allocation failed");
            assert_eq!(lexer.next().unwrap_err().to_string(), first.to_string());
            assert!(lexer.lookahead.is_none());
        }

        #[test]
        fn injected_batch_reservation_failure_fails_closed() {
            let mut lexer = source(b"x");
            lexer.inject_batch_reservation_failure();
            let error = match lexer.into_batch() {
                Ok(_) => panic!("batch reservation unexpectedly succeeded"),
                Err(error) => error,
            };
            assert_parse(&error, 1, "fixture: dictionary token allocation failed");
        }

        #[test]
        fn injected_source_diagnostic_reservation_failure_uses_sticky_fallback() {
            let mut lexer = source(b"]");
            lexer.inject_source_diagnostic_reservation_failure();
            let first = lexer.peek().unwrap_err();
            assert_parse(&first, 1, "");
            assert_eq!(lexer.next().unwrap_err().to_string(), first.to_string());
        }

        #[test]
        fn injected_cursor_diagnostic_reservation_failure_uses_sticky_fallback() {
            let mut cursor = super::tokenize(Path::new("cursor"), "")
                .unwrap()
                .into_cursor();
            cursor.inject_cursor_diagnostic_reservation_failure();
            let first = cursor.next_required().unwrap_err();
            assert_parse(&first, 1, "");
            assert_eq!(cursor.peek().unwrap_err().to_string(), first.to_string());
        }
    }
}

pub use streaming::{
    MAX_DICTIONARY_NESTING, MAX_DICTIONARY_PAYLOAD_BYTES, MAX_DICTIONARY_TOKENS, MAX_TOKEN_BYTES,
    Token, TokenBatch, TokenCursor, TokenProvenance, TokenSource, tokenize, tokenize_reader,
};

#[cfg(test)]
mod tests {
    use super::tokenize;
    use std::path::Path;

    #[test]
    fn keeps_function_style_dictionary_keys_together() {
        let tokens = tokenize(Path::new("test"), "grad(U) Gauss linear;").unwrap();
        let values = tokens
            .tokens()
            .iter()
            .map(|token| token.value.as_str())
            .collect::<Vec<_>>();

        assert_eq!(values, vec!["grad(U)", "Gauss", "linear", ";"]);
    }

    #[test]
    fn keeps_parenthesized_values_as_lists() {
        let tokens = tokenize(Path::new("test"), "internalField uniform (0 0 0);").unwrap();
        let values = tokens
            .tokens()
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
            Path::new("test"),
            "/* OpenFOAM\n   generated banner */\nFoamFile { class volVectorField; } /* tail */",
        )
        .unwrap();
        let values = tokens
            .tokens()
            .iter()
            .map(|token| token.value.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            values,
            vec!["FoamFile", "{", "class", "volVectorField", ";", "}"]
        );
    }
}
