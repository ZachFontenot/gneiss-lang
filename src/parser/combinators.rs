//! Generic parsing combinators for reusable parsing patterns

use crate::lexer::Token;

use super::cursor::TokenCursor;
use super::error::ParseResult;

/// Extension trait providing combinator methods on TokenCursor
pub trait Combinators {
    /// Parse a separated list of items: item (sep item)*
    /// Returns the items as a Vec
    fn separated_list<T, F>(
        &mut self,
        parse_item: F,
        separator: &Token,
        allow_trailing: bool,
        terminator: &Token,
    ) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Self) -> ParseResult<T>;

    /// Parse a delimited list: open item (sep item)* close
    fn delimited_list<T, F>(
        &mut self,
        open: Token,
        parse_item: F,
        separator: &Token,
        close: Token,
        allow_trailing: bool,
    ) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Self) -> ParseResult<T>;

    /// Parse zero or more items while a condition holds
    fn many_while<T, F, P>(&mut self, parse_item: F, condition: P) -> Vec<T>
    where
        F: FnMut(&mut Self) -> ParseResult<T>,
        P: Fn(&Self) -> bool;

    /// Parse an optional item based on a condition
    fn optional<T, F, P>(&mut self, parse_item: F, condition: P) -> ParseResult<Option<T>>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
        P: Fn(&Self) -> bool;
}

impl Combinators for TokenCursor {
    fn separated_list<T, F>(
        &mut self,
        mut parse_item: F,
        separator: &Token,
        allow_trailing: bool,
        terminator: &Token,
    ) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Self) -> ParseResult<T>,
    {
        let mut items = Vec::new();

        // Check for empty list
        if self.check(terminator) {
            return Ok(items);
        }

        // Parse first item
        items.push(parse_item(self)?);

        // Parse remaining items
        while self.match_token(separator) {
            // Check for trailing separator
            if self.check(terminator) {
                if allow_trailing {
                    break;
                } else {
                    return Err(self.unexpected("item after separator"));
                }
            }
            items.push(parse_item(self)?);
        }

        Ok(items)
    }

    fn delimited_list<T, F>(
        &mut self,
        open: Token,
        mut parse_item: F,
        separator: &Token,
        close: Token,
        allow_trailing: bool,
    ) -> ParseResult<Vec<T>>
    where
        F: FnMut(&mut Self) -> ParseResult<T>,
    {
        self.consume(open)?;
        let items = self.separated_list(&mut parse_item, separator, allow_trailing, &close)?;
        self.consume(close)?;
        Ok(items)
    }

    fn many_while<T, F, P>(&mut self, mut parse_item: F, condition: P) -> Vec<T>
    where
        F: FnMut(&mut Self) -> ParseResult<T>,
        P: Fn(&Self) -> bool,
    {
        let mut items = Vec::new();
        while condition(self) {
            match parse_item(self) {
                Ok(item) => items.push(item),
                Err(_) => break,
            }
        }
        items
    }

    fn optional<T, F, P>(&mut self, parse_item: F, condition: P) -> ParseResult<Option<T>>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
        P: Fn(&Self) -> bool,
    {
        if condition(self) {
            Ok(Some(parse_item(self)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn cursor(input: &str) -> TokenCursor {
        let tokens = Lexer::new(input).tokenize().unwrap();
        TokenCursor::new(tokens)
    }

    #[test]
    fn test_separated_list() {
        let mut c = cursor("a, b, c]");
        let items: Vec<String> = c
            .separated_list(
                |c| {
                    if let Token::Ident(s) = c.peek().clone() {
                        c.advance();
                        Ok(s)
                    } else {
                        Err(c.unexpected("identifier"))
                    }
                },
                &Token::Comma,
                false,
                &Token::RBracket,
            )
            .unwrap();
        assert_eq!(items, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_separated_list_trailing() {
        let mut c = cursor("a, b, c,]");
        let items: Vec<String> = c
            .separated_list(
                |c| {
                    if let Token::Ident(s) = c.peek().clone() {
                        c.advance();
                        Ok(s)
                    } else {
                        Err(c.unexpected("identifier"))
                    }
                },
                &Token::Comma,
                true,
                &Token::RBracket,
            )
            .unwrap();
        assert_eq!(items, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_separated_list_empty() {
        let mut c = cursor("]");
        let items: Vec<String> = c
            .separated_list(
                |c| {
                    if let Token::Ident(s) = c.peek().clone() {
                        c.advance();
                        Ok(s)
                    } else {
                        Err(c.unexpected("identifier"))
                    }
                },
                &Token::Comma,
                false,
                &Token::RBracket,
            )
            .unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn test_many_while() {
        let mut c = cursor("a b c =");
        let items: Vec<String> = c.many_while(
            |c| {
                if let Token::Ident(s) = c.peek().clone() {
                    c.advance();
                    Ok(s)
                } else {
                    Err(c.unexpected("identifier"))
                }
            },
            |c| matches!(c.peek(), Token::Ident(_)),
        );
        assert_eq!(items, vec!["a", "b", "c"]);
        assert!(c.check(&Token::Eq));
    }
}
