//! Type expression parsing

use std::rc::Rc;

use crate::ast::{Spanned, TypeExpr, TypeExprKind};
use crate::lexer::Token;

use super::cursor::TokenCursor;
use super::error::{ParseError, ParseResult};

/// Extension trait for type expression parsing
pub trait TypeParser {
    /// Parse a type expression
    fn parse_type_expr(&mut self) -> ParseResult<TypeExpr>;

    /// Parse a type arrow (right-associative): A -> B -> C
    /// With optional answer types: A/α -> B/β
    fn parse_type_arrow(&mut self) -> ParseResult<TypeExpr>;

    /// Helper for right-associative arrow parsing with answer types
    fn parse_type_arrow_continuation(&mut self, left: TypeExpr, left_ans: Option<Rc<TypeExpr>>) -> ParseResult<TypeExpr>;

    /// Parse type application: List Int, Map String Int
    fn parse_type_app(&mut self) -> ParseResult<TypeExpr>;

    /// Parse a type atom: variable, named type, tuple, list
    fn parse_type_atom(&mut self) -> ParseResult<TypeExpr>;
}

impl TypeParser for TokenCursor {
    fn parse_type_expr(&mut self) -> ParseResult<TypeExpr> {
        self.parse_type_arrow()
    }

    fn parse_type_arrow(&mut self) -> ParseResult<TypeExpr> {
        let start = self.current_span();
        let ty = self.parse_type_app()?;

        // Check for optional answer type annotation: from/ans_in -> to/ans_out
        let ans_in = if self.match_token(&Token::Slash) {
            Some(Rc::new(self.parse_type_app()?))
        } else {
            None
        };

        if self.match_token(&Token::Arrow) {
            let to_type = self.parse_type_app()?;

            // Check for answer-out annotation
            let ans_out = if self.match_token(&Token::Slash) {
                Some(Rc::new(self.parse_type_app()?))
            } else {
                None
            };

            // Handle right-associativity: if to is followed by ->, continue parsing
            // For chained arrows, the ans_out applies to the inner arrow, not the outer
            let (inner_to, outer_ans_out) = if self.peek() == &Token::Arrow {
                // Build the nested arrow chain, ans_out becomes ans_in of inner
                let inner = self.parse_type_arrow_continuation(to_type, ans_out)?;
                // For chained arrows, outer arrow has no explicit ans_out
                (inner, None)
            } else {
                // Simple case: to_type is the actual target type
                (to_type, ans_out)
            };

            // Always wrap with ty as the from type
            let span = start.merge(&inner_to.span);
            return Ok(Spanned::new(
                TypeExprKind::Arrow {
                    from: Rc::new(ty),
                    to: Rc::new(inner_to),
                    ans_in,
                    ans_out: outer_ans_out,
                },
                span,
            ));
        }

        // No arrow, but we might have parsed ans_in incorrectly as a type app
        // If we saw a slash, that's a parse error in non-arrow context
        if ans_in.is_some() {
            return Err(ParseError::unexpected(
                "'->' after answer type annotation",
                self.peek().clone(),
                start,
            ));
        }

        Ok(ty)
    }

    /// Helper for right-associative arrow parsing with answer types
    fn parse_type_arrow_continuation(&mut self, left: TypeExpr, left_ans: Option<Rc<TypeExpr>>) -> ParseResult<TypeExpr> {
        let start = left.span.clone();

        if self.match_token(&Token::Arrow) {
            let right = self.parse_type_app()?;
            let ans_out = if self.match_token(&Token::Slash) {
                Some(Rc::new(self.parse_type_app()?))
            } else {
                None
            };

            // Continue for right-associativity
            let to = if self.peek() == &Token::Arrow {
                self.parse_type_arrow_continuation(right, ans_out)?
            } else {
                Spanned::new(
                    TypeExprKind::Arrow {
                        from: Rc::new(left),
                        to: Rc::new(right),
                        ans_in: left_ans,
                        ans_out,
                    },
                    start.merge(&self.current_span()),
                )
            };

            Ok(to)
        } else {
            // This shouldn't happen if called correctly
            Err(ParseError::unexpected(
                "'->'",
                self.peek().clone(),
                start,
            ))
        }
    }

    fn parse_type_app(&mut self) -> ParseResult<TypeExpr> {
        let start = self.current_span();
        let base = self.parse_type_atom()?;

        let mut args = Vec::new();
        while self.is_type_atom_start() {
            args.push(self.parse_type_atom()?);
        }

        if args.is_empty() {
            Ok(base)
        } else {
            let span = start.merge(&args.last().unwrap().span);
            Ok(Spanned::new(
                TypeExprKind::App {
                    constructor: Rc::new(base),
                    args,
                },
                span,
            ))
        }
    }

    fn parse_type_atom(&mut self) -> ParseResult<TypeExpr> {
        let start = self.current_span();

        match self.peek().clone() {
            Token::Ident(name) => {
                self.advance();
                Ok(Spanned::new(TypeExprKind::Var(name), start))
            }
            Token::UpperIdent(name) => {
                self.advance();
                // Check for built-in Channel type
                if name == "Channel"
                    && self.is_type_atom_start() {
                        let inner = self.parse_type_atom()?;
                        let span = start.merge(&inner.span);
                        return Ok(Spanned::new(TypeExprKind::Channel(Rc::new(inner)), span));
                    }
                Ok(Spanned::new(TypeExprKind::Named(name), start))
            }
            Token::LParen => {
                self.advance();
                if self.match_token(&Token::RParen) {
                    // Unit type
                    let span = start.merge(&self.current_span());
                    return Ok(Spanned::new(TypeExprKind::Tuple(vec![]), span));
                }

                let first = self.parse_type_expr()?;

                if self.match_token(&Token::Comma) {
                    // Tuple type
                    let mut types = vec![first];
                    loop {
                        types.push(self.parse_type_expr()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                    self.consume(Token::RParen)?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(TypeExprKind::Tuple(types), span))
                } else {
                    self.consume(Token::RParen)?;
                    Ok(first)
                }
            }
            Token::LBracket => {
                // List type: [a]
                self.advance();
                let elem = self.parse_type_expr()?;
                self.consume(Token::RBracket)?;
                let span = start.merge(&self.current_span());
                Ok(Spanned::new(TypeExprKind::List(Rc::new(elem)), span))
            }
            _ => Err(self.unexpected("type")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse_type(input: &str) -> TypeExpr {
        let tokens = Lexer::new(input).tokenize().unwrap();
        let mut cursor = TokenCursor::new(tokens);
        cursor.parse_type_expr().unwrap()
    }

    #[test]
    fn test_simple_type() {
        let ty = parse_type("Int");
        assert!(matches!(ty.node, TypeExprKind::Named(ref s) if s == "Int"));
    }

    #[test]
    fn test_type_var() {
        let ty = parse_type("a");
        assert!(matches!(ty.node, TypeExprKind::Var(ref s) if s == "a"));
    }

    #[test]
    fn test_arrow_type() {
        let ty = parse_type("Int -> String");
        assert!(matches!(ty.node, TypeExprKind::Arrow { .. }));
    }

    #[test]
    fn test_arrow_type_right_assoc() {
        let ty = parse_type("Int -> String -> Bool");
        if let TypeExprKind::Arrow { from, to, ans_in, ans_out } = &ty.node {
            assert!(matches!(from.node, TypeExprKind::Named(ref s) if s == "Int"));
            assert!(matches!(to.node, TypeExprKind::Arrow { .. }));
            assert!(ans_in.is_none());
            assert!(ans_out.is_none());
        } else {
            panic!("expected arrow type");
        }
    }

    #[test]
    fn test_arrow_type_with_answer_types() {
        let ty = parse_type("Int/a -> String/b");
        if let TypeExprKind::Arrow { from, to, ans_in, ans_out } = &ty.node {
            assert!(matches!(from.node, TypeExprKind::Named(ref s) if s == "Int"));
            assert!(matches!(to.node, TypeExprKind::Named(ref s) if s == "String"));
            assert!(ans_in.is_some());
            assert!(ans_out.is_some());
            if let Some(ai) = ans_in {
                assert!(matches!(ai.node, TypeExprKind::Var(ref s) if s == "a"));
            }
            if let Some(ao) = ans_out {
                assert!(matches!(ao.node, TypeExprKind::Var(ref s) if s == "b"));
            }
        } else {
            panic!("expected arrow type with answer types");
        }
    }

    #[test]
    fn test_type_app() {
        let ty = parse_type("List Int");
        if let TypeExprKind::App { constructor, args } = &ty.node {
            assert!(matches!(constructor.node, TypeExprKind::Named(ref s) if s == "List"));
            assert_eq!(args.len(), 1);
        } else {
            panic!("expected type application");
        }
    }

    #[test]
    fn test_tuple_type() {
        let ty = parse_type("(Int, String)");
        if let TypeExprKind::Tuple(types) = &ty.node {
            assert_eq!(types.len(), 2);
        } else {
            panic!("expected tuple type");
        }
    }

    #[test]
    fn test_list_type() {
        let ty = parse_type("[Int]");
        assert!(matches!(ty.node, TypeExprKind::List(_)));
    }

    #[test]
    fn test_unit_type() {
        let ty = parse_type("()");
        if let TypeExprKind::Tuple(types) = &ty.node {
            assert!(types.is_empty());
        } else {
            panic!("expected unit type");
        }
    }

    #[test]
    fn test_channel_type() {
        let ty = parse_type("Channel Int");
        assert!(matches!(ty.node, TypeExprKind::Channel(_)));
    }
}
