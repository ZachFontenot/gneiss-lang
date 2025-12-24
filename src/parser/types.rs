//! Type expression parsing

use std::rc::Rc;

use crate::ast::{EffectExpr, EffectRowExpr, Spanned, TypeExpr, TypeExprKind};
use crate::lexer::Token;

use super::cursor::TokenCursor;
use super::error::{ParseError, ParseResult};

/// Extension trait for type expression parsing
pub trait TypeParser {
    /// Parse a type expression
    fn parse_type_expr(&mut self) -> ParseResult<TypeExpr>;

    /// Parse a type arrow (right-associative): A -> B -> C
    /// With optional effect row: A -> B { IO, State s | r }
    fn parse_type_arrow(&mut self) -> ParseResult<TypeExpr>;

    /// Parse type application: List Int, Map String Int
    fn parse_type_app(&mut self) -> ParseResult<TypeExpr>;

    /// Parse a type atom: variable, named type, tuple, list
    fn parse_type_atom(&mut self) -> ParseResult<TypeExpr>;

    /// Parse an effect row: { IO, State s | r }
    fn parse_effect_row(&mut self) -> ParseResult<EffectRowExpr>;

    /// Parse a single effect: IO, State s, Reader Config
    fn parse_effect(&mut self) -> ParseResult<EffectExpr>;
}

impl TypeParser for TokenCursor {
    fn parse_type_expr(&mut self) -> ParseResult<TypeExpr> {
        self.parse_type_arrow()
    }

    fn parse_type_arrow(&mut self) -> ParseResult<TypeExpr> {
        let start = self.current_span();
        let ty = self.parse_type_app()?;

        if self.match_token(&Token::Arrow) {
            // Parse the rest of the arrow chain (right-associative)
            let to_type = self.parse_type_arrow()?;

            // Check for optional effect row: { IO, State s | r }
            let effects = if self.peek() == &Token::LBrace {
                Some(self.parse_effect_row()?)
            } else {
                None
            };

            let span = if let Some(ref eff) = effects {
                start.merge(&eff.span)
            } else {
                start.merge(&to_type.span)
            };

            return Ok(Spanned::new(
                TypeExprKind::Arrow {
                    from: Rc::new(ty),
                    to: Rc::new(to_type),
                    effects,
                },
                span,
            ));
        }

        Ok(ty)
    }

    fn parse_effect_row(&mut self) -> ParseResult<EffectRowExpr> {
        let start = self.current_span();
        self.consume(Token::LBrace)?;

        let mut effects = Vec::new();
        let mut rest = None;

        // Handle empty effect row: {}
        if self.match_token(&Token::RBrace) {
            let span = start.merge(&self.current_span());
            return Ok(EffectRowExpr {
                effects,
                rest,
                span,
            });
        }

        // Parse first effect or row variable
        // If we see a lowercase identifier followed by } or nothing else, it's a row variable
        // Otherwise parse effects
        loop {
            // Check for row variable: | r }
            if self.match_token(&Token::Pipe) {
                match self.peek().clone() {
                    Token::Ident(name) => {
                        self.advance();
                        rest = Some(name);
                    }
                    _ => {
                        return Err(self.unexpected("row variable after '|'"));
                    }
                }
                break;
            }

            // Parse an effect
            effects.push(self.parse_effect()?);

            // Check for comma (more effects), pipe (row variable), or end
            if self.match_token(&Token::Comma) {
                continue;
            } else if self.match_token(&Token::Pipe) {
                // Row variable after effects: { IO, State s | r }
                match self.peek().clone() {
                    Token::Ident(name) => {
                        self.advance();
                        rest = Some(name);
                    }
                    _ => {
                        return Err(self.unexpected("row variable after '|'"));
                    }
                }
                break;
            } else {
                break;
            }
        }

        self.consume(Token::RBrace)?;
        let span = start.merge(&self.current_span());

        Ok(EffectRowExpr {
            effects,
            rest,
            span,
        })
    }

    fn parse_effect(&mut self) -> ParseResult<EffectExpr> {
        let start = self.current_span();

        // Effect name must be an UpperIdent (like IO, State, Reader)
        let name = match self.peek().clone() {
            Token::UpperIdent(n) => {
                self.advance();
                n
            }
            _ => {
                return Err(self.unexpected("effect name (capitalized identifier)"));
            }
        };

        // Parse optional type parameters (e.g., 's' in State s)
        let mut params = Vec::new();
        while self.is_type_atom_start() && self.peek() != &Token::Comma && self.peek() != &Token::Pipe && self.peek() != &Token::RBrace {
            params.push(self.parse_type_atom()?);
        }

        let span = if params.is_empty() {
            start
        } else {
            start.merge(&params.last().unwrap().span)
        };

        Ok(EffectExpr { name, params, span })
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
        if let TypeExprKind::Arrow { from, to, effects } = &ty.node {
            assert!(matches!(from.node, TypeExprKind::Named(ref s) if s == "Int"));
            assert!(matches!(to.node, TypeExprKind::Arrow { .. }));
            assert!(effects.is_none());
        } else {
            panic!("expected arrow type");
        }
    }

    #[test]
    fn test_arrow_type_with_single_effect() {
        let ty = parse_type("Int -> String { IO }");
        if let TypeExprKind::Arrow { from, to, effects } = &ty.node {
            assert!(matches!(from.node, TypeExprKind::Named(ref s) if s == "Int"));
            assert!(matches!(to.node, TypeExprKind::Named(ref s) if s == "String"));
            let eff = effects.as_ref().expect("expected effects");
            assert_eq!(eff.effects.len(), 1);
            assert_eq!(eff.effects[0].name, "IO");
            assert!(eff.rest.is_none());
        } else {
            panic!("expected arrow type with effects");
        }
    }

    #[test]
    fn test_arrow_type_with_multiple_effects() {
        let ty = parse_type("a -> b { IO, State s }");
        if let TypeExprKind::Arrow { effects, .. } = &ty.node {
            let eff = effects.as_ref().expect("expected effects");
            assert_eq!(eff.effects.len(), 2);
            assert_eq!(eff.effects[0].name, "IO");
            assert_eq!(eff.effects[1].name, "State");
            assert_eq!(eff.effects[1].params.len(), 1);
            assert!(eff.rest.is_none());
        } else {
            panic!("expected arrow type with effects");
        }
    }

    #[test]
    fn test_arrow_type_with_effect_row_variable() {
        let ty = parse_type("a -> b { IO | r }");
        if let TypeExprKind::Arrow { effects, .. } = &ty.node {
            let eff = effects.as_ref().expect("expected effects");
            assert_eq!(eff.effects.len(), 1);
            assert_eq!(eff.effects[0].name, "IO");
            assert_eq!(eff.rest, Some("r".to_string()));
        } else {
            panic!("expected arrow type with effect row variable");
        }
    }

    #[test]
    fn test_arrow_type_with_only_row_variable() {
        let ty = parse_type("a -> b { | r }");
        if let TypeExprKind::Arrow { effects, .. } = &ty.node {
            let eff = effects.as_ref().expect("expected effects");
            assert!(eff.effects.is_empty());
            assert_eq!(eff.rest, Some("r".to_string()));
        } else {
            panic!("expected arrow type with only row variable");
        }
    }

    #[test]
    fn test_arrow_type_empty_effects() {
        let ty = parse_type("a -> b {}");
        if let TypeExprKind::Arrow { effects, .. } = &ty.node {
            let eff = effects.as_ref().expect("expected effects");
            assert!(eff.effects.is_empty());
            assert!(eff.rest.is_none());
        } else {
            panic!("expected arrow type with empty effects");
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
