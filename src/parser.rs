//! Recursive descent parser for Gneiss

use crate::ast::*;
use crate::lexer::{SpannedToken, Token};
use std::rc::Rc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("unexpected token: expected {expected}, found {found:?}")]
    UnexpectedToken { expected: String, found: Token, span: Span },
    #[error("unexpected end of file")]
    UnexpectedEof { expected: String, last_span: Span },
    #[error("invalid pattern")]
    InvalidPattern { span: Span },
}

pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        while !self.is_at_end() {
            items.push(self.parse_item()?);
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        // First, optionally consume any leading ;;
        while self.match_token(&Token::DoubleSemi) {}

        match self.peek() {
            Token::Let => {
                // Could be a declaration (let x = e) or expression (let x = e in body)
                // Parse the common prefix, then check for `in`
                self.parse_let_item()
            }
            Token::Type | Token::Trait | Token::Impl => {
                let decl = self.parse_decl()?;
                // Optionally consume ;; after declaration
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Decl(decl))
            }
            _ => {
                let expr = self.parse_expr()?;
                // Optionally consume ;; after expression
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Expr(expr))
            }
        }
    }

    /// Parse a let that could be either a declaration or a let-expression.
    /// Returns Item::Decl if no `in`, Item::Expr if `in` is present.
    fn parse_let_item(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.consume(Token::Let)?;

        let name = self.parse_ident()?;

        // Collect parameters (function syntax: let f x y = ...)
        let mut params = Vec::new();
        while self.is_pattern_start() && !self.check(&Token::Eq) {
            params.push(self.parse_simple_pattern()?);
        }

        self.consume(Token::Eq)?;
        let value_expr = self.parse_expr()?;

        // KEY DECISION: check for `in`
        if self.check(&Token::In) {
            // It's a let-expression: let x = e in body
            self.advance(); // consume 'in'
            let body = self.parse_expr()?;
            let span = start.merge(&body.span);

            // Desugar function syntax if needed
            let (pattern, value) = if params.is_empty() {
                // Simple pattern: just the name as a variable pattern
                let name_pattern = Spanned::new(
                    PatternKind::Var(name),
                    start.clone(),
                );
                (name_pattern, value_expr)
            } else {
                // Function syntax: let f x y = e in body
                // becomes: let f = fun x y -> e in body
                let lambda_span = value_expr.span.clone();
                let lambda = Spanned::new(
                    ExprKind::Lambda {
                        params,
                        body: Rc::new(value_expr),
                    },
                    lambda_span,
                );
                let name_pattern = Spanned::new(
                    PatternKind::Var(name),
                    start.clone(),
                );
                (name_pattern, lambda)
            };

            let expr = Spanned::new(
                ExprKind::Let {
                    pattern,
                    value: Rc::new(value),
                    body: Some(Rc::new(body)),
                },
                span,
            );
            // Optionally consume ;; after expression
            self.match_token(&Token::DoubleSemi);
            Ok(Item::Expr(expr))
        } else {
            // It's a declaration: let x = e
            // Optionally consume ;; after declaration
            self.match_token(&Token::DoubleSemi);
            Ok(Item::Decl(Decl::Let {
                name,
                type_ann: None,
                params,
                body: value_expr,
            }))
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .map(|t| &t.token)
            .unwrap_or(&Token::Eof)
    }

    fn current_span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|t| t.span.clone())
            .unwrap_or_default()
    }

    fn advance(&mut self) -> &SpannedToken {
        if !self.is_at_end() {
            self.pos += 1;
        }
        &self.tokens[self.pos - 1]
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn check(&self, token: &Token) -> bool {
        self.peek() == token
    }

    fn consume(&mut self, expected: Token) -> Result<&SpannedToken, ParseError> {
        if self.check(&expected) {
            Ok(self.advance())
        } else {
            Err(ParseError::UnexpectedToken {
                expected: format!("{:?}", expected),
                found: self.peek().clone(),
                span: self.current_span(),
            })
        }
    }

    fn match_token(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Create an UnexpectedToken error at the current position
    fn unexpected_token(&self, expected: &str) -> ParseError {
        ParseError::UnexpectedToken {
            expected: expected.to_string(),
            found: self.peek().clone(),
            span: self.current_span(),
        }
    }

    // ========================================================================
    // Declarations
    // ========================================================================

    fn parse_decl(&mut self) -> Result<Decl, ParseError> {
        match self.peek() {
            Token::Let => self.parse_let_decl(),
            Token::Type => self.parse_type_decl(),
            Token::Trait => self.parse_trait_decl(),
            Token::Impl => self.parse_instance_decl(),
            _ => Err(self.unexpected_token("declaration")),
        }
    }

    fn parse_let_decl(&mut self) -> Result<Decl, ParseError> {
        self.consume(Token::Let)?;

        let name = self.parse_ident()?;

        // Collect parameters
        let mut params = Vec::new();
        while self.is_pattern_start() && !self.check(&Token::Eq) {
            params.push(self.parse_simple_pattern()?);
        }

        self.consume(Token::Eq)?;
        let body = self.parse_expr()?;

        Ok(Decl::Let {
            name,
            type_ann: None,
            params,
            body,
        })
    }

    fn parse_type_decl(&mut self) -> Result<Decl, ParseError> {
        self.consume(Token::Type)?;

        let name = self.parse_upper_ident()?;

        // Type parameters
        let mut params = Vec::new();
        while let Token::Ident(_) = self.peek() {
            params.push(self.parse_ident()?);
        }

        self.consume(Token::Eq)?;

        // Check if it's a variant type or alias
        if self.check(&Token::Pipe) || self.peek_is_upper_ident() {
            // Variant type
            let constructors = self.parse_constructors()?;
            Ok(Decl::Type {
                name,
                params,
                constructors,
            })
        } else {
            // Type alias
            let body = self.parse_type_expr()?;
            Ok(Decl::TypeAlias { name, params, body })
        }
    }

    fn parse_constructors(&mut self) -> Result<Vec<Constructor>, ParseError> {
        let mut constructors = Vec::new();

        // Optional leading pipe
        self.match_token(&Token::Pipe);

        loop {
            let name = self.parse_upper_ident()?;
            let mut fields = Vec::new();

            // Parse constructor fields
            while self.is_type_atom_start() {
                fields.push(self.parse_type_atom()?);
            }

            constructors.push(Constructor { name, fields });

            if !self.match_token(&Token::Pipe) {
                break;
            }
        }

        Ok(constructors)
    }

    fn peek_is_upper_ident(&self) -> bool {
        matches!(self.peek(), Token::UpperIdent(_))
    }

    // ========================================================================
    // Typeclass declarations
    // ========================================================================

    /// Parse a trait declaration:
    /// trait Show a = val show : a -> String end
    /// trait Ord a : Eq = val compare : a -> a -> Int end
    fn parse_trait_decl(&mut self) -> Result<Decl, ParseError> {
        self.consume(Token::Trait)?;

        // Trait name (uppercase, like a type constructor)
        let name = self.parse_upper_ident()?;

        // Type parameter (lowercase)
        let type_param = self.parse_ident()?;

        // Optional supertraits: : Eq or : Eq, Ord
        let supertraits = if self.match_token(&Token::Colon) {
            self.parse_supertrait_list()?
        } else {
            vec![]
        };

        self.consume(Token::Eq)?;

        // Parse method signatures: val show : a -> String
        let mut methods = Vec::new();
        while self.match_token(&Token::Val) {
            let method_name = self.parse_ident()?;
            self.consume(Token::Colon)?;
            let type_sig = self.parse_type_expr()?;
            methods.push(TraitMethod {
                name: method_name,
                type_sig,
            });
        }

        self.consume(Token::End)?;

        Ok(Decl::Trait {
            name,
            type_param,
            supertraits,
            methods,
        })
    }

    /// Parse a list of supertrait names: Eq or Eq, Ord
    fn parse_supertrait_list(&mut self) -> Result<Vec<Ident>, ParseError> {
        let mut traits = vec![self.parse_upper_ident()?];
        while self.match_token(&Token::Comma) {
            traits.push(self.parse_upper_ident()?);
        }
        Ok(traits)
    }

    /// Parse an instance declaration:
    /// impl Show for Int = let show n = int_to_string n end
    /// impl Show for (List a) where a : Show = let show xs = "[list]" end
    fn parse_instance_decl(&mut self) -> Result<Decl, ParseError> {
        self.consume(Token::Impl)?;

        // Trait name
        let trait_name = self.parse_upper_ident()?;

        self.consume(Token::For)?;

        // Target type (can be complex like (List a))
        let target_type = self.parse_type_expr()?;

        // Optional constraints: where a : Show, b : Eq
        let constraints = if self.match_token(&Token::Where) {
            self.parse_constraints()?
        } else {
            vec![]
        };

        self.consume(Token::Eq)?;

        // Parse method implementations: let show n = ...
        let mut methods = Vec::new();
        while self.match_token(&Token::Let) {
            let method_name = self.parse_ident()?;

            // Collect parameters
            let mut params = Vec::new();
            while self.is_pattern_start() && !self.check(&Token::Eq) {
                params.push(self.parse_simple_pattern()?);
            }

            self.consume(Token::Eq)?;
            let body = self.parse_expr()?;

            methods.push(InstanceMethod {
                name: method_name,
                params,
                body,
            });
        }

        self.consume(Token::End)?;

        Ok(Decl::Instance {
            trait_name,
            target_type,
            constraints,
            methods,
        })
    }

    /// Parse constraints: a : Show, b : Eq
    fn parse_constraints(&mut self) -> Result<Vec<Constraint>, ParseError> {
        let mut constraints = Vec::new();

        loop {
            // Parse: type_var : TraitName
            let type_var = self.parse_ident()?;
            self.consume(Token::Colon)?;
            let trait_name = self.parse_upper_ident()?;

            constraints.push(Constraint {
                type_var,
                trait_name,
            });

            if !self.match_token(&Token::Comma) {
                break;
            }
        }

        Ok(constraints)
    }

    // ========================================================================
    // Types
    // ========================================================================

    fn parse_type_expr(&mut self) -> Result<TypeExpr, ParseError> {
        self.parse_type_arrow()
    }

    fn parse_type_arrow(&mut self) -> Result<TypeExpr, ParseError> {
        let start = self.current_span();
        let mut ty = self.parse_type_app()?;

        if self.match_token(&Token::Arrow) {
            let to = self.parse_type_arrow()?;
            let span = start.merge(&to.span);
            ty = Spanned::new(
                TypeExprKind::Arrow {
                    from: Rc::new(ty),
                    to: Rc::new(to),
                },
                span,
            );
        }

        Ok(ty)
    }

    fn parse_type_app(&mut self) -> Result<TypeExpr, ParseError> {
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

    fn is_type_atom_start(&self) -> bool {
        matches!(
            self.peek(),
            Token::Ident(_) | Token::UpperIdent(_) | Token::LParen
        )
    }

    fn parse_type_atom(&mut self) -> Result<TypeExpr, ParseError> {
        let start = self.current_span();

        match self.peek().clone() {
            Token::Ident(name) => {
                self.advance();
                Ok(Spanned::new(TypeExprKind::Var(name), start))
            }
            Token::UpperIdent(name) => {
                self.advance();
                // Check for built-in Channel type
                if name == "Channel" {
                    if self.is_type_atom_start() {
                        let inner = self.parse_type_atom()?;
                        let span = start.merge(&inner.span);
                        return Ok(Spanned::new(
                            TypeExprKind::Channel(Rc::new(inner)),
                            span,
                        ));
                    }
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
            _ => Err(self.unexpected_token("type")),
        }
    }

    // ========================================================================
    // Expressions
    // ========================================================================

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_expr_seq()
    }

    fn parse_expr_seq(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut expr = self.parse_expr_let()?;

        while self.match_token(&Token::Semicolon) {
            let second = self.parse_expr_let()?;
            let span = start.merge(&second.span);
            expr = Spanned::new(
                ExprKind::Seq {
                    first: Rc::new(expr),
                    second: Rc::new(second),
                },
                span,
            );
        }

        Ok(expr)
    }

    fn parse_expr_let(&mut self) -> Result<Expr, ParseError> {
        if self.check(&Token::Let) {
            let start = self.current_span();
            self.advance();

            // First, try to parse a simple pattern
            let first_pattern = self.parse_simple_pattern()?;

            // Check if this is function syntax: let f x y = ... in ...
            // by looking for more patterns before the '='
            let mut params = Vec::new();
            while self.is_pattern_start() && !self.check(&Token::Eq) {
                params.push(self.parse_simple_pattern()?);
            }

            self.consume(Token::Eq)?;
            let value_expr = self.parse_expr()?;
            self.consume(Token::In)?;
            let body = self.parse_expr()?;

            let span = start.merge(&body.span);

            // If we have params, desugar: let f x y = e in body
            // becomes: let f = fun x y -> e in body
            let (pattern, value) = if params.is_empty() {
                (first_pattern, value_expr)
            } else {
                // Create a lambda wrapping the value
                let lambda_span = value_expr.span.clone();
                let lambda = Spanned::new(
                    ExprKind::Lambda {
                        params,
                        body: Rc::new(value_expr),
                    },
                    lambda_span,
                );
                (first_pattern, lambda)
            };

            Ok(Spanned::new(
                ExprKind::Let {
                    pattern,
                    value: Rc::new(value),
                    body: Some(Rc::new(body)),
                },
                span,
            ))
        } else {
            self.parse_expr_if()
        }
    }

    fn parse_expr_if(&mut self) -> Result<Expr, ParseError> {
        if self.check(&Token::If) {
            let start = self.current_span();
            self.advance();

            let cond = self.parse_expr()?;
            self.consume(Token::Then)?;
            let then_branch = self.parse_expr()?;
            self.consume(Token::Else)?;
            let else_branch = self.parse_expr()?;

            let span = start.merge(&else_branch.span);
            Ok(Spanned::new(
                ExprKind::If {
                    cond: Rc::new(cond),
                    then_branch: Rc::new(then_branch),
                    else_branch: Rc::new(else_branch),
                },
                span,
            ))
        } else if self.check(&Token::Match) {
            self.parse_match()
        } else if self.check(&Token::Fun) {
            self.parse_lambda()
        } else if self.check(&Token::Select) {
            self.parse_select()
        } else {
            self.parse_expr_pipe()
        }
    }

    fn parse_select(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        self.consume(Token::Select)?;

        let mut arms = Vec::new();

        // Require at least one arm
        // Each arm: | pattern <- channel -> body
        loop {
            if !self.match_token(&Token::Pipe) {
                break;
            }

            let pattern = self.parse_pattern()?;
            self.consume(Token::LArrow)?;
            let channel = self.parse_expr_app()?; // Parse channel expr (no operators to avoid <- ambiguity)
            self.consume(Token::Arrow)?;
            let body = self.parse_expr_pipe()?; // Body can have operators

            arms.push(SelectArm {
                channel,
                pattern,
                body,
            });
        }

        self.consume(Token::End)?;

        if arms.is_empty() {
            return Err(self.unexpected_token("select arm"));
        }

        let span = start.merge(&self.current_span());
        Ok(Spanned::new(ExprKind::Select { arms }, span))
    }

    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        self.consume(Token::Fun)?;

        let mut params = Vec::new();
        while self.is_pattern_start() && !self.check(&Token::Arrow) {
            params.push(self.parse_simple_pattern()?);
        }

        self.consume(Token::Arrow)?;
        let body = self.parse_expr()?;

        let span = start.merge(&body.span);
        Ok(Spanned::new(
            ExprKind::Lambda {
                params,
                body: Rc::new(body),
            },
            span,
        ))
    }

    fn parse_match(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        self.consume(Token::Match)?;
        let scrutinee = self.parse_expr_pipe()?;
        self.consume(Token::With)?;

        let mut arms = Vec::new();
        // Optional leading pipe
        self.match_token(&Token::Pipe);

        loop {
            let pattern = self.parse_pattern()?;

            // Optional guard
            let guard = if self.match_token(&Token::If) {
                Some(self.parse_expr_pipe()?)
            } else {
                None
            };

            self.consume(Token::Arrow)?;
            let body = self.parse_expr_pipe()?;

            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });

            if !self.match_token(&Token::Pipe) {
                break;
            }
        }

        let span = start.merge(&arms.last().unwrap().body.span);
        Ok(Spanned::new(
            ExprKind::Match {
                scrutinee: Rc::new(scrutinee),
                arms,
            },
            span,
        ))
    }

    // Pipe operators: |> <|
    fn parse_expr_pipe(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut left = self.parse_expr_or()?;

        loop {
            if self.match_token(&Token::PipeOp) {
                let right = self.parse_expr_or()?;
                let span = start.merge(&right.span);
                // x |> f  =>  f x
                left = Spanned::new(
                    ExprKind::App {
                        func: Rc::new(right),
                        arg: Rc::new(left),
                    },
                    span,
                );
            } else if self.match_token(&Token::PipeBack) {
                let right = self.parse_expr_or()?;
                let span = start.merge(&right.span);
                // f <| x  =>  f x
                left = Spanned::new(
                    ExprKind::App {
                        func: Rc::new(left),
                        arg: Rc::new(right),
                    },
                    span,
                );
            } else {
                break;
            }
        }

        Ok(left)
    }

    // Boolean or: ||
    fn parse_expr_or(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut left = self.parse_expr_and()?;

        while self.match_token(&Token::OrOr) {
            let right = self.parse_expr_and()?;
            let span = start.merge(&right.span);
            left = Spanned::new(
                ExprKind::BinOp {
                    op: BinOp::Or,
                    left: Rc::new(left),
                    right: Rc::new(right),
                },
                span,
            );
        }

        Ok(left)
    }

    // Boolean and: &&
    fn parse_expr_and(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut left = self.parse_expr_cmp()?;

        while self.match_token(&Token::AndAnd) {
            let right = self.parse_expr_cmp()?;
            let span = start.merge(&right.span);
            left = Spanned::new(
                ExprKind::BinOp {
                    op: BinOp::And,
                    left: Rc::new(left),
                    right: Rc::new(right),
                },
                span,
            );
        }

        Ok(left)
    }

    // Comparison: == != < > <= >=
    fn parse_expr_cmp(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let left = self.parse_expr_cons()?;

        let op = match self.peek() {
            Token::EqEq => BinOp::Eq,
            Token::Neq => BinOp::Neq,
            Token::Lt => BinOp::Lt,
            Token::Gt => BinOp::Gt,
            Token::Lte => BinOp::Lte,
            Token::Gte => BinOp::Gte,
            _ => return Ok(left),
        };

        self.advance();
        let right = self.parse_expr_cons()?;
        let span = start.merge(&right.span);

        Ok(Spanned::new(
            ExprKind::BinOp {
                op,
                left: Rc::new(left),
                right: Rc::new(right),
            },
            span,
        ))
    }

    // Cons and concat: :: ++
    fn parse_expr_cons(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let left = self.parse_expr_add()?;

        if self.match_token(&Token::Cons) {
            let right = self.parse_expr_cons()?; // Right associative
            let span = start.merge(&right.span);
            Ok(Spanned::new(
                ExprKind::BinOp {
                    op: BinOp::Cons,
                    left: Rc::new(left),
                    right: Rc::new(right),
                },
                span,
            ))
        } else if self.match_token(&Token::Concat) {
            let right = self.parse_expr_cons()?; // Right associative
            let span = start.merge(&right.span);
            Ok(Spanned::new(
                ExprKind::BinOp {
                    op: BinOp::Concat,
                    left: Rc::new(left),
                    right: Rc::new(right),
                },
                span,
            ))
        } else {
            Ok(left)
        }
    }

    // Addition and subtraction: + -
    fn parse_expr_add(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut left = self.parse_expr_mul()?;

        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_expr_mul()?;
            let span = start.merge(&right.span);
            left = Spanned::new(
                ExprKind::BinOp {
                    op,
                    left: Rc::new(left),
                    right: Rc::new(right),
                },
                span,
            );
        }

        Ok(left)
    }

    // Multiplication, division, modulo: * / %
    fn parse_expr_mul(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut left = self.parse_expr_compose()?;

        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_expr_compose()?;
            let span = start.merge(&right.span);
            left = Spanned::new(
                ExprKind::BinOp {
                    op,
                    left: Rc::new(left),
                    right: Rc::new(right),
                },
                span,
            );
        }

        Ok(left)
    }

    // Function composition: >> <<
    fn parse_expr_compose(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut left = self.parse_expr_unary()?;

        loop {
            if self.match_token(&Token::Compose) {
                let right = self.parse_expr_unary()?;
                let span = start.merge(&right.span);
                left = Spanned::new(
                    ExprKind::BinOp {
                        op: BinOp::Compose,
                        left: Rc::new(left),
                        right: Rc::new(right),
                    },
                    span,
                );
            } else if self.match_token(&Token::ComposeBack) {
                let right = self.parse_expr_unary()?;
                let span = start.merge(&right.span);
                left = Spanned::new(
                    ExprKind::BinOp {
                        op: BinOp::ComposeBack,
                        left: Rc::new(left),
                        right: Rc::new(right),
                    },
                    span,
                );
            } else {
                break;
            }
        }

        Ok(left)
    }

    // Unary: not, -
    fn parse_expr_unary(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();

        if self.match_token(&Token::Not) {
            let operand = self.parse_expr_unary()?;
            let span = start.merge(&operand.span);
            return Ok(Spanned::new(
                ExprKind::UnaryOp {
                    op: UnaryOp::Not,
                    operand: Rc::new(operand),
                },
                span,
            ));
        }

        if self.match_token(&Token::Minus) {
            let operand = self.parse_expr_unary()?;
            let span = start.merge(&operand.span);
            return Ok(Spanned::new(
                ExprKind::UnaryOp {
                    op: UnaryOp::Neg,
                    operand: Rc::new(operand),
                },
                span,
            ));
        }

        self.parse_expr_app()
    }

    // Function application
    fn parse_expr_app(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut func = self.parse_expr_atom()?;

        while self.is_atom_start() {
            let arg = self.parse_expr_atom()?;
            let span = start.merge(&arg.span);
            func = Spanned::new(
                ExprKind::App {
                    func: Rc::new(func),
                    arg: Rc::new(arg),
                },
                span,
            );
        }

        Ok(func)
    }

    fn is_atom_start(&self) -> bool {
        matches!(
            self.peek(),
            Token::Int(_)
                | Token::Float(_)
                | Token::String(_)
                | Token::Char(_)
                | Token::True
                | Token::False
                | Token::Ident(_)
                | Token::UpperIdent(_)
                | Token::LParen
                | Token::LBracket
                | Token::Spawn
                | Token::Reset
                | Token::Shift
        )
    }

    fn parse_expr_atom(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();

        match self.peek().clone() {
            Token::Int(n) => {
                self.advance();
                Ok(Spanned::new(ExprKind::Lit(Literal::Int(n)), start))
            }
            Token::Float(f) => {
                self.advance();
                Ok(Spanned::new(ExprKind::Lit(Literal::Float(f)), start))
            }
            Token::String(s) => {
                self.advance();
                Ok(Spanned::new(ExprKind::Lit(Literal::String(s)), start))
            }
            Token::Char(c) => {
                self.advance();
                Ok(Spanned::new(ExprKind::Lit(Literal::Char(c)), start))
            }
            Token::True => {
                self.advance();
                Ok(Spanned::new(ExprKind::Lit(Literal::Bool(true)), start))
            }
            Token::False => {
                self.advance();
                Ok(Spanned::new(ExprKind::Lit(Literal::Bool(false)), start))
            }
            Token::Ident(name) => {
                self.advance();
                // Check for qualified access: Module.func
                if self.check(&Token::Dot) {
                    self.advance();
                    let field = self.parse_ident()?;
                    // For now, treat Module.func as just "Module.func" identifier
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(
                        ExprKind::Var(format!("{}.{}", name, field)),
                        span,
                    ))
                } else {
                    Ok(Spanned::new(ExprKind::Var(name), start))
                }
            }
            Token::UpperIdent(name) => {
                self.advance();
                // Constructor or Module access
                if self.check(&Token::Dot) {
                    self.advance();
                    match self.peek().clone() {
                        Token::Ident(field) => {
                            self.advance();
                            // Module.func
                            let span = start.merge(&self.current_span());

                            // Special case: Channel.new, Channel.send, Channel.recv
                            if name == "Channel" {
                                match field.as_str() {
                                    "new" => return Ok(Spanned::new(ExprKind::NewChannel, span)),
                                    "send" => {
                                        // Need to parse two args
                                        let ch = self.parse_expr_atom()?;
                                        let val = self.parse_expr_atom()?;
                                        let span = start.merge(&val.span);
                                        return Ok(Spanned::new(
                                            ExprKind::ChanSend {
                                                channel: Rc::new(ch),
                                                value: Rc::new(val),
                                            },
                                            span,
                                        ));
                                    }
                                    "recv" => {
                                        let ch = self.parse_expr_atom()?;
                                        let span = start.merge(&ch.span);
                                        return Ok(Spanned::new(
                                            ExprKind::ChanRecv(Rc::new(ch)),
                                            span,
                                        ));
                                    }
                                    _ => {}
                                }
                            }

                            Ok(Spanned::new(
                                ExprKind::Var(format!("{}.{}", name, field)),
                                span,
                            ))
                        }
                        Token::UpperIdent(sub) => {
                            self.advance();
                            // Could be Module.Constructor
                            let span = start.merge(&self.current_span());
                            Ok(Spanned::new(
                                ExprKind::Constructor {
                                    name: format!("{}.{}", name, sub),
                                    args: vec![],
                                },
                                span,
                            ))
                        }
                        _ => Err(self.unexpected_token("identifier")),
                    }
                } else {
                    // Just a constructor
                    Ok(Spanned::new(
                        ExprKind::Constructor {
                            name,
                            args: vec![],
                        },
                        start,
                    ))
                }
            }
            Token::Spawn => {
                self.advance();
                let body = self.parse_expr_atom()?;
                let span = start.merge(&body.span);
                Ok(Spanned::new(ExprKind::Spawn(Rc::new(body)), span))
            }
            Token::Reset => {
                self.advance();
                let body = self.parse_expr_atom()?;
                let span = start.merge(&body.span);
                Ok(Spanned::new(ExprKind::Reset(Rc::new(body)), span))
            }
            Token::Shift => {
                self.advance();
                let func = self.parse_expr_atom()?;

                match &func.node {
                    ExprKind::Lambda { params, body } if params.len() == 1 => {
                        let span = start.merge(&func.span);
                        Ok(Spanned::new(ExprKind::Shift {
                            param: params[0].clone(),
                            body: body.clone(),
                        }, span))
                    }
                    _ => Err(self.unexpected_token("function (fun k -> ...)"))
                }
            }
            Token::LParen => {
                self.advance();
                if self.match_token(&Token::RParen) {
                    // Unit
                    let span = start.merge(&self.current_span());
                    return Ok(Spanned::new(ExprKind::Lit(Literal::Unit), span));
                }

                let first = self.parse_expr()?;

                if self.match_token(&Token::Comma) {
                    // Tuple
                    let mut exprs = vec![first];
                    loop {
                        if self.check(&Token::RParen) {
                            break;
                        }
                        exprs.push(self.parse_expr()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                    self.consume(Token::RParen)?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(ExprKind::Tuple(exprs), span))
                } else {
                    // Parenthesized expression
                    self.consume(Token::RParen)?;
                    Ok(first)
                }
            }
            Token::LBracket => {
                self.advance();
                let mut exprs = Vec::new();

                if !self.check(&Token::RBracket) {
                    loop {
                        exprs.push(self.parse_expr()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                        // Allow trailing comma
                        if self.check(&Token::RBracket) {
                            break;
                        }
                    }
                }

                self.consume(Token::RBracket)?;
                let span = start.merge(&self.current_span());
                Ok(Spanned::new(ExprKind::List(exprs), span))
            }
            _ => Err(self.unexpected_token("expression")),
        }
    }

    // ========================================================================
    // Patterns
    // ========================================================================

    fn is_pattern_start(&self) -> bool {
        matches!(
            self.peek(),
            Token::Ident(_)
                | Token::UpperIdent(_)
                | Token::Int(_)
                | Token::String(_)
                | Token::Char(_)
                | Token::True
                | Token::False
                | Token::LParen
                | Token::LBracket
                | Token::Underscore
        )
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        self.parse_pattern_cons()
    }

    fn parse_pattern_cons(&mut self) -> Result<Pattern, ParseError> {
        let start = self.current_span();
        let left = self.parse_simple_pattern()?;

        if self.match_token(&Token::Cons) {
            let right = self.parse_pattern_cons()?; // Right associative
            let span = start.merge(&right.span);
            Ok(Spanned::new(
                PatternKind::Cons {
                    head: Rc::new(left),
                    tail: Rc::new(right),
                },
                span,
            ))
        } else {
            Ok(left)
        }
    }

    fn parse_simple_pattern(&mut self) -> Result<Pattern, ParseError> {
        let start = self.current_span();

        match self.peek().clone() {
            Token::Underscore => {
                self.advance();
                Ok(Spanned::new(PatternKind::Wildcard, start))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Var(name), start))
            }
            Token::UpperIdent(name) => {
                self.advance();
                // Constructor pattern
                let mut args = Vec::new();
                while self.is_simple_pattern_atom() {
                    args.push(self.parse_pattern_atom()?);
                }
                let span = if args.is_empty() {
                    start
                } else {
                    start.merge(&args.last().unwrap().span)
                };
                Ok(Spanned::new(
                    PatternKind::Constructor { name, args },
                    span,
                ))
            }
            Token::Int(n) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Int(n)), start))
            }
            Token::String(s) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::String(s)), start))
            }
            Token::Char(c) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Char(c)), start))
            }
            Token::True => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Bool(true)), start))
            }
            Token::False => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Bool(false)), start))
            }
            Token::LParen => {
                self.advance();
                if self.match_token(&Token::RParen) {
                    // Unit pattern
                    let span = start.merge(&self.current_span());
                    return Ok(Spanned::new(PatternKind::Lit(Literal::Unit), span));
                }

                let first = self.parse_pattern()?;

                if self.match_token(&Token::Comma) {
                    // Tuple pattern
                    let mut pats = vec![first];
                    loop {
                        if self.check(&Token::RParen) {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                    self.consume(Token::RParen)?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(PatternKind::Tuple(pats), span))
                } else {
                    // Parenthesized pattern
                    self.consume(Token::RParen)?;
                    Ok(first)
                }
            }
            Token::LBracket => {
                self.advance();
                let mut pats = Vec::new();

                if !self.check(&Token::RBracket) {
                    loop {
                        pats.push(self.parse_pattern()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                        if self.check(&Token::RBracket) {
                            break;
                        }
                    }
                }

                self.consume(Token::RBracket)?;
                let span = start.merge(&self.current_span());
                Ok(Spanned::new(PatternKind::List(pats), span))
            }
            _ => Err(self.unexpected_token("pattern")),
        }
    }

    fn is_simple_pattern_atom(&self) -> bool {
        matches!(
            self.peek(),
            Token::Ident(_)
                | Token::Int(_)
                | Token::String(_)
                | Token::Char(_)
                | Token::True
                | Token::False
                | Token::LParen
                | Token::LBracket
                | Token::Underscore
        )
    }

    fn parse_pattern_atom(&mut self) -> Result<Pattern, ParseError> {
        let start = self.current_span();

        match self.peek().clone() {
            Token::Underscore => {
                self.advance();
                Ok(Spanned::new(PatternKind::Wildcard, start))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Var(name), start))
            }
            Token::Int(n) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Int(n)), start))
            }
            Token::String(s) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::String(s)), start))
            }
            Token::Char(c) => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Char(c)), start))
            }
            Token::True => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Bool(true)), start))
            }
            Token::False => {
                self.advance();
                Ok(Spanned::new(PatternKind::Lit(Literal::Bool(false)), start))
            }
            Token::LParen => {
                self.advance();
                if self.match_token(&Token::RParen) {
                    let span = start.merge(&self.current_span());
                    return Ok(Spanned::new(PatternKind::Lit(Literal::Unit), span));
                }

                let first = self.parse_pattern()?;

                if self.match_token(&Token::Comma) {
                    let mut pats = vec![first];
                    loop {
                        if self.check(&Token::RParen) {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                    self.consume(Token::RParen)?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(PatternKind::Tuple(pats), span))
                } else {
                    self.consume(Token::RParen)?;
                    Ok(first)
                }
            }
            Token::LBracket => {
                self.advance();
                let mut pats = Vec::new();

                if !self.check(&Token::RBracket) {
                    loop {
                        pats.push(self.parse_pattern()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                        if self.check(&Token::RBracket) {
                            break;
                        }
                    }
                }

                self.consume(Token::RBracket)?;
                let span = start.merge(&self.current_span());
                Ok(Spanned::new(PatternKind::List(pats), span))
            }
            _ => Err(self.unexpected_token("pattern")),
        }
    }

    // ========================================================================
    // Identifier helpers
    // ========================================================================

    fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        match self.peek().clone() {
            Token::Ident(name) => {
                self.advance();
                Ok(name)
            }
            _ => Err(self.unexpected_token("identifier")),
        }
    }

    fn parse_upper_ident(&mut self) -> Result<Ident, ParseError> {
        match self.peek().clone() {
            Token::UpperIdent(name) => {
                self.advance();
                Ok(name)
            }
            _ => Err(self.unexpected_token("constructor")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(input: &str) -> Program {
        let tokens = Lexer::new(input).tokenize().unwrap();
        Parser::new(tokens).parse_program().unwrap()
    }

    #[test]
    fn test_let_simple() {
        let prog = parse("let x = 42");
        assert_eq!(prog.items.len(), 1);
    }

    #[test]
    fn test_let_function() {
        let prog = parse("let add x y = x + y");
        assert_eq!(prog.items.len(), 1);
    }

    #[test]
    fn test_type_decl() {
        let prog = parse("type Option a = | Some a | None");
        assert_eq!(prog.items.len(), 1);
    }

    #[test]
    fn test_parse_trait_decl() {
        let prog = parse("trait Show a = val show : a -> String end");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Trait {
                name,
                type_param,
                supertraits,
                methods,
            }) => {
                assert_eq!(name, "Show");
                assert_eq!(type_param, "a");
                assert!(supertraits.is_empty());
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name, "show");
            }
            _ => panic!("expected trait decl"),
        }
    }

    #[test]
    fn test_parse_instance_decl() {
        let prog = parse("impl Show for Int = let show n = int_to_string n end");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Instance {
                trait_name,
                constraints,
                methods,
                ..
            }) => {
                assert_eq!(trait_name, "Show");
                assert!(constraints.is_empty());
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name, "show");
            }
            _ => panic!("expected instance decl"),
        }
    }

    #[test]
    fn test_parse_constrained_instance() {
        let prog = parse(r#"impl Show for (List a) where a : Show = let show xs = "list" end"#);
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Instance { constraints, .. }) => {
                assert_eq!(constraints.len(), 1);
                assert_eq!(constraints[0].type_var, "a");
                assert_eq!(constraints[0].trait_name, "Show");
            }
            _ => panic!("expected instance decl"),
        }
    }

    #[test]
    fn test_parse_trait_with_supertrait() {
        let prog = parse("trait Ord a : Eq = val compare : a -> a -> Int end");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Trait { supertraits, .. }) => {
                assert_eq!(supertraits.len(), 1);
                assert_eq!(supertraits[0], "Eq");
            }
            _ => panic!("expected trait decl"),
        }
    }

    #[test]
    fn test_parse_trait_multiple_methods() {
        let prog = parse(
            "trait Eq a = val eq : a -> a -> Bool val neq : a -> a -> Bool end",
        );
        match &prog.items[0] {
            Item::Decl(Decl::Trait { methods, .. }) => {
                assert_eq!(methods.len(), 2);
                assert_eq!(methods[0].name, "eq");
                assert_eq!(methods[1].name, "neq");
            }
            _ => panic!("expected trait decl"),
        }
    }

    #[test]
    fn test_let_expression_at_top_level() {
        // let x = e in body should parse as an expression, not a declaration
        let prog = parse("let x = 5 in x + 1");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => {
                match &expr.node {
                    ExprKind::Let { body, .. } => {
                        assert!(body.is_some(), "let-expression should have a body");
                    }
                    _ => panic!("expected let expression"),
                }
            }
            _ => panic!("expected Item::Expr, got Item::Decl"),
        }
    }

    #[test]
    fn test_let_decl_with_double_semi() {
        // let x = e;; followed by another expression
        let prog = parse("let x = 5;; x");
        assert_eq!(prog.items.len(), 2);
        assert!(matches!(&prog.items[0], Item::Decl(Decl::Let { .. })));
        assert!(matches!(&prog.items[1], Item::Expr(_)));
    }

    #[test]
    fn test_let_function_expression() {
        // let f x = e in f 1 should be an expression
        let prog = parse("let f x = x + 1 in f 5");
        assert_eq!(prog.items.len(), 1);
        assert!(matches!(&prog.items[0], Item::Expr(_)));
    }

    #[test]
    fn test_double_semi_lexer() {
        // Verify ;; is lexed as DoubleSemi, not two Semicolons
        let tokens = Lexer::new(";;").tokenize().unwrap();
        assert_eq!(tokens.len(), 2); // DoubleSemi + Eof
        assert!(matches!(tokens[0].token, Token::DoubleSemi));
    }
}
