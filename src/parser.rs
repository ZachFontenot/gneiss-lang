//! Recursive descent parser for Gneiss

use crate::ast::*;
use crate::lexer::{SpannedToken, Token};
use std::rc::Rc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("unexpected token: expected {expected}, found {found:?}")]
    UnexpectedToken { expected: String, found: Token },
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("invalid pattern")]
    InvalidPattern,
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
        let mut declarations = Vec::new();
        while !self.is_at_end() {
            declarations.push(self.parse_decl()?);
        }
        Ok(Program { declarations })
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

    // ========================================================================
    // Declarations
    // ========================================================================

    fn parse_decl(&mut self) -> Result<Decl, ParseError> {
        match self.peek() {
            Token::Let => self.parse_let_decl(),
            Token::Type => self.parse_type_decl(),
            _ => Err(ParseError::UnexpectedToken {
                expected: "declaration".into(),
                found: self.peek().clone(),
            }),
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
            _ => Err(ParseError::UnexpectedToken {
                expected: "type".into(),
                found: self.peek().clone(),
            }),
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

            let pattern = self.parse_pattern()?;
            self.consume(Token::Eq)?;
            let value = self.parse_expr()?;
            self.consume(Token::In)?;
            let body = self.parse_expr()?;

            let span = start.merge(&body.span);
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
            return Err(ParseError::UnexpectedToken {
                expected: "select arm".into(),
                found: self.peek().clone(),
            });
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
                        _ => Err(ParseError::UnexpectedToken {
                            expected: "identifier".into(),
                            found: self.peek().clone(),
                        }),
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
            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".into(),
                found: self.peek().clone(),
            }),
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
            _ => Err(ParseError::UnexpectedToken {
                expected: "pattern".into(),
                found: self.peek().clone(),
            }),
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
            _ => Err(ParseError::UnexpectedToken {
                expected: "pattern".into(),
                found: self.peek().clone(),
            }),
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
            _ => Err(ParseError::UnexpectedToken {
                expected: "identifier".into(),
                found: self.peek().clone(),
            }),
        }
    }

    fn parse_upper_ident(&mut self) -> Result<Ident, ParseError> {
        match self.peek().clone() {
            Token::UpperIdent(name) => {
                self.advance();
                Ok(name)
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "constructor".into(),
                found: self.peek().clone(),
            }),
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
        assert_eq!(prog.declarations.len(), 1);
    }

    #[test]
    fn test_let_function() {
        let prog = parse("let add x y = x + y");
        assert_eq!(prog.declarations.len(), 1);
    }

    #[test]
    fn test_type_decl() {
        let prog = parse("type Option a = | Some a | None");
        assert_eq!(prog.declarations.len(), 1);
    }
}
