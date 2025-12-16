//! Recursive descent parser for Gneiss

use crate::ast::*;
use crate::errors::Warning;
use crate::lexer::{SpannedToken, Token};
use crate::operators::{OpInfo, OperatorTable};
use std::rc::Rc;
use thiserror::Error;

/// What expression forms are allowed in the current parsing context.
/// This makes explicit where each expression form can appear.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExprContext {
    /// Full language: let, if, match, fun, seq, binary
    /// Used for: lambda bodies, top-level expressions
    Full,
    /// No sequences: let, if, match, fun, binary
    /// Used for: if branches, match arms, select arms
    NoSeq,
    /// Only binary/application expressions
    /// Used for: match guards (intentional restriction)
    BinaryOnly,
}

impl ExprContext {
    fn allows_seq(self) -> bool {
        matches!(self, ExprContext::Full)
    }

    fn allows_let(self) -> bool {
        matches!(self, ExprContext::Full | ExprContext::NoSeq)
    }
}

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("unexpected token: expected {expected}, found {found:?}")]
    UnexpectedToken {
        expected: String,
        found: Token,
        span: Span,
    },
    #[error("unexpected end of file")]
    UnexpectedEof { expected: String, last_span: Span },
    #[error("invalid pattern")]
    InvalidPattern { span: Span },
}

pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
    /// Operator precedence and associativity table
    op_table: OperatorTable,
    /// Warnings collected during parsing
    warnings: Vec<Warning>,
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Self {
            tokens,
            pos: 0,
            op_table: OperatorTable::new(),
            warnings: Vec::new(),
        }
    }

    /// Get collected warnings
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Take ownership of warnings, clearing the internal list
    pub fn take_warnings(&mut self) -> Vec<Warning> {
        std::mem::take(&mut self.warnings)
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
            Token::Import => {
                // Parse import statement
                let import = self.parse_import()?;
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Import(import))
            }
            Token::Pub => {
                // Parse public declaration
                let decl = self.parse_pub_decl()?;
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Decl(decl))
            }
            Token::Let => {
                // Could be a declaration (let x = e) or expression (let x = e in body)
                // Parse the common prefix, then check for `in`
                self.parse_let_item()
            }
            Token::Type | Token::Trait | Token::Impl | Token::Val => {
                let decl = self.parse_decl()?;
                // Optionally consume ;; after declaration
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Decl(decl))
            }
            // Fixity declarations: infixl, infixr, infix
            Token::Ident(ref s) if s == "infixl" || s == "infixr" || s == "infix" => {
                let decl = self.parse_fixity_decl()?;
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

    /// Parse an import statement
    /// Syntax:
    ///   import Module                      -- qualified access
    ///   import Module as M                 -- alias
    ///   import Module (item1, item2)       -- selective
    ///   import Module (item1 as alias1)   -- selective with alias
    fn parse_import(&mut self) -> Result<Spanned<ImportSpec>, ParseError> {
        let start = self.current_span();
        self.consume(Token::Import)?;

        // Parse module path (e.g., "Collections/HashMap" or just "List")
        let module_path = self.parse_module_path()?;

        // Check for alias: `as Name`
        let alias = if self.match_token(&Token::As) {
            Some(self.parse_upper_ident()?)
        } else {
            None
        };

        // Check for selective imports: (item1, item2, ...)
        let items = if self.check(&Token::LParen) {
            self.advance(); // consume (
            let mut imports = Vec::new();

            if !self.check(&Token::RParen) {
                loop {
                    let name = self.parse_ident()?;
                    let item_alias = if self.match_token(&Token::As) {
                        Some(self.parse_ident()?)
                    } else {
                        None
                    };
                    imports.push((name, item_alias));

                    if !self.match_token(&Token::Comma) {
                        break;
                    }
                }
            }

            self.consume(Token::RParen)?;
            Some(imports)
        } else {
            None
        };

        let end = self.current_span();
        Ok(Spanned::new(
            ImportSpec {
                module_path,
                alias,
                items,
            },
            start.merge(&end),
        ))
    }

    /// Parse a module path like "Collections/HashMap" or "List"
    fn parse_module_path(&mut self) -> Result<String, ParseError> {
        let first = self.parse_upper_ident()?;
        let mut path = first;

        // Check for path separator (using / for now, could also use .)
        while self.check(&Token::Slash) {
            self.advance();
            let next = self.parse_upper_ident()?;
            path.push('/');
            path.push_str(&next);
        }

        Ok(path)
    }

    /// Parse a public declaration: pub let, pub type, pub trait
    fn parse_pub_decl(&mut self) -> Result<Decl, ParseError> {
        self.consume(Token::Pub)?;

        match self.peek() {
            Token::Let => {
                let start = self.current_span();
                self.consume(Token::Let)?;

                // Check for 'rec'
                if self.match_token(&Token::Rec) {
                    return self.parse_let_rec_decl_with_visibility(Visibility::Public);
                }

                // Check for operator definition: pub let (<|>) a b = ...
                if self.check(&Token::LParen) {
                    if let Some(op) = self.peek_operator_in_parens() {
                        return self.parse_operator_def_decl_with_visibility(start, op, None, Visibility::Public);
                    }
                }

                // Parse regular let
                let name = self.parse_ident()?;

                // Check for infix operator: pub let a <|> b = ...
                if let Some(op) = self.try_peek_operator() {
                    return self.parse_operator_def_decl_with_visibility(start.clone(), op, Some((name.clone(), start)), Visibility::Public);
                }

                let mut params = Vec::new();
                while self.is_pattern_start() && !self.check(&Token::Eq) {
                    params.push(self.parse_simple_pattern()?);
                }

                self.consume(Token::Eq)?;
                let body = self.parse_expr()?;

                Ok(Decl::Let {
                    visibility: Visibility::Public,
                    name,
                    type_ann: None,
                    params,
                    body,
                })
            }
            Token::Type => {
                self.consume(Token::Type)?;
                let name = self.parse_upper_ident()?;

                let mut params = Vec::new();
                while let Token::Ident(_) = self.peek() {
                    params.push(self.parse_ident()?);
                }

                self.consume(Token::Eq)?;

                if self.check(&Token::Pipe) || self.peek_is_upper_ident() {
                    let constructors = self.parse_constructors()?;
                    Ok(Decl::Type {
                        visibility: Visibility::Public,
                        name,
                        params,
                        constructors,
                    })
                } else {
                    let body = self.parse_type_expr()?;
                    Ok(Decl::TypeAlias {
                        visibility: Visibility::Public,
                        name,
                        params,
                        body,
                    })
                }
            }
            Token::Trait => {
                self.consume(Token::Trait)?;
                let name = self.parse_upper_ident()?;
                let type_param = self.parse_ident()?;

                let supertraits = if self.match_token(&Token::Colon) {
                    self.parse_supertrait_list()?
                } else {
                    vec![]
                };

                self.consume(Token::Eq)?;

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
                    visibility: Visibility::Public,
                    name,
                    type_param,
                    supertraits,
                    methods,
                })
            }
            _ => Err(ParseError::UnexpectedToken {
                expected: "let, type, or trait after pub".to_string(),
                found: self.peek().clone(),
                span: self.current_span(),
            }),
        }
    }

    /// Parse let rec with specified visibility
    fn parse_let_rec_decl_with_visibility(&mut self, visibility: Visibility) -> Result<Decl, ParseError> {
        let mut bindings = Vec::new();
        bindings.push(self.parse_rec_binding()?);

        while self.match_token(&Token::And) {
            bindings.push(self.parse_rec_binding()?);
        }

        Ok(Decl::LetRec { visibility, bindings })
    }

    /// Parse operator definition with specified visibility
    fn parse_operator_def_decl_with_visibility(
        &mut self,
        _start: Span,
        op: String,
        first_param: Option<(String, Span)>,
        visibility: Visibility,
    ) -> Result<Decl, ParseError> {
        let mut params = Vec::new();

        if let Some((name, param_span)) = first_param {
            self.advance(); // consume operator
            params.push(Spanned::new(PatternKind::Var(name), param_span));
            params.push(self.parse_simple_pattern()?);
        } else {
            self.consume(Token::LParen)?;
            self.advance(); // consume operator
            self.consume(Token::RParen)?;
            while self.is_pattern_start() && !self.check(&Token::Eq) {
                params.push(self.parse_simple_pattern()?);
            }
        }

        self.consume(Token::Eq)?;
        let body = self.parse_expr()?;

        Ok(Decl::OperatorDef {
            visibility,
            op,
            params,
            body,
        })
    }

    /// Parse a let that could be either a declaration or a let-expression.
    /// Returns Item::Decl if no `in`, Item::Expr if `in` is present.
    ///
    /// Also handles operator definitions:
    /// - `let (<|>) a b = ...` (prefix syntax)
    /// - `let a <|> b = ...` (infix syntax)
    ///
    /// And mutual recursion:
    /// - `let rec f = ... and g = ...`
    fn parse_let_item(&mut self) -> Result<Item, ParseError> {
        let start = self.current_span();
        self.consume(Token::Let)?;

        // Check for 'rec' keyword for mutual recursion
        if self.match_token(&Token::Rec) {
            return self.parse_let_rec(start);
        }

        // Check for prefix operator syntax: let (<|>) a b = ...
        if self.check(&Token::LParen) {
            if let Some(op) = self.peek_operator_in_parens() {
                return self.parse_operator_def(start, op, None);
            }
        }

        // Parse first identifier
        let first_span = self.current_span();
        let first_name = self.parse_ident()?;

        // Check for infix operator syntax: let a <|> b = ...
        if let Some(op) = self.try_peek_operator() {
            return self.parse_operator_def(start, op, Some((first_name, first_span)));
        }

        // Regular let binding/function: let f x y = ...
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
                let name_pattern = Spanned::new(PatternKind::Var(first_name), start.clone());
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
                let name_pattern = Spanned::new(PatternKind::Var(first_name), start.clone());
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
                visibility: Visibility::Private,
                name: first_name,
                type_ann: None,
                params,
                body: value_expr,
            }))
        }
    }

    /// Check if the next tokens are `( <op> )` and return the operator symbol
    fn peek_operator_in_parens(&self) -> Option<String> {
        // Current token should be LParen
        if !matches!(self.peek(), Token::LParen) {
            return None;
        }
        // Look ahead: we need (op) where op is an operator
        if self.pos + 2 >= self.tokens.len() {
            return None;
        }
        let op_str = self.tokens[self.pos + 1].token.operator_symbol()?;
        // Check that token after operator is RParen
        if !matches!(self.tokens[self.pos + 2].token, Token::RParen) {
            return None;
        }
        Some(op_str)
    }

    /// Try to peek for an operator token (for infix syntax detection)
    fn try_peek_operator(&self) -> Option<String> {
        self.peek().operator_symbol()
    }

    /// Parse operator definition (both prefix and infix syntax)
    /// Prefix: let (<|>) a b = body
    /// Infix: let a <|> b = body
    fn parse_operator_def(
        &mut self,
        start: Span,
        op: String,
        first_param: Option<(String, Span)>,
    ) -> Result<Item, ParseError> {
        let mut params = Vec::new();

        if let Some((name, param_span)) = first_param {
            // Infix syntax: we have the first param already, consume the operator
            self.advance();
            params.push(Spanned::new(PatternKind::Var(name), param_span));
            // Parse second parameter
            params.push(self.parse_simple_pattern()?);
        } else {
            // Prefix syntax: consume ( op ) then parse all params
            self.consume(Token::LParen)?;
            self.advance(); // consume the operator token
            self.consume(Token::RParen)?;
            while self.is_pattern_start() && !self.check(&Token::Eq) {
                params.push(self.parse_simple_pattern()?);
            }
        }

        self.consume(Token::Eq)?;
        let body = self.parse_expr()?;

        self.match_token(&Token::DoubleSemi);

        Ok(Item::Decl(Decl::OperatorDef {
            visibility: Visibility::Private,
            op,
            params,
            body,
        }))
    }

    /// Parse mutually recursive let: let rec f x = ... and g y = ...
    fn parse_let_rec(&mut self, start: Span) -> Result<Item, ParseError> {
        let mut bindings = Vec::new();

        // Parse first binding
        bindings.push(self.parse_rec_binding()?);

        // Parse additional bindings with 'and'
        while self.match_token(&Token::And) {
            bindings.push(self.parse_rec_binding()?);
        }

        // Check for 'in' (expression) or end (declaration)
        if self.check(&Token::In) {
            self.advance(); // consume 'in'
            let body = self.parse_expr()?;
            let span = start.merge(&body.span);

            let expr = Spanned::new(
                ExprKind::LetRec {
                    bindings,
                    body: Some(Rc::new(body)),
                },
                span,
            );
            self.match_token(&Token::DoubleSemi);
            Ok(Item::Expr(expr))
        } else {
            // Top-level declaration
            self.match_token(&Token::DoubleSemi);
            Ok(Item::Decl(Decl::LetRec {
                visibility: Visibility::Private,
                bindings,
            }))
        }
    }

    /// Parse a single recursive binding: name params = expr
    fn parse_rec_binding(&mut self) -> Result<RecBinding, ParseError> {
        let name_start = self.current_span();
        let name = self.parse_ident()?;

        // Collect parameters
        let mut params = Vec::new();
        while self.is_pattern_start() && !self.check(&Token::Eq) {
            params.push(self.parse_simple_pattern()?);
        }

        self.consume(Token::Eq)?;
        let body = self.parse_expr()?;

        Ok(RecBinding {
            name: Spanned::new(name, name_start),
            params,
            body,
        })
    }

    /// Parse a fixity declaration: `infixl 6 +` or `infixr 5 :: ++`
    fn parse_fixity_decl(&mut self) -> Result<Decl, ParseError> {
        let start = self.current_span();

        // Parse associativity keyword
        let assoc = match self.peek() {
            Token::Ident(s) if s == "infixl" => {
                self.advance();
                Associativity::Left
            }
            Token::Ident(s) if s == "infixr" => {
                self.advance();
                Associativity::Right
            }
            Token::Ident(s) if s == "infix" => {
                self.advance();
                Associativity::None
            }
            other => {
                return Err(ParseError::UnexpectedToken {
                    expected: "infixl, infixr, or infix".to_string(),
                    found: other.clone(),
                    span: self.current_span(),
                });
            }
        };

        // Parse precedence (0-9)
        let precedence = match self.peek() {
            Token::Int(n) if *n >= 0 && *n <= 9 => {
                let p = *n as u8;
                self.advance();
                p
            }
            other => {
                return Err(ParseError::UnexpectedToken {
                    expected: "precedence (0-9)".to_string(),
                    found: other.clone(),
                    span: self.current_span(),
                });
            }
        };

        // Parse one or more operator symbols
        let mut operators = Vec::new();
        while let Some(op_str) = self.peek().operator_symbol() {
            operators.push(op_str);
            self.advance();
        }

        if operators.is_empty() {
            return Err(ParseError::UnexpectedToken {
                expected: "operator symbol".to_string(),
                found: self.peek().clone(),
                span: self.current_span(),
            });
        }

        let span = start.merge(&self.current_span());

        // Register operators in the operator table so they affect parsing
        for op in &operators {
            let info = OpInfo {
                precedence,
                assoc,
                is_builtin: false,
            };
            if let Some(warning) = self.op_table.register(op.clone(), info, span.clone()) {
                self.warnings.push(warning);
            }
        }

        Ok(Decl::Fixity(FixityDecl {
            assoc,
            precedence,
            operators,
            span,
        }))
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
            Token::Val => self.parse_val_decl(),
            _ => Err(self.unexpected_token("declaration")),
        }
    }

    /// Parse a standalone type signature: val xs : [Int]
    fn parse_val_decl(&mut self) -> Result<Decl, ParseError> {
        self.consume(Token::Val)?;
        let name = self.parse_ident()?;
        self.consume(Token::Colon)?;
        let type_sig = self.parse_type_expr()?;
        Ok(Decl::Val { name, type_sig })
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
            visibility: Visibility::Private,
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

        // Check if it's a variant type, record type, or alias
        if self.check(&Token::LBrace) {
            // Record type: type Request = { method : String, path : String }
            let fields = self.parse_record_fields()?;
            Ok(Decl::Record {
                visibility: Visibility::Private,
                name,
                params,
                fields,
            })
        } else if self.check(&Token::Pipe) || self.peek_is_upper_ident() {
            // Variant type
            let constructors = self.parse_constructors()?;
            Ok(Decl::Type {
                visibility: Visibility::Private,
                name,
                params,
                constructors,
            })
        } else {
            // Type alias
            let body = self.parse_type_expr()?;
            Ok(Decl::TypeAlias {
                visibility: Visibility::Private,
                name,
                params,
                body,
            })
        }
    }

    /// Parse record field declarations: { field1 : Type1, field2 : Type2, ... }
    fn parse_record_fields(&mut self) -> Result<Vec<RecordField>, ParseError> {
        self.consume(Token::LBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = self.parse_ident()?;
                self.consume(Token::Colon)?;
                let field_ty = self.parse_type_expr()?;
                fields.push(RecordField {
                    name: field_name,
                    ty: field_ty,
                });

                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
    }

    /// Parse record literal fields: { field1 = expr1, field2 = expr2, ... }
    fn parse_record_literal_fields(&mut self) -> Result<Vec<(Ident, Expr)>, ParseError> {
        self.consume(Token::LBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = self.parse_ident()?;
                self.consume(Token::Eq)?;
                let field_value = self.parse_expr()?;
                fields.push((field_name, field_value));

                if !self.match_token(&Token::Comma) {
                    break;
                }
                // Allow trailing comma
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
    }

    /// Parse record update fields: field1 = expr1, field2 = expr2, ...
    /// Note: The '{' and 'with' are already consumed, this just parses the field updates
    fn parse_record_update_fields(&mut self) -> Result<Vec<(Ident, Expr)>, ParseError> {
        let mut updates = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = self.parse_ident()?;
                self.consume(Token::Eq)?;
                let field_value = self.parse_expr()?;
                updates.push((field_name, field_value));

                if !self.match_token(&Token::Comma) {
                    break;
                }
                // Allow trailing comma
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        Ok(updates)
    }

    /// Parse record pattern fields: { field1, field2 = pat, ... }
    /// Each field is either:
    /// - Just a name (shorthand): { method } binds 'method' to the field value
    /// - Name with pattern: { method = m } binds 'm' to the field value
    fn parse_record_pattern_fields(&mut self) -> Result<Vec<(Ident, Option<Pattern>)>, ParseError> {
        self.consume(Token::LBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = self.parse_ident()?;
                let pattern = if self.match_token(&Token::Eq) {
                    // Explicit pattern binding: field = pattern
                    Some(self.parse_pattern()?)
                } else {
                    // Shorthand: just field name, binds to same name
                    None
                };
                fields.push((field_name, pattern));

                if !self.match_token(&Token::Comma) {
                    break;
                }
                // Allow trailing comma
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
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
            visibility: Visibility::Private,
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
                        return Ok(Spanned::new(TypeExprKind::Channel(Rc::new(inner)), span));
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
            Token::LBracket => {
                // List type: [a]
                self.advance();
                let elem = self.parse_type_expr()?;
                self.consume(Token::RBracket)?;
                let span = start.merge(&self.current_span());
                Ok(Spanned::new(TypeExprKind::List(Rc::new(elem)), span))
            }
            _ => Err(self.unexpected_token("type")),
        }
    }

    // ========================================================================
    // Expressions
    // ========================================================================

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_expr_in(ExprContext::Full)
    }

    /// Parse an expression in a specific context.
    /// This is the main entry point for context-aware expression parsing.
    fn parse_expr_in(&mut self, ctx: ExprContext) -> Result<Expr, ParseError> {
        if ctx.allows_seq() {
            self.parse_expr_seq()
        } else if ctx.allows_let() {
            self.parse_expr_let()
        } else {
            self.parse_expr_binary(0)
        }
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

            // Check for 'rec' keyword for mutual recursion
            if self.match_token(&Token::Rec) {
                return self.parse_let_rec_expr(start);
            }

            // Check for prefix operator syntax: let (<|>) a b = ... in ...
            if self.check(&Token::LParen) {
                if let Some(op) = self.peek_operator_in_parens() {
                    return self.parse_operator_let_expr(start, op, None);
                }
            }

            // First, try to parse a simple pattern (for regular let or function name)
            let first_span = self.current_span();
            let first_pattern = self.parse_simple_pattern()?;

            // Check for infix operator syntax: let a <|> b = ... in ...
            // But only if first_pattern is a Var (identifier)
            if let PatternKind::Var(ref name) = first_pattern.node {
                if let Some(op) = self.try_peek_operator() {
                    return self.parse_operator_let_expr(
                        start,
                        op,
                        Some((name.clone(), first_span)),
                    );
                }
            }

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

    /// Parse operator let-expression (both prefix and infix syntax)
    /// Prefix: let (<|>) a b = e in body
    /// Infix: let a <|> b = e in body
    fn parse_operator_let_expr(
        &mut self,
        start: Span,
        op: String,
        first_param: Option<(String, Span)>,
    ) -> Result<Expr, ParseError> {
        let mut params = Vec::new();

        if let Some((name, param_span)) = first_param {
            // Infix syntax: we have the first param already, consume the operator
            self.advance();
            params.push(Spanned::new(PatternKind::Var(name), param_span));
            // Parse second parameter
            params.push(self.parse_simple_pattern()?);
        } else {
            // Prefix syntax: consume ( op ) then parse all params
            self.consume(Token::LParen)?;
            self.advance(); // consume the operator token
            self.consume(Token::RParen)?;
            while self.is_pattern_start() && !self.check(&Token::Eq) {
                params.push(self.parse_simple_pattern()?);
            }
        }

        self.consume(Token::Eq)?;
        let value_expr = self.parse_expr()?;
        self.consume(Token::In)?;
        let in_body = self.parse_expr()?;

        let span = start.merge(&in_body.span);

        // Create a lambda for the operator: fun params -> value_expr
        let lambda_span = value_expr.span.clone();
        let lambda = Spanned::new(
            ExprKind::Lambda {
                params,
                body: Rc::new(value_expr),
            },
            lambda_span,
        );

        // Create a Var pattern for the operator name
        let op_pattern = Spanned::new(PatternKind::Var(op), start.clone());

        Ok(Spanned::new(
            ExprKind::Let {
                pattern: op_pattern,
                value: Rc::new(lambda),
                body: Some(Rc::new(in_body)),
            },
            span,
        ))
    }

    /// Parse let rec expression: let rec f x = ... and g y = ... in body
    fn parse_let_rec_expr(&mut self, start: Span) -> Result<Expr, ParseError> {
        let mut bindings = Vec::new();

        // Parse first binding
        bindings.push(self.parse_rec_binding()?);

        // Parse additional bindings with 'and'
        while self.match_token(&Token::And) {
            bindings.push(self.parse_rec_binding()?);
        }

        // Must have 'in' for expression form
        self.consume(Token::In)?;
        let body = self.parse_expr()?;
        let span = start.merge(&body.span);

        Ok(Spanned::new(
            ExprKind::LetRec {
                bindings,
                body: Some(Rc::new(body)),
            },
            span,
        ))
    }

    fn parse_expr_if(&mut self) -> Result<Expr, ParseError> {
        if self.check(&Token::If) {
            let start = self.current_span();
            self.advance();

            // Use NoSeq context for branches - sequences require parentheses
            let cond = self.parse_expr_in(ExprContext::NoSeq)?;
            self.consume(Token::Then)?;
            let then_branch = self.parse_expr_in(ExprContext::NoSeq)?;
            self.consume(Token::Else)?;
            let else_branch = self.parse_expr_in(ExprContext::NoSeq)?;

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
            self.parse_expr_binary(0)
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
            let body = self.parse_expr_in(ExprContext::Full)?; // Body allows full language including sequences

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
        let body = self.parse_expr_in(ExprContext::Full)?; // Lambda bodies allow full language

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
        let scrutinee = self.parse_expr_binary(0)?;
        self.consume(Token::With)?;

        let mut arms = Vec::new();
        // Optional leading pipe
        self.match_token(&Token::Pipe);

        loop {
            let pattern = self.parse_pattern()?;

            // Optional guard - uses BinaryOnly context (intentional restriction)
            let guard = if self.match_token(&Token::If) {
                Some(self.parse_expr_in(ExprContext::BinaryOnly)?)
            } else {
                None
            };

            self.consume(Token::Arrow)?;
            let body = self.parse_expr_in(ExprContext::Full)?; // Body allows full language including sequences

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

    /// Pratt parser for binary expressions.
    /// Uses precedence climbing to handle operator precedence and associativity.
    fn parse_expr_binary(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut left = self.parse_expr_unary()?;

        loop {
            // Check if current token is an operator
            let (op_str, _op_span) = match self.peek_operator_symbol() {
                Some(o) => o,
                None => break,
            };

            // Special case: |> and <| desugar to App, not BinOp
            if op_str == "|>" {
                let pipe_info = self
                    .op_table
                    .get("|>")
                    .cloned()
                    .unwrap_or_else(OpInfo::default);
                if pipe_info.precedence < min_prec {
                    break;
                }
                self.advance(); // consume |>
                let next_min = pipe_info.precedence + 1; // left-associative
                let right = self.parse_expr_binary(next_min)?;
                let span = start.merge(&right.span);
                // x |> f  =>  f x
                left = Spanned::new(
                    ExprKind::App {
                        func: Rc::new(right),
                        arg: Rc::new(left),
                    },
                    span,
                );
                continue;
            }

            if op_str == "<|" {
                let pipe_info = self
                    .op_table
                    .get("<|")
                    .cloned()
                    .unwrap_or_else(OpInfo::default);
                if pipe_info.precedence < min_prec {
                    break;
                }
                self.advance(); // consume <|
                let next_min = pipe_info.precedence; // right-associative
                let right = self.parse_expr_binary(next_min)?;
                let span = start.merge(&right.span);
                // f <| x  =>  f x
                left = Spanned::new(
                    ExprKind::App {
                        func: Rc::new(left),
                        arg: Rc::new(right),
                    },
                    span,
                );
                continue;
            }

            // Look up precedence in table
            let info = self
                .op_table
                .get(&op_str)
                .cloned()
                .unwrap_or_else(OpInfo::default);

            if info.precedence < min_prec {
                break;
            }

            self.advance(); // consume the operator

            // Calculate next min_prec based on associativity
            let next_min = match info.assoc {
                Associativity::Left => info.precedence + 1,
                Associativity::Right => info.precedence,
                Associativity::None => info.precedence + 1,
            };

            let right = self.parse_expr_binary(next_min)?;
            let span = start.merge(&right.span);

            // Build the BinOp node
            left = self.make_binop(&op_str, left, right, span);
        }

        Ok(left)
    }

    /// Check if the current token is an operator and return its string representation.
    fn peek_operator_symbol(&self) -> Option<(String, Span)> {
        let span = self.current_span();
        let op_str = self.peek().operator_symbol()?;
        Some((op_str, span))
    }

    /// Build a BinOp expression from an operator string.
    fn make_binop(&self, op_str: &str, left: Expr, right: Expr, span: Span) -> Expr {
        let op = match op_str {
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            "*" => BinOp::Mul,
            "/" => BinOp::Div,
            "%" => BinOp::Mod,
            "==" => BinOp::Eq,
            "!=" => BinOp::Neq,
            "<" => BinOp::Lt,
            ">" => BinOp::Gt,
            "<=" => BinOp::Lte,
            ">=" => BinOp::Gte,
            "&&" => BinOp::And,
            "||" => BinOp::Or,
            "::" => BinOp::Cons,
            "++" => BinOp::Concat,
            ">>" => BinOp::Compose,
            "<<" => BinOp::ComposeBack,
            // User-defined operator
            _ => BinOp::UserDefined(op_str.to_string()),
        };

        Spanned::new(
            ExprKind::BinOp {
                op,
                left: Rc::new(left),
                right: Rc::new(right),
            },
            span,
        )
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
        let mut func = self.parse_expr_postfix()?;

        while self.is_atom_start() {
            let arg = self.parse_expr_postfix()?;
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

    /// Parse postfix operations (field access): expr.field.another_field
    fn parse_expr_postfix(&mut self) -> Result<Expr, ParseError> {
        let start = self.current_span();
        let mut expr = self.parse_expr_atom()?;

        // Handle chained field access: expr.field1.field2
        while self.check(&Token::Dot) {
            self.advance(); // consume '.'
            let field = self.parse_ident()?;
            let span = start.merge(&self.current_span());
            expr = Spanned::new(
                ExprKind::FieldAccess {
                    record: Rc::new(expr),
                    field,
                },
                span,
            );
        }

        Ok(expr)
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
                // Just return the variable - field access (expr.field) is handled by parse_expr_postfix
                Ok(Spanned::new(ExprKind::Var(name), start))
            }
            Token::UpperIdent(name) => {
                self.advance();
                // Constructor, Record literal, or Module access
                if self.check(&Token::LBrace) {
                    // Record literal: TypeName { field = expr, ... }
                    let fields = self.parse_record_literal_fields()?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(
                        ExprKind::Record { name, fields },
                        span,
                    ))
                } else if self.check(&Token::Dot) {
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
                        ExprKind::Constructor { name, args: vec![] },
                        start,
                    ))
                }
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
                        Ok(Spanned::new(
                            ExprKind::Shift {
                                param: params[0].clone(),
                                body: body.clone(),
                            },
                            span,
                        ))
                    }
                    _ => Err(self.unexpected_token("function (fun k -> ...)")),
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
            Token::LBrace => {
                // Record update: { base_expr with field1 = val1, ... }
                self.advance(); // consume '{'
                let base = self.parse_expr()?;
                self.consume(Token::With)?;
                let updates = self.parse_record_update_fields()?;
                self.consume(Token::RBrace)?;
                let span = start.merge(&self.current_span());
                Ok(Spanned::new(
                    ExprKind::RecordUpdate {
                        base: Rc::new(base),
                        updates,
                    },
                    span,
                ))
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

    /// Parse a simple pattern (constructor with args allowed)
    fn parse_simple_pattern(&mut self) -> Result<Pattern, ParseError> {
        self.parse_pattern_primary(true)
    }

    /// Parse a pattern atom (no constructor args - used for constructor arguments themselves)
    fn parse_pattern_atom(&mut self) -> Result<Pattern, ParseError> {
        self.parse_pattern_primary(false)
    }

    fn is_pattern_atom(&self) -> bool {
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

    /// Unified pattern parsing - the allow_constructor_args parameter controls whether
    /// constructors can take arguments (true for simple patterns, false for atoms)
    fn parse_pattern_primary(
        &mut self,
        allow_constructor_args: bool,
    ) -> Result<Pattern, ParseError> {
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
                // Check for record pattern: Request { field1, field2 = pat }
                if self.check(&Token::LBrace) {
                    let fields = self.parse_record_pattern_fields()?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(PatternKind::Record { name, fields }, span))
                } else if allow_constructor_args {
                    // Constructor with args: "Some x" or "Pair a b"
                    let mut args = Vec::new();
                    while self.is_pattern_atom() {
                        args.push(self.parse_pattern_atom()?);
                    }
                    let span = if args.is_empty() {
                        start
                    } else {
                        start.merge(&args.last().unwrap().span)
                    };
                    Ok(Spanned::new(PatternKind::Constructor { name, args }, span))
                } else {
                    // Nullary constructor as atom: "Get" in "StateOp Get k"
                    Ok(Spanned::new(
                        PatternKind::Constructor { name, args: vec![] },
                        start,
                    ))
                }
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
            Token::Underscore => {
                self.advance();
                Ok("_".to_string())
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
                ..
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
        let prog = parse("trait Eq a = val eq : a -> a -> Bool val neq : a -> a -> Bool end");
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
            Item::Expr(expr) => match &expr.node {
                ExprKind::Let { body, .. } => {
                    assert!(body.is_some(), "let-expression should have a body");
                }
                _ => panic!("expected let expression"),
            },
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

    #[test]
    fn test_fixity_decl() {
        // Test basic fixity declaration
        let prog = parse("infixl 6 +");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Fixity(f)) => {
                assert_eq!(f.assoc, Associativity::Left);
                assert_eq!(f.precedence, 6);
                assert_eq!(f.operators, vec!["+"]);
            }
            _ => panic!("expected fixity decl"),
        }
    }

    #[test]
    fn test_fixity_decl_user_op() {
        // Test fixity declaration with user-defined operator
        let prog = parse("infixl 4 <|>");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Fixity(f)) => {
                assert_eq!(f.assoc, Associativity::Left);
                assert_eq!(f.precedence, 4);
                assert_eq!(f.operators, vec!["<|>"]);
            }
            _ => panic!("expected fixity decl"),
        }
    }

    #[test]
    fn test_fixity_decl_multiple_ops() {
        // Test fixity declaration with multiple operators
        let prog = parse("infixr 5 :: ++");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Fixity(f)) => {
                assert_eq!(f.assoc, Associativity::Right);
                assert_eq!(f.precedence, 5);
                assert_eq!(f.operators, vec!["::", "++"]);
            }
            _ => panic!("expected fixity decl"),
        }
    }

    #[test]
    fn test_fixity_decl_non_assoc() {
        // Test non-associative fixity
        let prog = parse("infix 4 ==");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Fixity(f)) => {
                assert_eq!(f.assoc, Associativity::None);
                assert_eq!(f.precedence, 4);
                assert_eq!(f.operators, vec!["=="]);
            }
            _ => panic!("expected fixity decl"),
        }
    }

    #[test]
    fn test_user_defined_operator_expr() {
        // Test that user-defined operators are parsed in expressions
        let prog = parse("a <|> b");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::BinOp { op, .. } => {
                    assert_eq!(*op, BinOp::UserDefined("<|>".to_string()));
                }
                _ => panic!("expected BinOp"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_user_defined_operator_precedence() {
        // Test that built-in operators still work correctly with Pratt parser
        // a + b * c should parse as a + (b * c)
        let prog = parse("a + b * c");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::BinOp { op, left, right } => {
                    assert_eq!(*op, BinOp::Add);
                    // left is just 'a'
                    assert!(matches!(left.node, ExprKind::Var(_)));
                    // right should be b * c
                    match &right.node {
                        ExprKind::BinOp { op: inner_op, .. } => {
                            assert_eq!(*inner_op, BinOp::Mul);
                        }
                        _ => panic!("expected inner BinOp"),
                    }
                }
                _ => panic!("expected BinOp"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_pipe_desugaring() {
        // Test that |> is desugared to application
        let prog = parse("x |> f");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::App { func, arg } => {
                    // f applied to x
                    assert!(matches!(func.node, ExprKind::Var(ref n) if n == "f"));
                    assert!(matches!(arg.node, ExprKind::Var(ref n) if n == "x"));
                }
                _ => panic!("expected App, got {:?}", expr.node),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_operator_def_prefix() {
        // Test prefix syntax: let (<|>) a b = a
        let prog = parse("let (<|>) a b = a");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::OperatorDef { op, params, .. }) => {
                assert_eq!(op, "<|>");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected OperatorDef"),
        }
    }

    #[test]
    fn test_operator_def_infix() {
        // Test infix syntax: let a <|> b = a
        let prog = parse("let a <|> b = a");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::OperatorDef { op, params, .. }) => {
                assert_eq!(op, "<|>");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected OperatorDef"),
        }
    }

    #[test]
    fn test_operator_def_builtin_op() {
        // Test redefining a built-in operator with prefix syntax
        let prog = parse("let (+) a b = a * b");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::OperatorDef { op, params, .. }) => {
                assert_eq!(op, "+");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected OperatorDef"),
        }
    }

    #[test]
    fn test_fixity_shadowing_warning() {
        // Test that shadowing a built-in operator produces a warning
        let tokens = Lexer::new("infixl 9 +").tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let _prog = parser.parse_program().unwrap();
        let warnings = parser.warnings();
        assert_eq!(warnings.len(), 1);
        match &warnings[0] {
            Warning::ShadowingBuiltinOperator { op, .. } => {
                assert_eq!(op, "+");
            }
        }
    }

    #[test]
    fn test_fixity_no_warning_for_new_op() {
        // Test that declaring a new operator produces no warning
        let tokens = Lexer::new("infixl 4 <|>").tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let _prog = parser.parse_program().unwrap();
        let warnings = parser.warnings();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_let_rec_simple() {
        // Test simple let rec
        let prog = parse("let rec f n = n");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::LetRec { bindings, .. }) => {
                assert_eq!(bindings.len(), 1);
                assert_eq!(bindings[0].name.node, "f");
                assert_eq!(bindings[0].params.len(), 1);
            }
            _ => panic!("expected LetRec"),
        }
    }

    #[test]
    fn test_let_rec_mutual() {
        // Test mutual recursion with 'and'
        let prog = parse("let rec f n = g n and g n = f n");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::LetRec { bindings, .. }) => {
                assert_eq!(bindings.len(), 2);
                assert_eq!(bindings[0].name.node, "f");
                assert_eq!(bindings[1].name.node, "g");
            }
            _ => panic!("expected LetRec"),
        }
    }

    #[test]
    fn test_let_rec_expr() {
        // Test let rec as expression (with 'in')
        let prog = parse("let rec f n = n in f 5");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::LetRec { bindings, body } => {
                    assert_eq!(bindings.len(), 1);
                    assert!(body.is_some());
                }
                _ => panic!("expected LetRec expression"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_let_in_match_arm() {
        // Match arm bodies should allow let expressions
        let prog = parse("match x with | Some y -> let z = y + 1 in z | None -> 0");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::Match { arms, .. } => {
                    assert_eq!(arms.len(), 2);
                    // First arm body should be a Let expression
                    match &arms[0].body.node {
                        ExprKind::Let { .. } => {}
                        _ => panic!("expected Let expression in match arm body"),
                    }
                }
                _ => panic!("expected Match expression"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_if_in_match_arm() {
        // Match arm bodies should allow if expressions
        let prog = parse("match x with | Some y -> if y > 0 then y else 0 | None -> 0");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::Match { arms, .. } => {
                    assert_eq!(arms.len(), 2);
                    // First arm body should be an If expression
                    match &arms[0].body.node {
                        ExprKind::If { .. } => {}
                        _ => panic!("expected If expression in match arm body"),
                    }
                }
                _ => panic!("expected Match expression"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_fun_in_match_arm() {
        // Match arm bodies should allow lambda expressions
        let prog = parse("match x with | Some y -> fun z -> y + z | None -> fun z -> z");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::Match { arms, .. } => {
                    assert_eq!(arms.len(), 2);
                    // First arm body should be a Lambda expression
                    match &arms[0].body.node {
                        ExprKind::Lambda { .. } => {}
                        _ => panic!("expected Lambda expression in match arm body"),
                    }
                }
                _ => panic!("expected Match expression"),
            },
            _ => panic!("expected expression"),
        }
    }

    // ========================================================================
    // Module system tests
    // ========================================================================

    #[test]
    fn test_import_simple() {
        let prog = parse("import List");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Import(import) => {
                assert_eq!(import.node.module_path, "List");
                assert!(import.node.alias.is_none());
                assert!(import.node.items.is_none());
            }
            _ => panic!("expected import"),
        }
    }

    #[test]
    fn test_import_with_alias() {
        let prog = parse("import List as L");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Import(import) => {
                assert_eq!(import.node.module_path, "List");
                assert_eq!(import.node.alias, Some("L".to_string()));
                assert!(import.node.items.is_none());
            }
            _ => panic!("expected import"),
        }
    }

    #[test]
    fn test_import_selective() {
        let prog = parse("import List (map, filter)");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Import(import) => {
                assert_eq!(import.node.module_path, "List");
                assert!(import.node.alias.is_none());
                let items = import.node.items.as_ref().unwrap();
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], ("map".to_string(), None));
                assert_eq!(items[1], ("filter".to_string(), None));
            }
            _ => panic!("expected import"),
        }
    }

    #[test]
    fn test_import_selective_with_alias() {
        let prog = parse("import List (map as m, filter as f)");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Import(import) => {
                let items = import.node.items.as_ref().unwrap();
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], ("map".to_string(), Some("m".to_string())));
                assert_eq!(items[1], ("filter".to_string(), Some("f".to_string())));
            }
            _ => panic!("expected import"),
        }
    }

    #[test]
    fn test_pub_let() {
        let prog = parse("pub let x = 42");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Let { visibility, name, .. }) => {
                assert_eq!(*visibility, Visibility::Public);
                assert_eq!(name, "x");
            }
            _ => panic!("expected pub let"),
        }
    }

    #[test]
    fn test_pub_let_function() {
        let prog = parse("pub let add x y = x + y");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Let { visibility, name, params, .. }) => {
                assert_eq!(*visibility, Visibility::Public);
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
            }
            _ => panic!("expected pub let"),
        }
    }

    #[test]
    fn test_pub_type() {
        let prog = parse("pub type Option a = | Some a | None");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Type { visibility, name, .. }) => {
                assert_eq!(*visibility, Visibility::Public);
                assert_eq!(name, "Option");
            }
            _ => panic!("expected pub type"),
        }
    }

    #[test]
    fn test_pub_trait() {
        let prog = parse("pub trait Show a = val show : a -> String end");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Trait { visibility, name, .. }) => {
                assert_eq!(*visibility, Visibility::Public);
                assert_eq!(name, "Show");
            }
            _ => panic!("expected pub trait"),
        }
    }

    #[test]
    fn test_private_by_default() {
        let prog = parse("let x = 42");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Let { visibility, .. }) => {
                assert_eq!(*visibility, Visibility::Private);
            }
            _ => panic!("expected let"),
        }
    }

    // ========================================================================
    // Record type tests
    // ========================================================================

    #[test]
    fn test_record_type_decl() {
        let prog = parse("type Request = { method : String, path : String }");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Record { name, params, fields, .. }) => {
                assert_eq!(name, "Request");
                assert!(params.is_empty());
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "method");
                assert_eq!(fields[1].name, "path");
            }
            _ => panic!("expected record type declaration"),
        }
    }

    #[test]
    fn test_record_type_with_params() {
        let prog = parse("type Pair a = { fst : a, snd : a }");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Decl(Decl::Record { name, params, fields, .. }) => {
                assert_eq!(name, "Pair");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0], "a");
                assert_eq!(fields.len(), 2);
            }
            _ => panic!("expected record type declaration"),
        }
    }

    #[test]
    fn test_record_literal() {
        let prog = parse("Request { method = \"GET\", path = \"/\" }");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::Record { name, fields } => {
                    assert_eq!(name, "Request");
                    assert_eq!(fields.len(), 2);
                    assert_eq!(fields[0].0, "method");
                    assert_eq!(fields[1].0, "path");
                }
                _ => panic!("expected record literal"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_field_access() {
        let prog = parse("req.method");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::FieldAccess { record, field } => {
                    assert_eq!(field, "method");
                    match &record.node {
                        ExprKind::Var(name) => assert_eq!(name, "req"),
                        _ => panic!("expected variable"),
                    }
                }
                _ => panic!("expected field access"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_chained_field_access() {
        let prog = parse("config.server.port");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::FieldAccess { record, field } => {
                    assert_eq!(field, "port");
                    match &record.node {
                        ExprKind::FieldAccess { record: inner, field: mid } => {
                            assert_eq!(mid, "server");
                            match &inner.node {
                                ExprKind::Var(name) => assert_eq!(name, "config"),
                                _ => panic!("expected variable"),
                            }
                        }
                        _ => panic!("expected nested field access"),
                    }
                }
                _ => panic!("expected field access"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_record_update() {
        let prog = parse("{ req with method = \"POST\" }");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::RecordUpdate { base, updates } => {
                    match &base.node {
                        ExprKind::Var(name) => assert_eq!(name, "req"),
                        _ => panic!("expected variable"),
                    }
                    assert_eq!(updates.len(), 1);
                    assert_eq!(updates[0].0, "method");
                }
                _ => panic!("expected record update"),
            },
            _ => panic!("expected expression"),
        }
    }

    #[test]
    fn test_record_pattern() {
        let prog = parse("match req with | Request { method, path } -> method");
        assert_eq!(prog.items.len(), 1);
        match &prog.items[0] {
            Item::Expr(expr) => match &expr.node {
                ExprKind::Match { arms, .. } => {
                    assert_eq!(arms.len(), 1);
                    match &arms[0].pattern.node {
                        PatternKind::Record { name, fields } => {
                            assert_eq!(name, "Request");
                            assert_eq!(fields.len(), 2);
                            assert_eq!(fields[0].0, "method");
                            assert!(fields[0].1.is_none()); // shorthand
                            assert_eq!(fields[1].0, "path");
                            assert!(fields[1].1.is_none()); // shorthand
                        }
                        _ => panic!("expected record pattern"),
                    }
                }
                _ => panic!("expected match expression"),
            },
            _ => panic!("expected expression"),
        }
    }
}
