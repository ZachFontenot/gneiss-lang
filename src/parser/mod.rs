//! Recursive descent parser for Gneiss
//!
//! This module provides the parser for the Gneiss programming language.
//! It uses a recursive descent approach with Pratt parsing for expressions.
//!
//! # Module Structure
//!
//! - `cursor` - Token stream navigation and lookahead
//! - `combinators` - Reusable parsing patterns
//! - `error` - Error types with source location tracking
//! - `types` - Type expression parsing
//! - `pattern` - Pattern parsing
//! - `record` - Unified record field parsing
//!
//! # Future modules (to be added):
//! - `expr` - Expression parsing with Pratt parser
//! - `decl` - Declaration parsing
//! - `item` - Top-level item parsing

pub mod combinators;
pub mod cursor;
pub mod error;
pub mod pattern;
pub mod record;
pub mod types;

// Re-export main types for convenience
pub use cursor::TokenCursor;
pub use error::{ParseError, ParseResult};

use crate::ast::*;
use crate::errors::Warning;
use crate::lexer::{SpannedToken, Token};
use crate::operators::{Associativity, OpInfo, OperatorTable};
use std::rc::Rc;

/// What expression forms are allowed in the current parsing context.
/// This makes explicit where each expression form can appear.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExprContext {
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

/// The Gneiss parser
pub struct Parser {
    cursor: TokenCursor,
    /// Operator precedence and associativity table
    op_table: OperatorTable,
    /// Warnings collected during parsing
    warnings: Vec<Warning>,
}

impl Parser {
    /// Create a new parser from a token stream
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Self {
            cursor: TokenCursor::new(tokens),
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

    // ========================================================================
    // Cursor delegation methods (for compatibility during transition)
    // ========================================================================

    fn peek(&self) -> &Token {
        self.cursor.peek()
    }

    fn current_span(&self) -> Span {
        self.cursor.current_span()
    }

    fn advance(&mut self) -> &SpannedToken {
        self.cursor.advance()
    }

    fn is_at_end(&self) -> bool {
        self.cursor.is_at_end()
    }

    fn check(&self, token: &Token) -> bool {
        self.cursor.check(token)
    }

    fn consume(&mut self, expected: Token) -> ParseResult<&SpannedToken> {
        self.cursor.consume(expected)
    }

    fn match_token(&mut self, token: &Token) -> bool {
        self.cursor.match_token(token)
    }

    fn unexpected_token(&self, expected: &str) -> ParseError {
        self.cursor.unexpected(expected)
    }

    fn is_atom_start(&self) -> bool {
        self.cursor.is_atom_start()
    }

    fn is_pattern_start(&self) -> bool {
        self.cursor.is_pattern_start()
    }

    fn is_type_atom_start(&self) -> bool {
        self.cursor.is_type_atom_start()
    }

    fn peek_is_upper_ident(&self) -> bool {
        self.cursor.is_upper_ident()
    }

    fn peek_operator_symbol(&self) -> Option<(String, Span)> {
        self.cursor.peek_operator_symbol()
    }

    fn try_peek_operator(&self) -> Option<String> {
        self.cursor.try_peek_operator()
    }

    fn peek_operator_in_parens(&self) -> Option<String> {
        self.cursor.peek_operator_in_parens()
    }

    // ========================================================================
    // Program parsing
    // ========================================================================

    pub fn parse_program(&mut self) -> ParseResult<Program> {
        // Parse optional export list (must be first if present)
        let exports = if self.check(&Token::Export) {
            Some(self.parse_export()?)
        } else {
            None
        };

        let mut items = Vec::new();
        while !self.is_at_end() {
            items.push(self.parse_item()?);
        }
        Ok(Program { exports, items })
    }

    fn parse_item(&mut self) -> ParseResult<Item> {
        // First, optionally consume any leading ;;
        while self.match_token(&Token::DoubleSemi) {}

        match self.peek() {
            Token::Import => {
                let import = self.parse_import()?;
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Import(import))
            }
            Token::Export => Err(ParseError::unexpected(
                "export declaration must appear at the beginning of the module",
                self.peek().clone(),
                self.current_span(),
            )),
            Token::Let => self.parse_let_item(),
            Token::Type | Token::Trait | Token::Impl | Token::Val => {
                let decl = self.parse_decl()?;
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Decl(decl))
            }
            Token::Ident(ref s) if s == "infixl" || s == "infixr" || s == "infix" => {
                let decl = self.parse_fixity_decl()?;
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Decl(decl))
            }
            _ => {
                let expr = self.parse_expr()?;
                self.match_token(&Token::DoubleSemi);
                Ok(Item::Expr(expr))
            }
        }
    }

    // ========================================================================
    // Import/Export parsing
    // ========================================================================

    fn parse_import(&mut self) -> ParseResult<Spanned<ImportSpec>> {
        let start = self.current_span();
        self.consume(Token::Import)?;

        let module_path = self.parse_module_path()?;

        let alias = if self.match_token(&Token::As) {
            Some(self.parse_upper_ident()?)
        } else {
            None
        };

        let items = if self.check(&Token::LParen) {
            self.advance();
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

    fn parse_module_path(&mut self) -> ParseResult<String> {
        let first = self.parse_upper_ident()?;
        let mut path = first;

        while self.check(&Token::Slash) {
            self.advance();
            let next = self.parse_upper_ident()?;
            path.push('/');
            path.push_str(&next);
        }

        Ok(path)
    }

    fn parse_export(&mut self) -> ParseResult<ExportDecl> {
        use crate::ast::ExportItem;

        let start = self.current_span();
        self.consume(Token::Export)?;
        self.consume(Token::LParen)?;

        let mut items = Vec::new();

        if !self.check(&Token::RParen) {
            loop {
                let item_start = self.current_span();
                let item = match self.peek() {
                    Token::Ident(name) => {
                        let name = name.clone();
                        self.advance();
                        ExportItem::Value(name)
                    }
                    Token::UpperIdent(name) => {
                        let name = name.clone();
                        self.advance();

                        if self.check(&Token::LParen) {
                            self.advance();

                            if self.match_token(&Token::Dot) {
                                self.consume(Token::Dot)?;
                                self.consume(Token::RParen)?;
                                ExportItem::TypeAll(name)
                            } else {
                                let mut constructors = Vec::new();
                                if !self.check(&Token::RParen) {
                                    loop {
                                        if let Token::UpperIdent(ctor) = self.peek() {
                                            constructors.push(ctor.clone());
                                            self.advance();
                                        } else {
                                            return Err(self.unexpected_token("constructor name"));
                                        }

                                        if !self.match_token(&Token::Comma) {
                                            break;
                                        }
                                    }
                                }
                                self.consume(Token::RParen)?;
                                ExportItem::TypeSome(name, constructors)
                            }
                        } else {
                            ExportItem::TypeOnly(name)
                        }
                    }
                    _ => {
                        return Err(
                            self.unexpected_token("identifier or type name in export list")
                        );
                    }
                };

                let item_end = self.current_span();
                items.push(Spanned::new(item, Span::merge(&item_start, &item_end)));

                if !self.match_token(&Token::Comma) {
                    break;
                }

                if self.check(&Token::RParen) {
                    break;
                }
            }
        }

        self.consume(Token::RParen)?;
        let end = self.current_span();

        Ok(ExportDecl {
            items,
            span: Span::merge(&start, &end),
        })
    }

    // ========================================================================
    // Let parsing
    // ========================================================================

    fn parse_let_item(&mut self) -> ParseResult<Item> {
        let start = self.current_span();
        self.consume(Token::Let)?;

        if self.match_token(&Token::Rec) {
            return self.parse_let_rec(start);
        }

        if self.check(&Token::LParen) {
            if let Some(op) = self.peek_operator_in_parens() {
                return self.parse_operator_def(start, op, None);
            }
        }

        let first_span = self.current_span();
        let first_name = self.parse_possibly_qualified_name()?;

        if let Some(op) = self.try_peek_operator() {
            return self.parse_operator_def(start, op, Some((first_name, first_span)));
        }

        let mut params = Vec::new();
        while self.is_pattern_start() && !self.check(&Token::Eq) {
            params.push(self.parse_simple_pattern()?);
        }

        self.consume(Token::Eq)?;
        let value_expr = self.parse_expr()?;

        if self.check(&Token::In) {
            self.advance();
            let body = self.parse_expr()?;
            let span = start.merge(&body.span);

            let (pattern, value) = if params.is_empty() {
                let name_pattern = Spanned::new(PatternKind::Var(first_name), start.clone());
                (name_pattern, value_expr)
            } else {
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
            self.match_token(&Token::DoubleSemi);
            Ok(Item::Expr(expr))
        } else {
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

    fn parse_operator_def(
        &mut self,
        _start: Span,
        op: String,
        first_param: Option<(String, Span)>,
    ) -> ParseResult<Item> {
        let mut params = Vec::new();

        if let Some((name, param_span)) = first_param {
            self.advance();
            params.push(Spanned::new(PatternKind::Var(name), param_span));
            params.push(self.parse_simple_pattern()?);
        } else {
            self.consume(Token::LParen)?;
            self.advance();
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

    fn parse_let_rec(&mut self, start: Span) -> ParseResult<Item> {
        let mut bindings = Vec::new();

        bindings.push(self.parse_rec_binding()?);

        while self.match_token(&Token::And) {
            bindings.push(self.parse_rec_binding()?);
        }

        if self.check(&Token::In) {
            self.advance();
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
            self.match_token(&Token::DoubleSemi);
            Ok(Item::Decl(Decl::LetRec {
                visibility: Visibility::Private,
                bindings,
            }))
        }
    }

    fn parse_rec_binding(&mut self) -> ParseResult<RecBinding> {
        let name_start = self.current_span();
        let name = self.parse_possibly_qualified_name()?;

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

    fn parse_fixity_decl(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();

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
                return Err(ParseError::unexpected(
                    "infixl, infixr, or infix",
                    other.clone(),
                    self.current_span(),
                ));
            }
        };

        let precedence = match self.peek() {
            Token::Int(n) if *n >= 0 && *n <= 9 => {
                let p = *n as u8;
                self.advance();
                p
            }
            other => {
                return Err(ParseError::unexpected(
                    "precedence (0-9)",
                    other.clone(),
                    self.current_span(),
                ));
            }
        };

        let mut operators = Vec::new();
        while let Some(op_str) = self.peek().operator_symbol() {
            operators.push(op_str);
            self.advance();
        }

        if operators.is_empty() {
            return Err(self.unexpected_token("operator symbol"));
        }

        let span = start.merge(&self.current_span());

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
    // Declarations
    // ========================================================================

    fn parse_decl(&mut self) -> ParseResult<Decl> {
        match self.peek() {
            Token::Let => self.parse_let_decl(),
            Token::Type => self.parse_type_decl(),
            Token::Trait => self.parse_trait_decl(),
            Token::Impl => self.parse_instance_decl(),
            Token::Val => self.parse_val_decl(),
            _ => Err(self.unexpected_token("declaration")),
        }
    }

    fn parse_val_decl(&mut self) -> ParseResult<Decl> {
        self.consume(Token::Val)?;
        let name = self.parse_possibly_qualified_name()?;
        self.consume(Token::Colon)?;
        let type_sig = self.parse_type_expr()?;
        Ok(Decl::Val { name, type_sig })
    }

    fn parse_let_decl(&mut self) -> ParseResult<Decl> {
        self.consume(Token::Let)?;

        let name = self.parse_possibly_qualified_name()?;

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

    fn parse_type_decl(&mut self) -> ParseResult<Decl> {
        self.consume(Token::Type)?;

        let name = self.parse_upper_ident()?;

        let mut params = Vec::new();
        while let Token::Ident(_) = self.peek() {
            params.push(self.parse_ident()?);
        }

        self.consume(Token::Eq)?;

        if self.check(&Token::LBrace) {
            // Record type: type Foo = { field : Type }
            let fields = self.parse_record_fields()?;
            Ok(Decl::Record {
                visibility: Visibility::Private,
                name,
                params,
                fields,
            })
        } else if self.check(&Token::Pipe) {
            // Variant type with leading pipe: type Foo = | A | B
            let constructors = self.parse_constructors()?;
            Ok(Decl::Type {
                visibility: Visibility::Private,
                name,
                params,
                constructors,
            })
        } else if self.peek_is_upper_ident() {
            // Could be variant or type alias - disambiguate by looking for `|`
            // Parse first constructor or type application
            let first_name = self.parse_upper_ident()?;
            let mut type_args = Vec::new();

            while self.is_type_atom_start() {
                type_args.push(self.parse_type_atom()?);
            }

            if self.check(&Token::Pipe) {
                // It's a variant type: type Foo = A args | B args
                let first_constructor = Constructor {
                    name: first_name,
                    fields: type_args,
                };
                let mut constructors = vec![first_constructor];
                while self.match_token(&Token::Pipe) {
                    let ctor_name = self.parse_upper_ident()?;
                    let mut ctor_fields = Vec::new();
                    while self.is_type_atom_start() {
                        ctor_fields.push(self.parse_type_atom()?);
                    }
                    constructors.push(Constructor {
                        name: ctor_name,
                        fields: ctor_fields,
                    });
                }
                Ok(Decl::Type {
                    visibility: Visibility::Private,
                    name,
                    params,
                    constructors,
                })
            } else {
                // It's a type alias: type IntList = List Int
                let body = if type_args.is_empty() {
                    Spanned::new(TypeExprKind::Named(first_name), Span::default())
                } else {
                    let constructor =
                        Spanned::new(TypeExprKind::Named(first_name), Span::default());
                    Spanned::new(
                        TypeExprKind::App {
                            constructor: Rc::new(constructor),
                            args: type_args,
                        },
                        Span::default(),
                    )
                };
                Ok(Decl::TypeAlias {
                    visibility: Visibility::Private,
                    name,
                    params,
                    body,
                })
            }
        } else {
            // Type alias with lowercase type var or other: type Foo a = a
            let body = self.parse_type_expr()?;
            Ok(Decl::TypeAlias {
                visibility: Visibility::Private,
                name,
                params,
                body,
            })
        }
    }

    fn parse_record_fields(&mut self) -> ParseResult<Vec<RecordField>> {
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

    fn parse_record_literal_fields(&mut self) -> ParseResult<Vec<(Ident, Expr)>> {
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
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
    }

    fn parse_record_update_fields(&mut self) -> ParseResult<Vec<(Ident, Expr)>> {
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
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        Ok(updates)
    }

    fn parse_record_pattern_fields(&mut self) -> ParseResult<Vec<(Ident, Option<Pattern>)>> {
        self.consume(Token::LBrace)?;
        let mut fields = Vec::new();

        if !self.check(&Token::RBrace) {
            loop {
                let field_name = self.parse_ident()?;
                let pattern = if self.match_token(&Token::Eq) {
                    Some(self.parse_pattern()?)
                } else {
                    None
                };
                fields.push((field_name, pattern));

                if !self.match_token(&Token::Comma) {
                    break;
                }
                if self.check(&Token::RBrace) {
                    break;
                }
            }
        }

        self.consume(Token::RBrace)?;
        Ok(fields)
    }

    fn parse_constructors(&mut self) -> ParseResult<Vec<Constructor>> {
        let mut constructors = Vec::new();

        self.match_token(&Token::Pipe);

        loop {
            let name = self.parse_upper_ident()?;
            let mut fields = Vec::new();

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

    // ========================================================================
    // Typeclass declarations
    // ========================================================================

    fn parse_trait_decl(&mut self) -> ParseResult<Decl> {
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
            visibility: Visibility::Private,
            name,
            type_param,
            supertraits,
            methods,
        })
    }

    fn parse_supertrait_list(&mut self) -> ParseResult<Vec<Ident>> {
        let mut traits = vec![self.parse_upper_ident()?];
        while self.match_token(&Token::Comma) {
            traits.push(self.parse_upper_ident()?);
        }
        Ok(traits)
    }

    fn parse_instance_decl(&mut self) -> ParseResult<Decl> {
        self.consume(Token::Impl)?;

        let trait_name = self.parse_upper_ident()?;

        self.consume(Token::For)?;

        let target_type = self.parse_type_expr()?;

        let constraints = if self.match_token(&Token::Where) {
            self.parse_constraints()?
        } else {
            vec![]
        };

        self.consume(Token::Eq)?;

        let mut methods = Vec::new();
        while self.match_token(&Token::Let) {
            let method_name = self.parse_ident()?;

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

    fn parse_constraints(&mut self) -> ParseResult<Vec<Constraint>> {
        let mut constraints = Vec::new();

        loop {
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

    fn parse_type_expr(&mut self) -> ParseResult<TypeExpr> {
        self.parse_type_arrow()
    }

    fn parse_type_arrow(&mut self) -> ParseResult<TypeExpr> {
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
                    let span = start.merge(&self.current_span());
                    return Ok(Spanned::new(TypeExprKind::Tuple(vec![]), span));
                }

                let first = self.parse_type_expr()?;

                if self.match_token(&Token::Comma) {
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

    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_expr_in(ExprContext::Full)
    }

    fn parse_expr_in(&mut self, ctx: ExprContext) -> ParseResult<Expr> {
        if ctx.allows_seq() {
            self.parse_expr_seq()
        } else if ctx.allows_let() {
            self.parse_expr_let()
        } else {
            self.parse_expr_binary(0)
        }
    }

    fn parse_expr_seq(&mut self) -> ParseResult<Expr> {
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

    fn parse_expr_let(&mut self) -> ParseResult<Expr> {
        if self.check(&Token::Let) {
            let start = self.current_span();
            self.advance();

            if self.match_token(&Token::Rec) {
                return self.parse_let_rec_expr(start);
            }

            if self.check(&Token::LParen) {
                if let Some(op) = self.peek_operator_in_parens() {
                    return self.parse_operator_let_expr(start, op, None);
                }
            }

            let first_span = self.current_span();
            let first_pattern = self.parse_simple_pattern()?;

            if let PatternKind::Var(ref name) = first_pattern.node {
                if let Some(op) = self.try_peek_operator() {
                    return self.parse_operator_let_expr(
                        start,
                        op,
                        Some((name.clone(), first_span)),
                    );
                }
            }

            let mut params = Vec::new();
            while self.is_pattern_start() && !self.check(&Token::Eq) {
                params.push(self.parse_simple_pattern()?);
            }

            self.consume(Token::Eq)?;
            let value_expr = self.parse_expr()?;
            self.consume(Token::In)?;
            let body = self.parse_expr()?;

            let span = start.merge(&body.span);

            let (pattern, value) = if params.is_empty() {
                (first_pattern, value_expr)
            } else {
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

    fn parse_operator_let_expr(
        &mut self,
        start: Span,
        op: String,
        first_param: Option<(String, Span)>,
    ) -> ParseResult<Expr> {
        let mut params = Vec::new();

        if let Some((name, param_span)) = first_param {
            self.advance();
            params.push(Spanned::new(PatternKind::Var(name), param_span));
            params.push(self.parse_simple_pattern()?);
        } else {
            self.consume(Token::LParen)?;
            self.advance();
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

        let lambda_span = value_expr.span.clone();
        let lambda = Spanned::new(
            ExprKind::Lambda {
                params,
                body: Rc::new(value_expr),
            },
            lambda_span,
        );

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

    fn parse_let_rec_expr(&mut self, start: Span) -> ParseResult<Expr> {
        let mut bindings = Vec::new();

        bindings.push(self.parse_rec_binding()?);

        while self.match_token(&Token::And) {
            bindings.push(self.parse_rec_binding()?);
        }

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

    fn parse_expr_if(&mut self) -> ParseResult<Expr> {
        if self.check(&Token::If) {
            let start = self.current_span();
            self.advance();

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

    fn parse_select(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.consume(Token::Select)?;

        let mut arms = Vec::new();

        loop {
            if !self.match_token(&Token::Pipe) {
                break;
            }

            let pattern = self.parse_pattern()?;
            self.consume(Token::LArrow)?;
            let channel = self.parse_expr_app()?;
            self.consume(Token::Arrow)?;
            let body = self.parse_expr_in(ExprContext::Full)?;

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

    fn parse_lambda(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.consume(Token::Fun)?;

        let mut params = Vec::new();
        while self.is_pattern_start() && !self.check(&Token::Arrow) {
            params.push(self.parse_simple_pattern()?);
        }

        self.consume(Token::Arrow)?;
        let body = self.parse_expr_in(ExprContext::Full)?;

        let span = start.merge(&body.span);
        Ok(Spanned::new(
            ExprKind::Lambda {
                params,
                body: Rc::new(body),
            },
            span,
        ))
    }

    fn parse_match(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.consume(Token::Match)?;
        let scrutinee = self.parse_expr_binary(0)?;
        self.consume(Token::With)?;

        let mut arms = Vec::new();
        self.match_token(&Token::Pipe);

        loop {
            let pattern = self.parse_pattern()?;

            let guard = if self.match_token(&Token::If) {
                Some(self.parse_expr_in(ExprContext::BinaryOnly)?)
            } else {
                None
            };

            self.consume(Token::Arrow)?;
            let body = self.parse_expr_in(ExprContext::Full)?;

            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });

            if !self.match_token(&Token::Pipe) {
                break;
            }
        }

        let end_span = self.current_span();
        self.consume(Token::End)?;

        let span = start.merge(&end_span);
        Ok(Spanned::new(
            ExprKind::Match {
                scrutinee: Rc::new(scrutinee),
                arms,
            },
            span,
        ))
    }

    fn parse_expr_binary(&mut self, min_prec: u8) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_expr_unary()?;

        loop {
            let (op_str, _op_span) = match self.peek_operator_symbol() {
                Some(o) => o,
                None => break,
            };

            if op_str == "|>" {
                let pipe_info = self
                    .op_table
                    .get("|>")
                    .cloned()
                    .unwrap_or_else(OpInfo::default);
                if pipe_info.precedence < min_prec {
                    break;
                }
                self.advance();
                let next_min = pipe_info.precedence + 1;
                let right = self.parse_expr_binary(next_min)?;
                let span = start.merge(&right.span);
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
                self.advance();
                let next_min = pipe_info.precedence;
                let right = self.parse_expr_binary(next_min)?;
                let span = start.merge(&right.span);
                left = Spanned::new(
                    ExprKind::App {
                        func: Rc::new(left),
                        arg: Rc::new(right),
                    },
                    span,
                );
                continue;
            }

            let info = self
                .op_table
                .get(&op_str)
                .cloned()
                .unwrap_or_else(OpInfo::default);

            if info.precedence < min_prec {
                break;
            }

            self.advance();

            let next_min = match info.assoc {
                Associativity::Left => info.precedence + 1,
                Associativity::Right => info.precedence,
                Associativity::None => info.precedence + 1,
            };

            let right = self.parse_expr_binary(next_min)?;
            let span = start.merge(&right.span);

            left = self.make_binop(&op_str, left, right, span);
        }

        Ok(left)
    }

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

    fn parse_expr_unary(&mut self) -> ParseResult<Expr> {
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

    fn parse_expr_app(&mut self) -> ParseResult<Expr> {
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

    fn parse_expr_postfix(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut expr = self.parse_expr_atom()?;

        while self.check(&Token::Dot) {
            self.advance();
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

    fn parse_expr_atom(&mut self) -> ParseResult<Expr> {
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
                Ok(Spanned::new(ExprKind::Var(name), start))
            }
            Token::UpperIdent(name) => {
                self.advance();
                if self.check(&Token::LBrace) {
                    let fields = self.parse_record_literal_fields()?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(ExprKind::Record { name, fields }, span))
                } else if self.check(&Token::Dot) {
                    self.advance();
                    match self.peek().clone() {
                        Token::Ident(field) => {
                            self.advance();
                            let span = start.merge(&self.current_span());

                            if name == "Channel" {
                                match field.as_str() {
                                    "new" => return Ok(Spanned::new(ExprKind::NewChannel, span)),
                                    "send" => {
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
                    let span = start.merge(&self.current_span());
                    return Ok(Spanned::new(ExprKind::Lit(Literal::Unit), span));
                }

                let first = self.parse_expr()?;

                if self.match_token(&Token::Comma) {
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
                self.advance();
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

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        self.parse_pattern_cons()
    }

    fn parse_pattern_cons(&mut self) -> ParseResult<Pattern> {
        let start = self.current_span();
        let left = self.parse_simple_pattern()?;

        if self.match_token(&Token::Cons) {
            let right = self.parse_pattern_cons()?;
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

    fn parse_simple_pattern(&mut self) -> ParseResult<Pattern> {
        self.parse_pattern_primary(true)
    }

    fn parse_pattern_atom(&mut self) -> ParseResult<Pattern> {
        self.parse_pattern_primary(false)
    }

    fn parse_pattern_primary(&mut self, allow_constructor_args: bool) -> ParseResult<Pattern> {
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
                if self.check(&Token::LBrace) {
                    let fields = self.parse_record_pattern_fields()?;
                    let span = start.merge(&self.current_span());
                    Ok(Spanned::new(PatternKind::Record { name, fields }, span))
                } else if allow_constructor_args {
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

    fn parse_ident(&mut self) -> ParseResult<Ident> {
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

    fn parse_upper_ident(&mut self) -> ParseResult<Ident> {
        match self.peek().clone() {
            Token::UpperIdent(name) => {
                self.advance();
                Ok(name)
            }
            _ => Err(self.unexpected_token("constructor")),
        }
    }

    fn parse_possibly_qualified_name(&mut self) -> ParseResult<Ident> {
        match self.peek().clone() {
            Token::UpperIdent(module_name) => {
                let pos = self.cursor.position();
                if pos + 1 < self.cursor.position() + 10 {
                    // peek ahead
                    if matches!(self.cursor.peek_nth(1), Token::Dot) {
                        self.advance(); // consume UpperIdent
                        self.advance(); // consume Dot
                        let func_name = self.parse_ident()?;
                        Ok(format!("{}.{}", module_name, func_name))
                    } else {
                        Err(self.unexpected_token("identifier"))
                    }
                } else {
                    Err(self.unexpected_token("identifier"))
                }
            }
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
}
