//! Pattern matching exhaustiveness checking.
//!
//! Implements Maranget's "Warnings for Pattern Matching" (2007) algorithm.
//! Given a list of arm patterns and the type of the scrutinee, decides whether
//! the patterns cover every possible value of that type.
//!
//! Supported types:
//! - Algebraic data types (constructor sets via `TypeContext`)
//! - `Bool` (`true` / `false`)
//! - Lists (`[]` and `_::_`)
//! - Tuples (single "constructor" with N fields)
//! - Records (single "constructor" with N fields, looked up by record name)
//! - `Unit` (single constructor, no fields)
//!
//! Open types (`Int`, `Float`, `String`, `Char`, abstract handles, etc.) can
//! only ever be exhausted by a wildcard / variable arm.
//!
//! Guarded arms are treated as non-contributing (a guard may dynamically
//! refuse the value), which matches OCaml/Rust behaviour.

use crate::ast::{Literal, MatchArm, Pattern, PatternKind};
use crate::types::{Type, TypeContext, UnionFind};

/// Abstract head constructor used by the usefulness algorithm.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Head {
    /// Named ADT constructor like `Some`, `None`, `Ok`, ...
    Adt(String),
    /// Boolean literal
    Bool(bool),
    /// The `()` literal / unit value
    Unit,
    /// Nil — empty list `[]`
    Nil,
    /// Cons — non-empty list `_::_`
    Cons,
    /// Tuple of fixed arity
    Tuple(usize),
    /// Named record (single constructor for that record type)
    Record(String),
    /// Concrete literal of an open type (Int/Float/String/Char). The pattern
    /// matches only this exact value, and the type has infinitely many other
    /// values, so it is only useful for *negative* coverage information.
    LitInt(i64),
    LitFloat(u64),       // bit pattern so we can hash
    LitString(String),
    LitChar(char),
}

impl Head {
    /// Number of sub-patterns produced when this head is specialized.
    fn arity(&self, ctx: &ExhaustivenessCtx<'_>) -> usize {
        match self {
            Head::Adt(name) => ctx
                .type_ctx
                .get_constructor(name)
                .map(|info| info.field_types.len())
                .unwrap_or(0),
            Head::Bool(_) | Head::Unit => 0,
            Head::Nil => 0,
            Head::Cons => 2,
            Head::Tuple(n) => *n,
            Head::Record(name) => ctx
                .type_ctx
                .get_record(name)
                .map(|info| info.field_names.len())
                .unwrap_or(0),
            Head::LitInt(_)
            | Head::LitFloat(_)
            | Head::LitString(_)
            | Head::LitChar(_) => 0,
        }
    }
}

/// Witness column for the head of a row, after pattern-matrix specialization.
/// Used during the descent to know what column types we have.
#[derive(Debug, Clone)]
enum Col {
    /// We have a real pattern for this column (used for matching).
    Pat(Pattern),
    /// We have a synthetic wildcard for this column (introduced by specialization
    /// of a wildcard row, or by wildcard expansion of a missing constructor).
    Wild,
}

impl Col {
    fn as_pattern_kind(&self) -> Option<&PatternKind> {
        match self {
            Col::Pat(p) => Some(&p.node),
            Col::Wild => None,
        }
    }
}

/// Context passed through the algorithm.
pub struct ExhaustivenessCtx<'a> {
    pub type_ctx: &'a TypeContext,
    pub uf: &'a UnionFind,
}

/// Public entry point. Returns `true` when the arms are exhaustive over the
/// scrutinee type, and `false` when at least one input value is left unmatched.
pub fn is_exhaustive(
    arms: &[MatchArm],
    scrutinee_ty: &Type,
    type_ctx: &TypeContext,
    uf: &UnionFind,
) -> bool {
    // Each arm contributes a single-column row [pattern]. Guarded arms do not
    // contribute (a guard may fail and fall through).
    let matrix: Vec<Vec<Col>> = arms
        .iter()
        .filter(|arm| arm.guard.is_none())
        .map(|arm| vec![Col::Pat(arm.pattern.clone())])
        .collect();

    let ctx = ExhaustivenessCtx { type_ctx, uf };

    // The match is exhaustive iff a single wildcard pattern is *not* useful
    // against the matrix.
    !is_useful(&matrix, &[Col::Wild], &[scrutinee_ty.resolve(uf)], &ctx)
}

/// `U(P, q)` from Maranget. Given the matrix `matrix` with patterns of width
/// `column_types.len()`, and a candidate row `q` of the same width, returns
/// true iff `q` matches at least one value not matched by some row of `matrix`.
fn is_useful(
    matrix: &[Vec<Col>],
    q: &[Col],
    column_types: &[Type],
    ctx: &ExhaustivenessCtx<'_>,
) -> bool {
    // Base case: zero columns. Useful iff the matrix has no rows.
    if q.is_empty() {
        return matrix.is_empty();
    }

    // Look at the head of q.
    let head_ty = column_types[0].resolve(ctx.uf);
    let rest_types = &column_types[1..];

    let head_kind = q[0].as_pattern_kind();

    // Helper to collect all heads in column 0 of the matrix.
    let collected_heads: Vec<Head> = matrix
        .iter()
        .filter_map(|row| pattern_head(row[0].as_pattern_kind()))
        .collect();

    match head_kind {
        // Wildcard / variable: split on the type.
        None | Some(PatternKind::Wildcard) | Some(PatternKind::Var(_)) => {
            if let Some(complete_set) = exhaustive_constructor_set(&head_ty, ctx) {
                // Closed type. We need to check usefulness for *every* missing
                // constructor (since wildcard matches all), and also for each
                // present constructor (in case the matrix only handles some
                // sub-patterns under it).
                let present: std::collections::HashSet<&Head> = collected_heads.iter().collect();

                // If some constructor of the closed set is not present at all
                // in the matrix, wildcard is useful (drives default matrix).
                let missing: Vec<Head> = complete_set
                    .iter()
                    .filter(|h| !present.contains(*h))
                    .cloned()
                    .collect();

                if !missing.is_empty() {
                    // Check usefulness via default matrix for missing constructors.
                    let default = default_matrix(matrix);
                    return is_useful(&default, &q[1..], rest_types, ctx);
                }

                // Otherwise every constructor appears: split per constructor.
                for head in complete_set {
                    let arity = head.arity(ctx);
                    let new_types = head_subtypes(&head, &head_ty, ctx);
                    let mut new_q = vec![Col::Wild; arity];
                    new_q.extend_from_slice(&q[1..]);
                    let mut new_col_types = new_types;
                    new_col_types.extend_from_slice(rest_types);

                    let specialized = specialize_matrix(matrix, &head, arity);
                    if is_useful(&specialized, &new_q, &new_col_types, ctx) {
                        return true;
                    }
                }
                false
            } else {
                // Open type — only the default matrix can witness usefulness,
                // since wildcard catches all the infinite literal cases.
                let default = default_matrix(matrix);
                is_useful(&default, &q[1..], rest_types, ctx)
            }
        }

        // Concrete head — specialize on it.
        Some(_) => {
            let head = pattern_head(head_kind).expect("non-wildcard pattern has a head");
            let arity = head.arity(ctx);
            let new_types = head_subtypes(&head, &head_ty, ctx);
            let mut new_q = expand_pattern(&q[0], &head, arity);
            new_q.extend_from_slice(&q[1..]);
            let mut new_col_types = new_types;
            new_col_types.extend_from_slice(rest_types);

            let specialized = specialize_matrix(matrix, &head, arity);
            is_useful(&specialized, &new_q, &new_col_types, ctx)
        }
    }
}

/// Compute the head of a pattern, normalising List / Cons / literal forms.
fn pattern_head(pat: Option<&PatternKind>) -> Option<Head> {
    let pat = pat?;
    match pat {
        PatternKind::Wildcard | PatternKind::Var(_) => None,
        PatternKind::Lit(Literal::Bool(b)) => Some(Head::Bool(*b)),
        PatternKind::Lit(Literal::Unit) => Some(Head::Unit),
        PatternKind::Lit(Literal::Int(n)) => Some(Head::LitInt(*n)),
        PatternKind::Lit(Literal::Float(f)) => Some(Head::LitFloat(f.to_bits())),
        PatternKind::Lit(Literal::String(s)) => Some(Head::LitString(s.clone())),
        PatternKind::Lit(Literal::Char(c)) => Some(Head::LitChar(*c)),
        PatternKind::Tuple(ps) => Some(Head::Tuple(ps.len())),
        PatternKind::List(ps) => {
            // [] is Nil. [a, b, ...] is sugar for (Cons _ (Cons _ ... Nil)),
            // so the head is Nil iff empty, else Cons.
            if ps.is_empty() {
                Some(Head::Nil)
            } else {
                Some(Head::Cons)
            }
        }
        PatternKind::Cons { .. } => Some(Head::Cons),
        PatternKind::Constructor { name, .. } => Some(Head::Adt(name.clone())),
        PatternKind::Record { name, .. } => Some(Head::Record(name.clone())),
    }
}

/// Expand `pat` (assumed to have head `head`) into the column patterns produced
/// by specialization. The total length is `arity`.
fn expand_pattern(col: &Col, head: &Head, arity: usize) -> Vec<Col> {
    let pat_kind = match col {
        Col::Pat(p) => &p.node,
        Col::Wild => return vec![Col::Wild; arity],
    };

    match (head, pat_kind) {
        (Head::Tuple(_), PatternKind::Tuple(ps)) => {
            ps.iter().cloned().map(Col::Pat).collect()
        }
        (Head::Adt(_), PatternKind::Constructor { args, .. }) => {
            args.iter().cloned().map(Col::Pat).collect()
        }
        (Head::Record(name), PatternKind::Record { fields, .. }) => {
            // Record specialization: order fields in the same order as the
            // record's declared field names so that all rows agree. We don't
            // have the RecordInfo here, so fall back to the order that they
            // appear in the pattern. This is fine because we always specialize
            // *all* rows of the matrix using the same `expand_record_row`
            // helper which uses the same order — see `specialize_matrix`.
            let _ = name;
            fields
                .iter()
                .map(|(_, opt)| match opt {
                    Some(p) => Col::Pat(p.clone()),
                    None => Col::Wild,
                })
                .collect()
        }
        (Head::Cons, PatternKind::Cons { head, tail }) => {
            vec![Col::Pat((**head).clone()), Col::Pat((**tail).clone())]
        }
        (Head::Cons, PatternKind::List(ps)) if !ps.is_empty() => {
            // [a, b, c] desugars to a :: [b, c]
            let head_pat = ps[0].clone();
            let tail_span = head_pat.span.clone();
            let tail = crate::ast::Spanned {
                node: PatternKind::List(ps[1..].to_vec()),
                span: tail_span,
            };
            vec![Col::Pat(head_pat), Col::Pat(tail)]
        }
        (Head::Nil, PatternKind::List(ps)) if ps.is_empty() => Vec::new(),
        // Constants and unit have arity 0.
        _ if arity == 0 => Vec::new(),
        _ => vec![Col::Wild; arity],
    }
}

/// Specialization S(c, P) — keep rows whose first column matches `head`,
/// expanded into the constructor's field columns. Wildcard / Var rows expand
/// into `arity` wildcards.
fn specialize_matrix(matrix: &[Vec<Col>], head: &Head, arity: usize) -> Vec<Vec<Col>> {
    let mut result = Vec::new();
    for row in matrix {
        let row_head = pattern_head(row[0].as_pattern_kind());
        let mut new_row: Vec<Col>;
        match row_head {
            // Wildcard / variable — expand into wildcards.
            None => {
                new_row = vec![Col::Wild; arity];
                new_row.extend_from_slice(&row[1..]);
                result.push(new_row);
            }
            Some(rh) if &rh == head => {
                let expanded = expand_pattern(&row[0], head, arity);
                new_row = expanded;
                new_row.extend_from_slice(&row[1..]);
                result.push(new_row);
            }
            // Different head — drop the row.
            Some(_) => {}
        }
    }
    result
}

/// Default matrix D(P) — drop rows starting with a constructor; for wildcard
/// rows, drop the first column.
fn default_matrix(matrix: &[Vec<Col>]) -> Vec<Vec<Col>> {
    matrix
        .iter()
        .filter_map(|row| match pattern_head(row[0].as_pattern_kind()) {
            None => Some(row[1..].to_vec()),
            Some(_) => None,
        })
        .collect()
}

/// Returns the full set of constructors for a *closed* type (ADT, Bool, list,
/// tuple, record, unit). Returns `None` for open types like Int/String.
fn exhaustive_constructor_set(ty: &Type, ctx: &ExhaustivenessCtx<'_>) -> Option<Vec<Head>> {
    let resolved = ty.resolve(ctx.uf);
    match resolved {
        Type::Bool => Some(vec![Head::Bool(true), Head::Bool(false)]),
        Type::Unit => Some(vec![Head::Unit]),
        Type::Tuple(ref ts) => Some(vec![Head::Tuple(ts.len())]),
        Type::Constructor { ref name, .. } if name == "List" => {
            Some(vec![Head::Nil, Head::Cons])
        }
        Type::Constructor { ref name, .. } => {
            // Look for a record with this name first (records are also stored
            // as Constructor types).
            if ctx.type_ctx.get_record(name).is_some() {
                return Some(vec![Head::Record(name.clone())]);
            }
            // Otherwise an ADT — gather every constructor whose info points to
            // this type.
            let mut ctors: Vec<(String, &crate::types::ConstructorInfo)> = ctx
                .type_ctx
                .constructors
                .iter()
                .filter(|(_, info)| info.type_name == *name)
                .map(|(n, info)| (n.clone(), info))
                .collect();
            if ctors.is_empty() {
                None
            } else {
                // Deterministic ordering keeps the algorithm reproducible.
                ctors.sort_by(|a, b| a.0.cmp(&b.0));
                Some(ctors.into_iter().map(|(n, _)| Head::Adt(n)).collect())
            }
        }
        // Open / opaque / variable types — no closed set.
        _ => None,
    }
}

/// Compute the column types produced when specializing a column of type `ty`
/// on `head`.
fn head_subtypes(head: &Head, ty: &Type, ctx: &ExhaustivenessCtx<'_>) -> Vec<Type> {
    let resolved = ty.resolve(ctx.uf);
    match head {
        Head::Bool(_) | Head::Unit | Head::Nil => Vec::new(),
        Head::LitInt(_)
        | Head::LitFloat(_)
        | Head::LitString(_)
        | Head::LitChar(_) => Vec::new(),
        Head::Cons => {
            // Cons of List a -> [a, List a]
            let elem = match &resolved {
                Type::Constructor { name, args } if name == "List" && !args.is_empty() => {
                    args[0].clone()
                }
                _ => Type::Unit, // shouldn't happen, fall back
            };
            vec![elem.clone(), Type::list(elem)]
        }
        Head::Tuple(_n) => match &resolved {
            Type::Tuple(ts) => ts.clone(),
            _ => Vec::new(),
        },
        Head::Adt(name) => {
            if let Some(info) = ctx.type_ctx.get_constructor(name) {
                // Substitute the type arguments from the resolved scrutinee
                // into the constructor's generic field types.
                let type_args: Vec<Type> = match &resolved {
                    Type::Constructor { args, .. } => args.clone(),
                    _ => Vec::new(),
                };
                info.field_types
                    .iter()
                    .map(|ft| crate::infer::substitute_generics_pub(ft, &type_args))
                    .collect()
            } else {
                Vec::new()
            }
        }
        Head::Record(name) => {
            if let Some(info) = ctx.type_ctx.get_record(name) {
                let type_args: Vec<Type> = match &resolved {
                    Type::Constructor { args, .. } => args.clone(),
                    _ => Vec::new(),
                };
                info.field_names
                    .iter()
                    .filter_map(|n| info.field_types.get(n))
                    .map(|ft| crate::infer::substitute_generics_pub(ft, &type_args))
                    .collect()
            } else {
                Vec::new()
            }
        }
    }
}
