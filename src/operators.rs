//! Operator precedence and associativity table for user-defined operators.

use std::collections::HashMap;

use crate::ast::Span;
// Re-export Associativity from ast to avoid duplicate definitions
pub use crate::ast::Associativity;
use crate::errors::Warning;

/// Information about an operator's fixity
#[derive(Debug, Clone)]
pub struct OpInfo {
    /// Precedence level (0-9, higher binds tighter)
    pub precedence: u8,
    /// Associativity
    pub assoc: Associativity,
    /// Whether this is a built-in operator
    pub is_builtin: bool,
}

impl Default for OpInfo {
    fn default() -> Self {
        // Default for unknown operators: highest precedence, left-associative
        OpInfo {
            precedence: 9,
            assoc: Associativity::Left,
            is_builtin: false,
        }
    }
}

/// Table of operator precedences and associativities
#[derive(Debug, Clone)]
pub struct OperatorTable {
    operators: HashMap<String, OpInfo>,
}

impl Default for OperatorTable {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorTable {
    /// Create a new operator table with built-in operators registered
    pub fn new() -> Self {
        let mut table = Self {
            operators: HashMap::new(),
        };
        table.register_builtins();
        table
    }

    /// Register built-in operators with their standard fixities
    fn register_builtins(&mut self) {
        use Associativity::*;

        // Precedence levels match current parser hierarchy:
        // Lower number = looser binding (evaluated later)
        //
        // Level 1: |> <| (pipe operators - but these desugar to App, not BinOp)
        // Level 2: || (boolean or)
        // Level 3: && (boolean and)
        // Level 4: == != < > <= >= (comparison, non-associative)
        // Level 5: :: ++ (cons, concat - right-associative)
        // Level 6: + - (additive)
        // Level 7: * / % (multiplicative)
        // Level 8: >> << (function composition)

        // Boolean
        self.operators
            .insert("||".into(), OpInfo { precedence: 2, assoc: Left, is_builtin: true });
        self.operators
            .insert("&&".into(), OpInfo { precedence: 3, assoc: Left, is_builtin: true });

        // Comparison (non-associative)
        self.operators
            .insert("==".into(), OpInfo { precedence: 4, assoc: None, is_builtin: true });
        self.operators
            .insert("!=".into(), OpInfo { precedence: 4, assoc: None, is_builtin: true });
        self.operators
            .insert("<".into(), OpInfo { precedence: 4, assoc: None, is_builtin: true });
        self.operators
            .insert(">".into(), OpInfo { precedence: 4, assoc: None, is_builtin: true });
        self.operators
            .insert("<=".into(), OpInfo { precedence: 4, assoc: None, is_builtin: true });
        self.operators
            .insert(">=".into(), OpInfo { precedence: 4, assoc: None, is_builtin: true });

        // List (right-associative)
        self.operators
            .insert("::".into(), OpInfo { precedence: 5, assoc: Right, is_builtin: true });
        self.operators
            .insert("++".into(), OpInfo { precedence: 5, assoc: Right, is_builtin: true });

        // Arithmetic
        self.operators
            .insert("+".into(), OpInfo { precedence: 6, assoc: Left, is_builtin: true });
        self.operators
            .insert("-".into(), OpInfo { precedence: 6, assoc: Left, is_builtin: true });
        self.operators
            .insert("*".into(), OpInfo { precedence: 7, assoc: Left, is_builtin: true });
        self.operators
            .insert("/".into(), OpInfo { precedence: 7, assoc: Left, is_builtin: true });
        self.operators
            .insert("%".into(), OpInfo { precedence: 7, assoc: Left, is_builtin: true });

        // Function composition
        self.operators
            .insert(">>".into(), OpInfo { precedence: 8, assoc: Left, is_builtin: true });
        self.operators
            .insert("<<".into(), OpInfo { precedence: 8, assoc: Left, is_builtin: true });

        // Pipe operators (these desugar to App in parser, but register for completeness)
        self.operators
            .insert("|>".into(), OpInfo { precedence: 1, assoc: Left, is_builtin: true });
        self.operators
            .insert("<|".into(), OpInfo { precedence: 1, assoc: Right, is_builtin: true });
    }

    /// Register a user-defined operator. Returns a warning if shadowing a built-in.
    pub fn register(&mut self, op: String, info: OpInfo, span: Span) -> Option<Warning> {
        let warning = if let Some(existing) = self.operators.get(&op) {
            if existing.is_builtin {
                Some(Warning::ShadowingBuiltinOperator { op: op.clone(), span })
            } else {
                None
            }
        } else {
            None
        };

        self.operators.insert(op, info);
        warning
    }

    /// Look up an operator's info
    pub fn get(&self, op: &str) -> Option<&OpInfo> {
        self.operators.get(op)
    }

    /// Check if an operator is registered
    pub fn contains(&self, op: &str) -> bool {
        self.operators.contains_key(op)
    }
}

/// Characters that can appear in operator symbols
pub fn is_operator_char(c: char) -> bool {
    matches!(c, '!' | '$' | '%' | '&' | '*' | '+' | '-' | '/' | '<' | '=' | '>' | '?' | '@' | '^' | '|' | '~')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_operators() {
        let table = OperatorTable::new();

        // Check some built-ins exist
        assert!(table.get("+").is_some());
        assert!(table.get("==").is_some());
        assert!(table.get("::").is_some());

        // Check precedences
        assert!(table.get("+").unwrap().precedence < table.get("*").unwrap().precedence);
        assert!(table.get("||").unwrap().precedence < table.get("&&").unwrap().precedence);

        // Check associativities
        assert_eq!(table.get("+").unwrap().assoc, Associativity::Left);
        assert_eq!(table.get("::").unwrap().assoc, Associativity::Right);
        assert_eq!(table.get("==").unwrap().assoc, Associativity::None);
    }

    #[test]
    fn test_user_operator() {
        let mut table = OperatorTable::new();

        // Register a new operator
        let warning = table.register(
            "<|>".into(),
            OpInfo {
                precedence: 4,
                assoc: Associativity::Left,
                is_builtin: false,
            },
            Span::new(0, 3),
        );

        assert!(warning.is_none());
        assert!(table.get("<|>").is_some());
        assert_eq!(table.get("<|>").unwrap().precedence, 4);
    }

    #[test]
    fn test_shadowing_builtin() {
        let mut table = OperatorTable::new();

        // Shadow a built-in
        let warning = table.register(
            "+".into(),
            OpInfo {
                precedence: 9,
                assoc: Associativity::Right,
                is_builtin: false,
            },
            Span::new(0, 1),
        );

        assert!(warning.is_some());
        // Shadowing should still work
        assert_eq!(table.get("+").unwrap().precedence, 9);
        assert_eq!(table.get("+").unwrap().assoc, Associativity::Right);
    }

    #[test]
    fn test_operator_chars() {
        assert!(is_operator_char('+'));
        assert!(is_operator_char('<'));
        assert!(is_operator_char('|'));
        assert!(is_operator_char('!'));

        assert!(!is_operator_char('a'));
        assert!(!is_operator_char(';'));
        assert!(!is_operator_char(','));
        assert!(!is_operator_char('('));
    }
}
