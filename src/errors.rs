//! Error formatting infrastructure for Elm-inspired error messages.
//!
//! This module provides:
//! - ANSI color support with TTY auto-detection
//! - Levenshtein distance for "did you mean?" suggestions
//! - Source snippet formatting with carets/underlines
//! - First-person error message templates
//! - Compiler warnings (non-fatal issues)

use crate::ast::{SourceMap, Span};

// ============================================================================
// Warnings
// ============================================================================

/// Compiler warnings (non-fatal issues)
#[derive(Debug, Clone)]
pub enum Warning {
    /// User-defined operator shadows a built-in operator
    ShadowingBuiltinOperator { op: String, span: Span },
}

impl Warning {
    /// Format the warning for display
    pub fn format(
        &self,
        source_map: &SourceMap,
        filename: Option<&str>,
        colors: &Colors,
    ) -> String {
        match self {
            Warning::ShadowingBuiltinOperator { op, span } => {
                let mut out = String::new();

                out.push_str(&format!(
                    "{}-- WARNING {}{}",
                    colors.yellow(),
                    "-".repeat(55),
                    colors.reset()
                ));
                out.push('\n');
                out.push('\n');

                let pos = source_map.position(span.start);
                let file = filename.unwrap_or("<input>");
                out.push_str(&format!(
                    "{}{}:{}{}\n\n",
                    colors.bold(),
                    file,
                    pos,
                    colors.reset()
                ));

                out.push_str(&format!(
                    "You are shadowing the built-in operator `{}{}{}`.\n\
                     This may cause confusion, as the operator will no longer have\n\
                     its default behavior in this scope.",
                    colors.bold(),
                    op,
                    colors.reset()
                ));
                out.push('\n');
                out.push('\n');
                out.push_str(&format_snippet(source_map, span, colors));
                out.push('\n');

                out
            }
        }
    }
}

/// ANSI color codes for terminal output
#[derive(Debug, Clone)]
pub struct Colors {
    pub enabled: bool,
}

impl Colors {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn red(&self) -> &'static str {
        if self.enabled {
            "\x1b[31m"
        } else {
            ""
        }
    }

    pub fn cyan(&self) -> &'static str {
        if self.enabled {
            "\x1b[36m"
        } else {
            ""
        }
    }

    pub fn yellow(&self) -> &'static str {
        if self.enabled {
            "\x1b[33m"
        } else {
            ""
        }
    }

    pub fn bold(&self) -> &'static str {
        if self.enabled {
            "\x1b[1m"
        } else {
            ""
        }
    }

    pub fn dim(&self) -> &'static str {
        if self.enabled {
            "\x1b[2m"
        } else {
            ""
        }
    }

    pub fn reset(&self) -> &'static str {
        if self.enabled {
            "\x1b[0m"
        } else {
            ""
        }
    }
}

impl Default for Colors {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Configuration for error display
#[derive(Debug, Clone)]
pub struct ErrorConfig {
    pub colors: Colors,
    pub filename: Option<String>,
}

impl ErrorConfig {
    pub fn new(use_color: bool) -> Self {
        Self {
            colors: Colors::new(use_color),
            filename: None,
        }
    }

    pub fn with_filename(mut self, name: impl Into<String>) -> Self {
        self.filename = Some(name.into());
        self
    }
}

impl Default for ErrorConfig {
    fn default() -> Self {
        Self::new(false)
    }
}

// ============================================================================
// Levenshtein Distance for "Did you mean?" suggestions
// ============================================================================

/// Compute the Levenshtein edit distance between two strings.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();

    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];

    for i in 0..=a.len() {
        dp[i][0] = i;
    }
    for j in 0..=b.len() {
        dp[0][j] = j;
    }

    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[a.len()][b.len()]
}

/// Find similar names from a list of candidates.
///
/// Returns up to 3 suggestions within the given max edit distance,
/// sorted by distance (closest first).
pub fn find_similar<'a>(
    name: &str,
    candidates: impl IntoIterator<Item = &'a str>,
    max_distance: usize,
) -> Vec<String> {
    let mut suggestions: Vec<(String, usize)> = candidates
        .into_iter()
        .filter_map(|c| {
            let dist = levenshtein_distance(name, c);
            // Only suggest if within max_distance and not identical
            if dist > 0 && dist <= max_distance {
                Some((c.to_string(), dist))
            } else {
                None
            }
        })
        .collect();

    // Sort by distance, then alphabetically for ties
    suggestions.sort_by(|(a, da), (b, db)| da.cmp(db).then_with(|| a.cmp(b)));

    // Return top 3
    suggestions.into_iter().map(|(s, _)| s).take(3).collect()
}

// ============================================================================
// Source Snippet Formatting
// ============================================================================

/// Format a source code snippet with line number and caret/underline.
///
/// For single-line spans:
/// ```text
/// 12 | let x = add 5 "hello"
///              ^^^^^^^^^^^^^
/// ```
///
/// For multi-line spans:
/// ```text
/// 12 | let x = match foo with
///              ^^^^^^^^^^^^^^
/// 13 | | Some y -> y
/// 14 | | None -> 0
///      ^
/// ```
pub fn format_snippet(source_map: &SourceMap, span: &Span, colors: &Colors) -> String {
    let loc = source_map.locate(span);

    // For single-line spans, use simple format
    if loc.start.line == loc.end.line {
        return format_single_line_snippet(source_map, span, colors);
    }

    // Multi-line span: show all affected lines
    let mut out = String::new();
    let max_line_num = loc.end.line;
    let gutter_width = format!("{}", max_line_num).len();

    for line_num in loc.start.line..=loc.end.line {
        let line_text = source_map.line(line_num).unwrap_or("");
        let gutter = format!("{:>width$}", line_num, width = gutter_width);

        // Source line with line number
        out.push_str(&format!(
            "{}{} |{} {}\n",
            colors.cyan(),
            gutter,
            colors.reset(),
            line_text
        ));

        // Marker line
        let marker_padding = " ".repeat(gutter_width + 3);

        if line_num == loc.start.line {
            // First line: caret at start column, underline to end of line content
            let spaces = " ".repeat(loc.start.column.saturating_sub(1));
            let underline_len = line_text
                .chars()
                .count()
                .saturating_sub(loc.start.column.saturating_sub(1))
                .max(1);
            let marks = "^".repeat(underline_len);
            out.push_str(&format!(
                "{}{}{}{}{}\n",
                marker_padding,
                spaces,
                colors.red(),
                marks,
                colors.reset()
            ));
        } else if line_num == loc.end.line {
            // Last line: caret at column 1 pointing to where error ends
            let marks = "^".repeat(loc.end.column.max(1));
            out.push_str(&format!(
                "{}{}{}{}\n",
                marker_padding,
                colors.red(),
                marks,
                colors.reset()
            ));
        } else {
            // Middle lines: vertical bar continuation marker
            out.push_str(&format!(
                "{}{}|{}\n",
                marker_padding,
                colors.red(),
                colors.reset()
            ));
        }
    }

    out.trim_end().to_string()
}

/// Format a single-line source snippet (internal helper)
fn format_single_line_snippet(source_map: &SourceMap, span: &Span, colors: &Colors) -> String {
    let loc = source_map.locate(span);
    let line_text = source_map.line(loc.start.line).unwrap_or("");
    let line_num = loc.start.line;

    let mut out = String::new();

    // Calculate gutter width for alignment
    let gutter = format!("{}", line_num);
    let gutter_width = gutter.len();

    // The source line with line number
    out.push_str(&format!(
        "{}{} |{} {}\n",
        colors.cyan(),
        gutter,
        colors.reset(),
        line_text
    ));

    // The caret/underline line
    // Pad to align with content after "NN | "
    let padding = " ".repeat(gutter_width + 3 + loc.start.column.saturating_sub(1));

    let len = (loc.end.column.saturating_sub(loc.start.column)).max(1);
    let underline = "^".repeat(len);

    out.push_str(&format!(
        "{}{}{}{}",
        padding,
        colors.red(),
        underline,
        colors.reset()
    ));

    out
}

/// Format the "did you mean?" hint.
pub fn format_suggestions(suggestions: &[String], colors: &Colors) -> String {
    if suggestions.is_empty() {
        return String::new();
    }

    if suggestions.len() == 1 {
        format!(
            "\n\nDid you mean {}{}{}?",
            colors.bold(),
            suggestions[0],
            colors.reset()
        )
    } else {
        let formatted: Vec<String> = suggestions
            .iter()
            .map(|s| format!("{}{}{}", colors.bold(), s, colors.reset()))
            .collect();
        format!("\n\nDid you mean one of: {}?", formatted.join(", "))
    }
}

/// Format the error header line.
///
/// Example: "-- TYPE ERROR ----------------------------------------------------------"
pub fn format_header(error_kind: &str, colors: &Colors) -> String {
    let dashes = "-".repeat(60 - error_kind.len() - 4);
    format!(
        "{}-- {} {}{}",
        colors.cyan(),
        error_kind,
        dashes,
        colors.reset()
    )
}

/// Format the location line.
///
/// Example: "examples/test.gn:12:15"
pub fn format_location(
    filename: Option<&str>,
    span: &Span,
    source_map: &SourceMap,
    colors: &Colors,
) -> String {
    let pos = source_map.position(span.start);
    let file = filename.unwrap_or("<input>");
    format!("{}{}:{}{}", colors.bold(), file, pos, colors.reset())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_one_char_diff() {
        assert_eq!(levenshtein_distance("hello", "hallo"), 1); // substitution
        assert_eq!(levenshtein_distance("print", "prit"), 1); // deletion of 'n'
        assert_eq!(levenshtein_distance("print", "priint"), 1); // insertion of 'i'
        assert_eq!(levenshtein_distance("print", "prnt"), 1); // deletion of 'i'
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        assert_eq!(levenshtein_distance("", ""), 0);
    }

    #[test]
    fn test_find_similar_typo() {
        let candidates = vec!["print", "println", "printf", "sprint", "map", "filter"];
        let suggestions = find_similar("prnt", candidates.into_iter(), 2);
        // "print" is distance 1, "printf" and "sprint" are distance 2
        assert!(suggestions.contains(&"print".to_string()));
        assert_eq!(suggestions[0], "print"); // closest should be first
    }

    #[test]
    fn test_find_similar_multiple() {
        let candidates = vec!["foo", "fop", "for", "bar", "baz"];
        let suggestions = find_similar("fo", candidates.into_iter(), 2);
        // "foo" and "for" are both distance 1, "fop" is distance 1
        assert!(suggestions.contains(&"foo".to_string()));
        assert!(suggestions.contains(&"for".to_string()));
    }

    #[test]
    fn test_find_similar_no_match() {
        let candidates = vec!["apple", "banana", "cherry"];
        let suggestions = find_similar("xyz", candidates.into_iter(), 2);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_find_similar_excludes_identical() {
        let candidates = vec!["print", "println"];
        let suggestions = find_similar("print", candidates.into_iter(), 2);
        // Should not suggest "print" itself
        assert!(!suggestions.contains(&"print".to_string()));
    }

    #[test]
    fn test_format_header() {
        let colors = Colors::new(false);
        let header = format_header("TYPE ERROR", &colors);
        assert!(header.contains("TYPE ERROR"));
        assert!(header.contains("--"));
    }

    #[test]
    fn test_format_suggestions_single() {
        let colors = Colors::new(false);
        let result = format_suggestions(&["print".to_string()], &colors);
        assert!(result.contains("Did you mean"));
        assert!(result.contains("print"));
    }

    #[test]
    fn test_format_suggestions_multiple() {
        let colors = Colors::new(false);
        let result = format_suggestions(&["print".to_string(), "println".to_string()], &colors);
        assert!(result.contains("one of"));
        assert!(result.contains("print"));
        assert!(result.contains("println"));
    }

    #[test]
    fn test_format_suggestions_empty() {
        let colors = Colors::new(false);
        let result = format_suggestions(&[], &colors);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_snippet_single_line() {
        use crate::ast::{SourceMap, Span};
        let source = "let x = 1 + \"hello\"";
        let map = SourceMap::new(source);
        let span = Span::new(8, 19); // "1 + \"hello\""
        let colors = Colors::new(false);
        let result = format_snippet(&map, &span, &colors);
        assert!(result.contains("let x = 1 + \"hello\""));
        assert!(result.contains("^^^")); // Should have carets
    }

    #[test]
    fn test_format_snippet_multiline() {
        use crate::ast::{SourceMap, Span};
        let source = "let x = match foo with\n| Some y -> y\n| None -> 0";
        let map = SourceMap::new(source);
        // Span covering "match foo with\n| Some y -> y\n| None"
        let span = Span::new(8, 48);
        let colors = Colors::new(false);
        let result = format_snippet(&map, &span, &colors);
        // Should show all three lines
        assert!(result.contains("match foo"));
        assert!(result.contains("Some y"));
        assert!(result.contains("None"));
        // Should have line numbers
        assert!(result.contains("1 |"));
        assert!(result.contains("2 |"));
        assert!(result.contains("3 |"));
    }
}
