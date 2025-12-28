//! Module system: file discovery, dependency resolution, and cycle detection

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::ast::{ImportSpec, Item, Program};
use crate::lexer::LexError;
use crate::parser::ParseError;

/// Unique identifier for a module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(u32);

impl ModuleId {
    pub fn new(id: u32) -> Self {
        ModuleId(id)
    }
}

/// Information about a single module
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub id: ModuleId,
    /// Module name (e.g., "List" or "Collections.HashMap")
    pub name: String,
    /// File path this module was loaded from
    pub path: PathBuf,
    /// Import specifications from this module
    pub imports: Vec<ImportSpec>,
    /// Names exported by this module (pub items)
    pub exports: HashSet<String>,
    /// The parsed program for this module
    pub program: Program,
}

/// Dependency graph for modules
#[derive(Debug, Default)]
pub struct ModuleGraph {
    /// All modules by ID
    modules: HashMap<ModuleId, ModuleInfo>,
    /// Module lookup by name
    by_name: HashMap<String, ModuleId>,
    /// Module lookup by path
    by_path: HashMap<PathBuf, ModuleId>,
    /// Next module ID
    next_id: u32,
    /// Dependency edges: module -> modules it imports
    dependencies: HashMap<ModuleId, Vec<ModuleId>>,
}

impl ModuleGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a module to the graph
    pub fn add_module(&mut self, info: ModuleInfo) -> ModuleId {
        let id = info.id;
        self.by_name.insert(info.name.clone(), id);
        self.by_path.insert(info.path.clone(), id);
        self.modules.insert(id, info);
        id
    }

    /// Get a module by ID
    pub fn get(&self, id: ModuleId) -> Option<&ModuleInfo> {
        self.modules.get(&id)
    }

    /// Get a module by name
    pub fn get_by_name(&self, name: &str) -> Option<&ModuleInfo> {
        self.by_name.get(name).and_then(|id| self.modules.get(id))
    }

    /// Get a module ID by name
    pub fn id_by_name(&self, name: &str) -> Option<ModuleId> {
        self.by_name.get(name).copied()
    }

    /// Allocate a new module ID
    pub fn next_id(&mut self) -> ModuleId {
        let id = ModuleId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a dependency edge
    pub fn add_dependency(&mut self, from: ModuleId, to: ModuleId) {
        self.dependencies.entry(from).or_default().push(to);
    }

    /// Get all modules
    pub fn modules(&self) -> impl Iterator<Item = &ModuleInfo> {
        self.modules.values()
    }
}

/// Error during module resolution
#[derive(Debug)]
pub enum ModuleError {
    /// Module file not found
    NotFound { module_path: String, search_paths: Vec<PathBuf> },
    /// Circular dependency detected
    CircularDependency { cycle: Vec<String> },
    /// IO error reading file
    IoError { path: PathBuf, message: String },
    /// Lex error in module (preserves error and source for formatting)
    LexError { path: PathBuf, error: LexError, source: String },
    /// Parse error in module (preserves error and source for formatting)
    ParseError { path: PathBuf, error: ParseError, source: String },
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::NotFound { module_path, search_paths } => {
                write!(f, "module '{}' not found in search paths: {:?}", module_path, search_paths)
            }
            ModuleError::CircularDependency { cycle } => {
                write!(f, "circular dependency detected: {}", cycle.join(" -> "))
            }
            ModuleError::IoError { path, message } => {
                write!(f, "error reading {}: {}", path.display(), message)
            }
            ModuleError::LexError { path, error, .. } => {
                write!(f, "lex error in {}: {}", path.display(), error)
            }
            ModuleError::ParseError { path, error, .. } => {
                write!(f, "parse error in {}: {}", path.display(), error)
            }
        }
    }
}

impl std::error::Error for ModuleError {}

/// Resolves module paths to files and builds the dependency graph
pub struct ModuleResolver {
    /// Paths to search for modules
    search_paths: Vec<PathBuf>,
    /// The dependency graph being built
    pub graph: ModuleGraph,
    /// Topologically sorted load order
    pub load_order: Vec<ModuleId>,
}

impl ModuleResolver {
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            graph: ModuleGraph::new(),
            load_order: Vec::new(),
        }
    }

    /// Resolve a module path to a file path
    ///
    /// Given "List", looks for:
    ///   - <search_path>/list.gn
    ///   - <search_path>/list/mod.gn
    ///
    /// Given "Collections/HashMap", looks for:
    ///   - <search_path>/collections/hash_map.gn
    ///   - <search_path>/collections/hash_map/mod.gn
    pub fn resolve_path(&self, module_path: &str) -> Option<PathBuf> {
        // Convert module path to file path
        // "List" -> "list"
        // "Collections/HashMap" -> "collections/hash_map"
        let file_path = module_path_to_file_path(module_path);

        for search_path in &self.search_paths {
            // Try direct file: list.gn
            let direct = search_path.join(format!("{}.gn", file_path));
            if direct.exists() {
                return Some(direct);
            }

            // Try directory module: list/mod.gn
            let dir_mod = search_path.join(&file_path).join("mod.gn");
            if dir_mod.exists() {
                return Some(dir_mod);
            }
        }

        None
    }

    /// Load a module and all its dependencies
    /// Returns the module ID for the entry module
    pub fn load_module(&mut self, entry_path: &Path) -> Result<ModuleId, ModuleError> {
        // Use a worklist algorithm with cycle detection
        let mut to_load: Vec<PathBuf> = vec![entry_path.to_path_buf()];
        let mut visiting: HashSet<PathBuf> = HashSet::new();
        let mut visited: HashSet<PathBuf> = HashSet::new();
        let mut path_stack: Vec<PathBuf> = Vec::new();

        while let Some(path) = to_load.pop() {
            if visited.contains(&path) {
                continue;
            }

            if visiting.contains(&path) {
                // Found a cycle - build cycle path from stack
                let cycle: Vec<String> = path_stack
                    .iter()
                    .map(|p| path_to_module_name(p))
                    .collect();
                return Err(ModuleError::CircularDependency { cycle });
            }

            visiting.insert(path.clone());
            path_stack.push(path.clone());

            // Load and parse the module
            let source = std::fs::read_to_string(&path)
                .map_err(|e| ModuleError::IoError {
                    path: path.clone(),
                    message: e.to_string(),
                })?;

            let tokens = crate::lexer::Lexer::new(&source)
                .tokenize()
                .map_err(|e| ModuleError::LexError {
                    path: path.clone(),
                    error: e,
                    source: source.clone(),
                })?;

            let program = crate::parser::Parser::new(tokens)
                .parse_program()
                .map_err(|e| ModuleError::ParseError {
                    path: path.clone(),
                    error: e,
                    source: source.clone(),
                })?;

            // Extract imports and exports
            let (imports, exports) = extract_module_info(&program);

            // Queue dependencies
            for import in &imports {
                if let Some(dep_path) = self.resolve_path(&import.module_path) {
                    if !visited.contains(&dep_path) {
                        to_load.push(dep_path);
                    }
                } else {
                    return Err(ModuleError::NotFound {
                        module_path: import.module_path.clone(),
                        search_paths: self.search_paths.clone(),
                    });
                }
            }

            // Create module info
            let id = self.graph.next_id();
            let name = path_to_module_name(&path);
            let info = ModuleInfo {
                id,
                name,
                path: path.clone(),
                imports,
                exports,
                program,
            };
            self.graph.add_module(info);

            visiting.remove(&path);
            visited.insert(path.clone());
            path_stack.pop();
            self.load_order.push(id);
        }

        // Reverse load_order so dependencies come before dependents
        // (the worklist algorithm naturally produces dependents-first order)
        self.load_order.reverse();

        // Return the ID of the entry module (looked up by path)
        let canonical_entry = entry_path.canonicalize()
            .unwrap_or_else(|_| entry_path.to_path_buf());
        self.graph.by_path.get(&canonical_entry)
            .or_else(|| self.graph.by_path.get(entry_path))
            .copied()
            .ok_or_else(|| ModuleError::NotFound {
                module_path: entry_path.display().to_string(),
                search_paths: self.search_paths.clone(),
            })
    }

    /// Topologically sort modules by dependencies
    /// Returns modules in load order (dependencies first)
    pub fn topological_sort(&self) -> Result<Vec<ModuleId>, ModuleError> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        let mut stack = Vec::new();

        for &module_id in self.graph.modules.keys() {
            if !visited.contains(&module_id) {
                self.topo_visit(module_id, &mut visited, &mut visiting, &mut stack, &mut result)?;
            }
        }

        Ok(result)
    }

    fn topo_visit(
        &self,
        id: ModuleId,
        visited: &mut HashSet<ModuleId>,
        visiting: &mut HashSet<ModuleId>,
        stack: &mut Vec<ModuleId>,
        result: &mut Vec<ModuleId>,
    ) -> Result<(), ModuleError> {
        if visited.contains(&id) {
            return Ok(());
        }

        if visiting.contains(&id) {
            // Cycle detected
            let cycle: Vec<String> = stack
                .iter()
                .filter_map(|&mid| self.graph.get(mid).map(|m| m.name.clone()))
                .collect();
            return Err(ModuleError::CircularDependency { cycle });
        }

        visiting.insert(id);
        stack.push(id);

        if let Some(deps) = self.graph.dependencies.get(&id) {
            for &dep_id in deps {
                self.topo_visit(dep_id, visited, visiting, stack, result)?;
            }
        }

        visiting.remove(&id);
        stack.pop();
        visited.insert(id);
        result.push(id);

        Ok(())
    }
}

/// Convert a module path like "Collections/HashMap" to a file path like "collections/hash_map"
fn module_path_to_file_path(module_path: &str) -> String {
    module_path
        .split('/')
        .map(to_snake_case)
        .collect::<Vec<_>>()
        .join("/")
}

/// Convert PascalCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

/// Convert a file path to a module name
/// "src/list.gn" -> "List"
/// "src/collections/hash_map.gn" -> "Collections/HashMap"
fn path_to_module_name(path: &Path) -> String {
    let stem = path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown");

    // Handle mod.gn - use parent directory name
    if stem == "mod" {
        if let Some(parent) = path.parent() {
            if let Some(parent_name) = parent.file_name().and_then(|s| s.to_str()) {
                return to_pascal_case(parent_name);
            }
        }
    }

    to_pascal_case(stem)
}

/// Convert snake_case to PascalCase
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        })
        .collect()
}

/// Extract imports and exports from a parsed program
/// If no explicit export list, all declarations are public.
/// If explicit export list, only those items are public.
fn extract_module_info(program: &Program) -> (Vec<ImportSpec>, HashSet<String>) {
    let mut imports = Vec::new();
    let mut exports = HashSet::new();

    // Check if there's an explicit export list
    let has_explicit_exports = program.exports.is_some();

    if let Some(ref export_decl) = program.exports {
        // Use the explicit export list
        use crate::ast::ExportItem;
        for item in &export_decl.items {
            match &item.node {
                ExportItem::Value(name) => {
                    exports.insert(name.clone());
                }
                ExportItem::TypeOnly(name) => {
                    exports.insert(name.clone());
                }
                ExportItem::TypeAll(name) => {
                    exports.insert(name.clone());
                    // Note: constructors would need to be looked up from declarations
                }
                ExportItem::TypeSome(name, ctors) => {
                    exports.insert(name.clone());
                    for ctor in ctors {
                        exports.insert(ctor.clone());
                    }
                }
            }
        }
    }

    for item in &program.items {
        match item {
            Item::Import(spanned) => {
                imports.push(spanned.node.clone());
            }
            Item::Decl(decl) => {
                // If no explicit export list, export all declarations
                if !has_explicit_exports {
                    use crate::ast::Decl;
                    match decl {
                        Decl::Let { name, .. } => {
                            exports.insert(name.clone());
                        }
                        Decl::LetRec { bindings, .. } => {
                            for binding in bindings {
                                exports.insert(binding.name.node.clone());
                            }
                        }
                        Decl::Type { name, constructors, .. } => {
                            exports.insert(name.clone());
                            // Also export constructors
                            for ctor in constructors {
                                exports.insert(ctor.name.clone());
                            }
                        }
                        Decl::TypeAlias { name, .. } => {
                            exports.insert(name.clone());
                        }
                        Decl::Record { name, .. } => {
                            exports.insert(name.clone());
                        }
                        Decl::Trait { name, methods, .. } => {
                            exports.insert(name.clone());
                            // Also export trait methods
                            for method in methods {
                                exports.insert(method.name.clone());
                            }
                        }
                        Decl::OperatorDef { op, .. } => {
                            exports.insert(op.clone());
                        }
                        Decl::EffectDecl { name, operations, .. } => {
                            exports.insert(name.clone());
                            // Also export effect operations
                            for op in operations {
                                exports.insert(op.name.clone());
                            }
                        }
                        Decl::Fixity { .. } | Decl::Val { .. } | Decl::Instance { .. } => {}
                    }
                }
            }
            Item::Expr(_) => {}
        }
    }

    (imports, exports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_path_to_file_path() {
        assert_eq!(module_path_to_file_path("List"), "list");
        assert_eq!(module_path_to_file_path("HashMap"), "hash_map");
        assert_eq!(module_path_to_file_path("Collections/HashMap"), "collections/hash_map");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("List"), "list");
        assert_eq!(to_snake_case("HashMap"), "hash_map");
        assert_eq!(to_snake_case("XMLParser"), "x_m_l_parser");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("list"), "List");
        assert_eq!(to_pascal_case("hash_map"), "HashMap");
    }

    #[test]
    fn test_path_to_module_name() {
        assert_eq!(path_to_module_name(Path::new("list.gn")), "List");
        assert_eq!(path_to_module_name(Path::new("hash_map.gn")), "HashMap");
        assert_eq!(path_to_module_name(Path::new("list/mod.gn")), "List");
    }
}
