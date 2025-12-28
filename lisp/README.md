# Gneiss Editor Support

This directory contains the Emacs major mode for Gneiss.

## Components

- `gneiss-ts-mode.el` - Emacs major mode using tree-sitter
- Tree-sitter grammar: separate repo at `tree-sitter-gneiss`

## Prerequisites

- Emacs 29.1+ (with tree-sitter support)

## Quick Start

### 1. Install the Grammar

Let Emacs compile the grammar from your local clone:

```elisp
(setq treesit-language-source-alist
      '((gneiss "file:///path/to/tree-sitter-gneiss")))

(treesit-install-language-grammar 'gneiss)
```

### 2. Configure Emacs

Add to your init.el:

```elisp
;; Add to load-path
(add-to-list 'load-path "/path/to/gneiss-lang/lisp")

;; Load the mode
(require 'gneiss-ts-mode)
```

## Usage

Open any `.gn` file and `gneiss-ts-mode` will activate automatically.

## Features

- **Syntax Highlighting**: Keywords, literals, types, operators, comments
- **Indentation**: Automatic indentation based on grammar structure
- **Navigation**: `M-.` / `M-,` for defun navigation
- **Imenu**: Jump to definitions with `M-x imenu`
- **Comments**: `M-;` to comment/uncomment regions

## Customization

```elisp
;; Change indentation width (default: 4)
(setq gneiss-ts-mode-indent-offset 2)
```

## Troubleshooting

### "Tree-sitter grammar for Gneiss is not available"

Run `(treesit-install-language-grammar 'gneiss)` in Emacs.

### Highlighting not working

Ensure all font-lock features are enabled:

```elisp
(setq treesit-font-lock-level 4)
```

## License

Same as the main Gneiss project.
