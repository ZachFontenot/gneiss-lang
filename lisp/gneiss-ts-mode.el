;;; gneiss-ts-mode.el --- Major mode for Gneiss using tree-sitter -*- lexical-binding: t; -*-

;; Copyright (C) 2024-2025
;; Author: Gneiss Contributors
;; Version: 0.1.0
;; Package-Requires: ((emacs "29.1"))
;; Keywords: languages, gneiss, tree-sitter
;; URL: https://github.com/your-username/gneiss-lang

;;; Commentary:

;; This package provides a major mode for editing Gneiss source files.
;; It uses tree-sitter for syntax highlighting and indentation.
;;
;; Features:
;; - Syntax highlighting via tree-sitter
;; - Automatic indentation
;; - Code navigation (imenu)
;; - Comment handling (line -- and block {- -})
;;
;; Installation:
;;   1. Build and install the tree-sitter-gneiss grammar
;;   2. Add this file to your load-path
;;   3. (require 'gneiss-ts-mode)
;;   4. Open a .gn file

;;; Code:

(require 'treesit)
(require 'prog-mode)

(declare-function treesit-parser-create "treesit.c")
(declare-function treesit-node-type "treesit.c")
(declare-function treesit-node-child-by-field-name "treesit.c")
(declare-function treesit-induce-sparse-tree "treesit.c")

(defgroup gneiss nil
  "Major mode for Gneiss programming language."
  :group 'languages
  :prefix "gneiss-")

(defcustom gneiss-ts-mode-indent-offset 4
  "Number of spaces for each indentation step in `gneiss-ts-mode'."
  :type 'integer
  :safe 'integerp
  :group 'gneiss)

(defvar gneiss-ts-mode--syntax-table
  (let ((table (make-syntax-table)))
    ;; Line comments: -- to end of line
    (modify-syntax-entry ?- ". 12" table)
    (modify-syntax-entry ?\n ">" table)

    ;; Block comments are handled by tree-sitter, but we set up basic syntax
    (modify-syntax-entry ?{ "(}" table)
    (modify-syntax-entry ?} "){" table)

    ;; Strings
    (modify-syntax-entry ?\" "\"" table)
    (modify-syntax-entry ?\' "\"" table)
    (modify-syntax-entry ?\\ "\\" table)

    ;; Operators
    (modify-syntax-entry ?+ "." table)
    (modify-syntax-entry ?* "." table)
    (modify-syntax-entry ?/ "." table)
    (modify-syntax-entry ?< "." table)
    (modify-syntax-entry ?> "." table)
    (modify-syntax-entry ?= "." table)
    (modify-syntax-entry ?| "." table)
    (modify-syntax-entry ?& "." table)
    (modify-syntax-entry ?! "." table)
    (modify-syntax-entry ?: "." table)
    (modify-syntax-entry ?% "." table)

    ;; Identifiers can contain underscore and apostrophe
    (modify-syntax-entry ?_ "_" table)
    (modify-syntax-entry ?' "_" table)

    table)
  "Syntax table for `gneiss-ts-mode'.")

(defvar gneiss-ts-mode--keywords
  '("let" "rec" "and" "in"
    "fun" "match" "with" "end"
    "if" "then" "else"
    "type" "trait" "impl" "for" "where" "val"
    "import" "export" "as"
    "spawn" "select"
    "reset" "shift"
    "infixl" "infixr" "infix")
  "Gneiss keywords for font-lock.")

(defvar gneiss-ts-mode--font-lock-settings
  (treesit-font-lock-rules
   :language 'gneiss
   :feature 'comment
   '((line_comment) @font-lock-comment-face
     (block_comment) @font-lock-comment-face)

   :language 'gneiss
   :feature 'keyword
   `(["let" "rec" "and" "in"
      "fun" "match" "with" "end"
      "if" "then" "else"
      "type" "trait" "impl" "for" "where" "val"
      "import" "export" "as"
      "spawn" "select"
      "reset" "shift"
      "infixl" "infixr" "infix"
      "not"]
     @font-lock-keyword-face)

   :language 'gneiss
   :feature 'constant
   '([(boolean) (wildcard)] @font-lock-constant-face)

   :language 'gneiss
   :feature 'string
   '((string) @font-lock-string-face
     (char) @font-lock-string-face
     (escape_sequence) @font-lock-escape-face)

   :language 'gneiss
   :feature 'number
   '((integer) @font-lock-number-face
     (float) @font-lock-number-face)

   :language 'gneiss
   :feature 'type
   '((upper_identifier) @font-lock-type-face
     (type_parameter) @font-lock-type-face)

   :language 'gneiss
   :feature 'function
   :override t
   '((binding name: (identifier) @font-lock-function-name-face)
     (impl_method name: (identifier) @font-lock-function-name-face)
     (val_signature name: (identifier) @font-lock-function-name-face)
     (application function: (identifier) @font-lock-function-call-face))

   :language 'gneiss
   :feature 'variable
   '((identifier) @font-lock-variable-name-face)

   :language 'gneiss
   :feature 'operator
   '(["=" "->" "<-" "|" "::" "++" "|>" "<|" ">>" "<<"
      "+" "-" "*" "/" "%" "==" "!=" "<" ">" "<=" ">="
      "&&" "||" ".."]
     @font-lock-operator-face)

   :language 'gneiss
   :feature 'property
   '((field_declaration name: (identifier) @font-lock-property-name-face)
     (field_init name: (identifier) @font-lock-property-use-face)
     (field_access field: (identifier) @font-lock-property-use-face))

   :language 'gneiss
   :feature 'delimiter
   '(["(" ")" "[" "]" "{" "}"] @font-lock-bracket-face
     ["," ";" ";;" ":" "."] @font-lock-delimiter-face))
  "Tree-sitter font-lock settings for `gneiss-ts-mode'.")

(defvar gneiss-ts-mode--indent-rules
  `((gneiss
     ;; Align closing delimiters with their opening line
     ((node-is ")") parent-bol 0)
     ((node-is "]") parent-bol 0)
     ((node-is "}") parent-bol 0)
     ((node-is "end") parent-bol 0)

     ;; Match and select arms
     ((node-is "match_arm") parent-bol 0)
     ((node-is "select_arm") parent-bol 0)

     ;; Indent after keywords
     ((parent-is "let_expression") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "if_expression") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "match_expression") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "lambda_expression") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "select_expression") parent-bol ,gneiss-ts-mode-indent-offset)

     ;; Match arm bodies
     ((parent-is "match_arm") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "select_arm") parent-bol ,gneiss-ts-mode-indent-offset)

     ;; Type declarations
     ((parent-is "type_declaration") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "trait_declaration") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "impl_declaration") parent-bol ,gneiss-ts-mode-indent-offset)

     ;; Records and lists
     ((parent-is "record_type") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "record_literal") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "list") parent-bol ,gneiss-ts-mode-indent-offset)
     ((parent-is "tuple") parent-bol ,gneiss-ts-mode-indent-offset)

     ;; Bindings continuation
     ((parent-is "binding") parent-bol ,gneiss-ts-mode-indent-offset)

     ;; Default: keep current indentation
     (no-node parent-bol 0)))
  "Tree-sitter indentation rules for `gneiss-ts-mode'.")

(defun gneiss-ts-mode--defun-name (node)
  "Return the name of the definition NODE for imenu."
  (treesit-node-text
   (treesit-node-child-by-field-name node "name")
   t))

;;;###autoload
(define-derived-mode gneiss-ts-mode prog-mode "Gneiss"
  "Major mode for editing Gneiss code, powered by tree-sitter.

\\{gneiss-ts-mode-map}"
  :syntax-table gneiss-ts-mode--syntax-table
  :group 'gneiss

  ;; Check tree-sitter availability
  (unless (treesit-ready-p 'gneiss)
    (error "Tree-sitter grammar for Gneiss is not available.
Please install the grammar from lisp/tree-sitter-gneiss/"))

  ;; Create parser
  (treesit-parser-create 'gneiss)

  ;; Font-lock
  (setq-local treesit-font-lock-settings gneiss-ts-mode--font-lock-settings)
  (setq-local treesit-font-lock-feature-list
              '((comment)
                (keyword string number)
                (type constant operator)
                (function variable property delimiter)))

  ;; Indentation
  (setq-local treesit-simple-indent-rules gneiss-ts-mode--indent-rules)
  (setq-local indent-tabs-mode nil)
  (setq-local tab-width gneiss-ts-mode-indent-offset)

  ;; Comments
  (setq-local comment-start "-- ")
  (setq-local comment-end "")
  (setq-local comment-start-skip "\\(?:--+\\|{-\\)[ \t]*")
  (setq-local comment-end-skip "[ \t]*\\(?:\n\\|-}\\)")
  (setq-local comment-multi-line t)

  ;; Navigation - defun recognition
  (setq-local treesit-defun-type-regexp
              (regexp-opt '("let_declaration"
                           "type_declaration"
                           "trait_declaration"
                           "impl_declaration")))
  (setq-local treesit-defun-name-function #'gneiss-ts-mode--defun-name)

  ;; Imenu
  (setq-local treesit-simple-imenu-settings
              '(("Function" "\\`let_declaration\\'" nil nil)
                ("Type" "\\`type_declaration\\'" nil nil)
                ("Trait" "\\`trait_declaration\\'" nil nil)
                ("Impl" "\\`impl_declaration\\'" nil nil)))

  ;; Electric
  (setq-local electric-indent-chars
              (append "()[]{};" electric-indent-chars))

  ;; Finalize setup
  (treesit-major-mode-setup))

;; Key bindings
(defvar gneiss-ts-mode-map
  (let ((map (make-sparse-keymap)))
    ;; Add keybindings here as needed
    ;; (define-key map (kbd "C-c C-z") #'gneiss-run-repl)
    ;; (define-key map (kbd "C-c C-c") #'gneiss-compile-file)
    map)
  "Keymap for `gneiss-ts-mode'.")

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.gn\\'" . gneiss-ts-mode))

;; Optional: Register with treesit-auto if available
(with-eval-after-load 'treesit-auto
  (add-to-list 'treesit-auto-langs 'gneiss))

(provide 'gneiss-ts-mode)
;;; gneiss-ts-mode.el ends here
