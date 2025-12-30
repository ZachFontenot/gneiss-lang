/*
 * Gneiss Runtime Library
 *
 * Value representation and runtime support for compiled Gneiss programs.
 * Uses tagged pointers for unboxed integers and heap-allocated objects.
 */

#ifndef GN_RUNTIME_H
#define GN_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * Value Representation
 *
 * All values are 64-bit. We use tagged pointers:
 * - If low bit is 1: immediate integer (shifted left by 1)
 * - If low bit is 0: pointer to heap object (aligned)
 * ============================================================================ */

typedef uint64_t gn_value;

/* Immediate integer encoding */
#define GN_INT(n)        (((gn_value)((int64_t)(n)) << 1) | 1)
#define GN_UNINT(v)      ((int64_t)(v) >> 1)
#define GN_IS_INT(v)     ((v) & 1)

/* Boolean encoding (as tagged integers) */
#define GN_FALSE         GN_INT(0)
#define GN_TRUE          GN_INT(1)
#define GN_IS_TRUE(v)    (GN_UNINT(v) != 0)

/* Unit value */
#define GN_UNIT          GN_INT(0)

/* Character encoding (as tagged integer with char value) */
#define GN_CHAR(c)       GN_INT((uint32_t)(c))
#define GN_UNCHAR(v)     ((char)GN_UNINT(v))
#define GN_CHAR_TO_INT(v) (v)  /* Already an int */
#define GN_INT_TO_CHAR(v) (v)  /* Already a char */

/* ============================================================================
 * Heap Objects
 * ============================================================================ */

typedef struct gn_object {
    uint32_t rc;           /* Reference count */
    uint32_t tag;          /* Constructor tag */
    uint32_t n_fields;     /* Number of fields */
    uint32_t _padding;     /* Alignment padding */
    gn_value fields[];     /* Flexible array of fields */
} gn_object;

/* Object access macros */
#define GN_OBJ(v)        ((gn_object*)(v))
#define GN_TAG(v)        (GN_IS_INT(v) ? 0 : GN_OBJ(v)->tag)
#define GN_FIELD(v, i)   (GN_OBJ(v)->fields[i])
#define GN_NFIELDS(v)    (GN_OBJ(v)->n_fields)

/* Constructor for singleton (nullary) constructors */
#define GN_CTOR(tag, dummy) gn_singleton(tag)

/* ============================================================================
 * Integer Arithmetic (on tagged integers)
 * ============================================================================ */

#define GN_INT_ADD(a, b)  GN_INT(GN_UNINT(a) + GN_UNINT(b))
#define GN_INT_SUB(a, b)  GN_INT(GN_UNINT(a) - GN_UNINT(b))
#define GN_INT_MUL(a, b)  GN_INT(GN_UNINT(a) * GN_UNINT(b))
#define GN_INT_DIV(a, b)  GN_INT(GN_UNINT(a) / GN_UNINT(b))
#define GN_INT_MOD(a, b)  GN_INT(GN_UNINT(a) % GN_UNINT(b))
#define GN_INT_NEG(a)     GN_INT(-GN_UNINT(a))

/* ============================================================================
 * Integer Comparison
 * ============================================================================ */

#define GN_INT_EQ(a, b)   ((a) == (b) ? GN_TRUE : GN_FALSE)
#define GN_INT_NE(a, b)   ((a) != (b) ? GN_TRUE : GN_FALSE)
#define GN_INT_LT(a, b)   (GN_UNINT(a) < GN_UNINT(b) ? GN_TRUE : GN_FALSE)
#define GN_INT_LE(a, b)   (GN_UNINT(a) <= GN_UNINT(b) ? GN_TRUE : GN_FALSE)
#define GN_INT_GT(a, b)   (GN_UNINT(a) > GN_UNINT(b) ? GN_TRUE : GN_FALSE)
#define GN_INT_GE(a, b)   (GN_UNINT(a) >= GN_UNINT(b) ? GN_TRUE : GN_FALSE)

/* ============================================================================
 * Boolean Operations
 * ============================================================================ */

#define GN_BOOL_AND(a, b) (GN_IS_TRUE(a) && GN_IS_TRUE(b) ? GN_TRUE : GN_FALSE)
#define GN_BOOL_OR(a, b)  (GN_IS_TRUE(a) || GN_IS_TRUE(b) ? GN_TRUE : GN_FALSE)
#define GN_BOOL_NOT(a)    (GN_IS_TRUE(a) ? GN_FALSE : GN_TRUE)

/* ============================================================================
 * Runtime Functions (implemented in gn_runtime.c)
 * ============================================================================ */

/* Initialization */
void gn_init(int argc, char** argv);
void gn_shutdown(void);

/* Memory allocation */
gn_value gn_alloc(uint32_t tag, uint32_t n_fields, gn_value* fields);
gn_value gn_singleton(uint32_t tag);

/* Reference counting (for future Perceus) */
gn_value gn_dup(gn_value v);
void gn_drop(gn_value v);

/* Printing */
void gn_print(gn_value v);
void gn_println(void);

/* Panic/error - returns gn_value for use in expressions (never actually returns) */
gn_value gn_panic(const char* msg);

/* String operations */
gn_value gn_string(const char* s);
gn_value gn_string_concat(gn_value a, gn_value b);
gn_value gn_string_length(gn_value s);
gn_value gn_string_eq(gn_value a, gn_value b);
gn_value gn_int_to_string(gn_value n);
gn_value gn_char_to_string(gn_value c);
gn_value gn_string_join(gn_value sep, gn_value list);
gn_value gn_string_split(gn_value sep, gn_value str);
gn_value gn_string_index_of(gn_value needle, gn_value haystack);
gn_value gn_string_substring(gn_value start, gn_value end, gn_value str);

/* I/O operations */
gn_value gn_io_print(gn_value s);
gn_value gn_io_read_line(gn_value unit);  /* unit arg ignored */

/* Float operations (boxed) */
gn_value gn_float(double f);
double gn_unfloat(gn_value v);
gn_value gn_float_add(gn_value a, gn_value b);
gn_value gn_float_sub(gn_value a, gn_value b);
gn_value gn_float_mul(gn_value a, gn_value b);
gn_value gn_float_div(gn_value a, gn_value b);
gn_value gn_float_neg(gn_value a);
gn_value gn_float_eq(gn_value a, gn_value b);
gn_value gn_float_ne(gn_value a, gn_value b);
gn_value gn_float_lt(gn_value a, gn_value b);
gn_value gn_float_le(gn_value a, gn_value b);
gn_value gn_float_gt(gn_value a, gn_value b);
gn_value gn_float_ge(gn_value a, gn_value b);
gn_value gn_int_to_float(gn_value n);
gn_value gn_float_to_int(gn_value f);
gn_value gn_float_to_string(gn_value f);

/* List operations */
gn_value gn_list_cons(gn_value head, gn_value tail);
gn_value gn_list_head(gn_value list);
gn_value gn_list_tail(gn_value list);
gn_value gn_list_is_empty(gn_value list);
gn_value gn_list_concat(gn_value left, gn_value right);

/* Closure creation and application (for higher-order functions) */
/* Create a closure with no captured environment (arity 1) */
gn_value gn_make_closure1(void* fn);
/* Create a closure with no captured environment (arity 2) */
gn_value gn_make_closure2(void* fn);
/* Create a closure with captured environment */
gn_value gn_make_closure(void* fn, uint32_t arity, uint32_t n_captures, gn_value* captures);
gn_value gn_apply(gn_value closure, gn_value arg);
gn_value gn_apply2(gn_value closure, gn_value arg1, gn_value arg2);

#endif /* GN_RUNTIME_H */
