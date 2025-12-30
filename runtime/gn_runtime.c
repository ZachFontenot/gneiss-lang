/*
 * Gneiss Runtime Library Implementation
 *
 * Minimal runtime for compiled Gneiss programs.
 * Focus: correctness first, optimize later.
 */

#include "gn_runtime.h"
#include <inttypes.h>

/* ============================================================================
 * Configuration
 * ============================================================================ */

/* Tag values for built-in types */
#define TAG_STRING   0xFFFF0001
#define TAG_FLOAT    0xFFFF0002
#define TAG_CLOSURE  0xFFFF0003

/* List constructors (standard Option/List encoding) */
#define TAG_NIL      6
#define TAG_CONS     7

/* Option constructors */
#define TAG_SOME     18
#define TAG_NONE     19

/* ============================================================================
 * Global State
 * ============================================================================ */

static int gn_argc = 0;
static char** gn_argv = NULL;

/* Singleton cache for nullary constructors */
#define MAX_SINGLETON_TAG 256
static gn_object* singleton_cache[MAX_SINGLETON_TAG] = {0};

/* ============================================================================
 * Initialization
 * ============================================================================ */

void gn_init(int argc, char** argv) {
    gn_argc = argc;
    gn_argv = argv;

    /* Pre-allocate common singletons */
    for (uint32_t tag = 0; tag < MAX_SINGLETON_TAG; tag++) {
        singleton_cache[tag] = NULL;
    }
}

void gn_shutdown(void) {
    /* Free singleton cache */
    for (uint32_t tag = 0; tag < MAX_SINGLETON_TAG; tag++) {
        if (singleton_cache[tag]) {
            free(singleton_cache[tag]);
            singleton_cache[tag] = NULL;
        }
    }
}

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

gn_value gn_alloc(uint32_t tag, uint32_t n_fields, gn_value* fields) {
    size_t size = sizeof(gn_object) + n_fields * sizeof(gn_value);
    gn_object* obj = (gn_object*)malloc(size);
    if (!obj) {
        gn_panic("out of memory");
    }

    obj->rc = 1;
    obj->tag = tag;
    obj->n_fields = n_fields;
    obj->_padding = 0;

    for (uint32_t i = 0; i < n_fields; i++) {
        obj->fields[i] = fields[i];
    }

    return (gn_value)obj;
}

gn_value gn_singleton(uint32_t tag) {
    if (tag < MAX_SINGLETON_TAG) {
        if (!singleton_cache[tag]) {
            gn_object* obj = (gn_object*)malloc(sizeof(gn_object));
            if (!obj) {
                gn_panic("out of memory");
            }
            obj->rc = 1;  /* Never freed */
            obj->tag = tag;
            obj->n_fields = 0;
            obj->_padding = 0;
            singleton_cache[tag] = obj;
        }
        return (gn_value)singleton_cache[tag];
    }

    /* Fallback for large tags */
    return gn_alloc(tag, 0, NULL);
}

/* ============================================================================
 * Reference Counting
 *
 * For now, these are mostly no-ops. Perceus transformation will insert
 * proper dup/drop calls, and we can implement real RC later.
 * ============================================================================ */

gn_value gn_dup(gn_value v) {
    if (!GN_IS_INT(v)) {
        gn_object* obj = GN_OBJ(v);
        obj->rc++;
    }
    return v;
}

void gn_drop(gn_value v) {
    if (!GN_IS_INT(v)) {
        gn_object* obj = GN_OBJ(v);
        if (obj->rc > 0) {
            obj->rc--;
            if (obj->rc == 0) {
                /* Recursively drop fields */
                for (uint32_t i = 0; i < obj->n_fields; i++) {
                    gn_drop(obj->fields[i]);
                }
                free(obj);
            }
        }
    }
}

/* ============================================================================
 * Printing
 * ============================================================================ */

static void print_value_impl(gn_value v, int depth);

static void print_value_impl(gn_value v, int depth) {
    if (depth > 100) {
        printf("...");
        return;
    }

    if (GN_IS_INT(v)) {
        printf("%" PRId64, GN_UNINT(v));
        return;
    }

    gn_object* obj = GN_OBJ(v);

    switch (obj->tag) {
        case TAG_STRING: {
            /* String: fields[0] is char*, fields[1] is length */
            char* str = (char*)obj->fields[0];
            printf("%s", str);
            break;
        }

        case TAG_FLOAT: {
            /* Float: stored as bits in fields[0] */
            union { uint64_t i; double d; } u;
            u.i = obj->fields[0];
            printf("%g", u.d);
            break;
        }

        case TAG_NIL:
            printf("[]");
            break;

        case TAG_CONS: {
            printf("[");
            gn_value current = v;
            int first = 1;
            while (!GN_IS_INT(current) && GN_OBJ(current)->tag == TAG_CONS) {
                if (!first) printf(", ");
                first = 0;
                print_value_impl(GN_OBJ(current)->fields[0], depth + 1);
                current = GN_OBJ(current)->fields[1];
            }
            printf("]");
            break;
        }

        default: {
            /* Generic ADT */
            if (obj->n_fields == 0) {
                printf("<ctor:%u>", obj->tag);
            } else {
                printf("<ctor:%u>(", obj->tag);
                for (uint32_t i = 0; i < obj->n_fields; i++) {
                    if (i > 0) printf(", ");
                    print_value_impl(obj->fields[i], depth + 1);
                }
                printf(")");
            }
            break;
        }
    }
}

void gn_print(gn_value v) {
    print_value_impl(v, 0);
}

void gn_println(void) {
    printf("\n");
    fflush(stdout);
}

/* ============================================================================
 * Panic/Error
 * ============================================================================ */

gn_value gn_panic(const char* msg) {
    fprintf(stderr, "panic: %s\n", msg);
    exit(1);
    return GN_UNIT;  /* Never reached, but satisfies return type */
}

/* ============================================================================
 * String Operations
 * ============================================================================ */

gn_value gn_string(const char* s) {
    size_t len = strlen(s);
    char* copy = (char*)malloc(len + 1);
    if (!copy) {
        gn_panic("out of memory");
    }
    memcpy(copy, s, len + 1);

    gn_value fields[2];
    fields[0] = (gn_value)copy;
    fields[1] = (gn_value)len;

    return gn_alloc(TAG_STRING, 2, fields);
}

gn_value gn_string_concat(gn_value a, gn_value b) {
    gn_object* obj_a = GN_OBJ(a);
    gn_object* obj_b = GN_OBJ(b);

    char* str_a = (char*)obj_a->fields[0];
    char* str_b = (char*)obj_b->fields[0];
    size_t len_a = (size_t)obj_a->fields[1];
    size_t len_b = (size_t)obj_b->fields[1];

    size_t total = len_a + len_b;
    char* result = (char*)malloc(total + 1);
    if (!result) {
        gn_panic("out of memory");
    }

    memcpy(result, str_a, len_a);
    memcpy(result + len_a, str_b, len_b + 1);

    gn_value fields[2];
    fields[0] = (gn_value)result;
    fields[1] = (gn_value)total;

    return gn_alloc(TAG_STRING, 2, fields);
}

gn_value gn_string_length(gn_value s) {
    gn_object* obj = GN_OBJ(s);
    return GN_INT((int64_t)obj->fields[1]);
}

gn_value gn_string_eq(gn_value a, gn_value b) {
    gn_object* obj_a = GN_OBJ(a);
    gn_object* obj_b = GN_OBJ(b);

    size_t len_a = (size_t)obj_a->fields[1];
    size_t len_b = (size_t)obj_b->fields[1];

    if (len_a != len_b) {
        return GN_FALSE;
    }

    char* str_a = (char*)obj_a->fields[0];
    char* str_b = (char*)obj_b->fields[0];

    return memcmp(str_a, str_b, len_a) == 0 ? GN_TRUE : GN_FALSE;
}

gn_value gn_int_to_string(gn_value n) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%" PRId64, GN_UNINT(n));
    return gn_string(buf);
}

gn_value gn_char_to_string(gn_value c) {
    char buf[2];
    buf[0] = GN_UNCHAR(c);
    buf[1] = '\0';
    return gn_string(buf);
}

gn_value gn_string_join(gn_value sep, gn_value list) {
    /* Join a list of strings with a separator */
    gn_object* sep_obj = GN_OBJ(sep);
    char* sep_str = (char*)sep_obj->fields[0];
    size_t sep_len = (size_t)sep_obj->fields[1];

    /* First pass: calculate total length */
    size_t total_len = 0;
    int count = 0;
    gn_value current = list;
    while (!GN_IS_INT(current) && GN_OBJ(current)->tag == TAG_CONS) {
        gn_object* elem = GN_OBJ(GN_OBJ(current)->fields[0]);
        total_len += (size_t)elem->fields[1];
        count++;
        current = GN_OBJ(current)->fields[1];
    }

    if (count == 0) {
        return gn_string("");
    }

    total_len += sep_len * (count - 1);

    /* Second pass: build the result */
    char* result = (char*)malloc(total_len + 1);
    if (!result) {
        gn_panic("out of memory");
    }

    char* ptr = result;
    int first = 1;
    current = list;
    while (!GN_IS_INT(current) && GN_OBJ(current)->tag == TAG_CONS) {
        if (!first) {
            memcpy(ptr, sep_str, sep_len);
            ptr += sep_len;
        }
        first = 0;

        gn_object* elem = GN_OBJ(GN_OBJ(current)->fields[0]);
        char* elem_str = (char*)elem->fields[0];
        size_t elem_len = (size_t)elem->fields[1];
        memcpy(ptr, elem_str, elem_len);
        ptr += elem_len;

        current = GN_OBJ(current)->fields[1];
    }
    *ptr = '\0';

    gn_value fields[2];
    fields[0] = (gn_value)result;
    fields[1] = (gn_value)total_len;
    return gn_alloc(TAG_STRING, 2, fields);
}

gn_value gn_string_split(gn_value sep, gn_value str) {
    /* Split a string by separator, return list of strings */
    gn_object* sep_obj = GN_OBJ(sep);
    gn_object* str_obj = GN_OBJ(str);

    char* sep_str = (char*)sep_obj->fields[0];
    size_t sep_len = (size_t)sep_obj->fields[1];
    char* src = (char*)str_obj->fields[0];
    size_t src_len = (size_t)str_obj->fields[1];

    if (sep_len == 0) {
        /* Empty separator - return list with original string */
        return gn_list_cons(str, gn_singleton(TAG_NIL));
    }

    /* Build list in reverse, then reverse it */
    gn_value result = gn_singleton(TAG_NIL);
    char* start = src;
    char* end = src + src_len;

    while (start < end) {
        char* found = strstr(start, sep_str);
        if (!found || found >= end) {
            /* No more separators - add rest of string */
            size_t len = end - start;
            char* copy = (char*)malloc(len + 1);
            memcpy(copy, start, len);
            copy[len] = '\0';
            gn_value fields[2] = {(gn_value)copy, (gn_value)len};
            gn_value part = gn_alloc(TAG_STRING, 2, fields);
            result = gn_list_cons(part, result);
            break;
        }

        /* Add part before separator */
        size_t len = found - start;
        char* copy = (char*)malloc(len + 1);
        memcpy(copy, start, len);
        copy[len] = '\0';
        gn_value fields[2] = {(gn_value)copy, (gn_value)len};
        gn_value part = gn_alloc(TAG_STRING, 2, fields);
        result = gn_list_cons(part, result);

        start = found + sep_len;
    }

    /* Reverse the list */
    gn_value reversed = gn_singleton(TAG_NIL);
    gn_value curr = result;
    while (!GN_IS_INT(curr) && GN_OBJ(curr)->tag == TAG_CONS) {
        gn_value head = GN_OBJ(curr)->fields[0];
        reversed = gn_list_cons(head, reversed);
        curr = GN_OBJ(curr)->fields[1];
    }

    return reversed;
}

gn_value gn_string_index_of(gn_value needle, gn_value haystack) {
    /* Find index of needle in haystack, return Some(index) or None */
    gn_object* needle_obj = GN_OBJ(needle);
    gn_object* haystack_obj = GN_OBJ(haystack);

    char* needle_str = (char*)needle_obj->fields[0];
    char* haystack_str = (char*)haystack_obj->fields[0];

    char* found = strstr(haystack_str, needle_str);
    if (found) {
        /* Return Some(index) */
        int64_t index = found - haystack_str;
        gn_value fields[1] = {GN_INT(index)};
        return gn_alloc(TAG_SOME, 1, fields);  /* TAG_SOME = 18 */
    } else {
        /* Return None */
        return gn_singleton(TAG_NONE);  /* TAG_NONE = 19 */
    }
}

gn_value gn_string_substring(gn_value start, gn_value end, gn_value str) {
    /* Extract substring from start to end (exclusive) */
    gn_object* str_obj = GN_OBJ(str);
    char* src = (char*)str_obj->fields[0];
    size_t src_len = (size_t)str_obj->fields[1];

    int64_t start_idx = GN_UNINT(start);
    int64_t end_idx = GN_UNINT(end);

    /* Clamp indices */
    if (start_idx < 0) start_idx = 0;
    if (end_idx < 0) end_idx = 0;
    if ((size_t)start_idx > src_len) start_idx = src_len;
    if ((size_t)end_idx > src_len) end_idx = src_len;
    if (start_idx > end_idx) start_idx = end_idx;

    size_t len = end_idx - start_idx;
    char* copy = (char*)malloc(len + 1);
    memcpy(copy, src + start_idx, len);
    copy[len] = '\0';

    gn_value fields[2] = {(gn_value)copy, (gn_value)len};
    return gn_alloc(TAG_STRING, 2, fields);
}

/* ============================================================================
 * I/O Operations
 * ============================================================================ */

gn_value gn_io_print(gn_value s) {
    /* Print a string without newline */
    gn_object* obj = GN_OBJ(s);
    char* str = (char*)obj->fields[0];
    printf("%s", str);
    fflush(stdout);
    return GN_UNIT;
}

gn_value gn_io_read_line(gn_value unit) {
    /* Read a line from stdin, return as string */
    (void)unit;  /* Ignore unit argument */
    char* line = NULL;
    size_t len = 0;
    ssize_t nread = getline(&line, &len, stdin);

    if (nread == -1) {
        free(line);
        return gn_string("");
    }

    /* Remove trailing newline if present */
    if (nread > 0 && line[nread - 1] == '\n') {
        line[nread - 1] = '\0';
        nread--;
    }

    gn_value result = gn_string(line);
    free(line);
    return result;
}

/* ============================================================================
 * Float Operations
 * ============================================================================ */

gn_value gn_float(double f) {
    union { uint64_t i; double d; } u;
    u.d = f;

    gn_value fields[1];
    fields[0] = u.i;

    return gn_alloc(TAG_FLOAT, 1, fields);
}

double gn_unfloat(gn_value v) {
    gn_object* obj = GN_OBJ(v);
    union { uint64_t i; double d; } u;
    u.i = obj->fields[0];
    return u.d;
}

gn_value gn_float_add(gn_value a, gn_value b) {
    return gn_float(gn_unfloat(a) + gn_unfloat(b));
}

gn_value gn_float_sub(gn_value a, gn_value b) {
    return gn_float(gn_unfloat(a) - gn_unfloat(b));
}

gn_value gn_float_mul(gn_value a, gn_value b) {
    return gn_float(gn_unfloat(a) * gn_unfloat(b));
}

gn_value gn_float_div(gn_value a, gn_value b) {
    return gn_float(gn_unfloat(a) / gn_unfloat(b));
}

gn_value gn_float_neg(gn_value a) {
    return gn_float(-gn_unfloat(a));
}

gn_value gn_float_eq(gn_value a, gn_value b) {
    return gn_unfloat(a) == gn_unfloat(b) ? GN_TRUE : GN_FALSE;
}

gn_value gn_float_ne(gn_value a, gn_value b) {
    return gn_unfloat(a) != gn_unfloat(b) ? GN_TRUE : GN_FALSE;
}

gn_value gn_float_lt(gn_value a, gn_value b) {
    return gn_unfloat(a) < gn_unfloat(b) ? GN_TRUE : GN_FALSE;
}

gn_value gn_float_le(gn_value a, gn_value b) {
    return gn_unfloat(a) <= gn_unfloat(b) ? GN_TRUE : GN_FALSE;
}

gn_value gn_float_gt(gn_value a, gn_value b) {
    return gn_unfloat(a) > gn_unfloat(b) ? GN_TRUE : GN_FALSE;
}

gn_value gn_float_ge(gn_value a, gn_value b) {
    return gn_unfloat(a) >= gn_unfloat(b) ? GN_TRUE : GN_FALSE;
}

gn_value gn_int_to_float(gn_value n) {
    return gn_float((double)GN_UNINT(n));
}

gn_value gn_float_to_int(gn_value f) {
    return GN_INT((int64_t)gn_unfloat(f));
}

gn_value gn_float_to_string(gn_value f) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g", gn_unfloat(f));
    return gn_string(buf);
}

/* ============================================================================
 * List Operations
 * ============================================================================ */

gn_value gn_list_nil(void) {
    return gn_singleton(TAG_NIL);
}

gn_value gn_list_cons(gn_value head, gn_value tail) {
    gn_value fields[2];
    fields[0] = head;
    fields[1] = tail;
    return gn_alloc(TAG_CONS, 2, fields);
}

gn_value gn_list_head(gn_value list) {
    gn_object* obj = GN_OBJ(list);
    if (obj->tag != TAG_CONS) {
        gn_panic("head of empty list");
    }
    return obj->fields[0];
}

gn_value gn_list_tail(gn_value list) {
    gn_object* obj = GN_OBJ(list);
    if (obj->tag != TAG_CONS) {
        gn_panic("tail of empty list");
    }
    return obj->fields[1];
}

gn_value gn_list_is_empty(gn_value list) {
    if (GN_IS_INT(list)) {
        return GN_TRUE;  /* Should not happen, but safe */
    }
    return GN_OBJ(list)->tag == TAG_NIL ? GN_TRUE : GN_FALSE;
}

gn_value gn_list_concat(gn_value left, gn_value right) {
    /* Concatenate two lists: left ++ right */
    /* If left is empty, return right */
    if (GN_IS_INT(left) || GN_OBJ(left)->tag == TAG_NIL) {
        return right;
    }

    /* Build reversed left list, then cons elements onto right */
    gn_value reversed = gn_singleton(TAG_NIL);
    gn_value curr = left;
    while (!GN_IS_INT(curr) && GN_OBJ(curr)->tag == TAG_CONS) {
        gn_value head = GN_OBJ(curr)->fields[0];
        reversed = gn_list_cons(head, reversed);
        curr = GN_OBJ(curr)->fields[1];
    }

    /* Now cons reversed elements onto right */
    gn_value result = right;
    curr = reversed;
    while (!GN_IS_INT(curr) && GN_OBJ(curr)->tag == TAG_CONS) {
        gn_value head = GN_OBJ(curr)->fields[0];
        result = gn_list_cons(head, result);
        curr = GN_OBJ(curr)->fields[1];
    }

    return result;
}

/* ============================================================================
 * Closure Application
 *
 * Closures are represented as:
 *   tag = TAG_CLOSURE
 *   fields[0] = function pointer (cast to gn_value)
 *   fields[1] = arity
 *   fields[2...] = captured environment
 * ============================================================================ */

typedef gn_value (*gn_fn1)(gn_value*, gn_value);
typedef gn_value (*gn_fn2)(gn_value*, gn_value, gn_value);
typedef gn_value (*gn_fn3)(gn_value*, gn_value, gn_value, gn_value);
typedef gn_value (*gn_fn4)(gn_value*, gn_value, gn_value, gn_value, gn_value);
typedef gn_value (*gn_fn5)(gn_value*, gn_value, gn_value, gn_value, gn_value, gn_value);

/*
 * Closure layout: [fn, arity, n_captures, capture0, ..., applied_arg0, ...]
 *
 * - fn: function pointer
 * - arity: remaining arguments needed
 * - n_captures: number of original captures (from lexical scope)
 * - captures: values captured from lexical scope (passed via env pointer)
 * - applied_args: partially applied arguments (passed as regular args)
 */

/* Create a closure with no captured environment (arity 1) */
gn_value gn_make_closure1(void* fn) {
    gn_value fields[3];
    fields[0] = (gn_value)fn;
    fields[1] = GN_INT(1);  /* arity = 1 */
    fields[2] = GN_INT(0);  /* n_captures = 0 */
    return gn_alloc(TAG_CLOSURE, 3, fields);
}

/* Create a closure with no captured environment (arity 2) */
gn_value gn_make_closure2(void* fn) {
    gn_value fields[3];
    fields[0] = (gn_value)fn;
    fields[1] = GN_INT(2);  /* arity = 2 */
    fields[2] = GN_INT(0);  /* n_captures = 0 */
    return gn_alloc(TAG_CLOSURE, 3, fields);
}

/* Create a closure with captured environment */
gn_value gn_make_closure(void* fn, uint32_t arity, uint32_t n_captures, gn_value* captures) {
    /* Closure layout: [fn, arity, n_captures, capture0, capture1, ...] */
    uint32_t n_fields = 3 + n_captures;
    gn_value* fields = (gn_value*)malloc(n_fields * sizeof(gn_value));
    if (!fields) {
        gn_panic("out of memory");
    }
    fields[0] = (gn_value)fn;
    fields[1] = GN_INT(arity);
    fields[2] = GN_INT(n_captures);
    for (uint32_t i = 0; i < n_captures; i++) {
        fields[3 + i] = captures[i];
    }
    gn_value result = gn_alloc(TAG_CLOSURE, n_fields, fields);
    free(fields);
    return result;
}

/*
 * gn_apply - Apply a single argument to a closure
 *
 * Handles partial application for n-arity closures:
 * - If arity == 1: call the function directly with captures + accumulated args + new arg
 * - If arity > 1: return a new closure with arity-1, appending arg to applied args
 */
gn_value gn_apply(gn_value closure, gn_value arg) {
    gn_object* obj = GN_OBJ(closure);
    if (obj->tag != TAG_CLOSURE) {
        gn_panic("apply: not a closure");
    }

    void* fn = (void*)obj->fields[0];
    int64_t arity = GN_UNINT(obj->fields[1]);
    int64_t n_captures = GN_UNINT(obj->fields[2]);
    gn_value* captures = &obj->fields[3];
    gn_value* applied_args = &obj->fields[3 + n_captures];
    uint32_t n_applied = obj->n_fields - 3 - n_captures;

    if (arity == 1) {
        /* Last argument - call the function */
        /* Function signature: fn(env*, arg0, arg1, ...) where env points to captures */
        switch (n_applied) {
            case 0:
                return ((gn_fn1)fn)(captures, arg);
            case 1:
                return ((gn_fn2)fn)(captures, applied_args[0], arg);
            case 2:
                return ((gn_fn3)fn)(captures, applied_args[0], applied_args[1], arg);
            case 3:
                return ((gn_fn4)fn)(captures, applied_args[0], applied_args[1], applied_args[2], arg);
            case 4:
                return ((gn_fn5)fn)(captures, applied_args[0], applied_args[1], applied_args[2], applied_args[3], arg);
            default:
                gn_panic("apply: too many accumulated arguments (max 5)");
        }
    } else {
        /* Partial application - create new closure with arg appended */
        uint32_t new_n_fields = obj->n_fields + 1;
        gn_value* new_fields = (gn_value*)malloc(new_n_fields * sizeof(gn_value));
        if (!new_fields) {
            gn_panic("out of memory");
        }
        new_fields[0] = (gn_value)fn;
        new_fields[1] = GN_INT(arity - 1);
        new_fields[2] = GN_INT(n_captures);
        /* Copy captures */
        for (int64_t i = 0; i < n_captures; i++) {
            new_fields[3 + i] = captures[i];
        }
        /* Copy existing applied args */
        for (uint32_t i = 0; i < n_applied; i++) {
            new_fields[3 + n_captures + i] = applied_args[i];
        }
        /* Append new arg */
        new_fields[3 + n_captures + n_applied] = arg;

        gn_value result = gn_alloc(TAG_CLOSURE, new_n_fields, new_fields);
        free(new_fields);
        return result;
    }
    return GN_UNIT; /* unreachable */
}

gn_value gn_apply2(gn_value closure, gn_value arg1, gn_value arg2) {
    return gn_apply(gn_apply(closure, arg1), arg2);
}
