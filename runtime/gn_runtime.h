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
#include <pthread.h>
#include <stdatomic.h>

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
#define GN_INT_NEG(a)     GN_INT(-GN_UNINT(a))

/* Division and modulo - panic with clear message on zero divisor */
gn_value gn_int_div(gn_value a, gn_value b);
gn_value gn_int_mod(gn_value a, gn_value b);

/* Safe division/modulo - return Result Int String */
gn_value gn_safe_div(gn_value a, gn_value b);
gn_value gn_safe_mod(gn_value a, gn_value b);

/* Helper to create Result types */
gn_value gn_ok(gn_value v);
gn_value gn_err(gn_value v);
gn_value gn_some(gn_value v);
gn_value gn_none(void);

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
gn_value gn_print(gn_value* env, gn_value v);
gn_value gn_println(void);

/* Panic/error - returns gn_value for use in expressions (never actually returns) */
gn_value gn_panic(const char* msg);
gn_value gn_panic_str(gn_value msg);  /* panic with String value */

/* Assert - panics if condition is false */
gn_value gn_assert(gn_value cond);

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
gn_value gn_string_to_upper(gn_value s);
gn_value gn_string_to_lower(gn_value s);
gn_value gn_string_trim(gn_value s);
gn_value gn_string_replace(gn_value old_str, gn_value new_str, gn_value s);
gn_value gn_string_starts_with(gn_value prefix, gn_value s);
gn_value gn_string_ends_with(gn_value suffix, gn_value s);
gn_value gn_string_contains(gn_value needle, gn_value haystack);
gn_value gn_string_char_at(gn_value index, gn_value s);
gn_value gn_string_to_chars(gn_value s);
gn_value gn_chars_to_string(gn_value chars);
gn_value gn_bytes_to_string(gn_value bytes);
gn_value gn_char_to_int(gn_value c);

/* I/O operations */
gn_value gn_io_print(gn_value s);
gn_value gn_io_read_line(gn_value unit);  /* unit arg ignored */

/* File operations */
gn_value gn_file_open(gn_value path, gn_value mode);
gn_value gn_file_read(gn_value handle, gn_value max_bytes);
gn_value gn_file_write(gn_value handle, gn_value data);
gn_value gn_file_close(gn_value handle);

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

/* Dictionary operations (association list) */
gn_value gn_Dict_new(gn_value unit);
gn_value gn_Dict_insert(gn_value key, gn_value value, gn_value dict);
gn_value gn_Dict_get(gn_value key, gn_value dict);
gn_value gn_Dict_remove(gn_value key, gn_value dict);
gn_value gn_Dict_contains(gn_value key, gn_value dict);
gn_value gn_Dict_keys(gn_value dict);
gn_value gn_Dict_values(gn_value dict);
gn_value gn_Dict_size(gn_value dict);
gn_value gn_Dict_isEmpty(gn_value dict);
gn_value gn_Dict_toList(gn_value dict);
gn_value gn_Dict_fromList(gn_value list);
gn_value gn_Dict_merge(gn_value d1, gn_value d2);
gn_value gn_Dict_getOrDefault(gn_value default_val, gn_value key, gn_value dict);

/* ============================================================================
 * Fiber Scheduler (Cooperative Concurrency)
 *
 * Single-threaded cooperative scheduler for fibers (lightweight processes).
 * Fibers communicate via rendezvous channels and effects.
 * ============================================================================ */

/* Forward declaration for effect types (defined below) */
typedef uint32_t gn_op_id;

/* Tags for fiber-related values */
#define TAG_FIBER_HANDLE 0xFFFF0010
#define TAG_CHANNEL      0xFFFF0011

/* Special effect ID for built-in async operations */
#define EFFECT_ASYNC     0xFFFFFFFF

/* Async operation IDs */
#define ASYNC_OP_SPAWN   0
#define ASYNC_OP_JOIN    1
#define ASYNC_OP_YIELD   2
#define ASYNC_OP_CHAN_NEW 3
#define ASYNC_OP_SEND    4
#define ASYNC_OP_RECV    5

/* Fiber states */
typedef enum {
    FIBER_READY,           /* In run queue, ready to execute */
    FIBER_RUNNING,         /* Currently executing */
    FIBER_BLOCKED_JOIN,    /* Waiting for another fiber to complete */
    FIBER_BLOCKED_SEND,    /* Waiting to send on channel */
    FIBER_BLOCKED_RECV,    /* Waiting to receive on channel */
    FIBER_COMPLETED,       /* Done, has result */
    FIBER_ABORTED          /* Failed */
} gn_fiber_state;

/* How to resume a fiber */
typedef enum {
    FIBER_CONT_START,      /* Initial spawn, run thunk */
    FIBER_CONT_RESUME      /* Resume with value */
} gn_fiber_cont_kind;

/* Forward declarations */
struct gn_fiber;
struct gn_channel;

/* Fiber structure */
typedef struct gn_fiber {
    uint64_t id;
    gn_fiber_state state;
    gn_fiber_cont_kind cont_kind;
    gn_value cont_value;           /* Thunk (START) or resume value (RESUME) */
    gn_value saved_k;              /* CPS continuation with handler context */
    gn_value result;               /* Result when COMPLETED */

    /* For join semantics */
    struct gn_fiber* join_target;  /* Fiber we're waiting to join */
    struct gn_fiber** joiners;     /* Fibers waiting for us to complete */
    size_t joiner_count;
    size_t joiner_capacity;

    /* For channel blocking */
    struct gn_channel* blocked_channel;
    gn_value send_value;           /* Value being sent (for blocked senders) */
} gn_fiber;

/* Growable run queue (FIFO) */
typedef struct gn_run_queue {
    gn_fiber** data;
    size_t head;                   /* Index of first element */
    size_t count;                  /* Number of elements */
    size_t capacity;               /* Allocated size (doubles when full) */
} gn_run_queue;

/* Channel waiter (for blocking send/recv) */
typedef struct gn_waiter {
    gn_fiber* fiber;
    gn_value value;                /* For senders: value to send */
    struct gn_waiter* next;
} gn_waiter;

/* Channel (rendezvous - no buffer) */
typedef struct gn_channel {
    uint64_t id;
    gn_waiter* waiting_senders;
    gn_waiter* waiting_receivers;
    bool closed;
} gn_channel;

/* Scheduler state */
typedef struct gn_scheduler {
    gn_fiber** all_fibers;         /* Array of all fibers by ID */
    size_t fiber_count;
    size_t fiber_capacity;
    uint64_t next_fiber_id;

    gn_run_queue ready_queue;
    gn_fiber* main_fiber;          /* Main fiber (for final result) */

    /* Channel tracking */
    gn_channel** all_channels;
    size_t channel_count;
    size_t channel_capacity;
    uint64_t next_channel_id;

    /* Threading support (Phase 2) */
    pthread_mutex_t lock;          /* Protects all scheduler state */
    pthread_cond_t work_available; /* Signals when work is available */
    pthread_t* workers;            /* Worker thread handles */
    size_t n_workers;              /* Number of worker threads */
    atomic_bool shutdown;          /* Shutdown flag */

    /* Per-thread current fiber (thread-local in implementation) */
    /* Note: current fiber is now thread-local, not in scheduler */
} gn_scheduler;

/* Scheduler API */
void gn_sched_init(void);
void gn_sched_shutdown(void);
gn_value gn_sched_run(gn_value main_thunk);

/* Fiber operations */
gn_fiber* gn_fiber_create(gn_value thunk);
void gn_fiber_add_joiner(gn_fiber* target, gn_fiber* joiner);
gn_value gn_make_fiber_handle(uint64_t id);
uint64_t gn_get_fiber_id(gn_value handle);

/* Run queue operations */
void gn_run_queue_push(gn_run_queue* q, gn_fiber* f);
gn_fiber* gn_run_queue_pop(gn_run_queue* q);

/* Channel operations */
gn_channel* gn_channel_create(void);
gn_channel* gn_get_channel(gn_value v);
gn_value gn_make_channel_handle(gn_channel* ch);

/* Async effect handler (called by gn_perform for EFFECT_ASYNC) */
gn_value gn_handle_async(gn_op_id op, uint32_t n_args, gn_value* args, gn_value cont);

/* Fiber/Channel wrapper functions (non-CPS synchronous mode) */
gn_value gn_Fiber_spawn(gn_value thunk);
gn_value gn_Fiber_join(gn_value handle);
gn_value gn_Fiber_yield(gn_value unit);
gn_value gn_Channel_new(gn_value unit);
gn_value gn_Channel_send(gn_value handle, gn_value value);
gn_value gn_Channel_recv(gn_value handle);

/* ============================================================================
 * Algebraic Effects Runtime Support
 *
 * Implements handler stack and continuation capture/resume for algebraic effects.
 * Uses CPS-style where effectful functions take continuation parameters.
 * ============================================================================ */

/* Effect and operation identifiers */
typedef uint32_t gn_effect_id;
/* gn_op_id is forward-declared in scheduler section above */

/* Handler structure - installed when entering a 'handle' block */
typedef struct gn_handler {
    gn_effect_id effect;           /* Which effect this handles */
    gn_value return_fn;            /* Return handler: (value, outer_k) -> result */
    uint32_t n_ops;                /* Number of operations */
    gn_value* op_fns;              /* Array of op handlers: (args..., k, outer_k) -> result */
    gn_value outer_cont;           /* Continuation to resume when handler completes */
    struct gn_handler* parent;     /* Parent handler (for nested handlers) */
} gn_handler;

/* Continuation structure - captured when performing an effect */
typedef struct gn_continuation {
    uint32_t rc;                   /* Reference count */
    uint32_t tag;                  /* TAG_CONTINUATION */
    gn_handler* captured_stack_top; /* Top of handler stack when captured (for deep semantics) */
    gn_handler* stack_bottom;       /* Where the stack was after popping (to know when to stop restoring) */
    gn_value resume_fn;            /* Function to call to resume: (value) -> result */
} gn_continuation;

/* Handler stack operations */
void gn_push_handler(gn_handler* h);
gn_handler* gn_pop_handler(void);
gn_handler* gn_find_handler(gn_effect_id effect);
gn_handler* gn_current_handler(void);

/* Continuation operations */
gn_value gn_make_continuation(gn_value resume_fn, gn_handler* stack_top, gn_handler* stack_bottom);
gn_value gn_resume(gn_value cont, gn_value value);
gn_value gn_resume_multi(gn_value cont, gn_value value);  /* Multi-shot resume (copies cont) */

/* Effect operations */
gn_value gn_perform(gn_effect_id effect, gn_op_id op,
                    uint32_t n_args, gn_value* args,
                    gn_value current_k);

/* Handler creation helper */
gn_handler* gn_create_handler(gn_effect_id effect, gn_value return_fn,
                              uint32_t n_ops, gn_value* op_fns,
                              gn_value outer_cont);
void gn_free_handler(gn_handler* h);

#endif /* GN_RUNTIME_H */
