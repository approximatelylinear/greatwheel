# Guide: Codebase Optimization

General-purpose instructions for improving runtime performance in a Rust
codebase. Focus on measurable impact, not micro-optimization.

---

## Principles

- **Profile first.** `cargo flamegraph`, `perf`, or `tokio-console` will
  tell you where time actually goes. Don't optimize based on intuition —
  the bottleneck is almost never where you think it is.
- **Optimize hot paths, leave cold paths alone.** Plugin init, config
  parsing, server startup — these run once. Readability wins. The text
  processing pipeline that runs per-request is where allocation savings
  compound.
- **The biggest wins are algorithmic, not mechanical.** Switching from
  `O(n^2)` to `O(n log n)` dwarfs any number of avoided allocations.
  Check algorithmic complexity before reaching for `Cow`.

---

## 1. Allocation Avoidance

The most broadly applicable category. Every heap allocation (`String`,
`Vec`, `Box`, `HashMap`) involves a call to the allocator, and every
unnecessary clone doubles the cost.

### Cow<'a, str> / Cow<'a, [T]>

Borrow when the data passes through unmodified, clone only when mutation
is needed. Most useful in text processing where most strings don't need
transformation:

```rust
fn normalize_query(input: &str) -> Cow<'_, str> {
    if input.contains('\n') {
        Cow::Owned(input.replace('\n', " "))
    } else {
        Cow::Borrowed(input)  // no allocation
    }
}
```

**Where to apply:** any function that takes a string/slice, sometimes
modifies it, and returns it. Common in normalization, sanitization, and
formatting pipelines.

**Where not to apply:** if the function always modifies the input, just
return `String`. `Cow` adds cognitive overhead for no benefit when the
borrowed path is never taken.

### Arc<str> for Shared Immutable Strings

Strings that are created once and shared across threads (plugin names,
capability strings, function names, error messages). `Arc<str>` is one
allocation + cheap refcount bumps on clone, vs. `String::clone()` which
allocates and copies every time:

```rust
// Before: every clone copies the string
name: String,

// After: clones bump a refcount
name: Arc<str>,
```

**Where to apply:** identifiers, keys, and names stored in registries
or passed to multiple consumers. Anything that lives for the duration
of the process and is read-only after creation.

### Borrow at API Boundaries

Functions should accept the most general borrowed form of their input.
Don't force callers to allocate:

```rust
// Before: caller must own a String
fn search(&self, query: String, k: usize) -> Vec<Result>

// After: caller can pass &str, String, or Cow
fn search(&self, query: &str, k: usize) -> Vec<Result>
```

Similarly, accept `&[T]` instead of `Vec<T>`, `&Path` instead of
`PathBuf`, and `impl AsRef<str>` when multiple string types should
be accepted.

### Pre-allocate Collections

When the size is known or estimable, allocate once:

```rust
// Before: starts at 0, reallocates at 1, 2, 4, 8, 16...
let mut results = Vec::new();

// After: one allocation
let mut results = Vec::with_capacity(expected_count);
```

Also applies to `String::with_capacity`, `HashMap::with_capacity`,
and `HashSet::with_capacity`. The estimate doesn't need to be exact —
even a rough guess avoids most reallocations.

### SmallVec for Usually-Small Collections

Stack-allocate collections that almost always have fewer than N items
but occasionally grow beyond:

```rust
use smallvec::SmallVec;

// Stack-allocated for up to 4 items, heap-allocated beyond
pub provides: SmallVec<[String; 4]>,
```

**Where to apply:** struct fields and function-local collections where
the typical size is small and known. Plugin manifests, search result
lists, capability sets.

**Where not to apply:** large collections, collections with unknown
size, or collections that are already heap-allocated for other reasons
(e.g., deserialized from JSON).

---

## 2. Clone Audit

Every `.clone()` on a heap type is an allocation. Systematically audit
clone calls and replace where possible:

| Pattern | Replacement |
|---------|-------------|
| `x.clone()` passed to a function that only reads it | `&x` — borrow instead |
| `String` cloned into multiple owners | `Arc<str>` — clone bumps a refcount |
| `Vec<T>` cloned to iterate | `&[T]` — borrow the slice |
| `value.clone()` moved into an `Arc`-wrapped closure | `Arc::clone(&value)` — explicit, cheap |
| `hashmap.clone()` to avoid borrow conflicts | Restructure to separate the mutable and immutable parts |

**How to find them:** `grep -n '\.clone()'` across the workspace. Focus
on hot paths — clones in init code or tests are fine.

**Guardrails:**
- Don't fight the borrow checker by introducing `unsafe` to avoid a clone.
  If the borrow checker demands a clone, the code structure needs changing,
  not the safety guarantees.
- Some clones are correct and necessary (e.g., cloning into a spawned task
  that outlives the current scope). Don't remove these.

---

## 3. Serialization

JSON serialization and deserialization is expensive relative to Rust function
calls. Audit the serde path:

### Avoid Unnecessary Round-Trips

If both sides of an interface know the concrete type, don't serialize to
`serde_json::Value` and back:

```rust
// Expensive: serialize to JSON, deserialize back
let value = serde_json::to_value(&result)?;
let parsed: MyType = serde_json::from_value(value)?;

// If both sides are Rust, just pass the type directly
```

This is common at internal boundaries that were designed for a polyglot
interface (e.g., host function dispatch) but are also used for Rust-to-Rust
calls. Consider a typed fast-path alongside the JSON path.

### Stream Serialization

Write directly to the output buffer instead of building an intermediate
string:

```rust
// Before: allocates a String, then writes it
let json = serde_json::to_string(&response)?;
writer.write_all(json.as_bytes())?;

// After: writes directly to the buffer
serde_json::to_writer(&mut writer, &response)?;
```

### Derive Efficiently

- Use `#[serde(borrow)]` on `Cow<'a, str>` fields in deserialized structs
  to borrow from the input buffer instead of allocating.
- Use `#[serde(skip)]` on fields that are computed, not serialized.
- Consider `simd-json` as a drop-in replacement for `serde_json` if JSON
  parsing is a measured bottleneck.

---

## 4. Compute

### Compile Regexes Once

If a regex is used per-request, compile it once at startup:

```rust
use std::sync::LazyLock;
use regex::Regex;

static ENTITY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#""[^"]+"|[A-Z][a-z]+ [A-Z][a-z]+"#).unwrap()
});
```

A `Regex::new()` call compiles the pattern into an NFA. Doing this per-call
is orders of magnitude slower than reusing a compiled regex.

### Prefer String Methods Over Regex

For fixed-string operations, `str` methods are faster than regex:

```rust
// Regex (compiles a pattern engine):
Regex::new("error")?.is_match(line)

// String method (direct byte comparison):
line.contains("error")
```

Use regex for patterns. Use `contains`, `starts_with`, `ends_with`,
`split`, `trim`, `replace` for literal strings.

### Avoid Repeated Hash Lookups

When reading and writing the same key in a `HashMap`, use the entry API
to hash once:

```rust
// Before: hashes "key" twice
if !map.contains_key("key") {
    map.insert("key", compute_value());
}

// After: hashes "key" once
map.entry("key").or_insert_with(|| compute_value());
```

### Use Iterators Over Collect

If you're collecting into a `Vec` just to iterate it again, chain the
iterators instead:

```rust
// Before: allocates an intermediate Vec
let filtered: Vec<_> = items.iter().filter(|x| x.active).collect();
for item in &filtered { process(item); }

// After: no intermediate allocation
for item in items.iter().filter(|x| x.active) {
    process(item);
}
```

---

## 5. Async and I/O

In a system with LLM calls and database queries, I/O dominates wall-clock
time. Optimizing I/O patterns often yields bigger wins than all allocation
savings combined.

### Batch Database Queries

Replace N round-trips with one:

```rust
// Before: N round-trips
for id in &ids {
    let row = sqlx::query!("SELECT ... WHERE id = $1", id)
        .fetch_one(&pool).await?;
}

// After: 1 round-trip
let rows = sqlx::query!("SELECT ... WHERE id = ANY($1)", &ids[..])
    .fetch_all(&pool).await?;
```

Also applies to inserts (`INSERT ... VALUES ... ON CONFLICT`),
updates, and deletes. Every round-trip adds network latency.

### Parallelize Independent I/O

Use `tokio::join!` or `futures::join_all` for independent async operations:

```rust
// Before: sequential, total time = A + B + C
let a = fetch_a().await?;
let b = fetch_b().await?;
let c = fetch_c().await?;

// After: parallel, total time = max(A, B, C)
let (a, b, c) = tokio::try_join!(fetch_a(), fetch_b(), fetch_c())?;
```

### Stream Large Results

Don't load entire result sets into memory when processing row-by-row:

```rust
// Before: loads all rows into Vec
let rows = sqlx::query!("SELECT ...").fetch_all(&pool).await?;
for row in rows { process(row); }

// After: processes rows as they arrive
let mut stream = sqlx::query!("SELECT ...").fetch(&pool);
while let Some(row) = stream.try_next().await? {
    process(row);
}
```

### Connection Pool Sizing

Ensure pool sizes match concurrency. Too few connections = requests queue.
Too many = connection overhead and database contention. A reasonable
starting point is 2-4x the number of CPU cores for Postgres.

### Buffer Reuse with Bytes/BytesMut

For network I/O, `bytes::Bytes` is a reference-counted byte buffer that
avoids copying when slicing. Axum and reqwest use it internally — make
sure you're not converting `Bytes` → `String` → `Bytes` at intermediate
layers:

```rust
// Before: copies bytes into a String, then back to bytes for response
let body = String::from_utf8(bytes.to_vec())?;
let response_bytes = body.into_bytes();

// After: work with Bytes directly if content is passed through
let response_bytes = bytes;  // zero-copy
```

---

## 6. Data Structure Selection

### HashMap vs BTreeMap

- `HashMap`: O(1) average lookup. Use for large maps with frequent lookups.
- `BTreeMap`: O(log n) lookup but sorted keys and cache-friendly iteration.
  Use when you need sorted output or when the map is small (< ~20 entries)
  where cache locality wins.

### HashSet for Membership Checks

If you're checking `vec.contains(x)` in a loop, it's O(n) per check.
Convert to `HashSet` for O(1):

```rust
// Before: O(n * m)
for item in &items {
    if allowed_names.contains(&item.name) { ... }
}

// After: O(n) — build set once, O(1) lookups
let allowed: HashSet<&str> = allowed_names.iter().map(|s| s.as_str()).collect();
for item in &items {
    if allowed.contains(item.name.as_str()) { ... }
}
```

### FxHashMap for String Keys

The default `HashMap` uses SipHash (resistant to hash-flooding attacks).
If your keys are not from untrusted input (internal identifiers, plugin
names), `rustc_hash::FxHashMap` is significantly faster:

```rust
use rustc_hash::FxHashMap;

let mut registry: FxHashMap<String, Handler> = FxHashMap::default();
```

**Only use when:** keys are not from user input and hash-flooding is not
a concern. For user-facing APIs, keep the default hasher.

---

## 7. What Not to Optimize

- **Code that runs once** — startup, shutdown, config parsing, plugin init.
  Readability wins over performance.
- **Code behind I/O** — if a function allocates 10 strings but then makes
  an HTTP call that takes 200ms, the allocations are invisible.
- **Code in tests** — tests should be clear, not fast.
- **Micro-optimizations without measurement** — replacing `to_string()`
  with `to_owned()` saves nothing measurable. Don't.
- **Unsafe for performance** — almost never worth it in application code.
  The compiler optimizes safe code well. `unsafe` is for FFI and
  data-structure internals, not for shaving nanoseconds.

---

## 8. Measurement Tools

| Tool | Purpose |
|------|---------|
| `cargo flamegraph` | CPU profiling — where is time spent? |
| `cargo bench` (criterion) | Micro-benchmarks — is this change faster? |
| `perf stat` / `perf record` | Low-level CPU counters, cache misses |
| `tokio-console` | Async task profiling — where are tasks blocked? |
| `heaptrack` / `DHAT` | Heap allocation profiling — where are allocations? |
| `cargo bloat` | Binary size — what's taking space? |
| `cargo tree -d` | Duplicate transitive dependencies |
| `/usr/bin/time -v` | Peak RSS, wall-clock time for end-to-end runs |

Always measure before and after. "I think this is faster" is not evidence.
A benchmark that shows a 2% improvement on a cold path is not worth the
added complexity.

---

## 9. Optimization Prompt

Structured prompt for running parallel optimization agents across a Rust
codebase. Each agent researches their area, writes a critical assessment,
and implements high-confidence improvements.

```
I want to optimize the performance of this codebase.

This is a complex task, so we'll need 8 subagents, one for each area below.

Each subagent should: profile or audit their area, write a critical
assessment of the current state with evidence (not intuition), and then
implement all high-confidence improvements. Do not optimize cold paths
(startup, config, plugin init, tests). Focus on code that runs per-request
or per-turn.

1. **Allocation audit.** Find functions that allocate (`String`, `Vec`,
   `Box`) on hot paths where they could borrow instead. Look for functions
   that accept `String` where `&str` would work, functions that return
   `String` where `Cow<'_, str>` would avoid allocation on the common path,
   and `Vec::new()` where `Vec::with_capacity()` is feasible. Don't
   introduce `Cow` where the function always modifies — that adds complexity
   for zero benefit.

2. **Clone audit.** Grep for `.clone()` across the workspace. For each
   clone on a hot path: determine whether it can be replaced with a borrow,
   an `Arc`, or a restructuring that avoids the need. Classify each as
   necessary (spawned task, borrow checker constraint) or unnecessary
   (could borrow, could share via Arc). Implement removals for the
   unnecessary ones. Don't introduce `unsafe` to avoid a clone — restructure
   instead.

3. **Shared string optimization.** Find strings that are created once and
   read many times from multiple threads — identifiers, registry keys,
   capability names, error messages. Replace `String` with `Arc<str>` where
   the string is immutable after creation and cloned into multiple owners.
   Also look for string interning opportunities where the same string
   values are repeatedly allocated (e.g., repeated plugin names or event
   names in logs).

4. **Serialization audit.** Find all serde serialization/deserialization on
   hot paths. Look for: unnecessary `serde_json::Value` round-trips where
   both sides know the concrete type, `to_string` + write where `to_writer`
   would avoid an intermediate allocation, and missing `#[serde(borrow)]`
   on `Cow` fields. Check whether the host function dispatch path can use a
   typed fast-path for Rust-to-Rust calls alongside the JSON path used for
   the Python bridge.

5. **Async I/O patterns.** Find sequential `.await` calls that could run in
   parallel via `tokio::try_join!`. Find N+1 query patterns (queries in
   loops) that could be batched with `ANY($1)` or bulk inserts. Find places
   where entire result sets are loaded into `Vec` but could be streamed via
   `fetch` instead of `fetch_all`. Check connection pool sizing relative to
   expected concurrency.

6. **Compute efficiency.** Find regexes compiled per-call that should be
   `LazyLock<Regex>` statics. Find `HashMap` lookups using `contains_key`
   + `insert` that should use the entry API. Find `.collect()` into
   intermediate `Vec`s that are only iterated — chain iterators instead.
   Find linear searches (`vec.contains()`, `.find()`) in loops that should
   use `HashSet` for O(1) membership. Don't optimize compute that's behind
   I/O — it's invisible next to network latency.

7. **Data structure selection.** Audit `HashMap` usage: replace with
   `FxHashMap` where keys are internal (not user input) and hash-flooding
   is not a concern. Find small collections (typically < 8 items) stored
   as `Vec` or `HashMap` on hot paths where `SmallVec` or inline arrays
   would avoid heap allocation. Check for `BTreeMap` opportunities where
   sorted iteration is needed. Look for `Vec<T>` used as a set (with
   `.contains()`) that should be `HashSet<T>`.

8. **Buffer and I/O efficiency.** Find `Bytes` → `String` → `Bytes`
   conversions in network/response paths that could work with `Bytes`
   directly. Find places where response bodies are fully buffered in memory
   that could stream instead. Check for unnecessary `.to_vec()` or
   `.to_owned()` calls on byte slices that could use zero-copy slicing.
   Audit large string building (concatenation in loops) that should use
   `String::with_capacity` or `write!` to a pre-sized buffer.

Important constraints for all subagents:
- Do not optimize code that runs once (startup, shutdown, config, init).
- Do not introduce `unsafe` for performance gains.
- Do not change public API signatures unless the crate is internal.
- Every change must preserve existing behavior — run tests after each
  modification.
- If unsure whether a path is hot, check before optimizing. Measure,
  don't guess.
```
