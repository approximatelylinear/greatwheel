# Guide: Codebase Cleanup

General-purpose instructions for systematically improving code quality in a
Rust workspace. Adapt the scope and language-specific details to your codebase.

---

## Principles

- **Fix what's wrong, don't gold-plate what works.** Cleanup removes defects
  and friction. It does not add features, refactor working code for style
  preferences, or introduce abstractions "for the future."
- **The compiler is your first tool.** In Rust, many cleanup categories that
  require third-party tools in other languages (circular dependencies, unused
  imports, dead code) are already compiler errors or warnings. Don't reinvent
  what `rustc` already enforces.
- **Every change must be justified.** "This could be cleaner" is not
  justification. "This duplication has caused bugs" or "this hides errors in
  production" is. When in doubt, leave it alone.

---

## 1. Deduplication

Find duplicated logic across crates and consolidate where it reduces
complexity. Look for:

- Near-identical error handling patterns across multiple modules
- Duplicated config parsing or validation logic
- Builder patterns or constructors that repeat the same setup
- Copy-pasted utility functions with minor variations

**Guardrails:**
- Three similar lines of code are fine. Three similar 20-line blocks are not.
  Don't create abstractions for trivial duplication.
- Don't merge things that happen to look similar but serve different purposes
  or evolve independently. Duplication is cheaper than the wrong abstraction.
- If consolidating, put shared code in the core/common crate. Don't create
  new utility crates for a single function.

---

## 2. Type Consolidation

Find type definitions that are duplicated or could be shared. Look for:

- Identical or near-identical structs defined in multiple crates
- Raw primitives (`String`, `u64`, `Uuid`) used where a newtype would prevent
  misuse at call sites (e.g., passing an `OrgId` where a `SessionId` is
  expected)
- Type aliases that obscure rather than clarify (e.g.,
  `type Result = std::result::Result<Value, Box<dyn Error>>`)

**Guardrails:**
- Newtypes are worth it when two values have the same underlying type but
  different semantics. They are not worth it for one-off uses.
- Consolidate into the core crate only if the type is used across crate
  boundaries. Crate-internal types stay crate-internal.
- Don't introduce generic type parameters to unify types that are "almost
  the same." If they differ, they're different types.

---

## 3. Dead Code and Unused Dependencies

The Rust compiler already catches unused imports, variables, and unreachable
code (especially with `deny` lints). Focus on what the compiler can't catch:

- **Unused Cargo dependencies.** Run `cargo machete` (heuristic, fast,
  works on stable) or `cargo +nightly udeps` (compilation-based, more
  accurate). Remove what's unused.
- **Orphaned source files.** `.rs` files not declared in any `mod` tree are
  silently ignored by the compiler. Find them with a glob + grep for `mod`
  declarations.
- **Dead feature flags.** Cargo features defined in `Cargo.toml` that no
  code checks for via `#[cfg(feature = "...")]`, or features that are always
  enabled and never optional.
- **`pub` items with no external consumers.** In a workspace, a `pub`
  function in a library crate that no other crate imports is effectively
  dead code. The compiler won't flag it because it's part of the "public
  API." Search for unused `pub` exports across the workspace.

**Guardrails:**
- Before removing a `pub` item, grep the entire workspace. It might be
  used in tests, examples, or integration test crates.
- Don't remove dependencies that are used only via re-export, proc macro,
  or build script — `cargo machete` can false-positive on these.

---

## 4. Crate Dependency Hygiene

Map the workspace dependency graph and identify unnecessary coupling:

- **Heavy dependencies for light use.** A crate pulled in for a single
  function or type. Consider whether the functionality can be inlined or
  replaced with a lighter alternative.
- **Items that belong in core.** Types or traits imported across many crates
  that should live in the shared core crate instead of creating cross-crate
  dependencies.
- **Unnecessary crate boundaries.** Two crates that always change together
  and have no independent consumers might be better as one crate.
- **Transitive dependency weight.** Use `cargo tree -d` to find duplicated
  transitive dependencies (multiple versions of the same crate). Align
  versions where possible.

**Guardrails:**
- Rust prevents circular crate dependencies at compile time. No tool needed.
- Don't merge crates just to reduce count. Crate boundaries that enforce
  separation of concerns are valuable even if they add a few edges to the
  dependency graph.
- Use `cargo tree` and `cargo tree --invert <crate>` to understand the
  actual graph before making changes.

---

## 5. Type Erasure Audit

Find all uses of runtime type erasure and evaluate whether each is justified:

- `dyn Any` / `Box<dyn Any>` — justified at plugin boundaries where the
  core can't know plugin-defined types. Unjustified as a lazy substitute
  for an enum or generic.
- `serde_json::Value` used as an internal type — justified at serialization
  boundaries (API input/output, config parsing). Unjustified when passed
  between Rust functions that both know the concrete type.
- Raw pointers (`*const ()`, `*mut ()`) — justified in FFI. Unjustified
  almost everywhere else.

**Do not touch:**
- `dyn Trait` — this is intentional polymorphism, not type erasure. A
  `Box<dyn Plugin>` or `Arc<dyn Fn(...)>` is correct design.
- `()` (unit type) — this is Rust's void, a real type. It's fine.
- `PhantomData<T>` — this is a compile-time marker, not erasure.

**Guardrails:**
- Type erasure at system boundaries (plugins, FFI, serialization) is
  expected and correct. Don't try to eliminate it.
- If replacing `serde_json::Value` with a concrete type, make sure the
  concrete type is correct for all producers — don't introduce a type that
  only covers 80% of cases.

---

## 6. Error Handling Audit

Find defensive patterns that hide errors or mask bugs:

- **`.unwrap_or_default()`** — silently produces an empty value when
  something failed. Is the failure expected, or is this hiding a bug?
- **`.ok()`** — discards the error. Is the caller intentionally ignoring
  the failure, or did someone avoid dealing with it?
- **`_ => {}` catch-all match arms** — silently drops unhandled variants.
  New enum variants will be silently ignored instead of causing a compile
  error.
- **Fallback values that mask bugs** — returning a default or empty
  collection when a query fails, making the caller think there's no data
  rather than reporting an error.
- **`eprintln!` / `println!` for error reporting** — should use
  `tracing::warn!` or `tracing::error!` so errors are captured by the
  observability stack.

**Replace with:**
- `?` propagation for errors that should reach the caller
- Explicit error variants in the crate's error enum
- `tracing::warn!` at minimum so failures are visible
- Exhaustive matches on internal enums

**Do not add error handling where:**
- The code correctly relies on type-system guarantees (e.g., a `match`
  on a two-variant enum doesn't need a default)
- Framework invariants ensure the condition can't occur (e.g., an Axum
  extractor that has already validated input)
- The `.unwrap()` is in a test — tests should panic on failure

---

## 7. Dead Codepaths and Legacy Code

Find code that is no longer reachable or references removed systems:

- **`#[allow(dead_code)]`** — why is this suppressed? Either the code is
  used (remove the annotation) or it's dead (remove the code).
- **`todo!()` / `unimplemented!()`** — is this a real planned feature or
  an abandoned stub? If planned, file an issue and leave it. If abandoned,
  remove it or replace with a proper error.
- **`#[cfg(...)]` for dead features** — code gated behind features or
  flags that no longer exist or are never enabled.
- **Commented-out code blocks** — remove. Git has the history.
- **References to removed systems** — imports, config fields, or
  documentation that mentions components that no longer exist.

**Guardrails:**
- Check `git log` if intent is unclear. A `todo!()` from last week is
  probably in-progress work. A `todo!()` from six months ago with no
  related commits is abandoned.
- Don't remove `#[allow(dead_code)]` on items that are used only via
  FFI, proc macros, or conditional compilation that isn't active in
  your default build.

---

## 8. Comment and Documentation Quality

Remove or fix comments that reduce signal-to-noise:

**Remove:**
- Comments that restate what the code does
  (`// increment counter` above `counter += 1`)
- Comments referencing past states ("previously we used X",
  "this replaced the old Y", "TODO: migrate from Z")
- Markers or boilerplate left by AI code generation
- Section dividers or decorative comments that add no information
- Commented-out code (git has the history)

**Keep:**
- Comments explaining *why* — non-obvious business logic, performance
  tradeoffs, safety invariants, links to issues or specs
- `// SAFETY:` comments on unsafe blocks (required by convention)
- Module-level doc comments (`//!`) that explain the module's role
  in the system

**Fix:**
- Outdated comments that describe behavior the code no longer has
- Comments that explain a workaround without linking to the issue
  that tracks fixing it

**Guardrails:**
- Write for someone with junior-level experience in the language
  joining the project. They know the language; they don't know the
  system.
- Be concise. A comment longer than the code it describes is almost
  always too long.
- If you're unsure whether a comment is helpful, remove it and see
  if the code is clear without it. If not, the code needs refactoring,
  not more comments.
