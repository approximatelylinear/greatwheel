# Design: Apps — Composing Plugins into Deployable Agents

**Status:** Proposal
**Date:** 2026-04-12
**Depends on:** [Plugin & Lifecycle Framework](design-plugin-framework.md) (Phases 1-2 done, 3+ pending)

---

## 0. Problem

The plugin system answers "how does someone extend Greatwheel without forking
it?" But extension is not the same as application. A plugin adds a capability;
an **app** composes capabilities into a purpose-built agent that serves a
specific audience through a specific surface.

Today, building "an Acme support bot" on Greatwheel requires:

1. Writing the relevant plugins (KB, CRM connector, guardrails) — this works.
2. Hardcoding their registration in `gw-server/main.rs` — not reusable.
3. Defining the agent persona somewhere — no standard format.
4. Exposing it to users — no app-scoped routing or UI.
5. Isolating it from other apps — no scoping mechanism.

There is no unit of deployment between "a single plugin" and "the entire
server." The apps layer fills this gap.

---

## 1. What Is an App?

An app is a **manifest that composes existing primitives** — plugins, an agent
definition, capabilities, and a channel surface — into something independently
deployable. It is not new runtime code. It is configuration that the engine
interprets.

```
┌────────────────────────────────────┐
│  App: "acme-support"               │
│                                    │
│  ┌──────────┐  ┌────────────────┐  │
│  │ Plugins  │  │ Agent Config   │  │
│  │ - kb     │  │ - persona      │  │
│  │ - crm    │  │ - model        │  │
│  │ - guard  │  │ - temperature  │  │
│  └──────────┘  └────────────────┘  │
│                                    │
│  ┌──────────┐  ┌────────────────┐  │
│  │ Caps     │  │ Surface        │  │
│  │ allow:   │  │ - HTTP /acme   │  │
│  │  kb.read │  │ - WS /ws/acme  │  │
│  │ deny:    │  │ - UI: chat     │  │
│  │  crm.wrt │  │                │  │
│  └──────────┘  └────────────────┘  │
└────────────────────────────────────┘
```

An app does **not** own its plugins. Plugins are shared infrastructure. An app
declares which plugins it uses and how they're configured for this context.

---

## 2. Design Principles

**Apps are data, not code.** An app manifest is a TOML file interpreted by the
engine. No Rust code needed to define an app. This is the layer where
non-engine-developers operate.

**Plugins remain shared, apps scope their use.** Two apps can both use the KB
plugin. The plugin is loaded once; the app provides per-app configuration
(which index, which capabilities) through scoped config sections.

**Progressive surface complexity.** An app can be API-only (no UI), use a
built-in chat UI (no frontend code), extend the built-in UI with plugin
widgets, or serve a fully custom frontend. Most apps should never need to
write frontend code.

**Capability enforcement is the isolation boundary.** Apps don't get process
isolation or separate memory spaces. Isolation comes from the capability
system: an app's agent can only call host functions it's been granted access
to. This is lightweight and sufficient — we're not running untrusted code
(agents run in ouros sandboxes, plugins are compiled in).

---

## 3. App Manifest

```toml
# apps/acme-support.toml

[app]
name = "acme-support"
description = "Customer support agent for Acme Corp"
version = "1.0.0"

[agent]
persona = """
You are a helpful support agent for Acme Corp. You have access to the
product knowledge base and can look up customer records. Always be polite
and concise. If you cannot resolve an issue, offer to escalate.
"""
model = "qwen3.5:9b"
temperature = 0.3
max_turns = 20                     # safety limit per session

# Plugin configuration scoped to this app.
# Each key must match a plugin loaded by the engine.
# Values are merged with (and override) the global [plugins.X] config.
[plugins.kb]
index_path = "/data/acme-docs"
search_k = 5

[plugins.crm]
source = "salesforce"
instance = "acme.my.salesforce.com"

[plugins.guardrails]
block_patterns = ["SSN", "credit card"]
max_tokens_per_session = 50000

# What the agent is allowed to do.
[capabilities]
allow = ["kb.read", "crm.read", "crm.lookup_customer"]
deny  = ["crm.write", "crm.delete"]

# How users reach this app.
[surface]
type = "chat"                      # built-in chat UI
path = "/apps/acme-support"        # HTTP mount point

[surface.branding]
title = "Acme Support"
welcome = "Hi! How can I help with your Acme account?"
theme_color = "#1a73e8"

[surface.suggestions]
prompts = [
    "Check my order status",
    "I need to return something",
    "Talk to a human",
]
```

### Manifest Sections

| Section | Required | Purpose |
|---------|----------|---------|
| `[app]` | Yes | Identity: name, description, version |
| `[agent]` | Yes | Agent behavior: persona, model, temperature, limits |
| `[plugins.*]` | No | Per-app plugin config (merged with global config) |
| `[capabilities]` | No | Allow/deny list for host functions (default: all allowed) |
| `[surface]` | No | How users interact (API-only if omitted) |

---

## 4. Surface Types

An app's surface defines how it's exposed to users. Four levels, progressively
more custom:

### Level 0: API-only

No `[surface]` section. The app is accessible only via the engine's session
API. The caller manages their own UI.

```
POST /api/v1/apps/acme-support/sessions
POST /api/v1/apps/acme-support/sessions/{id}/messages
GET  /api/v1/apps/acme-support/sessions/{id}/events  (SSE)
```

### Level 1: Built-in Chat UI

```toml
[surface]
type = "chat"
path = "/apps/acme-support"
```

The engine serves a generic chat shell at the mount path. The shell is a
single-page application (shipped with the engine) that adapts to the app's
branding, welcome message, and suggested prompts. No frontend code needed.

The chat UI connects to the app's session API and renders the conversation.
It supports:
- Streaming responses (SSE)
- Markdown rendering
- Code blocks
- Typing indicators (via lifecycle events)

### Level 2: Chat UI + Plugin Widgets

Plugins can contribute UI fragments to named slots in the chat shell:

```toml
[surface]
type = "chat"
path = "/apps/acme-support"

[[surface.panels]]
slot = "sidebar"
plugin = "kb"
widget = "browser"          # KB plugin's built-in "browser" widget

[[surface.panels]]
slot = "header"
plugin = "crm"
widget = "customer-card"    # CRM plugin renders a customer info card
```

Widgets are served by the plugin via `register_routes` (Phase 3 of the plugin
framework). The chat shell loads them in iframes or via a fragment protocol.
The engine handles the plumbing — the app manifest just declares which widgets
go where.

### Level 3: Custom Frontend

For apps that need a fully custom UI (dashboards, multi-panel workflows,
visual editors):

```toml
[surface]
type = "custom"
static_dir = "./apps/acme-support/dist"    # serve a built SPA
path = "/apps/acme-support"
```

The engine serves the static files at the mount path. The SPA uses the
standard session API for conversation. This is an escape hatch — most apps
should not need it.

---

## 5. App Lifecycle

### Loading

```
1. Engine reads apps/ directory (or [apps] config section)
2. For each app manifest:
   a. Validate: required plugins are loaded, capabilities are satisfiable
   b. Merge plugin configs (global + app-scoped overrides)
   c. Build per-app CallContext template (org, capabilities, limits)
   d. Mount routes (API + surface)
3. Emit AppLoaded event for each app
```

### Session Creation

When a session is created for an app, the engine:

1. Creates a `CallContext` with the app's capability allow/deny list
2. Builds the system prompt from the agent persona + available host functions
   (filtered to allowed capabilities)
3. Starts the conversation loop with the app's model and temperature

The session is **scoped to the app**. Host function calls are checked against
the app's capability list before dispatch.

### Request Routing

```
GET /apps/acme-support           → serve chat UI (if surface.type = "chat")
POST /api/v1/apps/acme-support/sessions        → create session
POST /api/v1/apps/acme-support/sessions/{id}/messages → send message
GET  /api/v1/apps/acme-support/sessions/{id}/events   → SSE stream

GET /apps/research-tool          → serve different app
POST /api/v1/apps/research-tool/sessions       → create session (different plugins, caps)
```

Each app gets its own route namespace. The engine's existing routes remain
at `/api/v1/sessions/*` for direct, app-less access.

---

## 6. Capability Enforcement

This is the mechanism deferred in the plugin framework (noted as TODO in
`host_fn_router.rs`). The apps layer makes it necessary.

### How It Works

1. Plugin registers a host function with an optional capability:
   ```rust
   ctx.register_host_fn_async("crm.lookup_customer", Some("crm.read"), handler);
   ```

2. App manifest declares allowed capabilities:
   ```toml
   [capabilities]
   allow = ["crm.read"]
   deny  = ["crm.write"]
   ```

3. At session creation, the engine builds a `CapabilitySet` from the app's
   allow/deny lists.

4. When the agent calls a host function, `HostFnRouter::dispatch` checks
   the function's required capability against the session's `CapabilitySet`.
   If denied, the call returns an error without invoking the handler.

### Resolution Rules

```
1. If capability is in deny  → blocked (deny wins)
2. If capability is in allow → permitted
3. If neither               → permitted (open by default)
4. If allow is non-empty and capability is not in it → blocked (allowlist mode)
```

Rule 4 means: if an app specifies *any* allow entries, it's operating in
allowlist mode — only explicitly allowed capabilities are available. If no
`[capabilities]` section exists, everything is allowed (backwards compatible).

### CallContext Integration

`CallContext` already carries `permissions: Permissions`. Extend this to
include the resolved `CapabilitySet`:

```rust
pub struct CallContext {
    pub org_id: OrgId,
    pub user_id: UserId,
    pub session_id: SessionId,
    pub agent_id: AgentId,
    pub app_id: Option<AppId>,          // NEW
    pub capabilities: CapabilitySet,    // NEW (replaces or extends permissions)
    // ...
}
```

---

## 7. Plugin Scoping

A subtle but important point: plugins are loaded **globally** but configured
**per-app**. Two apps can use the KB plugin with different indexes.

### How Scoping Works

During app loading, the engine merges the global plugin config with the
app's overrides:

```
global:  [plugins.kb] { index_path = "/data/default", search_k = 10 }
app:     [plugins.kb] { index_path = "/data/acme-docs", search_k = 5 }
merged:  { index_path = "/data/acme-docs", search_k = 5 }
```

This merged config is attached to the app's `CallContext`. When a host
function is called, the handler receives the context and can read the
app-scoped config.

This means host function handlers need access to the calling context:

```rust
// Current signature:
Fn(Vec<Value>, HashMap<String, Value>) -> Result<Value, PluginError>

// Needed:
Fn(&CallContext, Vec<Value>, HashMap<String, Value>) -> Result<Value, PluginError>
```

This is a **breaking change** to the host function handler signature. It
should be bundled with the capability enforcement work since both require
threading `CallContext` through dispatch.

### What This Does Not Do

This is **not** plugin instance isolation. The KB plugin has one tantivy
index, one LanceDB connection. If two apps need different KB indexes, the
plugin must support multi-tenancy internally (e.g., by reading `index_path`
from the call context at query time rather than at init time).

Plugins that hold per-init state (a single connection pool, a single index)
will need refactoring to support per-app config. This is intentional — full
process isolation per app is overkill for a system where plugins are
compiled-in trusted code.

---

## 8. Plugin-Contributed Routes & Widgets

This materializes the deferred `register_routes` from the plugin framework.

### Route Registration

```rust
impl<'a> PluginContext<'a> {
    /// Mount an Axum router under /plugins/{plugin_name}/{prefix}
    pub fn register_routes(&mut self, prefix: &str, router: axum::Router) {
        self.registrations.routes.push(RouteRegistration {
            prefix: prefix.to_string(),
            router,
        });
    }
}
```

The engine collects these and merges them into the server's Axum router
during init. A KB plugin registering routes at `"browse"` would be mounted
at `/plugins/kb/browse/*`.

### Widget Protocol

For Level 2 surfaces (chat + widgets), the engine needs a way to load
plugin UI fragments into the chat shell. Two options:

**Option A: iframe embedding.** The chat shell renders an iframe pointing
at the plugin's route. Simple, fully isolated, but limited interaction
between widget and chat.

**Option B: Fragment protocol.** The plugin route returns an HTML fragment.
The chat shell fetches it and injects it into a slot. Richer integration
(shared styles, event passing via `postMessage`) but more coupling.

**Recommendation:** Start with iframes (Option A). It's simpler, avoids
CSS/JS conflicts, and the interaction between chat and sidebar widgets is
low-bandwidth (e.g., "show customer card for this session"). Add the
fragment protocol later if iframe limitations become painful.

### Widget Registration

Plugins declare available widgets in their manifest:

```rust
fn manifest(&self) -> PluginManifest {
    PluginManifest {
        provides: vec!["host_fn:kb_search".into(), "widget:kb:browser".into()],
        // ...
    }
}
```

The engine validates that widgets referenced in app manifests actually exist.

---

## 9. Built-in Chat UI

The engine ships a generic chat frontend that apps can use without writing
any frontend code. This is the Level 1 surface.

### Architecture

```
/apps/{app-name}/                → serves index.html (chat shell SPA)
/apps/{app-name}/assets/*        → serves JS/CSS/fonts
/api/v1/apps/{app-name}/config   → returns app branding + widget config as JSON
```

The SPA is a single built artifact shipped with the engine. At load time
it fetches `/config` to learn the app's title, theme, welcome message,
suggested prompts, and widget layout. Everything else is standard chat
behavior (create session, send messages, render SSE stream).

### Technology Choice

The chat UI should be minimal — this is not a product frontend, it's a
generic shell. Candidates:

- **Vanilla JS + minimal CSS.** Smallest bundle, no build step, easy to
  embed. Sufficient for a chat interface.
- **Preact + HTM.** Slightly richer (components, state management) without
  a build step. Good if we add widget slots.

Recommendation: start with vanilla JS. A chat box, a message list, an
input field, and an SSE listener is ~500 lines. Add a framework only
if widget integration demands it.

### Streaming

The UI connects to the SSE endpoint and renders tokens as they arrive.
The SSE stream maps directly to lifecycle events:

```
event: turn_start
data: {}

event: token
data: {"content": "I can help"}

event: tool_use
data: {"function": "kb_search", "args": {"query": "return policy"}}

event: tool_result
data: {"function": "kb_search", "result": {...}}

event: turn_end
data: {"input_tokens": 1200, "output_tokens": 350}
```

This gives the UI enough information to render typing indicators, tool-use
cards, and token counts without polling.

---

## 10. Multi-App Server Configuration

The engine needs to know which apps to load. Two options:

### Option A: Directory Convention

```
apps/
  acme-support.toml
  research-tool.toml
  devops-copilot.toml
```

The engine scans `apps/` at startup and loads all manifests. Simple,
file-system based, works with version control.

### Option B: Config Section

```toml
# greatwheel.toml

[apps]
dir = "apps/"                      # scan this directory
# or explicit:
# manifests = ["apps/acme-support.toml", "apps/research-tool.toml"]
```

**Recommendation:** Option A as default, Option B as override. The engine
scans `apps/` unless `[apps]` in the main config says otherwise.

### App Index Endpoint

```
GET /api/v1/apps → [
  { "name": "acme-support", "description": "...", "path": "/apps/acme-support" },
  { "name": "research-tool", "description": "...", "path": "/apps/research-tool" },
]
```

Useful for building a launcher or app directory page.

---

## 11. What Changes in Existing Code

| Component | Change | Scope |
|-----------|--------|-------|
| `CallContext` | Add `app_id: Option<AppId>`, `capabilities: CapabilitySet` | `gw-core` |
| `HostFnRouter::dispatch` | Accept `&CallContext`, enforce capabilities | `gw-engine` |
| Host function handler signature | Add `&CallContext` parameter | Breaking — all plugins |
| `PluginContext` | Implement `register_routes` | `gw-core` + `gw-engine` |
| `gw-server/main.rs` | Load app manifests, mount per-app routes | `gw-server` |
| `SessionManager` | Accept `AppId` on session creation, apply app config | `gw-loop` |
| New: `AppManifest` parser | Parse TOML manifests, validate against loaded plugins | `gw-engine` or new `gw-apps` |
| New: Chat UI | Static assets for built-in chat shell | `gw-server/static/` |

---

## 12. Implementation Plan

### Phase A — Foundations (prerequisite: Plugin Framework Phase 3)

1. **`CallContext` threading.** Add `app_id` and `capabilities` to
   `CallContext`. Update host function handler signature to receive
   `&CallContext`. Update all existing plugins.

2. **Capability enforcement.** Implement allow/deny resolution in
   `HostFnRouter`. Wire it into the dispatch path.

3. **`register_routes`.** Implement route registration in `PluginContext`.
   Merge plugin routes into the Axum router at startup.

### Phase B — App Manifest

4. **Manifest format.** Define `AppManifest` struct, TOML parser,
   validation (required plugins exist, capabilities are satisfiable).

5. **App loading.** Scan `apps/` directory, parse manifests, merge
   plugin configs, mount API routes per app.

6. **Scoped sessions.** `SessionManager::create_session` accepts an
   `AppId`, builds `CallContext` with app's capabilities and agent
   config (persona, model, temperature).

### Phase C — Built-in Chat UI

7. **Chat shell.** Vanilla JS chat UI: message list, input, SSE
   streaming, markdown rendering.

8. **App config endpoint.** `GET /api/v1/apps/{name}/config` returns
   branding, welcome message, suggested prompts.

9. **Surface mounting.** For apps with `surface.type = "chat"`, serve
   the chat shell at the app's path with config pointing at the right
   app.

### Phase D — Widgets & Custom Surfaces

10. **Widget slots.** Define slot names (sidebar, header, footer) in the
    chat shell. Load plugin widgets via iframe.

11. **Custom frontend.** For `surface.type = "custom"`, serve static
    files from `static_dir`.

12. **App index.** `GET /api/v1/apps` endpoint, optional launcher page.

---

## 13. Open Questions

| Question | Options | Leaning |
|----------|---------|---------|
| Should apps support auth/identity? | (a) Engine-level auth applies to all apps, (b) per-app auth config | (a) first — per-app auth is a Phase D+ concern |
| Should the chat UI be a separate crate or bundled in gw-server? | (a) Embedded static assets in gw-server, (b) separate `gw-ui` crate | (a) — it's small and tightly coupled to the server |
| How should app-scoped plugin config reach host function handlers? | (a) Via `CallContext`, (b) via `SharedState` keyed by `AppId` | (a) — it's the natural place and already threaded through |
| Should apps support multiple agents? | (a) One agent per app, (b) multi-agent with bus | (a) first — multi-agent is a composition of apps, not a feature of one app |
| Hot-reload of app manifests? | (a) Restart required, (b) watch `apps/` directory | (a) first — hot-reload is nice but not essential and adds complexity |

---

## 14. Example: Three Apps on One Server

```toml
# greatwheel.toml

[server]
bind = "0.0.0.0:3000"

[plugins.kb]
enabled = true
default_index = "/data/general"

[plugins.crm]
enabled = true
provider = "salesforce"
api_key = "${SALESFORCE_API_KEY}"

[plugins.guardrails]
enabled = true
```

```toml
# apps/acme-support.toml
[app]
name = "acme-support"
description = "Customer support for Acme Corp"
[agent]
persona = "You are Acme's support agent..."
model = "qwen3.5:9b"
[plugins.kb]
index_path = "/data/acme-docs"
[capabilities]
allow = ["kb.read", "crm.read"]
[surface]
type = "chat"
path = "/apps/acme-support"
```

```toml
# apps/research-tool.toml
[app]
name = "research-tool"
description = "Internal research assistant"
[agent]
persona = "You are a research assistant..."
model = "qwen3.5:9b"
temperature = 0.7
[plugins.kb]
index_path = "/data/papers"
[capabilities]
allow = ["kb.read", "kb.explore"]
[surface]
type = "chat"
path = "/apps/research"
```

```toml
# apps/devops-copilot.toml
[app]
name = "devops-copilot"
description = "Incident triage assistant"
[agent]
persona = "You are a DevOps copilot..."
model = "qwen3.5:9b"
temperature = 0.1
max_turns = 50
[capabilities]
allow = ["kb.read"]
[surface]
type = "api"               # no UI — accessed via PagerDuty integration
```

Result: one Greatwheel process serving three purpose-built agents, each
with different knowledge bases, capabilities, and surfaces. No Rust code
written — just TOML.
