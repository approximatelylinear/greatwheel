use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use gw_core::{EntryType, LlmMessage, LlmResponse, SessionId};
use gw_loop::error::LoopError;
use gw_loop::llm::LlmClient;
use gw_loop::{ConversationLoop, LoopConfig, SnapshotPolicy};
use gw_runtime::{AgentError, HostBridge, ReplAgent};
use ouros::Object;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Mock LLM that returns pre-scripted responses in order.
struct MockLlm {
    responses: Arc<Mutex<Vec<String>>>,
}

impl MockLlm {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

impl LlmClient for MockLlm {
    fn chat<'a>(
        &'a self,
        _messages: &'a [LlmMessage],
        _model: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LoopError>> + Send + 'a>> {
        let responses = self.responses.clone();
        Box::pin(async move {
            let content = {
                let mut r = responses.lock().unwrap();
                if r.is_empty() {
                    return Err(LoopError::Llm("no more mock responses".into()));
                }
                r.remove(0)
            };
            Ok(LlmResponse {
                content,
                model: Some("mock".into()),
                input_tokens: Some(10),
                output_tokens: Some(20),
            })
        })
    }
}

struct NullBridge;

impl HostBridge for NullBridge {
    fn call(
        &mut self,
        function: &str,
        _args: Vec<Value>,
        _kwargs: HashMap<String, Value>,
    ) -> Result<Object, AgentError> {
        Err(AgentError::UnknownFunction(function.to_string()))
    }
}

#[tokio::test]
async fn test_handle_turn_sets_variable() {
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    let llm = Box::new(MockLlm::new(vec![
        "Setting x.\n\n```python\nx = 42\nFINAL(str(x))\n```".into(),
    ]));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig {
        system_prompt: "You are a test assistant.".into(),
        ..Default::default()
    };

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);

    let result = loop_.handle_turn("Set x to 42").await.unwrap();

    assert!(result.is_final);
    assert_eq!(result.response.as_deref(), Some("42"));
    assert_eq!(result.iterations, 1);

    // Check REPL state.
    let x = loop_
        .repl
        .get_variable("x")
        .map(|o| gw_runtime::object_to_json(&o));
    assert_eq!(x, Some(serde_json::json!(42)));

    // Check tree has entries.
    assert!(loop_.tree.entries().len() >= 3); // user + code_exec + assistant

    // Drain events (should not block).
    drop(loop_);
    let mut events = Vec::new();
    while let Ok(e) = event_rx.try_recv() {
        events.push(e);
    }
}

#[tokio::test]
async fn test_multi_iteration_turn() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    let llm = Box::new(MockLlm::new(vec![
        // First iteration: sets a variable but no FINAL.
        "```python\ndata = [1, 2, 3]\n```".into(),
        // Second iteration: produces FINAL.
        "```python\nresult = sum(data)\nFINAL(str(result))\n```".into(),
    ]));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig::default();

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);
    let result = loop_.handle_turn("Sum [1,2,3]").await.unwrap();

    assert!(result.is_final);
    assert_eq!(result.response.as_deref(), Some("6"));
    assert_eq!(result.iterations, 2);
}

#[tokio::test]
async fn test_text_only_response_ends_turn() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    let llm = Box::new(MockLlm::new(vec![
        "I don't need code for this. The answer is 42.".into(),
    ]));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig::default();

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);
    let result = loop_.handle_turn("What is 42?").await.unwrap();

    assert!(!result.is_final);
    assert!(result.response.unwrap().contains("42"));
    assert_eq!(result.iterations, 1);
}

#[tokio::test]
async fn test_steering_injection() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));

    // We'll capture what messages the LLM sees.
    let seen_messages: Arc<Mutex<Vec<Vec<LlmMessage>>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_clone = seen_messages.clone();

    struct SpyLlm {
        seen: Arc<Mutex<Vec<Vec<LlmMessage>>>>,
        responses: Arc<Mutex<Vec<String>>>,
    }

    impl LlmClient for SpyLlm {
        fn chat<'a>(
            &'a self,
            messages: &'a [LlmMessage],
            _model: Option<&'a str>,
        ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LoopError>> + Send + 'a>> {
            let seen = self.seen.clone();
            let responses = self.responses.clone();
            let msgs = messages.to_vec();
            Box::pin(async move {
                seen.lock().unwrap().push(msgs);
                let content = {
                    let mut r = responses.lock().unwrap();
                    if r.is_empty() {
                        "FINAL(\"done\")".to_string()
                    } else {
                        r.remove(0)
                    }
                };
                Ok(LlmResponse {
                    content,
                    model: None,
                    input_tokens: None,
                    output_tokens: None,
                })
            })
        }
    }

    let llm = Box::new(SpyLlm {
        seen: seen_clone.clone(),
        responses: Arc::new(Mutex::new(vec![
            "```python\nx = 1\n```".into(), // iter 1: no FINAL, loops
            "FINAL(\"done\")".into(),       // iter 2: ends
        ])),
    });

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig::default();

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);

    // Inject steering before the turn starts — it should appear in the 2nd LLM call.
    loop_.inject_steering("Focus on auth");

    let _result = loop_.handle_turn("Analyze code").await.unwrap();

    let calls = seen_clone.lock().unwrap();
    assert_eq!(calls.len(), 2);

    // First call should NOT have steering (it was consumed before first LLM call).
    // Actually, steering is injected before every LLM call, so first call has it.
    let first_call_text: String = calls[0].iter().map(|m| m.content.clone()).collect();
    assert!(
        first_call_text.contains("Focus on auth"),
        "steering should be in first LLM call"
    );
}

#[tokio::test]
async fn test_auto_snapshot_policy() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    // 6 responses: 3 turns × (code + FINAL needed, but mock gives both in one)
    let llm = Box::new(MockLlm::new(vec![
        "```python\na = 1\n```".into(),
        "FINAL(\"done\")".into(),
        "```python\nb = 2\n```".into(),
        "FINAL(\"done\")".into(),
        "```python\nc = 3\n```".into(),
        "FINAL(\"done\")".into(),
    ]));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig {
        snapshot_policy: SnapshotPolicy {
            every_n_turns: 2,
            before_compaction: true,
        },
        ..Default::default()
    };

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);

    // Turn 1 — no snapshot yet.
    loop_.handle_turn("Set a").await.unwrap();
    let snapshot_count = count_snapshots(&loop_);
    assert_eq!(snapshot_count, 0, "no snapshot after 1 turn");

    // Turn 2 — should auto-snapshot (every_n_turns=2).
    loop_.handle_turn("Set b").await.unwrap();
    let snapshot_count = count_snapshots(&loop_);
    assert_eq!(snapshot_count, 1, "snapshot after 2 turns");

    // Turn 3 — counter reset, no new snapshot.
    loop_.handle_turn("Set c").await.unwrap();
    let snapshot_count = count_snapshots(&loop_);
    assert_eq!(snapshot_count, 1, "no new snapshot after 3 turns");
}

fn count_snapshots(loop_: &ConversationLoop) -> usize {
    loop_
        .tree
        .entries()
        .iter()
        .filter(|e| matches!(e.entry_type, EntryType::ReplSnapshot(_)))
        .count()
}

#[tokio::test]
async fn test_compaction() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    // 8 turns worth of responses + 1 for the compaction summary LLM call
    let mut responses: Vec<String> = Vec::new();
    for i in 1..=8 {
        responses.push(format!("```python\nvar_{i} = {i}\n```"));
        responses.push("FINAL(\"done\")".into());
    }
    // The compaction will call summarize_entries which makes an LLM call.
    responses.push("Summary of turns 1-3.".into());

    let llm = Box::new(MockLlm::new(responses));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig {
        compaction_keep_count: 5,
        snapshot_policy: SnapshotPolicy {
            every_n_turns: 0, // disable auto-snapshot to keep tree clean
            before_compaction: true,
        },
        ..Default::default()
    };

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);

    // Run 8 turns.
    for i in 1..=8 {
        loop_.handle_turn(&format!("Set var_{i}")).await.unwrap();
    }

    // All variables should exist.
    for i in 1..=8 {
        let v = loop_
            .repl
            .get_variable(&format!("var_{i}"))
            .map(|o| gw_runtime::object_to_json(&o));
        assert_eq!(v, Some(serde_json::json!(i)), "var_{i} should be {i}");
    }

    // Compact — keep last 5 user messages.
    loop_.compact().await.unwrap();

    // Variables should still exist (compaction doesn't erase REPL state).
    for i in 1..=8 {
        let v = loop_
            .repl
            .get_variable(&format!("var_{i}"))
            .map(|o| gw_runtime::object_to_json(&o));
        assert_eq!(
            v,
            Some(serde_json::json!(i)),
            "var_{i} should survive compaction"
        );
    }

    // Tree should have a Compaction entry.
    let has_compaction = loop_
        .tree
        .entries()
        .iter()
        .any(|e| matches!(e.entry_type, EntryType::Compaction { .. }));
    assert!(has_compaction, "tree should have a compaction entry");
}

#[tokio::test]
async fn test_branch_switch_restores_state() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    // Responses for turns 1-4, plus a summary LLM call for branch switch
    let responses = vec![
        // Turn 1: x = 1
        "```python\nx = 1\n```".into(),
        "FINAL(\"done\")".into(),
        // Turn 2: y = 2
        "```python\ny = 2\n```".into(),
        "FINAL(\"done\")".into(),
        // Turn 3: x = 100 (the divergent branch)
        "```python\nx = 100\n```".into(),
        "FINAL(\"done\")".into(),
        // Summary for abandoned branch
        "Tried approach A: set x to 100.".into(),
    ];
    let llm = Box::new(MockLlm::new(responses));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig {
        snapshot_policy: SnapshotPolicy {
            every_n_turns: 1, // snapshot after every turn for reliable restore
            before_compaction: true,
        },
        ..Default::default()
    };

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);

    // Turn 1: x = 1
    loop_.handle_turn("Set x to 1").await.unwrap();
    // Turn 2: y = 2
    loop_.handle_turn("Set y to 2").await.unwrap();

    // Record the entry ID after turn 2 (before divergence).
    // The active leaf after turn 2 is the snapshot entry (auto-snapshot fires).
    // We want the snapshot entry as our branch point.
    let branch_point = loop_.tree.active_leaf().unwrap();

    // Turn 3: x = 100 (diverge)
    loop_.handle_turn("Set x to 100").await.unwrap();

    // Verify x was changed.
    let x = loop_
        .repl
        .get_variable("x")
        .map(|o| gw_runtime::object_to_json(&o));
    assert_eq!(x, Some(serde_json::json!(100)));

    // Switch back to the branch point.
    loop_.switch_branch(branch_point, true).await.unwrap();

    // x should be restored to 1 (from the snapshot at turn 2).
    let x = loop_
        .repl
        .get_variable("x")
        .map(|o| gw_runtime::object_to_json(&o));
    assert_eq!(x, Some(serde_json::json!(1)), "x should be restored to 1");

    // y should still be 2.
    let y = loop_
        .repl
        .get_variable("y")
        .map(|o| gw_runtime::object_to_json(&o));
    assert_eq!(y, Some(serde_json::json!(2)), "y should be restored to 2");

    // Tree should have a BranchSummary entry.
    let has_branch_summary = loop_
        .tree
        .entries()
        .iter()
        .any(|e| matches!(e.entry_type, EntryType::BranchSummary(_)));
    assert!(has_branch_summary, "tree should have a branch summary");
}

#[tokio::test]
async fn test_auto_compaction() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    // 8 turns of responses + 1 for the compaction summary LLM call
    let mut responses: Vec<String> = Vec::new();
    for i in 1..=8 {
        responses.push(format!("```python\nvar_{i} = {i}\n```"));
        responses.push("FINAL(\"done\")".into());
    }
    // Auto-compaction will call summarize_entries which makes an LLM call.
    responses.push("Summary of earlier turns.".into());

    let llm = Box::new(MockLlm::new(responses));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig {
        compaction_keep_count: 5,
        auto_compact_after_turns: Some(8), // compact after 8 turns
        snapshot_policy: SnapshotPolicy {
            every_n_turns: 0,
            before_compaction: true,
        },
        ..Default::default()
    };

    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);

    // Run 7 turns — should NOT trigger auto-compaction.
    for i in 1..=7 {
        loop_.handle_turn(&format!("Set var_{i}")).await.unwrap();
    }
    let has_compaction = loop_
        .tree
        .entries()
        .iter()
        .any(|e| matches!(e.entry_type, EntryType::Compaction { .. }));
    assert!(!has_compaction, "no compaction before threshold");

    // Turn 8 — should trigger auto-compaction.
    loop_.handle_turn("Set var_8").await.unwrap();
    loop_.check_auto_compact().await;

    let has_compaction = loop_
        .tree
        .entries()
        .iter()
        .any(|e| matches!(e.entry_type, EntryType::Compaction { .. }));
    assert!(has_compaction, "compaction should trigger after 8 turns");

    // All variables should still exist.
    for i in 1..=8 {
        let v = loop_
            .repl
            .get_variable(&format!("var_{i}"))
            .map(|o| gw_runtime::object_to_json(&o));
        assert_eq!(v, Some(serde_json::json!(i)), "var_{i} survives compaction");
    }
}

#[tokio::test]
async fn test_channel_ask_reply() {
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

    // Create a bridge with ask support.
    let ask_handle = gw_loop::bridge::new_ask_handle();

    let conv_bridge =
        gw_loop::bridge::ConversationBridge::new(event_tx.clone(), ask_handle.clone(), None);

    let repl = ReplAgent::new(
        vec!["FINAL".into(), "ask_user".into()],
        Box::new(conv_bridge),
    );

    // Mock LLM: generates code that calls ask_user().
    let llm = Box::new(MockLlm::new(vec![
        "```python\nname = ask_user(\"What is your name?\")\n```".into(),
        "FINAL(\"done\")".into(),
    ]));

    let session_id = SessionId(Uuid::new_v4());
    let config = LoopConfig::default();
    let mut loop_ = ConversationLoop::new(session_id, repl, llm, config, event_tx);

    // Run the turn in a separate std::thread with its own runtime,
    // since channel.ask() blocks the thread and we need the test thread
    // free to deliver the reply.
    let ask_handle_clone = ask_handle.clone();
    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(loop_.handle_turn("Ask for my name"))
    });

    // Wait for the ask to be registered, then reply.
    let mut replied = false;
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        if let Some(prompt) = gw_loop::bridge::get_pending_ask(&ask_handle_clone) {
            assert_eq!(prompt, "What is your name?");
            let delivered = gw_loop::bridge::reply_to_ask(&ask_handle_clone, "Alice".into());
            assert!(delivered);
            replied = true;
            break;
        }
    }

    assert!(replied, "ask should have been registered");

    // The turn should complete now.
    let result = handle.join().unwrap().unwrap();
    assert!(result.is_final);

    // Check that the InputRequest event was emitted.
    let mut found_input_request = false;
    while let Ok(event) = event_rx.try_recv() {
        if let gw_core::LoopEvent::InputRequest(prompt) = event {
            assert_eq!(prompt, "What is your name?");
            found_input_request = true;
        }
    }
    assert!(found_input_request, "InputRequest event should be emitted");
}

// ─── Plugin host function router dispatch ─────────────────────────────────

/// Exercises the plugin host function router wiring added in phase 1 of
/// the KB agent integration work. Builds a router with both a sync and
/// an async registered handler, wires it into a `ConversationBridge`,
/// and asserts that `HostBridge::call` dispatches through the router
/// before falling back to the inner bridge. This is the test that
/// validates the whole sync↔async bridge (block_in_place + block_on).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_plugin_router_dispatch_sync_and_async() {
    use gw_core::{HostFnHandler, HostFnRegistration, PluginError};
    use gw_engine::HostFnRouter;
    use std::sync::Arc;

    // Build the router by hand. In production this happens inside
    // GreatWheelEngine::init via plugin registrations, but a direct
    // construction keeps the test hermetic.
    let mut registrations = HashMap::new();

    // Sync handler — returns a constant.
    registrations.insert(
        "test.sync_echo".to_string(),
        HostFnRegistration {
            handler: HostFnHandler::Sync(Arc::new(|args, _kwargs| {
                let msg = args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                Ok(Value::String(format!("sync:{msg}")))
            })),
            capability: Some("test.read".into()),
        },
    );

    // Async handler — simulates a real async workload by yielding.
    registrations.insert(
        "test.async_echo".to_string(),
        HostFnRegistration {
            handler: HostFnHandler::Async(Arc::new(|args, _kwargs| {
                Box::pin(async move {
                    tokio::task::yield_now().await;
                    let msg = args
                        .first()
                        .and_then(|v| v.as_str())
                        .unwrap_or("default")
                        .to_string();
                    Ok::<_, PluginError>(Value::String(format!("async:{msg}")))
                })
            })),
            capability: Some("test.read".into()),
        },
    );

    let router = Arc::new(HostFnRouter::new(registrations));

    // Build a ConversationBridge wired to the router. We pass a
    // throwaway event channel and ask handle.
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let ask_handle = gw_loop::bridge::new_ask_handle();
    let mut bridge = gw_loop::bridge::ConversationBridge::with_plugin_router(
        event_tx,
        ask_handle,
        None,
        Some(Arc::clone(&router)),
    );

    // Sync handler dispatches through the router and returns the right value.
    let result = bridge
        .call(
            "test.sync_echo",
            vec![Value::String("hello".into())],
            HashMap::new(),
        )
        .expect("sync call should succeed");
    match result {
        Object::String(s) => assert_eq!(s, "sync:hello"),
        other => panic!("expected Object::String, got {:?}", other),
    }

    // Async handler dispatches through the router via block_in_place.
    // This is the real test of the phase 1 foundation — if it deadlocks
    // or panics, we've got the runtime setup wrong.
    let result = bridge
        .call(
            "test.async_echo",
            vec![Value::String("world".into())],
            HashMap::new(),
        )
        .expect("async call should succeed");
    match result {
        Object::String(s) => assert_eq!(s, "async:world"),
        other => panic!("expected Object::String, got {:?}", other),
    }

    // A function that isn't in the router falls through to UnknownFunction
    // because there's no inner bridge configured.
    let result = bridge.call("test.missing", vec![], HashMap::new());
    match result {
        Err(AgentError::UnknownFunction(name)) => assert_eq!(name, "test.missing"),
        other => panic!("expected UnknownFunction error, got {:?}", other),
    }
}
