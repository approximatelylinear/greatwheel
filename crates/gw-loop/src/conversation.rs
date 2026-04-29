use std::sync::Arc;

use gw_core::{
    EntryId, EntryType, LlmMessage, LoopEvent, ReplSnapshotData, SessionId, SpineEntityLink,
    SpineRelation, SpineSegmentSnapshot,
};
use gw_runtime::{extract_code_blocks, HostBridge, ReplAgent};
use tokio::sync::mpsc;
use tracing::{info, info_span, warn};
use uuid::Uuid;

/// Emit an AG-UI text-message trilogy (START, DELTA, END) for a
/// complete assistant-authored string. Phase 5: today we emit the
/// whole string as one DELTA — wire format is already forward-
/// compatible with real token streaming, which would fire many
/// DELTAs between the same START/END pair.
fn emit_text_message(
    tx: &mpsc::UnboundedSender<LoopEvent>,
    content: String,
    entry_id: Option<EntryId>,
) {
    let message_id = Uuid::new_v4().to_string();
    let _ = tx.send(LoopEvent::TextMessageStart {
        message_id: message_id.clone(),
        entry_id,
    });
    let _ = tx.send(LoopEvent::TextMessageDelta {
        message_id: message_id.clone(),
        delta: content,
        model: None,
    });
    let _ = tx.send(LoopEvent::TextMessageEnd { message_id });
}

use crate::context::{build_turn_context_with_opts, ContextOptions};
use crate::error::LoopError;
use crate::llm::LlmClient;
use crate::tree::SessionTree;

/// Boxed predicate that checks whether a FINAL answer is acceptable.
type AnswerValidator = Box<dyn Fn(&str) -> bool + Send>;

/// Auto-snapshot policy — when to save REPL state.
#[derive(Debug, Clone)]
pub struct SnapshotPolicy {
    /// Snapshot every N user turns.
    pub every_n_turns: u32,
    /// Always snapshot before compaction.
    pub before_compaction: bool,
}

impl Default for SnapshotPolicy {
    fn default() -> Self {
        Self {
            every_n_turns: 3,
            before_compaction: true,
        }
    }
}

/// Callback for per-iteration control of the rLM loop.
///
/// Used by benchmark harnesses to inject iteration-specific prompts
/// (e.g., "EXPLORE FIRST", "⚠️ LAST CHANCE") and handle max-iterations
/// fallback extraction.
pub trait IterationCallback: Send {
    /// Called before each LLM call within a turn.
    /// If Some(prompt) is returned, it's appended as a UserMessage entry
    /// in the tree before the LLM call.
    fn before_iteration(
        &mut self,
        iteration: usize,
        max_iterations: usize,
        repl: &ReplAgent,
    ) -> Option<String>;

    /// Called when max iterations reached without FINAL.
    /// Return Some(answer) to use as the fallback response.
    fn on_max_iterations(&mut self, _query: &str) -> Option<String> {
        None
    }
}

/// Configuration for the conversation loop.
pub struct LoopConfig {
    /// System prompt prepended to every LLM call.
    pub system_prompt: String,
    /// Number of recent user/assistant pairs to include in context.
    pub recency_window: usize,
    /// Maximum rLM iterations per turn (LLM call → execute → repeat).
    pub max_iterations: usize,
    /// Auto-snapshot policy.
    pub snapshot_policy: SnapshotPolicy,
    /// Number of recent user/assistant pairs to keep after compaction.
    pub compaction_keep_count: usize,
    /// Auto-compact after this many user turns. None = disabled.
    pub auto_compact_after_turns: Option<u32>,
    /// Include CodeExecution output in the LLM prompt.
    /// Required for benchmark-style execution where the LLM needs to
    /// see search results and document text.
    pub include_code_output: bool,
    /// Maximum chars for REPL output when include_code_output is true.
    /// 0 = no limit.
    pub repl_output_max_chars: usize,
    /// Strip `<think>...</think>` tags from LLM responses before
    /// code extraction. For thinking models (qwen3.5, etc.).
    pub strip_think_tags: bool,
    /// Optional answer validator. When FINAL is called and this returns
    /// false, the loop continues with a rejection message instead of
    /// returning. Used to reject refusal/hedge answers.
    pub answer_validator: Option<AnswerValidator>,
    /// Optional per-iteration callback for benchmark-style coaching.
    pub iteration_callback: Option<Box<dyn IterationCallback>>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            recency_window: 5,
            max_iterations: 10,
            snapshot_policy: SnapshotPolicy::default(),
            compaction_keep_count: 5,
            auto_compact_after_turns: None,
            include_code_output: false,
            repl_output_max_chars: 0,
            strip_think_tags: false,
            answer_validator: None,
            iteration_callback: None,
        }
    }
}

/// Result of a single turn in the conversation loop.
#[derive(Debug)]
pub struct TurnResult {
    /// Final response text (from FINAL or last assistant message).
    pub response: Option<String>,
    /// Whether FINAL() was called.
    pub is_final: bool,
    /// Number of rLM iterations (LLM calls) in this turn.
    pub iterations: usize,
    /// Total input tokens across all LLM calls.
    pub input_tokens: u32,
    /// Total output tokens across all LLM calls.
    pub output_tokens: u32,
    /// Persistent `session_entries.id` of the assistant message that
    /// produced `response` (when there is one). Forwarded into
    /// `LoopEvent::TextMessageStart.entry_id` so the frontend can
    /// anchor each chat row to the entry the spine indexes by.
    pub assistant_entry_id: Option<EntryId>,
}

/// The multi-turn conversation loop.
///
/// Drives the rLM pattern: user message → build context → LLM call →
/// extract code → execute in REPL → check FINAL → repeat or respond.
pub struct ConversationLoop {
    pub tree: SessionTree,
    pub repl: ReplAgent,
    session_id: SessionId,
    config: LoopConfig,
    llm: Box<dyn LlmClient>,
    /// Outbound events (responses, input requests, etc.).
    event_tx: mpsc::UnboundedSender<LoopEvent>,
    /// Pending steering messages to inject before the next LLM call.
    pending_steering: Vec<LlmMessage>,
    /// Pending follow-up messages to process after the current turn.
    pending_follow_ups: Vec<String>,
    /// Tracks turns since last snapshot for auto-snapshot policy.
    turns_since_snapshot: u32,
    /// Turn counter for tracing.
    turn_number: u32,
    /// Optional handle to the semantic-spine extractor. When present,
    /// each `flush_tree` fans out background tasks that read each
    /// freshly-persisted prose entry, run the joint extractor, and
    /// write `session_entry_entities` + `session_entry_relations`
    /// rows. Off the chat path; per-entry failures are logged.
    spine_extractor: Option<Arc<crate::spine::SpineExtractor>>,
}

impl ConversationLoop {
    pub fn new(
        session_id: SessionId,
        repl: ReplAgent,
        llm: Box<dyn LlmClient>,
        config: LoopConfig,
        event_tx: mpsc::UnboundedSender<LoopEvent>,
    ) -> Self {
        Self {
            tree: SessionTree::new(session_id),
            repl,
            session_id,
            config,
            llm,
            event_tx,
            pending_steering: Vec::new(),
            pending_follow_ups: Vec::new(),
            turns_since_snapshot: 0,
            turn_number: 0,
            spine_extractor: None,
        }
    }

    /// Create a ConversationLoop with a pre-built SessionTree (e.g., loaded from Postgres).
    pub fn with_tree(
        tree: SessionTree,
        repl: ReplAgent,
        llm: Box<dyn LlmClient>,
        config: LoopConfig,
        event_tx: mpsc::UnboundedSender<LoopEvent>,
    ) -> Self {
        let session_id = tree.session_id();
        // Count existing user turns for the turn counter.
        let turn_number = tree.user_turn_count() as u32;
        Self {
            tree,
            repl,
            session_id,
            config,
            llm,
            event_tx,
            pending_steering: Vec::new(),
            pending_follow_ups: Vec::new(),
            turns_since_snapshot: 0,
            turn_number,
            spine_extractor: None,
        }
    }

    /// Attach a semantic-spine extractor. With this set, every call
    /// to `flush_tree` after a turn fans out background extractions
    /// for the newly-persisted prose entries (UserMessage,
    /// AssistantMessage). Per-entry extraction failures don't bubble
    /// up — they're warned and the spine catches up on the next
    /// successful entry.
    pub fn with_spine_extractor(
        mut self,
        extractor: Arc<crate::spine::SpineExtractor>,
    ) -> Self {
        self.spine_extractor = Some(extractor);
        self
    }

    /// Inject a steering message to be included before the next LLM call.
    pub fn inject_steering(&mut self, content: &str) {
        self.pending_steering.push(LlmMessage {
            role: "user".into(),
            content: format!("[Steering] {content}"),
        });
    }

    /// Queue a follow-up message for after the current turn completes.
    pub fn queue_follow_up(&mut self, content: String) {
        self.pending_follow_ups.push(content);
    }

    /// Drain all pending follow-up messages, returning them for external processing.
    pub fn drain_follow_ups(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_follow_ups)
    }

    /// Run the event loop until SessionEnd.
    pub async fn run(
        &mut self,
        mut event_rx: mpsc::UnboundedReceiver<LoopEvent>,
    ) -> Result<(), LoopError> {
        loop {
            let event = event_rx.recv().await.ok_or(LoopError::ChannelClosed)?;

            match event {
                LoopEvent::UserMessage(content) => {
                    self.run_input_turn(&content).await?;
                }

                LoopEvent::WidgetInteraction(event) => {
                    // Spine action menu (revisit / expand / compare):
                    // build a server-side templated prompt from the
                    // segment's content and run it as the next turn.
                    // Falls through to the default to_user_message
                    // path on lookup failure or non-spine actions —
                    // the existing literature_assistant follow-up
                    // buttons (Method details / Vs neighbors / What
                    // followed) and other widget interactions stay
                    // on the original code path.
                    let prompt = match self.translate_spine_action(&event).await {
                        Ok(Some(p)) => p,
                        Ok(None) => event.to_user_message(),
                        Err(e) => {
                            warn!(error = %e, "spine action translation failed; falling back");
                            event.to_user_message()
                        }
                    };
                    self.run_input_turn(&prompt).await?;
                }

                LoopEvent::FollowUp(content) => {
                    self.queue_follow_up(content);
                }

                LoopEvent::Compact => {
                    self.compact().await?;
                }

                LoopEvent::SwitchBranch(target) => {
                    self.switch_branch(target, true).await?;
                }

                LoopEvent::SessionEnd => {
                    info!("session ended");
                    return Ok(());
                }

                // Other events are outbound or handled elsewhere.
                LoopEvent::TextMessageStart { .. }
                | LoopEvent::TextMessageDelta { .. }
                | LoopEvent::TextMessageEnd { .. }
                | LoopEvent::InputRequest(_)
                | LoopEvent::HostCallStarted { .. }
                | LoopEvent::HostCallArgs { .. }
                | LoopEvent::HostCallCompleted { .. }
                | LoopEvent::TurnStarted
                | LoopEvent::TurnComplete
                | LoopEvent::TurnError { .. }
                | LoopEvent::WidgetEmitted(_)
                | LoopEvent::WidgetSuperseded { .. }
                | LoopEvent::CodeExecuted { .. }
                | LoopEvent::SpineEntryExtracted { .. }
                | LoopEvent::SpineSegmentsUpdated { .. } => {}
            }
        }
    }

    /// Run a single user-side input (text message or projected widget
    /// interaction) as a turn, emit the resulting events, drain any
    /// follow-ups the agent queued, and apply auto-compaction policy.
    /// Used by both the `UserMessage` and `WidgetInteraction` arms of
    /// `run` — kept as one helper so the two paths cannot drift.
    /// Translate a spine action-menu click (`revisit` / `expand` /
    /// `compare`) into a server-side templated prompt built from the
    /// focused segment's entities + label. Returns `Ok(None)` for
    /// non-spine widget events so the caller falls back to the
    /// default `to_user_message()` path.
    ///
    /// Prompt templates match `docs/design-semantic-spine.md` §5.5:
    ///   revisit  → "Revisit our discussion about {labels}. Summarize
    ///              what we concluded and what's still open."
    ///   expand   → "Go deeper on {labels} — what didn't we cover?"
    ///   compare  → "Compare {segment_entities} with what's still on
    ///              the table elsewhere in this conversation."
    /// `{labels}` is the segment's label; `{segment_entities}` is the
    /// top entity labels in the segment, joined with commas.
    async fn translate_spine_action(
        &self,
        event: &gw_core::WidgetEvent,
    ) -> Result<Option<String>, LoopError> {
        let action = event.action.as_str();
        if !matches!(action, "revisit" | "expand" | "compare") {
            return Ok(None);
        }
        // Need a PgPool to read segment detail; fall through to the
        // default path if the spine isn't wired up.
        let Some(extractor) = self.spine_extractor.as_ref() else {
            return Ok(None);
        };
        let pool = &extractor.stores.pg;

        let segment_id = event
            .data
            .get("segment_id")
            .and_then(|v| v.as_str())
            .and_then(|s| uuid::Uuid::parse_str(s).ok());
        let Some(seg_id) = segment_id else {
            return Ok(None);
        };
        let detail = match crate::spine::query::fetch_segment_detail(pool, seg_id).await? {
            Some(d) => d,
            None => return Ok(None),
        };

        // Top-N entity labels for the prompt. Query is already
        // ranked by mentions_in_segment desc, so slicing is enough.
        let entity_labels: Vec<String> = detail
            .entities
            .iter()
            .take(6)
            .map(|e| e.label.clone())
            .collect();
        let entities_phrase = if entity_labels.is_empty() {
            detail.segment.label.clone()
        } else {
            entity_labels.join(", ")
        };
        let prompt = match action {
            "revisit" => format!(
                "Revisit our discussion about {entities_phrase}. \
                Summarize what we concluded and what's still open."
            ),
            "expand" => format!(
                "Go deeper on {entities_phrase} — what didn't we cover yet?"
            ),
            "compare" => format!(
                "Compare {entities_phrase} with what's still on the table \
                elsewhere in this conversation. Where do they overlap, \
                where do they diverge?"
            ),
            _ => unreachable!("filtered above"),
        };
        info!(
            action = action,
            segment_id = %seg_id,
            "spine action → synthetic prompt"
        );
        Ok(Some(prompt))
    }

    async fn run_input_turn(&mut self, initial: &str) -> Result<(), LoopError> {
        let _ = self.event_tx.send(LoopEvent::TurnStarted);
        let result = match self.handle_turn(initial).await {
            Ok(r) => r,
            Err(e) => {
                let _ = self.event_tx.send(LoopEvent::TurnError {
                    message: e.to_string(),
                });
                return Err(e);
            }
        };
        let _ = self.event_tx.send(LoopEvent::TurnComplete);
        if let Some(response) = result.response {
            emit_text_message(&self.event_tx, response, result.assistant_entry_id);
        }

        while let Some(follow_up) = self.pending_follow_ups.pop() {
            let _ = self.event_tx.send(LoopEvent::TurnStarted);
            let result = match self.handle_turn(&follow_up).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = self.event_tx.send(LoopEvent::TurnError {
                        message: e.to_string(),
                    });
                    return Err(e);
                }
            };
            let _ = self.event_tx.send(LoopEvent::TurnComplete);
            if let Some(response) = result.response {
                emit_text_message(&self.event_tx, response, result.assistant_entry_id);
            }
        }

        self.check_auto_compact().await;
        Ok(())
    }

    // ─── Turn execution ─────────────────────────────────────────────

    /// Execute a single user turn through the rLM loop.
    #[tracing::instrument(
        name = "conversation.turn",
        skip(self, user_message),
        fields(
            gw.session_id = %self.session_id.0,
            gw.turn_number = tracing::field::Empty,
            gw.iterations = tracing::field::Empty,
            gw.input_tokens = tracing::field::Empty,
            gw.output_tokens = tracing::field::Empty,
            gw.is_final = tracing::field::Empty,
        )
    )]
    pub async fn handle_turn(&mut self, user_message: &str) -> Result<TurnResult, LoopError> {
        self.turn_number += 1;
        let span = tracing::Span::current();
        span.record("gw.turn_number", self.turn_number);

        // 1. Append user message to tree.
        self.tree
            .append(EntryType::UserMessage(user_message.to_string()));

        let mut iterations = 0;
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;

        let ctx_opts = ContextOptions {
            include_code_output: self.config.include_code_output,
            repl_output_max_chars: self.config.repl_output_max_chars,
        };

        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                warn!(
                    iterations,
                    "max rLM iterations reached, ending turn without FINAL"
                );
                break;
            }

            // 2a. Call iteration callback (if configured) to inject coaching prompts.
            if let Some(callback) = &mut self.config.iteration_callback {
                if let Some(prompt) =
                    callback.before_iteration(iterations, self.config.max_iterations, &self.repl)
                {
                    self.tree.append(EntryType::UserMessage(prompt));
                }
            }

            // 2b. Build context.
            {
                let _ctx_span = info_span!(
                    "conversation.build_context",
                    gw.entries_count = tracing::field::Empty,
                    gw.has_compaction = tracing::field::Empty,
                )
                .entered();

                let path = self.tree.path_to_leaf();
                let entries_count = path.len();
                let has_compaction = path
                    .iter()
                    .any(|e| matches!(e.entry_type, EntryType::Compaction { .. }));

                tracing::Span::current().record("gw.entries_count", entries_count);
                tracing::Span::current().record("gw.has_compaction", has_compaction);

                let state_summary = self.repl.state_summary();
                let mut context = build_turn_context_with_opts(
                    &path,
                    &self.config.system_prompt,
                    &state_summary,
                    self.config.recency_window,
                    &ctx_opts,
                );

                // Inject pending steering messages.
                if !self.pending_steering.is_empty() {
                    context
                        .messages
                        .extend(std::mem::take(&mut self.pending_steering));
                }

                // 3. Call LLM.
                let model_ref = context.model.as_deref();
                let llm_response = self.llm.chat(&context.messages, model_ref).await?;

                total_input_tokens += llm_response.input_tokens.unwrap_or(0);
                total_output_tokens += llm_response.output_tokens.unwrap_or(0);

                let mut response_content = llm_response.content.clone();
                let response_model = llm_response.model.clone();

                // 3a. Strip think tags if configured.
                if self.config.strip_think_tags {
                    response_content = gw_runtime::strip_think_tags(&response_content);
                }

                // 4. Extract code blocks.
                let code_blocks = extract_code_blocks(&response_content);

                // 5. Execute code blocks.
                let mut had_final = false;
                let mut final_response = None;

                for block in &code_blocks {
                    let _exec_span = info_span!(
                        "repl.execute",
                        gw.code_length = block.len(),
                        gw.is_final = tracing::field::Empty,
                    )
                    .entered();

                    match self.repl.execute(block) {
                        Ok(exec_result) => {
                            self.tree.append(EntryType::CodeExecution {
                                code: block.clone(),
                                stdout: exec_result.stdout.clone(),
                                result: exec_result.value.clone(),
                            });

                            let _ = self.event_tx.send(LoopEvent::CodeExecuted {
                                code: block.clone(),
                                stdout: exec_result.stdout.clone(),
                                is_final: exec_result.is_final,
                                error: None,
                            });

                            if exec_result.is_final {
                                let answer = exec_result
                                    .final_value
                                    .map(|v| v.to_string().trim_matches('"').to_string());

                                // 5a. Validate the answer if a validator is configured.
                                if let Some(validator) = &self.config.answer_validator {
                                    if let Some(ref ans) = answer {
                                        if !validator(ans) {
                                            warn!(answer = %ans, "FINAL answer rejected by validator");
                                            self.tree.append(EntryType::UserMessage(
                                                "That answer was rejected (refusal/hedge). Please provide a specific, concrete answer.".into()
                                            ));
                                            // Don't set had_final — continue the loop.
                                            continue;
                                        }
                                    }
                                }

                                tracing::Span::current().record("gw.is_final", true);
                                had_final = true;
                                final_response = answer;
                                break;
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "code execution failed");
                            self.tree.append(EntryType::CodeExecution {
                                code: block.clone(),
                                stdout: String::new(),
                                result: serde_json::json!({"error": e.to_string()}),
                            });
                            let _ = self.event_tx.send(LoopEvent::CodeExecuted {
                                code: block.clone(),
                                stdout: String::new(),
                                is_final: false,
                                error: Some(e.to_string()),
                            });
                        }
                    }
                }

                // 6. Append assistant message to tree (the LLM's
                //    raw response — typically code for tool-using
                //    agents). Narration (the user-visible FINAL
                //    prose) is appended below as a separate entry so
                //    the spine extractor reads from prose, not code.
                let assistant_entry_id = self.tree.append(EntryType::AssistantMessage {
                    content: response_content.clone(),
                    model: response_model,
                });

                // 7. Check for FINAL or text-only response (no code blocks).
                let turn_result = if had_final {
                    Some(TurnResult {
                        response: final_response,
                        is_final: true,
                        iterations,
                        input_tokens: total_input_tokens,
                        output_tokens: total_output_tokens,
                        assistant_entry_id: Some(assistant_entry_id),
                    })
                } else if let Some(answer) = gw_runtime::extract_final_answer(&response_content) {
                    // Check validator for text-based FINAL too.
                    if let Some(validator) = &self.config.answer_validator {
                        if !validator(&answer) {
                            warn!(answer = %answer, "text FINAL answer rejected by validator");
                            self.tree.append(EntryType::UserMessage(
                                "That answer was rejected. Please provide a specific answer."
                                    .into(),
                            ));
                            continue; // Skip to next iteration.
                        }
                    }
                    Some(TurnResult {
                        response: Some(answer),
                        is_final: true,
                        iterations,
                        input_tokens: total_input_tokens,
                        output_tokens: total_output_tokens,
                        assistant_entry_id: Some(assistant_entry_id),
                    })
                } else if code_blocks.is_empty() {
                    Some(TurnResult {
                        response: Some(response_content),
                        is_final: false,
                        iterations,
                        input_tokens: total_input_tokens,
                        output_tokens: total_output_tokens,
                        assistant_entry_id: Some(assistant_entry_id),
                    })
                } else {
                    None // Continue rLM loop.
                };

                if let Some(mut result) = turn_result {
                    // Persist the resolved user-visible prose so the
                    // spine has narrative text to extract from (the
                    // raw AssistantMessage is mostly Python source).
                    // Re-point the chat row's data-entry-id to this
                    // narration entry so segments anchored to it
                    // line up with the chat row that displays the
                    // same prose.
                    if let Some(text) = result.response.as_ref() {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            let narration_id = self.tree.append(
                                EntryType::AssistantNarration {
                                    content: text.clone(),
                                },
                            );
                            result.assistant_entry_id = Some(narration_id);
                        }
                    }
                    self.post_turn_snapshot();
                    span.record("gw.iterations", iterations);
                    span.record("gw.input_tokens", total_input_tokens);
                    span.record("gw.output_tokens", total_output_tokens);
                    span.record("gw.is_final", result.is_final);
                    info!(
                        is_final = result.is_final,
                        has_follow_up = !self.pending_follow_ups.is_empty(),
                        "turn complete"
                    );
                    self.flush_tree().await;
                    return Ok(result);
                }
            }

            // Code was executed but no FINAL — continue rLM loop.
        }

        // Max iterations reached — try fallback via callback.
        let fallback_response = if let Some(callback) = &mut self.config.iteration_callback {
            callback.on_max_iterations(user_message)
        } else {
            None
        };

        self.post_turn_snapshot();
        span.record("gw.iterations", iterations);
        span.record("gw.input_tokens", total_input_tokens);
        span.record("gw.output_tokens", total_output_tokens);
        let has_fallback = fallback_response.is_some();
        span.record("gw.is_final", has_fallback);
        info!(
            is_final = has_fallback,
            has_fallback, "turn complete (max iterations)"
        );
        self.flush_tree().await;
        Ok(TurnResult {
            response: fallback_response,
            is_final: has_fallback,
            iterations,
            input_tokens: total_input_tokens,
            output_tokens: total_output_tokens,
            // The max-iterations fallback path doesn't append an
            // assistant entry, so there's no entry_id to forward.
            assistant_entry_id: None,
        })
    }

    // ─── Compaction ─────────────────────────────────────────────────

    /// Compact the session: snapshot REPL state, summarize old entries, prune context.
    #[tracing::instrument(
        name = "conversation.compact",
        skip(self),
        fields(
            gw.session_id = %self.session_id.0,
            gw.compacted_count = tracing::field::Empty,
        )
    )]
    pub async fn compact(&mut self) -> Result<(), LoopError> {
        // Collect entry data we need before mutating the tree.
        let (first_kept_id, entries_to_summarize) = {
            let path = self.tree.path_to_leaf();

            // Find the split point: keep last N user messages.
            let mut user_count = 0;
            let mut split_idx = None;
            for (i, entry) in path.iter().enumerate().rev() {
                if matches!(entry.entry_type, EntryType::UserMessage(_)) {
                    user_count += 1;
                    if user_count == self.config.compaction_keep_count {
                        split_idx = Some(i);
                        break;
                    }
                }
            }

            let split_idx = match split_idx {
                Some(idx) => idx,
                None => return Err(LoopError::NothingToCompact),
            };

            let first_kept_id = path[split_idx].id;

            if split_idx == 0 {
                return Err(LoopError::NothingToCompact);
            }

            // Clone entries so we can release the borrow on self.tree.
            let entries: Vec<gw_core::SessionEntry> =
                path[..split_idx].iter().map(|e| (*e).clone()).collect();
            (first_kept_id, entries)
        };

        // 1. Snapshot REPL state.
        let snapshot = {
            let _snap_span = info_span!(
                "repl.snapshot",
                gw.variable_count = self.repl.get_all_variables().len(),
            )
            .entered();
            self.capture_snapshot()?
        };

        // 2. Summarize old entries via LLM.
        let entry_refs: Vec<&gw_core::SessionEntry> = entries_to_summarize.iter().collect();
        let compacted_count = entry_refs.len();
        tracing::Span::current().record("gw.compacted_count", compacted_count);
        let summary = self.summarize_entries(&entry_refs).await?;

        // 3. Append Compaction entry.
        self.tree.append(EntryType::Compaction {
            summary,
            first_kept_id,
            snapshot: Box::new(snapshot),
        });

        info!(compacted = compacted_count, "session compacted");
        self.flush_tree().await;
        Ok(())
    }

    /// Ask the LLM to summarize a sequence of entries.
    async fn summarize_entries(
        &self,
        entries: &[&gw_core::SessionEntry],
    ) -> Result<String, LoopError> {
        let mut transcript = String::new();
        for entry in entries {
            match &entry.entry_type {
                EntryType::UserMessage(content) => {
                    transcript.push_str(&format!("User: {content}\n"));
                }
                EntryType::AssistantMessage { content, .. } => {
                    transcript.push_str(&format!("Assistant: {content}\n"));
                }
                EntryType::AssistantNarration { content } => {
                    // Resolved user-visible prose — include in
                    // compaction summaries since this is what the
                    // user actually saw, regardless of whether the
                    // raw assistant entry above contained the literal
                    // or an f-string template.
                    transcript.push_str(&format!("Assistant (narration): {content}\n"));
                }
                EntryType::CodeExecution {
                    code,
                    stdout,
                    result,
                    ..
                } => {
                    transcript.push_str(&format!("Code: {code}\n"));
                    if !stdout.is_empty() {
                        transcript.push_str(&format!("Output: {stdout}\n"));
                    }
                    transcript.push_str(&format!("Result: {result}\n"));
                }
                EntryType::HostCall { .. }
                | EntryType::ReplSnapshot(_)
                | EntryType::Compaction { .. }
                | EntryType::BranchSummary(_)
                | EntryType::System(_) => {}
            }
        }

        let messages = vec![
            LlmMessage {
                role: "system".into(),
                content: "Summarize the following conversation transcript concisely. \
                          Focus on what was accomplished, key decisions, and any state \
                          that was established. Keep it under 200 words."
                    .into(),
            },
            LlmMessage {
                role: "user".into(),
                content: transcript,
            },
        ];

        let response = self.llm.chat(&messages, None).await?;
        Ok(response.content)
    }

    // ─── Branch navigation ──────────────────────────────────────────

    /// Switch the active branch to a different entry.
    #[tracing::instrument(
        name = "conversation.branch",
        skip(self),
        fields(
            gw.session_id = %self.session_id.0,
            gw.from_leaf = tracing::field::Empty,
            gw.to_leaf = ?target,
        )
    )]
    pub async fn switch_branch(
        &mut self,
        target: EntryId,
        summarize_abandoned: bool,
    ) -> Result<(), LoopError> {
        // Verify target exists.
        if self.tree.find_entry(target).is_none() {
            return Err(LoopError::EntryNotFound(target));
        }

        let old_leaf = self.tree.active_leaf();
        if let Some(old) = old_leaf {
            tracing::Span::current().record("gw.from_leaf", tracing::field::debug(old));
        }

        // 1. Optionally summarize the abandoned branch.
        if summarize_abandoned {
            if let Some(old_leaf_id) = old_leaf {
                let abandoned = self.tree.branch_entries(old_leaf_id, target);
                if !abandoned.is_empty() {
                    let summary = self.summarize_entries(&abandoned).await?;
                    self.tree
                        .append_at(target, EntryType::BranchSummary(summary));
                }
            }
        }

        // 2. Set new active leaf.
        self.tree.set_active_leaf(target);

        // 3. Restore REPL state from snapshot on the target path.
        {
            let _restore_span = info_span!("repl.restore").entered();
            self.restore_repl_to(target)?;
        }

        info!("switched branch");
        self.flush_tree().await;
        Ok(())
    }

    /// Restore the REPL to the state at a given entry by finding the
    /// most recent snapshot on its path.
    fn restore_repl_to(&mut self, entry_id: EntryId) -> Result<(), LoopError> {
        let snapshot_data = self
            .tree
            .find_latest_snapshot(entry_id)
            .ok_or(LoopError::NoSnapshot)?;

        if let Some(raw_bytes) = &snapshot_data.raw_bytes {
            self.repl = ReplAgent::restore_snapshot(raw_bytes, Box::new(NullBridge))
                .map_err(LoopError::Agent)?;
        } else {
            // Slow path: replay code executions from the path.
            let path = self.tree.path_to(entry_id);
            let mut new_repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));

            for entry in &path {
                if let EntryType::CodeExecution { code, .. } = &entry.entry_type {
                    let _ = new_repl.execute(code);
                }
            }

            self.repl = new_repl;
        }

        Ok(())
    }

    // ─── Snapshots ──────────────────────────────────────────────────

    /// Capture the current REPL state as a ReplSnapshotData.
    pub fn capture_snapshot(&self) -> Result<ReplSnapshotData, LoopError> {
        let variables = serde_json::to_value(self.repl.get_all_variables())
            .unwrap_or(serde_json::Value::Object(Default::default()));
        let definitions = self.repl.get_definitions();
        let raw_bytes = self.repl.save_snapshot().ok();

        Ok(ReplSnapshotData {
            variables,
            definitions,
            raw_bytes,
        })
    }

    /// Append a ReplSnapshot entry to the tree.
    pub fn save_snapshot(&mut self) -> Result<EntryId, LoopError> {
        let _span = info_span!(
            "repl.snapshot",
            gw.variable_count = self.repl.get_all_variables().len(),
        )
        .entered();

        let data = self.capture_snapshot()?;
        let id = self.tree.append(EntryType::ReplSnapshot(data));
        self.turns_since_snapshot = 0;
        info!("REPL snapshot saved");
        Ok(id)
    }

    /// Check auto-snapshot policy after a turn and snapshot if warranted.
    fn post_turn_snapshot(&mut self) {
        self.turns_since_snapshot += 1;
        let policy = &self.config.snapshot_policy;
        if policy.every_n_turns > 0 && self.turns_since_snapshot >= policy.every_n_turns {
            if let Err(e) = self.save_snapshot() {
                warn!(error = %e, "auto-snapshot failed");
            }
        }
    }

    /// Check auto-compaction policy after a turn. Compacts if the user
    /// turn count exceeds the configured threshold and there are enough
    /// turns to actually compact (more than compaction_keep_count).
    pub async fn check_auto_compact(&mut self) {
        let threshold = match self.config.auto_compact_after_turns {
            Some(n) if n > 0 => n,
            _ => return,
        };

        let user_turns = self.tree.user_turn_count() as u32;
        if user_turns >= threshold {
            // Only compact if there are enough turns beyond what we keep.
            let keep = self.config.compaction_keep_count as u32;
            if user_turns > keep {
                info!(user_turns, threshold, "auto-compaction triggered");
                if let Err(e) = self.compact().await {
                    // NothingToCompact is expected if we already compacted recently.
                    if !matches!(e, LoopError::NothingToCompact) {
                        warn!(error = %e, "auto-compaction failed");
                    }
                }
            }
        }
    }

    /// Flush new tree entries to Postgres (no-op if no pg store attached).
    /// When a spine extractor is attached, fan out per-entry extraction
    /// tasks for freshly-persisted prose entries, then a single
    /// coordinator awaits them all and runs **one** resegment pass.
    ///
    /// Why debounce resegment: a turn typically flushes 3+ extractable
    /// entries (user_message, assistant_message, one or more
    /// assistant_narration). If each runs its own resegment, we pay
    /// the LLM label cost on every intermediate state — a turn with
    /// 4 entries means 4 resegment passes, 3 of which see partial
    /// state, create transient segments, then get invalidated by the
    /// next pass. Coordinating one pass per flush eliminates the
    /// invalidate→recreate churn (the user-facing log used to show
    /// `created=15 invalidated=15` style passes).
    ///
    /// Non-prose entry types (CodeExecution, HostCall, ReplSnapshot,
    /// Compaction, BranchSummary, System) are skipped at the type
    /// filter so we don't pay the task-spawn cost for entries the
    /// extractor would no-op anyway.
    pub async fn flush_tree(&mut self) {
        let new_entries = match self.tree.flush_to_pg().await {
            Ok(entries) => entries,
            Err(e) => {
                warn!(error = %e, "failed to flush session tree to Postgres");
                return;
            }
        };

        let Some(extractor) = self.spine_extractor.as_ref() else {
            return;
        };
        let session_id = self.session_id;

        let mut extract_handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();
        for entry in new_entries {
            if !is_extractable_entry_type(&entry.entry_type) {
                continue;
            }
            let extractor = Arc::clone(extractor);
            let event_tx = self.event_tx.clone();
            extract_handles.push(tokio::spawn(async move {
                run_entry_extraction(extractor, event_tx, entry).await;
            }));
        }

        if extract_handles.is_empty() {
            return;
        }

        // Single coordinator: await all per-entry extractions, then
        // run exactly one resegment pass. Spawned so flush_tree
        // returns immediately (the chat path doesn't block on spine
        // work). Per-entry failures inside the join loop are already
        // logged by run_entry_extraction; we don't gate the resegment
        // on every entry succeeding.
        let extractor = Arc::clone(extractor);
        let event_tx = self.event_tx.clone();
        tokio::spawn(async move {
            for h in extract_handles {
                let _ = h.await;
            }
            run_resegment(extractor, event_tx, session_id).await;
        });
    }
}

/// Per-entry: extract, persist, emit `SpineEntryExtracted`. No
/// resegment — the coordinator handles that once after all extractions
/// in the flush batch complete.
async fn run_entry_extraction(
    extractor: Arc<crate::spine::SpineExtractor>,
    event_tx: mpsc::UnboundedSender<LoopEvent>,
    entry: gw_core::SessionEntry,
) {
    let entry_id = entry.id;
    let extraction = match extractor.extract_entry(&entry).await {
        Ok(x) => x,
        Err(e) => {
            warn!(
                entry_id = %entry_id.0,
                error = %e,
                "spine extract failed; entry left without spine rows"
            );
            return;
        }
    };
    if extraction.entities.is_empty() && extraction.relations.is_empty() {
        return;
    }

    let pool = extractor.stores.pg.clone();
    if let Err(e) =
        crate::spine::persist::persist_entry_extraction(&pool, &extraction).await
    {
        warn!(
            entry_id = %entry_id.0,
            error = %e,
            "spine persist failed; entry left without spine rows"
        );
        return;
    }
    info!(
        entry_id = %entry_id.0,
        entity_links = extraction.entities.len(),
        relations = extraction.relations.len(),
        "spine extraction persisted"
    );

    // Outbound diagnostic event so subscribers see per-entry deltas
    // without polling Postgres.
    let entities_wire: Vec<SpineEntityLink> = extraction
        .entities
        .iter()
        .map(|l| SpineEntityLink {
            entry_id: l.entry_id,
            entity_id: l.entity_id,
            surface: l.surface.clone(),
            role: l.role.clone(),
            status: l.status.clone(),
            confidence: l.confidence,
            span_start: l.span_start,
            span_end: l.span_end,
        })
        .collect();
    let relations_wire: Vec<SpineRelation> = extraction
        .relations
        .iter()
        .map(|r| SpineRelation {
            entry_id: r.entry_id,
            subject_id: r.subject_id,
            object_id: r.object_id,
            predicate: r.predicate.clone(),
            directed: r.directed,
            surface: r.surface.clone(),
            confidence: r.confidence,
            span_start: r.span_start,
            span_end: r.span_end,
        })
        .collect();
    let _ = event_tx.send(LoopEvent::SpineEntryExtracted {
        entry_id,
        entities: entities_wire,
        relations: relations_wire,
    });
}

/// Single resegment pass + outbound `SpineSegmentsUpdated`. Called
/// once per flush_tree batch by the coordinator task.
async fn run_resegment(
    extractor: Arc<crate::spine::SpineExtractor>,
    event_tx: mpsc::UnboundedSender<LoopEvent>,
    session_id: SessionId,
) {
    let pool = extractor.stores.pg.clone();
    let llm = extractor.stores.llm.clone();
    match crate::spine::resegment::resegment(
        &pool,
        llm.as_ref(),
        session_id.0,
        &crate::spine::resegment::ResegmentOpts::default(),
    )
    .await
    {
        Ok((report, snapshots)) => {
            info!(
                session_id = %session_id.0,
                created = report.segments_created,
                updated = report.segments_updated,
                invalidated = report.segments_invalidated,
                "spine resegment complete"
            );
            let segments_wire: Vec<SpineSegmentSnapshot> = snapshots
                .into_iter()
                .map(|s| SpineSegmentSnapshot {
                    segment_id: s.segment_id,
                    session_id: s.session_id,
                    label: s.label,
                    kind: s.kind,
                    entry_first: s.entry_first,
                    entry_last: s.entry_last,
                    entity_ids: s.entity_ids,
                    summary: s.summary,
                })
                .collect();
            let _ = event_tx.send(LoopEvent::SpineSegmentsUpdated {
                session_id,
                segments: segments_wire,
            });
        }
        Err(e) => warn!(
            session_id = %session_id.0,
            error = %e,
            "spine resegment failed; segments stale"
        ),
    }
}

/// Predicate used by `flush_tree` to skip entry types that the spine
/// extractor would no-op anyway. Mirrors the inner match in
/// `SpineExtractor::raw_extract_entry` — keep them in sync.
fn is_extractable_entry_type(t: &EntryType) -> bool {
    matches!(
        t,
        EntryType::UserMessage(_)
            | EntryType::AssistantMessage { .. }
            | EntryType::AssistantNarration { .. }
    )
}

/// Minimal HostBridge that rejects all calls.
struct NullBridge;

impl HostBridge for NullBridge {
    fn call(
        &mut self,
        function: &str,
        _args: Vec<serde_json::Value>,
        _kwargs: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<ouros::Object, gw_runtime::AgentError> {
        Err(gw_runtime::AgentError::UnknownFunction(
            function.to_string(),
        ))
    }
}
