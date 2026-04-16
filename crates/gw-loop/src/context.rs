use gw_core::{EntryType, LlmMessage, SessionEntry};

/// The assembled context for an LLM turn.
#[derive(Debug, Clone)]
pub struct TurnContext {
    pub messages: Vec<LlmMessage>,
    pub model: Option<String>,
}

/// Options for context building.
#[derive(Debug, Clone, Default)]
pub struct ContextOptions {
    /// When true, include CodeExecution entries as user messages
    /// showing REPL output. Required for benchmark-style execution
    /// where the LLM needs to see search results, document text, etc.
    pub include_code_output: bool,
    /// Maximum chars for code output. 0 = no limit.
    pub repl_output_max_chars: usize,
}

/// Build the LLM prompt from the session path.
///
/// - If a Compaction entry exists, uses its summary and skips entries before `first_kept_id`.
/// - Keeps only the last `recency_window` user/assistant pairs.
/// - Prepends a system message with the system prompt and REPL state summary.
/// - When `opts.include_code_output` is true, CodeExecution entries become user messages.
pub fn build_turn_context(
    path: &[&SessionEntry],
    system_prompt: &str,
    repl_state_summary: &str,
    recency_window: usize,
) -> TurnContext {
    build_turn_context_with_opts(
        path,
        system_prompt,
        repl_state_summary,
        recency_window,
        &ContextOptions::default(),
    )
}

/// Build context with explicit options.
pub fn build_turn_context_with_opts(
    path: &[&SessionEntry],
    system_prompt: &str,
    repl_state_summary: &str,
    recency_window: usize,
    opts: &ContextOptions,
) -> TurnContext {
    let mut model = None;
    let mut skip_before: Option<gw_core::EntryId> = None;
    let mut compaction_summary: Option<String> = None;

    // Scan for the most recent Compaction entry.
    for entry in path.iter().rev() {
        if let EntryType::Compaction {
            summary,
            first_kept_id,
            ..
        } = &entry.entry_type
        {
            compaction_summary = Some(summary.clone());
            skip_before = Some(*first_kept_id);
            break;
        }
    }

    // Collect messages, respecting compaction boundary.
    let mut conversation: Vec<LlmMessage> = Vec::new();
    let mut past_boundary = skip_before.is_none();

    for entry in path {
        if !past_boundary {
            if entry.id == skip_before.unwrap() {
                past_boundary = true;
            } else {
                continue;
            }
        }

        match &entry.entry_type {
            EntryType::UserMessage(content) => {
                conversation.push(LlmMessage {
                    role: "user".into(),
                    content: content.clone(),
                });
            }
            EntryType::AssistantMessage { content, model: m } => {
                if let Some(m) = m {
                    model = Some(m.clone());
                }
                conversation.push(LlmMessage {
                    role: "assistant".into(),
                    content: content.clone(),
                });
            }
            EntryType::CodeExecution { stdout, result, .. } if opts.include_code_output => {
                let output = format_code_output(stdout, result, opts.repl_output_max_chars);
                if !output.is_empty() {
                    conversation.push(LlmMessage {
                        role: "user".into(),
                        content: format!("REPL output:\n```\n{output}\n```"),
                    });
                }
            }
            EntryType::BranchSummary(summary) => {
                conversation.push(LlmMessage {
                    role: "system".into(),
                    content: format!("[Branch summary] {summary}"),
                });
            }
            // Skip other entry types (CodeExecution when not include_code_output,
            // HostCall, ReplSnapshot, System, Compaction)
            EntryType::CodeExecution { .. }
            | EntryType::HostCall { .. }
            | EntryType::ReplSnapshot(_)
            | EntryType::Compaction { .. }
            | EntryType::System(_) => {}
        }
    }

    // Apply recency window: keep last N user/assistant pairs.
    if recency_window > 0 {
        let mut user_count = 0;
        let mut cutoff = None;
        for (i, msg) in conversation.iter().enumerate().rev() {
            if msg.role == "user" {
                user_count += 1;
                if user_count == recency_window {
                    cutoff = Some(i);
                    break;
                }
            }
        }
        if let Some(idx) = cutoff {
            if idx > 0 {
                conversation = conversation.split_off(idx);
            }
        }
    }

    // Build the system message.
    let mut system_content = system_prompt.to_string();
    if let Some(summary) = compaction_summary {
        system_content.push_str(&format!("\n\n[Conversation summary]\n{summary}"));
    }
    if !repl_state_summary.is_empty() {
        system_content.push_str(&format!("\n\n[REPL state]\n{repl_state_summary}"));
    }

    let mut messages = vec![LlmMessage {
        role: "system".into(),
        content: system_content,
    }];
    messages.extend(conversation);

    TurnContext { messages, model }
}

/// Format CodeExecution output for inclusion in the LLM prompt.
fn format_code_output(stdout: &str, result: &serde_json::Value, max_chars: usize) -> String {
    let mut output = String::new();

    if !stdout.is_empty() {
        output.push_str(stdout);
        if !stdout.ends_with('\n') {
            output.push('\n');
        }
    }

    // Include the return value if it's not null/None.
    if !result.is_null() {
        let result_str = result.to_string();
        if result_str != "null" && result_str != "\"None\"" {
            output.push_str(&format!("→ {result_str}\n"));
        }
    }

    // Truncate if needed (at char boundary, not byte boundary).
    if max_chars > 0 && output.len() > max_chars {
        let truncate_at = output
            .char_indices()
            .take(max_chars)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        output.truncate(truncate_at);
        output.push_str("\n...\n[truncated]");
    }

    output
}
