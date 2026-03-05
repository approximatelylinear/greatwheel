"""
Triage agent — classifies incoming tasks and dispatches to specialist agents.

This is a stub. The rLM will receive tasks and use the SDK to classify
and route them to the appropriate worker agent.
"""


def handle(task):
    """Entry point called by the runtime with the incoming task."""
    # Classify the task
    classification = llm.complete(
        system="Classify this task into one of: research, code, general. Reply with just the category.",
        messages=[{"role": "user", "content": task["payload"]}],
    )

    category = classification.strip().lower()

    # Dispatch to specialist (or handle directly for 'general')
    if category == "research":
        result = agent.call("research", task)
    else:
        # Handle directly
        result = llm.complete(
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": task["payload"]}],
        )

    channel.send(result)
