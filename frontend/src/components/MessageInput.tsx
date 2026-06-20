import { useState } from 'react';

interface Props {
  onSend: (content: string) => void;
  /** Text-field disabled (no queuing typed input while a turn is in
   *  flight). Independent of `running` — caller decides which turns
   *  the user can pre-type into. */
  disabled?: boolean;
  /** True while the conversation loop is mid-turn. Drives the
   *  Send → Stop button morph. Decoupled from `disabled` so callers
   *  can allow pre-typing (wiki drawer open, etc.) while still
   *  showing the Stop affordance. */
  running?: boolean;
  /** Click handler for the Stop button. Required for the morph to
   *  appear; absent means we render Send even when running. */
  onCancel?: () => void;
}

export function MessageInput({ onSend, disabled, running, onCancel }: Props) {
  const [value, setValue] = useState('');

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue('');
  };

  const showStop = running && onCancel;

  return (
    <form
      className="message-input"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Say something…"
        disabled={disabled}
        autoFocus
      />
      {showStop ? (
        <button
          type="button"
          className="message-input-stop"
          onClick={onCancel}
          aria-label="Stop current turn"
        >
          Stop
        </button>
      ) : (
        <button type="submit" disabled={disabled || !value.trim()}>
          Send
        </button>
      )}
    </form>
  );
}
