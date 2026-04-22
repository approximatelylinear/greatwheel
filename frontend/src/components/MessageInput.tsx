import { useState } from 'react';

interface Props {
  onSend: (content: string) => void;
  disabled?: boolean;
}

export function MessageInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState('');

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue('');
  };

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
      <button type="submit" disabled={disabled || !value.trim()}>
        Send
      </button>
    </form>
  );
}
