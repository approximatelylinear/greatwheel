interface Props {
  content: string;
  language: string | null;
  diff: boolean;
  title: string | null;
}

export function CodeBlock({ content, language, diff, title }: Props) {
  const lines = content.split('\n');
  return (
    <div className="a2ui-code">
      {(title || language) && (
        <div className="a2ui-code-head">
          {title && <span className="a2ui-code-title">{title}</span>}
          {language && <span className="a2ui-code-lang">{language}</span>}
        </div>
      )}
      <pre className="a2ui-code-body">
        {diff
          ? lines.map((line, i) => (
              <div key={i} className={`a2ui-code-line ${diffClass(line)}`}>
                {line || ' '}
              </div>
            ))
          : lines.map((line, i) => (
              <div key={i} className="a2ui-code-line">
                {line || ' '}
              </div>
            ))}
      </pre>
    </div>
  );
}

function diffClass(line: string): string {
  if (line.startsWith('+++') || line.startsWith('---')) return 'diff-meta';
  if (line.startsWith('@@')) return 'diff-hunk';
  if (line.startsWith('+')) return 'diff-add';
  if (line.startsWith('-')) return 'diff-del';
  return '';
}
