interface CostRow {
  slug: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  est_usd?: number | null;
  mtime: string;
}

interface Props {
  rows: CostRow[];
  metric: 'tokens' | 'usd';
}

const W = 360;
const H = 80;
const PAD = 8;

export function CostTrendWidget({ rows, metric }: Props) {
  const sorted = [...rows].sort(
    (a, b) => Date.parse(a.mtime) - Date.parse(b.mtime),
  );

  const points = sorted
    .map((r, i) => {
      const v =
        metric === 'usd'
          ? r.est_usd ?? null
          : r.input_tokens + r.output_tokens;
      return v == null ? null : { i, v, row: r };
    })
    .filter((p): p is { i: number; v: number; row: CostRow } => p !== null);

  const max = points.reduce((m, p) => Math.max(m, p.v), 0);
  const denom = sorted.length > 1 ? sorted.length - 1 : 1;

  const xy = (p: { i: number; v: number }) => {
    const x = PAD + (p.i / denom) * (W - 2 * PAD);
    const y = H - PAD - (max > 0 ? (p.v / max) * (H - 2 * PAD) : 0);
    return [x, y] as const;
  };

  const path = points
    .map((p, idx) => {
      const [x, y] = xy(p);
      return `${idx === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  const total =
    metric === 'usd'
      ? points.reduce((s, p) => s + p.v, 0)
      : sorted.reduce(
          (s, r) => s + r.input_tokens + r.output_tokens,
          0,
        );

  return (
    <div className="a2ui-costtrend">
      <div className="a2ui-costtrend-head">
        <span className="a2ui-costtrend-metric">
          {metric === 'usd' ? 'est. spend' : 'tokens'} · {sorted.length} runs
        </span>
        <span className="a2ui-costtrend-total">
          {metric === 'usd'
            ? `$${total.toFixed(2)}`
            : total.toLocaleString()}
        </span>
      </div>
      <svg
        className="a2ui-costtrend-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
      >
        {points.length >= 2 && (
          <path d={path} className="a2ui-costtrend-line" fill="none" />
        )}
        {points.map((p) => {
          const [x, y] = xy(p);
          return (
            <circle
              key={p.row.slug}
              cx={x}
              cy={y}
              r={2.5}
              className="a2ui-costtrend-dot"
            >
              <title>
                {p.row.slug} · {p.row.model} ·{' '}
                {metric === 'usd'
                  ? `$${p.v.toFixed(3)}`
                  : `${p.v.toLocaleString()} tok`}
              </title>
            </circle>
          );
        })}
      </svg>
      <div className="a2ui-costtrend-rows">
        {sorted.slice(-4).map((r) => (
          <div key={r.slug} className="a2ui-costtrend-row">
            <span className="a2ui-costtrend-slug">{r.slug}</span>
            <span className="a2ui-costtrend-val">
              {metric === 'usd' && r.est_usd != null
                ? `$${r.est_usd.toFixed(3)}`
                : metric === 'usd'
                  ? '—'
                  : (r.input_tokens + r.output_tokens).toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
