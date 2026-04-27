import { useMemo, useState } from 'react';

interface CloudPoint {
  id: string;
  label: string;
  x: number;
  y: number;
  kind?: string;
  cluster?: number;
  year?: string;
  category?: string;
}

interface CloudCluster {
  id: number;
  label: string;
  x: number;
  y: number;
}

interface Props {
  points: CloudPoint[];
  clusters: CloudCluster[] | null;
  highlight: Record<string, boolean> | null;
  onPointClick: (id: string) => void;
}

const CLUSTER_COLORS = [
  '#7ba5ff',
  '#b48cde',
  '#7bd38f',
  '#d6a56e',
  '#6ec3d2',
  '#e8a0c4',
  '#d4d4a0',
  '#a8b1bf',
];

function clusterColor(id: number | undefined): string {
  if (id == null) return CLUSTER_COLORS[0]!;
  return CLUSTER_COLORS[((id % CLUSTER_COLORS.length) + CLUSTER_COLORS.length) %
    CLUSTER_COLORS.length]!;
}

/**
 * SVG scatter plot of typed entities. Auto-fits a viewBox to the
 * incoming `(x, y)` coords with padding. Hover expands the focal dot,
 * dims out-of-cluster siblings, and shows a wrapping HTML card with
 * title + year + cluster + category. Click fires `onPointClick(id)`.
 */
export function EntityCloudWidget({
  points,
  clusters,
  highlight,
  onPointClick,
}: Props) {
  const [hover, setHover] = useState<CloudPoint | null>(null);

  const view = useMemo(() => {
    if (points.length === 0) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const span = Math.max(maxX - minX, maxY - minY, 0.001);
    const margin = span * 0.18;
    return {
      minX: minX - margin,
      maxX: maxX + margin,
      minY: minY - margin,
      maxY: maxY + margin,
      span,
    };
  }, [points]);

  const clusterNameById = useMemo(() => {
    const m = new Map<number, string>();
    for (const c of clusters ?? []) m.set(c.id, c.label);
    return m;
  }, [clusters]);

  if (points.length === 0 || !view) {
    return (
      <div className="entity-cloud entity-cloud-empty">
        No points to render yet.
      </div>
    );
  }

  const W = 600;
  const H = 400;
  const sx = (x: number) =>
    ((x - view.minX) / (view.maxX - view.minX)) * W;
  const sy = (y: number) =>
    ((y - view.minY) / (view.maxY - view.minY)) * H;

  const dimmed = (p: CloudPoint) =>
    hover != null && p.id !== hover.id && p.cluster !== hover.cluster;
  const sameCluster = (p: CloudPoint) =>
    hover != null && p.id !== hover.id && p.cluster === hover.cluster;

  return (
    <div className="entity-cloud">
      <div className="entity-cloud-stage">
        <svg
          className="entity-cloud-svg"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="xMidYMid meet"
          onMouseLeave={() => setHover(null)}
        >
          <line x1={W / 2} y1={0} x2={W / 2} y2={H} className="cloud-axis" />
          <line x1={0} y1={H / 2} x2={W} y2={H / 2} className="cloud-axis" />
          {clusters?.map((c) => (
            <text
              key={`cluster-${c.id}`}
              x={sx(c.x)}
              y={sy(c.y)}
              className={`cloud-cluster-label${
                hover != null && hover.cluster !== c.id ? ' faded' : ''
              }`}
              textAnchor="middle"
              fill={clusterColor(c.id)}
            >
              {c.label}
            </text>
          ))}
          {points.map((p) => {
            const cx = sx(p.x);
            const cy = sy(p.y);
            const isHi = highlight?.[p.id] === true;
            const isHover = hover?.id === p.id;
            const kind = p.kind ?? 'paper';
            const color = clusterColor(p.cluster);
            const classes = [
              'cloud-point',
              `cloud-point-${kind}`,
              isHi ? 'hi' : '',
              isHover ? 'focal' : '',
              dimmed(p) ? 'dim' : '',
              sameCluster(p) ? 'sib' : '',
            ]
              .filter(Boolean)
              .join(' ');
            return (
              <g
                key={p.id}
                className={classes}
                onMouseEnter={() => setHover(p)}
                onClick={() => onPointClick(p.id)}
              >
                <circle cx={cx} cy={cy} r={isHover ? 11 : 6} fill={color} />
                <circle cx={cx} cy={cy} r={2} className="cloud-point-core" />
              </g>
            );
          })}
        </svg>
        {hover &&
          (() => {
            const cx = sx(hover.x);
            const cy = sy(hover.y);
            // Anchor in % of the SVG viewBox; the SVG itself fills
            // the stage width, so percentages map cleanly to pixels.
            const leftPct = (cx / W) * 100;
            const topPct = (cy / H) * 100;
            const above = cy > H * 0.55;
            // Horizontal flip near the edges so the card never runs
            // off-stage.
            const xAlign =
              leftPct < 22 ? 'left' : leftPct > 78 ? 'right' : 'center';
            const meta = [
              hover.year,
              hover.cluster != null
                ? clusterNameById.get(hover.cluster)
                : undefined,
              hover.category,
            ]
              .filter((s): s is string => !!s && s.length > 0)
              .join(' · ');
            return (
              <div
                className={`cloud-hovercard ha-${xAlign} ${
                  above ? 'va-above' : 'va-below'
                }`}
                style={{ left: `${leftPct}%`, top: `${topPct}%` }}
              >
                <div className="cloud-hovercard-title">{hover.label}</div>
                {meta && (
                  <div className="cloud-hovercard-meta">{meta}</div>
                )}
              </div>
            );
          })()}
      </div>
      <div className="entity-cloud-legend">
        {points.length} points · click to inspect · hover for label
      </div>
    </div>
  );
}
