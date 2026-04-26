import { useMemo, useState } from 'react';

interface CloudPoint {
  id: string;
  label: string;
  x: number;
  y: number;
  kind?: string;
}

interface Props {
  points: CloudPoint[];
  highlight: Record<string, boolean> | null;
  onPointClick: (id: string) => void;
}

/**
 * SVG scatter plot of typed entities. Auto-fits a viewBox to the
 * incoming `(x, y)` coords (server already scales them roughly to
 * [-1, 1]) with padding for labels. Hover reveals a tooltip; click
 * fires `onPointClick(id)`.
 *
 * Pan/zoom is deliberately deferred — we render at fit-to-view and
 * let the agent re-emit a smaller subset (or supersede with a
 * sub-cloud) if density is unreadable. Adding pan/zoom is one of the
 * follow-up items in `docs/design-demo-literature-assistant.md`.
 */
export function EntityCloudWidget({ points, highlight, onPointClick }: Props) {
  const [hover, setHover] = useState<CloudPoint | null>(null);

  const view = useMemo(() => {
    if (points.length === 0) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    // Add a margin so labels and points don't kiss the edge.
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

  if (points.length === 0 || !view) {
    return (
      <div className="entity-cloud entity-cloud-empty">
        No points to render yet.
      </div>
    );
  }

  // Render in a ~600x400 logical viewBox so absolute pixel sizes for
  // strokes and dots feel right; the SVG itself scales to its
  // container width via CSS.
  const W = 600;
  const H = 400;
  const sx = (x: number) =>
    ((x - view.minX) / (view.maxX - view.minX)) * W;
  const sy = (y: number) =>
    ((y - view.minY) / (view.maxY - view.minY)) * H;

  return (
    <div className="entity-cloud">
      <svg
        className="entity-cloud-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        onMouseLeave={() => setHover(null)}
      >
        {/* faint grid for orientation */}
        <line x1={W / 2} y1={0} x2={W / 2} y2={H} className="cloud-axis" />
        <line x1={0} y1={H / 2} x2={W} y2={H / 2} className="cloud-axis" />
        {points.map((p) => {
          const cx = sx(p.x);
          const cy = sy(p.y);
          const isHi = highlight?.[p.id] === true;
          const kind = p.kind ?? 'paper';
          return (
            <g
              key={p.id}
              className={`cloud-point cloud-point-${kind}${isHi ? ' hi' : ''}`}
              onMouseEnter={() => setHover(p)}
              onClick={() => onPointClick(p.id)}
            >
              <circle cx={cx} cy={cy} r={6} />
              <circle cx={cx} cy={cy} r={2} className="cloud-point-core" />
            </g>
          );
        })}
        {hover &&
          (() => {
            const cx = sx(hover.x);
            const cy = sy(hover.y);
            const above = cy > H / 2;
            return (
              <g className="cloud-tooltip">
                <text
                  x={cx}
                  y={above ? cy - 12 : cy + 18}
                  textAnchor="middle"
                >
                  {hover.label}
                </text>
              </g>
            );
          })()}
      </svg>
      <div className="entity-cloud-legend">
        {points.length} points · click to inspect · hover for label
      </div>
    </div>
  );
}
