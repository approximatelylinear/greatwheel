type Cell = 'exact' | 'fuzzy' | 'wrong' | 'error' | 'missing';

interface Props {
  queries: string[];
  runs: string[];
  cells: Cell[][];
  onCellClick: (queryId: string, slug: string) => void;
}

export function DifficultyMatrixWidget({
  queries,
  runs,
  cells,
  onCellClick,
}: Props) {
  return (
    <div className="a2ui-difmat-wrap">
      <table className="a2ui-difmat">
        <thead>
          <tr>
            <th className="a2ui-difmat-corner">query</th>
            {runs.map((slug) => (
              <th key={slug} className="a2ui-difmat-runhead" title={slug}>
                <span>{slug}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {queries.map((qid, qi) => (
            <tr key={qid}>
              <th className="a2ui-difmat-qhead">{qid}</th>
              {runs.map((slug, ri) => {
                const cell = cells[qi]?.[ri] ?? 'missing';
                return (
                  <td
                    key={slug}
                    className={`a2ui-difmat-cell cell-${cell}`}
                    title={`${slug} · ${qid} · ${cell}`}
                    onClick={() => onCellClick(qid, slug)}
                  />
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="a2ui-difmat-legend">
        <span className="cell-exact" /> exact
        <span className="cell-fuzzy" /> fuzzy
        <span className="cell-wrong" /> wrong
        <span className="cell-error" /> error
        <span className="cell-missing" /> missing
      </div>
    </div>
  );
}
