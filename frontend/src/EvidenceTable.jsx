import { displayValue, evidenceColumns, fieldLabel, tableLabel } from './evidenceDisplay.js'

export default function EvidenceTable({ item }) {
  const rows = Array.isArray(item.shown_rows) ? item.shown_rows : []
  const columns = evidenceColumns(item.table, rows)
  const hiddenCount = Math.max(0, Number(item.total_rows) - rows.length)

  return (
    <section className="evidence-table" aria-label={`${tableLabel(item.table)} records`}>
      <div className="evidence-title">
        <div className="evidence-heading">
          <h3 className="evidence-table-name">{tableLabel(item.table)}</h3>
          <span className="evidence-table-code">{item.table}</span>
        </div>
        <span className="evidence-count">{item.total_rows} matching {item.total_rows === 1 ? 'row' : 'rows'}</span>
      </div>

      {rows.length ? (
        <>
          <div className="evidence-scroll">
            <table className="records-table">
              <thead className="records-head">
                <tr className="records-row">
                  {columns.map((field) => <th className="records-cell" scope="col" key={field}>{fieldLabel(field)}</th>)}
                </tr>
              </thead>
              <tbody className="records-body">
                {rows.map((row, index) => (
                  <tr className="records-row" key={index}>
                    {columns.map((field) => (
                      <td className="records-cell" key={field} title={displayValue(row[field])}>
                        {displayValue(row[field])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {hiddenCount > 0 && <p className="evidence-note">Showing {rows.length} of {item.total_rows} matching rows.</p>}
        </>
      ) : (
        <p className="evidence-note">{item.total_rows ? 'No rows were included in the model evidence.' : 'No matching rows.'}</p>
      )}
    </section>
  )
}
