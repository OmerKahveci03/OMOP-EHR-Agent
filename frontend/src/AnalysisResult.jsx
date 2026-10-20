function formatNumber(value) {
  return value == null ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function TimeSeriesChart({ points }) {
  const data = points.map((point) => ({ ...point, time: Date.parse(point.date) }))
    .filter((point) => Number.isFinite(point.time) && Number.isFinite(point.value))
  if (!data.length) return <p className="analysis-empty">No dated numeric values to plot.</p>

  const width = 600
  const height = 220
  const left = 54
  const right = 18
  const top = 18
  const bottom = 38
  const times = data.map((point) => point.time)
  const values = data.map((point) => point.value)
  const minTime = Math.min(...times)
  const maxTime = Math.max(...times)
  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const x = (time) => left + (maxTime === minTime ? (width - left - right) / 2 : (time - minTime) / (maxTime - minTime) * (width - left - right))
  const y = (value) => top + (maxValue === minValue ? (height - top - bottom) / 2 : (maxValue - value) / (maxValue - minValue) * (height - top - bottom))

  return (
    <svg className="analysis-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Time series of ${data.length} measurements`}>
      <line x1={left} y1={top} x2={left} y2={height - bottom} className="chart-axis" />
      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} className="chart-axis" />
      <text x={left - 7} y={top + 4} textAnchor="end" className="chart-label">{formatNumber(maxValue)}</text>
      <text x={left - 7} y={height - bottom + 4} textAnchor="end" className="chart-label">{formatNumber(minValue)}</text>
      <text x={left} y={height - 10} className="chart-label">{new Date(minTime).toLocaleDateString()}</text>
      <text x={width - right} y={height - 10} textAnchor="end" className="chart-label">{new Date(maxTime).toLocaleDateString()}</text>
      {data.length > 1 && <polyline points={data.map((point) => `${x(point.time)},${y(point.value)}`).join(' ')} className="chart-line" />}
      {data.map((point, index) => (
        <circle key={`${point.measurement_id}-${index}`} cx={x(point.time)} cy={y(point.value)} r="4" className="chart-point">
          <title>{`${point.date}: ${formatNumber(point.value)} (measurement ${point.measurement_id})`}</title>
        </circle>
      ))}
    </svg>
  )
}

function DistributionChart({ bins }) {
  if (!bins.length) return <p className="analysis-empty">No numeric values to plot.</p>
  const width = 600
  const height = 220
  const left = 54
  const right = 18
  const top = 18
  const bottom = 38
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count))
  const barWidth = plotWidth / bins.length

  return (
    <svg className="analysis-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Distribution histogram with ${bins.length} bins`}>
      <line x1={left} y1={top} x2={left} y2={height - bottom} className="chart-axis" />
      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} className="chart-axis" />
      <text x={left - 7} y={top + 4} textAnchor="end" className="chart-label">{maxCount}</text>
      <text x={left - 7} y={height - bottom + 4} textAnchor="end" className="chart-label">0</text>
      <text x={left} y={height - 10} className="chart-label">{formatNumber(bins[0].start)}</text>
      <text x={width - right} y={height - 10} textAnchor="end" className="chart-label">{formatNumber(bins.at(-1).end)}</text>
      {bins.map((bin, index) => {
        const barHeight = bin.count / maxCount * plotHeight
        return (
          <rect key={index} x={left + index * barWidth + 1} y={height - bottom - barHeight}
            width={Math.max(1, barWidth - 2)} height={barHeight} className="chart-bar">
            <title>{`${formatNumber(bin.start)}–${formatNumber(bin.end)}: ${bin.count}`}</title>
          </rect>
        )
      })}
    </svg>
  )
}

export default function AnalysisResult({ analysis }) {
  if (!analysis) return null
  return (
    <div className="analysis-results">
      <p className="analysis-context">{analysis.source_rows} matching records · {analysis.scope === 'history' ? 'All encounters' : 'Latest encounter'}</p>
      {analysis.series.length === 0 && <p className="analysis-empty">No matching measurements.</p>}
      {analysis.series.map((series) => (
        <section className="analysis-series" key={`${series.concept_id}-${series.unit_concept_id}-${series.unit}`}>
          <h3>{series.label} <span>({series.concept_id})</span></h3>
          <p>{series.numeric_count} numeric values · {series.excluded_count} excluded · Unit: {series.unit}</p>
          {analysis.kind === 'time_series' && <TimeSeriesChart points={series.points} />}
          {analysis.kind === 'distribution' && <DistributionChart bins={series.bins} />}
          {analysis.kind === 'statistics' && (
            <dl className="analysis-stats">
              {Object.entries(series.stats).map(([name, value]) => (
                <div key={name}><dt>{name.replace('_', ' ')}</dt><dd>{formatNumber(value)}</dd></div>
              ))}
            </dl>
          )}
        </section>
      ))}
    </div>
  )
}
