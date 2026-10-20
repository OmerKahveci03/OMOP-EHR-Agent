import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function formatNumber(value) {
  return value == null ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function formatBinBound(value, width) {
  const decimals = width > 0 ? Math.min(8, Math.max(0, 1 - Math.floor(Math.log10(width)))) : 2
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: decimals })
}

function valueAxisLabel(label, unit) {
  return `${label} (${unit === 'Unspecified' ? 'unit unspecified' : unit})`
}

function formatDate(timestamp, includeTime = false) {
  const date = new Date(timestamp)
  return includeTime
    ? date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
    : date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function MeasurementTooltip({ active, payload, unit }) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return (
    <div className="analysis-tooltip">
      <strong>{formatDate(point.time, true)}</strong>
      <span>{formatNumber(point.value)} {unit === 'Unspecified' ? '(unit unspecified)' : unit}</span>
      <small>Measurement ID: {point.measurement_id}</small>
    </div>
  )
}

function BinTooltip({ active, payload, unit, lastEnd }) {
  if (!active || !payload?.length) return null
  const bin = payload[0].payload
  const endBracket = bin.end === lastEnd ? ']' : ')'
  const width = bin.end - bin.start
  return (
    <div className="analysis-tooltip">
      <strong>{`[${formatBinBound(bin.start, width)}, ${formatBinBound(bin.end, width)}${endBracket} ${unit === 'Unspecified' ? '' : unit}`}</strong>
      <span>{formatNumber(bin.count)} measurements</span>
    </div>
  )
}

function TimeSeriesChart({ points, label, unit }) {
  const data = points.map((point) => ({ ...point, time: Date.parse(point.date) }))
    .filter((point) => Number.isFinite(point.time) && Number.isFinite(point.value))
    .sort((left, right) => left.time - right.time)
  if (!data.length) return <p className="analysis-empty">No dated numeric values to plot.</p>

  const minTime = data[0].time
  const maxTime = data.at(-1).time
  const includeTime = maxTime - minTime <= 48 * 60 * 60 * 1000
  const timeDomain = minTime === maxTime ? [minTime - 60 * 60 * 1000, maxTime + 60 * 60 * 1000] : [minTime, maxTime]

  return (
    <div className="analysis-chart" role="img" aria-label={`${label} over time, ${data.length} measurements in ${unit}`}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 18, bottom: 18, left: 18 }} accessibilityLayer>
          <CartesianGrid stroke="#e3eae6" strokeDasharray="3 3" />
          <XAxis dataKey="time" type="number" domain={timeDomain} scale="time"
            tickFormatter={(time) => formatDate(time, includeTime)} minTickGap={32}
            tick={{ fill: '#667d77', fontSize: 11 }} height={55}
            label={{ value: 'Measurement date and time', position: 'insideBottom', offset: -12, fill: '#526b62', fontSize: 12 }} />
          <YAxis type="number" domain={['auto', 'auto']} tickFormatter={formatNumber}
            tick={{ fill: '#667d77', fontSize: 11 }} width={62}
            label={{ value: valueAxisLabel(label, unit), angle: -90, position: 'insideLeft', fill: '#526b62', fontSize: 12 }} />
          <Tooltip content={<MeasurementTooltip unit={unit} />} />
          <Line dataKey="value" type="linear" stroke="#24745e" strokeWidth={2}
            dot={{ r: 3, fill: '#24745e' }} activeDot={{ r: 5 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function DistributionChart({ bins, stats, label, unit }) {
  if (!bins.length) return <p className="analysis-empty">No numeric values to plot.</p>
  const data = bins.map((bin) => ({
    ...bin,
    range: `${formatBinBound(bin.start, bin.end - bin.start)}–${formatBinBound(bin.end, bin.end - bin.start)}`,
  }))
  const mostCommon = data.reduce((best, bin) => bin.count > best.count ? bin : best, data[0])

  return (
    <>
      {stats?.count > 0 && (
        <dl className="analysis-highlights">
          <div><dt>Median</dt><dd>{formatNumber(stats.median)} {unit}</dd></div>
          <div><dt>Mean</dt><dd>{formatNumber(stats.mean)} {unit}</dd></div>
          <div><dt>Middle 50% (Q1–Q3)</dt><dd>{formatNumber(stats.q1)}–{formatNumber(stats.q3)} {unit}</dd></div>
          <div><dt>Minimum–maximum</dt><dd>{formatNumber(stats.min)}–{formatNumber(stats.max)} {unit}</dd></div>
        </dl>
      )}
      <div className="analysis-chart" role="img" aria-label={`Distribution of ${label}: measurement count by ${unit} value range`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 20, right: 18, bottom: 0, left: 18 }} barCategoryGap="24%" accessibilityLayer>
            <CartesianGrid stroke="#e3eae6" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="range" tick={{ fill: '#667d77', fontSize: 11 }} height={28} minTickGap={8} />
            <YAxis allowDecimals={false} domain={[0, 'auto']} tick={{ fill: '#667d77', fontSize: 11 }} width={62}
              label={{ value: 'Number of measurements', angle: -90, position: 'insideLeft', fill: '#526b62', fontSize: 12 }} />
            <Tooltip content={<BinTooltip unit={unit} lastEnd={data.at(-1).end} />} />
            <Bar dataKey="count" fill="#24745e" maxBarSize={86} isAnimationActive={false}>
              <LabelList dataKey="count" position="top" formatter={(count) => count || ''} fill="#526b62" fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="analysis-axis-title">{valueAxisLabel(label, unit)} ranges</p>
      <p className="analysis-chart-note">Highest-count range: {mostCommon.range} {unit} ({mostCommon.count} of {stats?.count ?? data.reduce((count, bin) => count + bin.count, 0)} measurements).</p>
      <p className="analysis-chart-note">Each bar counts measurements in its labeled range. Hover for the precise bounds; each upper bound is excluded except the last.</p>
    </>
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
          {analysis.kind === 'time_series' && <TimeSeriesChart points={series.points} label={series.label} unit={series.unit} />}
          {analysis.kind === 'distribution' && <DistributionChart bins={series.bins} stats={series.stats} label={series.label} unit={series.unit} />}
          {analysis.kind === 'statistics' && (
            <dl className="analysis-stats">
              {Object.entries(series.stats).map(([name, value]) => (
                <div key={name}><dt>{name.replace('_', ' ')}</dt><dd>{formatNumber(value)}</dd></div>
              ))}
            </dl>
          )}
        </section>
      ))}
      {analysis.series.length > 1 && <p className="analysis-context">Each chart retains its concept and source unit label; values are not converted or combined.</p>}
    </div>
  )
}
