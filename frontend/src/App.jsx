import { useEffect, useState } from 'react'
import { serializeSearchRequest } from './searchPayload.js'
import EvidenceTable from './EvidenceTable.jsx'

const statusLabels = {
  offline: 'Local server offline',
  not_running: 'Not running',
  connecting: 'Connecting',
  checking: 'Checking HiPerGator',
  booting: 'Booting up',
  ready: 'Running',
  disconnected: 'Disconnected',
  error: 'Connection error',
}

export default function App() {
  const [personId, setPersonId] = useState('')
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [randomLoading, setRandomLoading] = useState(false)
  const [tunnelStatus, setTunnelStatus] = useState({
    state: 'offline',
    detail: 'Connecting to the local server…',
  })
  const [retrying, setRetrying] = useState(false)
  const tunnelReady = tunnelStatus.state === 'ready'

  useEffect(() => {
    let socket
    let retryTimer
    let stopped = false

    function connect() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/tunnel/status`)
      socket.onmessage = (event) => {
        try {
          setTunnelStatus(JSON.parse(event.data))
        } catch {
          setTunnelStatus({ state: 'error', detail: 'Invalid tunnel status received.' })
        }
      }
      socket.onclose = () => {
        if (stopped) return
        setTunnelStatus({ state: 'offline', detail: 'Cannot reach the local server.' })
        retryTimer = window.setTimeout(connect, 3000)
      }
      socket.onerror = () => socket.close()
    }

    connect()
    return () => {
      stopped = true
      window.clearTimeout(retryTimer)
      socket?.close()
    }
  }, [])

  async function retryTunnel() {
    setRetrying(true)
    try {
      const response = await fetch('/tunnel/reconnect', { method: 'POST' })
      const body = await response.json()
      if (!response.ok) throw new Error(body.error || 'Could not reconnect.')
      setTunnelStatus(body)
    } catch (cause) {
      setTunnelStatus({ state: 'error', detail: cause.message })
    } finally {
      setRetrying(false)
    }
  }

  async function chooseRandomPerson() {
    setRandomLoading(true)
    setError('')
    try {
      const response = await fetch('/random-person')
      const body = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(body?.error || `Could not get a patient ID (HTTP ${response.status}).`)
      }
      if (typeof body?.person_id !== 'string' || !/^-?[0-9]+$/.test(body.person_id)) {
        throw new Error('The server returned an invalid patient ID.')
      }
      setPersonId(body.person_id)
      setResult(null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not get a patient ID.')
    } finally {
      setRandomLoading(false)
    }
  }

  async function search(event) {
    event.preventDefault()
    let requestBody
    try {
      requestBody = serializeSearchRequest(personId, question.trim())
    } catch (cause) {
      setError(cause.message)
      return
    }
    if (!question.trim()) {
      setError('Enter a question about the patient.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await fetch('/patient-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: requestBody,
      })
      const responseBody = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(responseBody?.error || `Search failed (HTTP ${response.status}).`)
      }
      setResult(responseBody)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Search failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page">
      <header className="masthead">
        <div className="brand-mark" aria-hidden="true">✳</div>
        <div className="masthead-copy">
          <p className="eyebrow">OMOP EHR AGENT</p>
          <h1 className="page-title">Patient search</h1>
          <p className="subtitle">Ask a question and review the records behind the answer.</p>
        </div>
        <button
          className="random-person-button"
          type="button"
          onClick={chooseRandomPerson}
          disabled={!tunnelReady || loading || randomLoading}
        >
          {randomLoading ? 'Choosing…' : 'Random patient'}
        </button>
      </header>

      <div className={`status-banner ${tunnelStatus.state}`} role="status" aria-live="polite">
        <span className="status-dot" aria-hidden="true" />
        <div className="status-copy">
          <strong className="status-title">HiPerGator tunnel · {statusLabels[tunnelStatus.state] || 'Unknown'}</strong>
          <p className="status-detail">{tunnelStatus.detail}</p>
        </div>
        {(tunnelStatus.state === 'error' || tunnelStatus.state === 'disconnected') && (
          <button className="status-retry" type="button" onClick={retryTunnel} disabled={retrying}>
            {retrying ? 'Retrying…' : 'Retry'}
          </button>
        )}
      </div>

      <section className="panel" aria-labelledby="search-heading">
        <div className="section-heading">
          <span className="step">01</span>
          <h2 className="section-title" id="search-heading">New search</h2>
        </div>
        <form className="search-form" onSubmit={search}>
          <div className="field">
            <label className="field-label" htmlFor="person-id">Patient ID</label>
            <input
              className="field-control"
              id="person-id"
              type="text"
              inputMode="numeric"
              placeholder="e.g. 1001"
              value={personId}
              onChange={(event) => setPersonId(event.target.value)}
              disabled={!tunnelReady || loading || randomLoading}
              required
            />
          </div>
          <div className="field field-question">
            <label className="field-label" htmlFor="question">Your question</label>
            <textarea
              className="field-control"
              id="question"
              rows="4"
              placeholder="What were this patient's recent visits?"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              disabled={!tunnelReady || loading}
              required
            />
          </div>
          <div className="form-footer">
            <p className="form-hint">Answers are based on retrieved OMOP records.</p>
            <button className="submit-button" type="submit" disabled={!tunnelReady || loading || randomLoading}>
              {loading ? 'Searching…' : 'Search patient'}
              {!loading && <span className="button-arrow" aria-hidden="true"> →</span>}
            </button>
          </div>
        </form>
      </section>

      {error && <div className="message error" role="alert">{error}</div>}
      {loading && <div className="message" role="status">The model is reviewing patient records. This may take a moment.</div>}

      {result && (
        <section className="panel result" aria-labelledby="answer-heading">
          <div className="section-heading">
            <span className="step">02</span>
            <h2 className="section-title" id="answer-heading">Answer</h2>
          </div>
          <p className="answer">{result.answer}</p>
          <details className="evidence">
            <summary className="evidence-toggle">View retrieved records ({result.evidence?.length ?? 0} tables)</summary>
            <p className="evidence-date-note">
              {result.evidence?.[0]?.retrieval_scope === 'history'
                ? 'Evidence includes earlier encounters.'
                : 'Evidence is limited to the latest encounter unless the question requests history.'}
              {' '}Dates are de-identified and may be shifted by more than a century.
            </p>
            {(result.evidence ?? []).map((item, index) => (
              <EvidenceTable item={item} key={`${item.table}-${index}`} />
            ))}
          </details>
        </section>
      )}
    </main>
  )
}
