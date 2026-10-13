import { useState } from 'react'
import { serializeSearchRequest } from './searchPayload.js'

export default function App() {
  const [personId, setPersonId] = useState('')
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

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
      </header>

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
              disabled={loading}
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
              disabled={loading}
              required
            />
          </div>
          <div className="form-footer">
            <p className="form-hint">Answers are based on retrieved OMOP records.</p>
            <button className="submit-button" type="submit" disabled={loading}>
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
            {(result.evidence ?? []).map((item, index) => (
              <div className="evidence-table" key={`${item.table}-${index}`}>
                <div className="evidence-title">
                  <strong className="evidence-table-name">{item.table}</strong>
                  <span className="evidence-count">{item.total_rows} matching rows</span>
                </div>
                <pre className="evidence-json">{JSON.stringify(item.shown_rows, null, 2)}</pre>
              </div>
            ))}
          </details>
        </section>
      )}
    </main>
  )
}
