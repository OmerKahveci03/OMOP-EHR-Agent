function ConceptItem({ concept, selected = false, source }) {
  return (
    <li className={`concept-item${selected ? ' selected' : ''}`}>
      <span className="concept-item-name">{concept.name}</span>
      <span className="concept-item-meta">
        <code>{concept.concept_id}</code>
        <span>{concept.row_count} patient records</span>
        {source && <span className="concept-source">{source === 'ancestor' ? 'via ancestor' : 'direct'}</span>}
        {selected && !source && <span className="concept-source">selected</span>}
      </span>
    </li>
  )
}

export default function ConceptSelection({ selection }) {
  const selectedIds = new Set(selection.selected_concepts.map((concept) => concept.concept_id))
  const selectedAncestorIds = new Set(selection.selected_ancestors.map((ancestor) => ancestor.ancestor_id))

  return (
    <aside className="concept-sidebar" aria-labelledby="concept-sidebar-title">
      <h3 id="concept-sidebar-title">Concept selection</h3>
      <p className="concept-sidebar-intro">Patient measurement concepts considered for this question.</p>

      <div className="concept-sidebar-section">
        <h4>Used for this result</h4>
        {selection.selection_mode === 'all_measurements' && <p>All patient measurements; no concept filter was applied.</p>}
        {selection.selection_mode === 'none' && <p>No measurement concepts were selected.</p>}
        {selection.selected_ancestors.length > 0 && (
          <div className="concept-ancestor-selection">
            <span>Selected ancestor groups</span>
            <ul>
              {selection.selected_ancestors.map((ancestor) => (
                <li key={ancestor.ancestor_id}>{ancestor.name} <code>{ancestor.ancestor_id}</code></li>
              ))}
            </ul>
          </div>
        )}
        {selection.selected_concepts.length > 0 && (
          <ul className="concept-list selected-list">
            {selection.selected_concepts.map((concept) => (
              <ConceptItem concept={concept} source={concept.source} selected key={concept.concept_id} />
            ))}
          </ul>
        )}
      </div>

      <div className="concept-sidebar-section">
        <h4>Candidate concepts <span>({selection.candidates.length} shown)</span></h4>
        <p>Ranked from this patient’s measurement history; the shortlist is capped at 40.</p>
        {selection.candidates.length > 0 ? (
          <ul className="concept-list candidate-list">
            {selection.candidates.map((concept) => (
              <ConceptItem concept={concept} selected={selectedIds.has(concept.concept_id)} key={concept.concept_id} />
            ))}
          </ul>
        ) : <p>No candidate measurement concepts were found.</p>}
      </div>

      {selection.ancestor_candidates.length > 0 && (
        <details className="concept-sidebar-section ancestor-candidates">
          <summary>Ancestor groups considered ({selection.ancestor_candidates.length})</summary>
          <ul className="concept-list">
            {selection.ancestor_candidates.map((ancestor) => (
              <li className={`concept-item${selectedAncestorIds.has(ancestor.ancestor_id) ? ' selected' : ''}`} key={ancestor.ancestor_id}>
                <span className="concept-item-name">{ancestor.name}</span>
                <span className="concept-item-meta">
                  <code>{ancestor.ancestor_id}</code>
                  <span>{ancestor.patient_descendant_count} patient concepts</span>
                  {selectedAncestorIds.has(ancestor.ancestor_id) && <span className="concept-source">selected</span>}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </aside>
  )
}
