// Show every returned column, with the most useful clinical fields first.
// Preferred fields are chosen from documents/metadata_master.csv.
export const tableColumns = {
  person: ['person_id', 'year_of_birth', 'gender_source_value', 'race_source_value', 'ethnicity_source_value'],
  visit_occurrence: ['visit_occurrence_id', 'visit_start_datetime', 'visit_end_datetime', 'visit_source_value', 'admitting_source_value', 'discharge_to_source_value'],
  visit_detail: ['visit_detail_id', 'visit_detail_start_datetime', 'visit_detail_end_datetime', 'visit_detail_source_value', 'admitting_source_value', 'discharge_to_source_value'],
  condition_occurrence: ['condition_occurrence_id', 'condition_start_datetime', 'condition_end_datetime', 'condition_source_value', 'condition_concept_id'],
  drug_exposure: ['drug_exposure_id', 'drug_exposure_start_datetime', 'drug_exposure_end_datetime', 'drug_source_value', 'quantity', 'dose_unit_source_value', 'route_source_value'],
  procedure_occurrence: ['procedure_occurrence_id', 'procedure_datetime', 'procedure_source_value', 'quantity'],
  device_exposure: ['device_exposure_id', 'device_exposure_start_datetime', 'device_exposure_end_datetime', 'device_source_value', 'quantity', 'unique_device_id'],
  measurement: ['measurement_id', 'measurement_datetime', 'measurement_source_value', 'value_as_number', 'value_source_value', 'unit_source_value', 'range_low', 'range_high'],
  measurement_summary: ['summary_id', 'visit_occurrence_id', 'measurement_concept_id', 'measurement_source_value', 'unit_concept_id', 'unit_source_value', 'result_type', 'count', 'min', 'mean', 'max', 'std', 'latest_value', 'first_date', 'last_date', 'latest_measurement_id'],
  observation: ['observation_id', 'observation_datetime', 'observation_source_value', 'value_as_string', 'value_as_number', 'value_as_concept_id'],
  death: ['person_id', 'death_datetime', 'cause_concept_id', 'death_type_concept_id'],
  specimen: ['specimen_id', 'specimen_datetime', 'specimen_source_value', 'anatomic_site_concept_id', 'disease_status_concept_id'],
  observation_period: ['observation_period_id', 'observation_period_start_date', 'observation_period_end_date', 'period_type_concept_id'],
}

const labels = {
  person_id: 'Person ID',
  birth_datetime: 'Born',
  year_of_birth: 'Birth year',
  gender_source_value: 'Gender',
  race_source_value: 'Race',
  ethnicity_source_value: 'Ethnicity',
  value_as_number: 'Value',
  value_as_string: 'Value',
  value_source_value: 'Source value',
  unit_source_value: 'Unit',
  range_low: 'Range low',
  range_high: 'Range high',
  sig: 'Instructions',
  unique_device_id: 'Device ID',
  result_type: 'Result type',
  std: 'Std. dev.',
  latest_measurement_id: 'Latest measurement ID',
  summary_id: 'Summary ID',
  first_date: 'First date',
  last_date: 'Last date',
}

export function fieldLabel(field) {
  if (field.endsWith('_concept_id_name')) {
    return `${fieldLabel(field.slice(0, -'_concept_id_name'.length))} Concept Name`
  }
  if (field.endsWith('_concept_id')) {
    return `${fieldLabel(field.slice(0, -'_concept_id'.length))} Concept ID`
  }
  return labels[field] || field.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function tableLabel(table) {
  return fieldLabel(table)
}

export function evidenceColumns(table, rows) {
  const available = [...new Set(rows.flatMap((row) => Object.keys(row || {})))]
  const preferred = tableColumns[table] || []
  const ordered = [...preferred.filter((field) => available.includes(field)), ...available.filter((field) => !preferred.includes(field))]
  const columns = []
  for (const field of ordered) {
    if (columns.includes(field)) continue
    columns.push(field)
    const nameField = `${field}_name`
    if (field.endsWith('_concept_id') && available.includes(nameField)) columns.push(nameField)
  }
  return columns
}

export function displayValue(value) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
