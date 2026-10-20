// Keep this catalog in sync with documents/EHR_AGENT_TEST_QUESTIONS.md.
export const testQuestions = [
  { label: 'Visits', question: "Show me this patient's visits, with the most recent visit first." },
  { label: 'Latest encounter measurements', question: 'What measurements were recorded during the latest encounter? Show the values, units, and dates.' },
  { label: 'Creatinine history', question: "Show this patient's creatinine lab results across all encounters." },
  { label: 'Blood pressure over time', question: "Show this patient's systolic and diastolic blood pressure measurements over time during the latest encounter." },
  { label: 'Heart rate distribution', question: "Show the distribution of this patient's heart rate measurements across all encounters." },
  { label: 'Glucose statistics', question: "For the latest encounter, what are the count, minimum, median, maximum, and standard deviation of this patient's glucose measurements?" },
  { label: 'Condition history', question: "Which conditions appear in this patient's record, and when were they recorded?" },
  { label: 'Latest encounter medications', question: "What medications were recorded during this patient's latest encounter? Show the underlying records." },
  { label: 'Urine output rate', question: "Calculate and summarize this patient's urine output rate during the latest encounter. Show the source measurements and explain the units and time intervals used." },
  { label: 'Creatinine encounter comparison', question: "Compare this patient's creatinine measurements between their latest encounter and earlier encounters. Show the counts and distributions for each." },
]
