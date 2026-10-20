# EHR agent test questions

Use the same patient ID and the exact wording below when comparing agent versions. Record whether each result shows the requested source records or computed data, uses the requested encounter scope, and handles missing data honestly.

1. Show me this patient's visits, with the most recent visit first.
2. What measurements were recorded during the latest encounter? Show the values, units, and dates.
3. Show this patient's creatinine lab results across all encounters.
4. Plot this patient's systolic and diastolic blood pressure over time during the latest encounter.
5. Show the distribution of this patient's heart rate measurements across all encounters.
6. For the latest encounter, what are the count, minimum, median, maximum, and standard deviation of this patient's glucose measurements?
7. Which conditions appear in this patient's record, and when were they recorded?
8. What medications were recorded during this patient's latest encounter? Show the underlying records.
9. Calculate and summarize this patient's urine output rate during the latest encounter. Show the source measurements and explain the units and time intervals used.
10. Compare this patient's creatinine measurements between their latest encounter and earlier encounters. Show the counts and distributions for each.

If matching records are absent or a calculation lacks required fields, the expected behavior is to say so rather than invent a result.
