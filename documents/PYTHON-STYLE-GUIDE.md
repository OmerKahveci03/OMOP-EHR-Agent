Overall, the code needs to be easy to follow, the purpose of a function should be clear,
and we should know of the return variable and potential state changes at the call site.

I've prepared some concrete rules to facilitate this.

1. Type hint everything
2. Always pass in parameters to functions as kwargs when possible.
3. Every function needs a docstring. We can keep docstrings minimal, but we should
    be able to tell what it needs / does based on the docstring and parameters alone.
4. Line spacing between different blocks of logic is important. Constant comments breaking
    steps apart is crucial. The goal is to make the code spacious and ncie to look at.
    For example:

    ```python
    # --- Optionally Use Latest Encounter ---
    encounter: Optional[dict] = (
        latest_encounter(data_dir, person_id) 
        if scope == "latest_encounter" 
        else None
    )

    if scope == "latest_encounter" and encounter is None:
        raise ValueError(f"person_id {person_id} has no VISIT_OCCURRENCE record")

    # --- Pull the Encounters Involved in the Scope ---
    visits: dict[RowsKeys, Any] = patient_rows(data_dir=data_dir, 
                                               table="visit_occurrence", 
                                               person_id=person_id,
                                               scope=scope, 
                                               encounter=encounter)
    ```
5. Strings with a known/fixed set of possible values need to be type hinted as a Literal.
6. Use Optional instead of '| None':
    ```python 
        # Bad
        type | None

        # Good
        Optional[type]
    ```
