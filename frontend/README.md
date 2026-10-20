```powershell
cd frontend
npm install
npm run dev
```

Start the local backend in another terminal with `python -m src.local.server`. The
frontend shows the HiPerGator tunnel's live status while the backend connects,
submits the Slurm job, and loads the model. Search becomes available when the
status says **Running**. If the connection fails, use **Retry**; SSH password and
Duo prompts appear in the backend terminal.

Patient search defaults to the most recent visit. Questions about earlier visits
or trends can request history; the model selects that scope before retrieval.
Numeric measurements are summarized by visit, measurement type, and unit with
count, minimum, mean, maximum, sample standard deviation, and latest value.
Non-numeric measurement results are counted separately. The records panel shows
the scope and how many measurements contributed to each summary table.
