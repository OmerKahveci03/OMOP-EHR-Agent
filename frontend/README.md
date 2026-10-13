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
