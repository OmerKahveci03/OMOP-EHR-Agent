# Patient search frontend

Start the local Python server from the repository root:

```powershell
python -m src.local.server
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite. The development server forwards `/patient-search`
to the local Python server on port 8765. If you change `LOCAL_SERVER_PORT` in
`.env`, update the target in `vite.config.js` too.
