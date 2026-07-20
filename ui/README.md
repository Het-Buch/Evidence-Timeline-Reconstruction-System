# UI

This folder contains the standalone Flask web app for video analysis.

## Run

From the project root:

```bash
python ui/app.py
```

Open `http://localhost:5000`.

## Structure

- `app.py` exposes the Flask API and analysis routes.
- `templates/index.html` renders the shell page.
- `static/css/app.css` contains the UI styling.
- `static/js/app.js` handles API calls and result rendering.
- `uploads/` stores temporary uploaded videos.
- `results/` stores saved analysis JSON files.
