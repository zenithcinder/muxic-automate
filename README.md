## YouTube Search to MP3 (Batch)

Download the first YouTube result for each query as MP3 using yt-dlp.

### Requirements
- Python 3.8+
- ffmpeg installed and on PATH
- pip packages in `requirements.txt`

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Project Structure
The project has been refactored into a clean, modular package structure:

```
yt_search_dl/           # Main package
├── __init__.py         # Package exports
├── config.py           # Configuration and data models
├── utils.py            # Utility functions
├── matching.py         # Text matching and scoring
├── spotify.py          # Spotify integration
├── search.py           # YouTube search
└── download.py         # Download processing
main.py                 # CLI entry point
```

### Usage
Prepare a queries file with one search per line. Lines beginning with `#` are ignored.

Example (`queries.txt`):
```
I Really Want to Stay at Your House by Rosa Walton, Hallie Coggins
EPILEPSY by Mista
```

Run the script:
```bash
python main.py --input queries.txt --output downloads --log-level INFO
```

Optional flags:
- `--delay 1.5` delay in seconds between downloads (default 1.0)
- `--rate-limit-kbps 512` throttle speed
- `--log-file logs/run.log` log file path
- `--deep-search` enable deeper search by doubling results considered (up to 50)
- `--search-without-authors` remove artist names from queries for broader search

Downloads will be saved to the output directory as MP3 with embedded metadata and thumbnail (when available).

### Notes
- Ensure `ffmpeg` is installed (e.g., `sudo apt install ffmpeg`).
- This project uses yt-dlp under the hood for search and downloads.

---

## Enhanced matching and performance

This tool was improved to better match your query titles and run faster:

- Best/closest selection (default):
  - Strong exact matching via normalization (case, punctuation) and noise removal (e.g., "official video", "lyrics").
  - Fuzzy fallback using multiple similarity signals when no exact match is found.
  - No view-count heuristics are used.
- Progressive search for speed: tries 5, then 10, then the full `--search-count` only if needed.
- Flat extraction for metadata-only search to reduce overhead.

### Advanced search features

#### Deep Search (`--deep-search`)
When enabled, doubles the number of search results considered (capped at 50). This helps find more perfect matches by searching deeper into YouTube results. Useful when:
- Your queries are specific and you want to ensure the best match
- You're dealing with obscure or less popular content
- You want to maximize the chance of finding the exact track

#### Title-Only Search (`--search-without-authors`)
Intelligently removes artist/author information from queries before searching YouTube. This helps find the song even if the artist name doesn't exactly match what's on YouTube. Handles common patterns:
- "Song Title by Artist" → searches for "Song Title"
- "Song Title - Artist" → searches for "Song Title"
- "Song Title (Artist)" → searches for "Song Title"
- "Song Title feat. Artist" → searches for "Song Title"

Useful when:
- Artist names in your queries don't match YouTube uploader names
- You want broader search results to find covers or different versions
- You're unsure about the exact artist name

### Result filtering
- Excludes YouTube Shorts automatically.
- Excludes long-form videos (default: longer than 10 minutes).
- Excludes non-single content unless requested in your query: full albums, mixes/playlists, concerts, visualizers.
- Excludes live, covers, and remixes unless your query mentions them.
- Author-aware matching: boosts results whose `artist`/`channel`/`uploader` align with your query.

### Current defaults
- `--select-strategy best`
- `--search-count 20` (bounded 1–50)

### CLI options (full)
```text
--input                 Path to queries file (required)
--output                Output directory for MP3 files (default: downloads)
--delay                 Delay between queries in seconds (default: 1.0)
--search-count          Number of search results to consider (1–50, default: 20)
--select-strategy       How to pick results: best | first (default: best)
--deep-search           Enable deeper search by doubling results considered (up to 50)
--search-without-authors Remove artist names from queries for broader search
--rate-limit-kbps       Limit download speed in KiB/s (e.g., 512). Omit to disable
--log-level             Console log level: DEBUG | INFO | WARNING | ERROR (default: INFO)
--log-file              Path to write detailed logs (default: logs/run.log)
--list-only             List search result links to a file instead of downloading
--list-output           Path to write links when using --list-only (default: links.txt)
--from-links            Path to a file containing one URL per line to download as MP3
--concurrency           Number of queries/URLs to process in parallel (default: 1)
--use-ai-match          Enable AI semantic matching for ranking (requires extra deps)
--ai-model              Sentence-transformers model name when --use-ai-match
--use-spotify           Use Spotify to enrich queries (Client Credentials flow)
--spotify-client-id     Spotify Client ID (required with --use-spotify)
--spotify-client-secret Spotify Client Secret (required with --use-spotify)
--debug-matching        Log top candidates with score breakdown
--debug-topk            How many top candidates to show with --debug-matching (default: 10)
```

### Examples
Use the default best/closest selection over 20 results:
```bash
python main.py --input queries.txt
```

Use the first result as-is (fastest):
```bash
python main.py --input queries.txt --select-strategy first
```

Search more broadly when exact titles are tricky:
```bash
python main.py --input queries.txt --search-count 40
```

Use deep search for better matches (doubles results considered):
```bash
python main.py --input queries.txt --deep-search --search-count 25
```

Search with title only (removes artist names for broader results):
```bash
python main.py --input queries.txt --search-without-authors
```

Combine both features for maximum coverage:
```bash
python main.py --input queries.txt --deep-search --search-without-authors
```

Throttle downloads if your network is sensitive:
```bash
python main.py --input queries.txt --rate-limit-kbps 512
```

List results to a file without downloading (review first):
```bash
python main.py --input queries.txt --list-only --list-output links.txt
```

Download directly from a prepared links file:
```bash
python main.py --from-links links.txt --output downloads --concurrency 3
```

List mode writes incrementally while searching, so you can open the file and watch it fill:
```bash
python main.py --input queries.txt --list-only --list-output links.txt
```

### Troubleshooting
- If downloads fail with audio extraction errors, ensure `ffmpeg` is installed and on PATH.
- If a query does not find a good match, try simplifying the text (remove extra words like "official video" or add the artist name). The improved matching often handles this automatically, but inputs still matter.

### Parallel processing
You can process multiple queries concurrently. This is effective because searching/downloading are I/O-bound.

Enable with the `--concurrency` flag:
```bash
python main.py --input queries.txt --concurrency 4
```

Notes:
- Each worker respects `--delay` between its own requests.
- Very high concurrency may trigger throttling; start with 2–4 and adjust.
- GPU acceleration is not used here; it does not meaningfully speed up yt-dlp searches or MP3 encoding.

### AI semantic matching (optional)
Improves ranking when text is messy or varies (e.g., multiple languages, extra words). Disabled by default.

Install extra dependencies:
```bash
pip install sentence-transformers numpy
```

Enable:
```bash
python yt_search_dl.py --input queries.txt --use-ai-match
```

Pick a different model (recommendations):
- Default fast model: `sentence-transformers/all-MiniLM-L6-v2` (good balance)
- Higher accuracy: `sentence-transformers/all-mpnet-base-v2`
- Multilingual focus: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Example:
```bash
python main.py --input queries.txt --use-ai-match --ai-model sentence-transformers/all-mpnet-base-v2

python main.py --input queries.txt --search-count 50  --output downloads --use-spotify  --concurrency 4 --select-strategy best --debug-matching --debug-topk 10 --log-level DEBUG
```

### Spotify enrichment (optional)
Improves YouTube search by first resolving the track on Spotify, then using the Spotify title, artists, and duration to guide matching (including a duration similarity boost).

Setup:
1. Create a Spotify application at the Spotify Developer Dashboard and obtain a Client ID and Client Secret.
2. Enable the flags:
```bash
python main.py \
  --input queries.txt \
  --use-spotify \
  --spotify-client-id YOUR_CLIENT_ID \
  --spotify-client-secret YOUR_CLIENT_SECRET
```

You can combine Spotify with AI matching and debug logs:
```bash
python main.py \
  --input queries.txt \
  --use-spotify --spotify-client-id YOUR_ID --spotify-client-secret YOUR_SECRET \
  --use-ai-match --debug-matching --debug-topk 10 --log-level DEBUG
```

### Ranking signals (summary)
- Exact title match (strong preference)
- Title similarity (multiple signals: normalized, token sort/set, Jaccard, substring)
- Author match and author token overlap
- Duration similarity (when Spotify enrichment is enabled)
- Popularity via views (log-scaled)
- Bonuses/penalties: prefer official; downrank live, mixes/playlists, full albums, visualizers; filter Shorts and >10min by default