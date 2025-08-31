## YouTube Search to MP3 (Batch)

Download the first YouTube result for each query as MP3 using yt-dlp.

### Requirements
- Python 3.8+
- ffmpeg installed and on PATH
- pip packages in `requirements.txt`

### Install

#### Full installation (recommended)
Includes all features including AI semantic matching:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Minimal installation
For core functionality only (no AI matching):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
```

#### System dependencies
Install ffmpeg for audio extraction:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

#### Verify installation
Test that all dependencies are working:
```bash
python test_dependencies.py
```

Test GPU detection for AI matching:
```bash
python test_gpu_detection.py
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
--audio-quality         Audio quality in kbps (e.g., 192, 320) or 'best' for highest available
--audio-format          Audio format for output files: mp3, m4a, opus, flac (default: mp3)
--cookies-file          Path to cookies.txt file for YouTube authentication (handles age restrictions)
--use-google-search     Use Google search to enrich queries with additional context
--google-api-key        Google Custom Search API key (required when --use-google-search)
--google-search-engine-id Google Custom Search Engine ID (required when --use-google-search)
--google-min-confidence Minimum confidence score for Google search enrichment (0.0-1.0, default: 0.3)
--use-google-search-fallback Use web scraping fallback for Google search (when API is not available)
--filter-queries-with-google Filter queries through Google search first and use first result details as input
--use-llm-google-parsing Use LLM to parse Google search results (requires --llm-api-key or --llm-base-url)
--llm-api-key API key for LLM service (OpenAI, Anthropic, etc.) for parsing Google results
--llm-model LLM model to use for parsing (default: gpt-3.5-turbo)
--llm-base-url Base URL for local LLM service (e.g., http://localhost:11434 for Ollama)
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

Filter queries through Google search for better accuracy:
```bash
python main.py --input queries.txt --filter-queries-with-google --google-api-key YOUR_API_KEY --google-search-engine-id YOUR_ENGINE_ID
```

Combine multiple features for maximum coverage:
```bash
python main.py --input queries.txt --deep-search --search-without-authors --filter-queries-with-google
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

Download with higher audio quality (320 kbps):
```bash
python main.py --input queries.txt --audio-quality 320
```

Download in lossless FLAC format:
```bash
python main.py --input queries.txt --audio-format flac --audio-quality best
```

Download in M4A format (better quality than MP3):
```bash
python main.py --input queries.txt --audio-format m4a --audio-quality best
```

Use authentication to bypass age restrictions:
```bash
python main.py --input queries.txt --cookies-file cookies.txt
```

Use Google search enrichment for better results:
```bash
python main.py --input queries.txt --use-google-search --google-api-key YOUR_API_KEY --google-search-engine-id YOUR_ENGINE_ID
```

Use Google search with fallback (web scraping):
```bash
python main.py --input queries.txt --use-google-search --use-google-search-fallback
```

Filter queries through Google search first:
```bash
python main.py --input queries.txt --filter-queries-with-google --google-api-key YOUR_API_KEY --google-search-engine-id YOUR_ENGINE_ID
```

Combine multiple Google features for maximum accuracy:
```bash
python main.py --input queries.txt --filter-queries-with-google --use-google-search --google-api-key YOUR_API_KEY --google-search-engine-id YOUR_ENGINE_ID
```

Use LLM-based parsing for more reliable results:
```bash
python main.py --input queries.txt --filter-queries-with-google --use-llm-google-parsing --llm-api-key YOUR_OPENAI_KEY
```

Use local LLM for privacy and cost savings:
```bash
python main.py --input queries.txt --filter-queries-with-google --use-llm-google-parsing --llm-base-url http://localhost:11434 --llm-model llama2
```

### Audio Quality and Formats

The tool supports multiple audio formats and quality levels:

- **MP3**: Default format, good compatibility (default: 192 kbps)
- **M4A**: Better quality than MP3 at same bitrate, smaller file size
- **Opus**: Excellent quality at low bitrates, modern codec
- **FLAC**: Lossless format, highest quality but larger files

Quality options:
- Specific bitrate: `192`, `256`, `320` kbps
- `best`: Automatically selects the highest available quality

Examples:
```bash
# High quality MP3
python main.py --input songs.txt --audio-quality 320

# Best available quality in M4A format
python main.py --input songs.txt --audio-format m4a --audio-quality best

# Lossless FLAC
python main.py --input songs.txt --audio-format flac
```

### YouTube Authentication (Age Restrictions)

Some YouTube videos have age restrictions that prevent downloading. You can bypass this by using authentication:

1. **Generate cookies automatically** (recommended):
   ```bash
   python -m yt_search_dl.generate_cookies --output cookies.txt
   ```

2. **Create manual template**:
   ```bash
   python -m yt_search_dl.generate_cookies --output cookies.txt --manual
   ```
   Then edit the file and add your YouTube cookies manually.

3. **Use with downloads**:
   ```bash
   python main.py --input songs.txt --cookies-file cookies.txt
   ```

**Note**: You need to be logged into YouTube in your browser for this to work. The cookies file contains your authentication data, so keep it secure.

### Google Search Enrichment

Google search enrichment can improve query accuracy by finding additional context, correct song titles, and artist information from web search results.

#### Setup (API Method - Recommended)

1. **Create a Google Custom Search Engine**:
   - Go to https://cse.google.com/
   - Create a new search engine
   - Add sites like YouTube, Spotify, music databases
   - Note your Search Engine ID

2. **Get Google API Key**:
   - Go to https://console.cloud.google.com/
   - Enable Custom Search API
   - Create credentials (API Key)

3. **Use with downloads**:
   ```bash
   python main.py --input songs.txt --use-google-search --google-api-key YOUR_API_KEY --google-search-engine-id YOUR_ENGINE_ID
   ```

#### Setup (Fallback Method)

If you don't have API credentials, you can use web scraping fallback:
```bash
python main.py --input songs.txt --use-google-search --use-google-search-fallback
```

**Note**: The fallback method is less reliable and may be rate-limited by Google.

### Google Query Filtering

Google query filtering takes your input queries and refines them using Google search results before searching YouTube. This helps improve accuracy by using Google's search results to correct and enhance your queries.

#### How it works

1. **Reads queries** from your input file
2. **Searches Google** for each query using Google Custom Search API
3. **Extracts the first result's title** from Google search results
4. **Uses that title** as the new query for YouTube search
5. **Falls back gracefully** if Google search fails (uses original query)

#### Setup

Same setup as Google Search Enrichment (requires Google Custom Search API):

1. **Create a Google Custom Search Engine**:
   - Go to https://cse.google.com/
   - Create a new search engine
   - Add sites like YouTube, Spotify, music databases
   - Note your Search Engine ID

2. **Get Google API Key**:
   - Go to https://console.cloud.google.com/
   - Enable Custom Search API
   - Create credentials (API Key)

3. **Use with downloads**:
   ```bash
   python main.py --input songs.txt --filter-queries-with-google --google-api-key YOUR_API_KEY --google-search-engine-id YOUR_ENGINE_ID
   ```

#### Example workflow

- **Input file contains**: `"bohemian rhapsody"`
- **Google search finds**: `"Bohemian Rhapsody - Queen (Official Video Remastered)"`
- **YouTube search uses**: `"Bohemian Rhapsody - Queen (Official Video Remastered)"`
- **Result**: Better, more accurate YouTube search results!

#### When to use

This feature is particularly useful when you have:
- **Incomplete song titles** in your input file
- **Misspelled artist names** or song titles
- **Ambiguous queries** that need disambiguation
- **Want to ensure you get the most popular/accurate version** of a song

#### Fallback support

If you don't have API credentials, you can use web scraping fallback:
```bash
python main.py --input songs.txt --filter-queries-with-google --use-google-search-fallback
```

**Note**: The fallback method is less reliable and may be rate-limited by Google.

### LLM-Based Google Parsing

For more reliable parsing without requiring Google Custom Search API credentials, you can use LLM-based parsing. This approach uses an LLM (like GPT-3.5) to intelligently parse Google search results and extract the first result title.

#### Setup

1. **Get an OpenAI API key**:
   - Go to https://platform.openai.com/
   - Create an account and get an API key

2. **Install OpenAI library**:
   ```bash
   pip install openai
   ```

3. **Use with downloads**:
   ```bash
   python main.py --input songs.txt --filter-queries-with-google --use-llm-google-parsing --llm-api-key YOUR_OPENAI_KEY
   ```

#### How it works

1. **Fetches Google search results** using web scraping
2. **Sends HTML content to LLM** with specific instructions
3. **LLM analyzes the page** and extracts the first search result title
4. **Returns the enhanced query** for YouTube search

#### Advantages

- **No Google API required**: Works without Google Custom Search API
- **Adaptive parsing**: LLM can handle Google's layout changes automatically
- **Intelligent filtering**: Ignores ads, navigation, and irrelevant content
- **More reliable**: Better success rate than traditional web scraping

#### Example workflow

- **Input**: `"bohemian rhapsody"`
- **LLM analyzes Google results** and extracts: `"Bohemian Rhapsody - Queen (Official Video Remastered)"`
- **YouTube search uses**: The enhanced title
- **Result**: Better, more accurate results!

#### Supported models

- **OpenAI GPT models**: `gpt-3.5-turbo`, `gpt-4`, etc.
- **Local LLMs**: Ollama, LM Studio, and other OpenAI-compatible APIs
- **Custom models**: Can be extended to support other LLM providers

### Local LLM Support

For privacy, cost savings, and offline operation, you can use local LLMs like Ollama.

#### Setup with Ollama

1. **Install Ollama**:
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

2. **Pull a model**:
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   # or
   ollama pull codellama
   ```

3. **Start Ollama service**:
   ```bash
   ollama serve
   ```

4. **Use with downloads**:
   ```bash
   python main.py --input songs.txt --filter-queries-with-google --use-llm-google-parsing \
     --llm-base-url http://localhost:11434 --llm-model llama2
   ```

#### Advantages of Local LLMs

- **Privacy**: No data sent to external services
- **Cost**: No API costs for inference
- **Offline**: Works without internet connection
- **Customization**: Use any model you prefer
- **Speed**: Lower latency for local processing

#### Supported Local LLM Services

- **Ollama**: Easy-to-use local LLM runner
- **LM Studio**: GUI-based local LLM manager
- **Any OpenAI-compatible API**: Custom local deployments

#### How it works

- Searches for song information using Google
- Extracts song titles, artists, albums, and years from search results
- Uses pattern matching to identify music-related information
- Only applies enrichment if confidence score is above threshold
- Falls back to original query if no good matches found

### Troubleshooting
- If downloads fail with audio extraction errors, ensure `ffmpeg` is installed and on PATH.
- If a query does not find a good match, try simplifying the text (remove extra words like "official video" or add the artist name). The improved matching often handles this automatically, but inputs still matter.
- If you get age restriction errors, use the `--cookies-file` option with authentication.

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
- For AI matching, GPU acceleration is automatically detected and used when available.

### AI semantic matching (optional)
Improves ranking when text is messy or varies (e.g., multiple languages, extra words). Disabled by default.

Install extra dependencies:
```bash
pip install sentence-transformers numpy
```

For GPU acceleration (optional):
```bash
# CUDA 11.8 (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only PyTorch
pip install torch torchvision torchaudio
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