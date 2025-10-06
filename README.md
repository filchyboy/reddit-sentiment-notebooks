# Reddit Sentiment Analysis Toolchain

A comprehensive Reddit sentiment analysis tool with OAuth authentication, Tailscale Funnel support, and automated reporting.

## Features

- **OAuth Authentication**: Secure Reddit API access with token management
- **Tailscale Funnel Integration**: Works with your existing Tailscale setup
- **Comprehensive Analysis**: VADER sentiment analysis with optional transformer models
- **Text Processing**: Advanced cleaning, tokenization, lemmatization, and n-gram analysis
- **Rich Visualizations**: Automated charts and plots for sentiment trends
- **Jupyter Notebooks**: Auto-generated analysis notebooks with methodology
- **CLI Interface**: Simple command-line tools for all operations
- **Data Export**: Multiple formats (JSON, CSV, Parquet) for further analysis

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd reddit-sentiment-notebooks

# Install dependencies
pip install -r requirements.txt

# Install spaCy English model (recommended)
python -m spacy download en_core_web_sm
```

### 2. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your Reddit app credentials
# REDDIT_CLIENT_ID=your_client_id
# REDDIT_CLIENT_SECRET=your_client_secret
# FUNNEL_HOST=your-tailscale-host.ts.net
```

### 3. Authentication

```bash
# Authenticate with Reddit
python -m reddit_tool auth
```

### 4. Fetch and Analyze Data

```bash
# Fetch posts from r/python
python -m reddit_tool fetch --subreddit python --limit 300

# Analyze the data
python -m reddit_tool analyze --subreddit python --ngrams 1,2

# Generate report and visualizations
python -m reddit_tool report --subreddit python
```

## Setup Instructions

### Reddit App Setup

1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "web app" as the app type
4. Set the redirect URI to: `https://your-funnel-host.ts.net/callback`
5. Note your client ID and client secret

### Tailscale Funnel Setup

This tool is designed to work with Tailscale Funnel for secure OAuth callbacks. If you already have Tailscale Funnel configured:

1. Ensure your funnel is serving:
   - `/` → `http://127.0.0.1:9090` (your main app)
   - `/callback` → `http://127.0.0.1:5000/callback` (OAuth callback)

2. Update your `.env` file with your Tailscale Funnel host:
   ```
   FUNNEL_HOST=your-device.tail4a3811.ts.net
   REDIRECT_URI=https://${FUNNEL_HOST}/callback
   ```

If you don't have Tailscale Funnel set up, the tool will fall back to a local OAuth server on port 5000.

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Reddit OAuth app (web app)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# Public redirect host (Tailscale Funnel)
FUNNEL_HOST=macbook-pro.tail4a3811.ts.net
REDIRECT_URI=https://${FUNNEL_HOST}/callback

# App behavior
USER_AGENT=your-app:reddit-analytics:v0.1 (by /u/your_username)
SUBREDDIT_DEFAULT=python
DATA_DIR=./data
REPORT_DIR=./reports
USE_TRANSFORMER=false

# Optional settings
LOG_LEVEL=INFO
REQUESTS_PER_MINUTE=60
```

## CLI Commands

### Authentication
```bash
# Authenticate with Reddit OAuth
python -m reddit_tool auth
```

### Data Fetching
```bash
# Basic fetch
python -m reddit_tool fetch --subreddit python --limit 200

# Include comments
python -m reddit_tool fetch --subreddit python --limit 200 --include-comments

# Different listing types
python -m reddit_tool fetch --subreddit python --listing hot --limit 100
python -m reddit_tool fetch --subreddit python --listing top --time-filter week --limit 100

# Verbose output
python -m reddit_tool fetch --subreddit python --limit 200 --verbose
```

### Analysis
```bash
# Basic analysis
python -m reddit_tool analyze --subreddit python

# Custom n-grams
python -m reddit_tool analyze --subreddit python --ngrams 1,2,3

# Use transformer models (requires additional setup)
python -m reddit_tool analyze --subreddit python --use-transformer

# Verbose analysis
python -m reddit_tool analyze --subreddit python --verbose
```

### Reporting
```bash
# Generate report and visualizations
python -m reddit_tool report --subreddit python

# Verbose reporting
python -m reddit_tool report --subreddit python --verbose
```

### Utility Commands
```bash
# List available data
python -m reddit_tool list-data
```

## Project Structure

```
reddit-sentiment-notebooks/
├── reddit_tool/              # Main package
│   ├── __init__.py
│   ├── __main__.py           # Module entry point
│   ├── cli.py                # Command-line interface
│   ├── auth.py               # OAuth authentication
│   ├── client.py             # Reddit API client
│   ├── config.py             # Configuration management
│   ├── io_utils.py           # Data I/O utilities
│   ├── text_clean.py         # Text processing
│   ├── sentiment.py          # Sentiment analysis
│   ├── analyze.py            # Data analysis
│   ├── plotting.py           # Visualizations
│   └── notebook_gen.py       # Notebook generation
├── tests/                    # Test suite
│   ├── test_text_clean.py
│   └── test_ngrams.py
├── data/                     # Data storage
│   ├── raw/                  # Raw JSONL files
│   └── processed/            # Processed Parquet/CSV files
├── reports/                  # Generated reports and plots
├── notebooks/                # Generated Jupyter notebooks
├── secrets/                  # OAuth tokens (gitignored)
├── logs/                     # Application logs
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Data Processing Pipeline

### 1. Data Fetching
- Connects to Reddit API using PRAW with OAuth authentication
- Fetches submissions and optionally comments from specified subreddits
- Supports different listing types: new, hot, top, rising
- Implements rate limiting and retry logic for reliability
- Saves raw data in JSONL format for reproducibility

### 2. Text Processing
- **Cleaning**: Removes URLs, Reddit formatting, excessive punctuation
- **Tokenization**: Uses spaCy (preferred) or NLTK for word tokenization
- **Lemmatization**: Reduces words to their base forms
- **Stopword Removal**: Filters common words and Reddit-specific terms
- **N-gram Generation**: Creates unigrams, bigrams, and trigrams

### 3. Sentiment Analysis
- **VADER**: Primary sentiment analysis using VADER lexicon
- **Transformer Models**: Optional support for BERT-based models
- **Scoring**: Compound scores from -1 (negative) to +1 (positive)
- **Labeling**: Automatic categorization into positive/neutral/negative

### 4. Analysis and Reporting
- **KPI Calculation**: Basic statistics, engagement metrics, temporal patterns
- **Visualization**: Automated charts for sentiment trends, distributions, n-grams
- **Notebook Generation**: Creates executable Jupyter notebooks with analysis
- **Export**: Multiple formats for further analysis

## Output Files

### Raw Data
- `data/raw/{subreddit}/{subreddit}_submissions_{timestamp}.jsonl`
- `data/raw/{subreddit}/{subreddit}_comments_{timestamp}.jsonl`

### Processed Data
- `data/processed/{subreddit}_posts.parquet` (primary format)
- `data/processed/{subreddit}_posts.csv` (compatibility)
- `data/processed/{subreddit}_comments.parquet`
- `data/processed/{subreddit}_comments.csv`

### Reports and Analysis
- `reports/{subreddit}_ngrams.csv` (n-gram frequency tables)
- `reports/{subreddit}_sentiment_distribution.png`
- `reports/{subreddit}_sentiment_timeline.png`
- `reports/{subreddit}_score_distribution.png`
- `reports/{subreddit}_top_authors.png`
- `reports/{subreddit}_ngram_frequencies.png`
- `reports/{subreddit}_report.html` (HTML export of notebook)

### Notebooks
- `notebooks/{subreddit}_analysis.ipynb` (executable analysis notebook)

## Configuration Options

### Text Processing
- `USE_TRANSFORMER`: Enable transformer-based sentiment analysis
- Custom stopwords can be added in `text_clean.py`
- N-gram sizes configurable via CLI (default: 1,2)

### Rate Limiting
- `REQUESTS_PER_MINUTE`: Control API request rate (default: 60)
- Automatic retry with exponential backoff for failed requests

### Data Storage
- `DATA_DIR`: Change data storage location (default: ./data)
- `REPORT_DIR`: Change report output location (default: ./reports)

### Logging
- `LOG_LEVEL`: Control verbosity (DEBUG, INFO, WARNING, ERROR)
- Logs saved to `logs/app.log`

## Advanced Usage

### Custom Analysis
```python
from reddit_tool.io_utils import DataManager
from reddit_tool.sentiment import analyze_sentiment
from reddit_tool.text_clean import process_dataframe_text

# Load your data
dm = DataManager()
posts_df, comments_df = dm.load_processed_data('python')

# Custom text processing
posts_df, ngrams = process_dataframe_text(posts_df, ngram_sizes=[1,2,3])

# Custom sentiment analysis
posts_df = analyze_sentiment(posts_df, use_transformer=True)
```

### Batch Processing
```bash
# Process multiple subreddits
for sub in python programming MachineLearning; do
    python -m reddit_tool fetch --subreddit $sub --limit 500
    python -m reddit_tool analyze --subreddit $sub
    python -m reddit_tool report --subreddit $sub
done
```

### Integration with Existing Workflows
The tool outputs standard formats (CSV, Parquet) that can be easily integrated with:
- Pandas/NumPy data analysis workflows
- Jupyter notebook environments
- Business intelligence tools
- Machine learning pipelines

## Troubleshooting

### Authentication Issues
```bash
# Check if tokens exist
ls -la secrets/token_store.json

# Re-authenticate if needed
python -m reddit_tool auth

# Test connection
python -c "from reddit_tool.client import RedditClient; RedditClient().test_connection()"
```

### Missing Dependencies
```bash
# Install spaCy model
python -m spacy download en_core_web_sm

# Install transformer dependencies (optional)
pip install torch transformers

# Verify NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

### Data Issues
```bash
# Check available data
python -m reddit_tool list-data

# Verify file integrity
python -c "from reddit_tool.io_utils import DataManager; dm = DataManager(); print(dm.get_file_info(dm.get_processed_data_path('python', 'posts')))"
```

### Performance Optimization
- Use `--limit` to control data volume
- Enable transformer models only when needed (`--use-transformer`)
- Process data in smaller batches for large subreddits
- Use Parquet format for faster loading of processed data

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_text_clean.py
pytest tests/test_ngrams.py

# Run with coverage
pytest --cov=reddit_tool tests/

# Verbose output
pytest -v tests/
```

## Development

### Code Quality
```bash
# Format code
black reddit_tool/ tests/

# Lint code
ruff reddit_tool/ tests/

# Type checking
mypy reddit_tool/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `logs/app.log`
3. Open an issue on the repository
4. Include relevant log output and configuration details