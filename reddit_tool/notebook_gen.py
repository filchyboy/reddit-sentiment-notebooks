"""Generate and execute Jupyter notebooks for Reddit sentiment analysis."""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import pandas as pd
from rich.console import Console

console = Console()


def create_notebook_cells(subreddit: str, posts_df: pd.DataFrame, comments_df: pd.DataFrame, ngram_data: Dict[int, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Create notebook cells for the analysis."""
    
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Reddit Sentiment Analysis: r/{subreddit}\n",
            f"\n",
            f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"\n",
            f"This notebook contains sentiment analysis and text processing results for the r/{subreddit} subreddit.\n",
            f"\n",
            f"## Dataset Overview\n",
            f"- **Posts:** {len(posts_df)} submissions\n",
            f"- **Comments:** {len(comments_df)} comments\n",
            f"- **Total Records:** {len(posts_df) + len(comments_df)}\n"
        ]
    })
    
    # Imports cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "\n",
            "# Set plotting style\n",
            "plt.style.use('default')\n",
            "plt.rcParams['figure.figsize'] = (12, 8)\n",
            "plt.rcParams['font.size'] = 10\n",
            "\n",
            "# Suppress warnings\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "print('📚 Libraries imported successfully!')"
        ]
    })
    
    # Load data cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load processed data\n",
            f"subreddit = '{subreddit}'\n",
            "data_dir = Path('data/processed')\n",
            "\n",
            "# Load posts and comments\n",
            f"posts_df = pd.read_parquet(data_dir / f'{subreddit}_posts.parquet')\n",
            f"comments_df = pd.read_parquet(data_dir / f'{subreddit}_comments.parquet') if (data_dir / f'{subreddit}_comments.parquet').exists() else pd.DataFrame()\n",
            "\n",
            "print(f'📊 Loaded {len(posts_df)} posts and {len(comments_df)} comments')\n",
            "print(f'📅 Date range: {posts_df[\"created_datetime\"].min()} to {posts_df[\"created_datetime\"].max()}' if not posts_df.empty else 'No posts data')"
        ]
    })
    
    # Basic statistics
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Basic Statistics\n",
            "\n",
            "Let's start with some basic statistics about the dataset."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Combine data for overall statistics\n",
            "combined_df = pd.concat([posts_df, comments_df], ignore_index=True) if not posts_df.empty or not comments_df.empty else pd.DataFrame()\n",
            "\n",
            "if not combined_df.empty:\n",
            "    print('📈 Dataset Statistics:')\n",
            "    print(f'   Total records: {len(combined_df):,}')\n",
            "    print(f'   Unique authors: {combined_df[\"author\"].nunique():,}')\n",
            "    print(f'   Average score: {combined_df[\"score\"].mean():.2f}')\n",
            "    print(f'   Score range: {combined_df[\"score\"].min()} to {combined_df[\"score\"].max()}')\n",
            "    \n",
            "    if 'sentiment_compound' in combined_df.columns:\n",
            "        print(f'\\n🎭 Sentiment Statistics:')\n",
            "        print(f'   Average sentiment: {combined_df[\"sentiment_compound\"].mean():.3f}')\n",
            "        print(f'   Positive ratio: {(combined_df[\"sentiment_compound\"] > 0.05).mean():.1%}')\n",
            "        print(f'   Negative ratio: {(combined_df[\"sentiment_compound\"] < -0.05).mean():.1%}')\n",
            "        print(f'   Neutral ratio: {combined_df[\"sentiment_compound\"].between(-0.05, 0.05).mean():.1%}')\n",
            "else:\n",
            "    print('❌ No data available')"
        ]
    })
    
    # Sentiment distribution plot
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Sentiment Analysis\n",
            "\n",
            "### Sentiment Distribution"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if not combined_df.empty and 'sentiment_compound' in combined_df.columns:\n",
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
            "    \n",
            "    # Histogram of compound scores\n",
            "    ax1.hist(combined_df['sentiment_compound'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')\n",
            "    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')\n",
            "    ax1.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='Positive threshold')\n",
            "    ax1.axvline(x=-0.05, color='orange', linestyle='--', alpha=0.7, label='Negative threshold')\n",
            "    ax1.set_xlabel('Sentiment Compound Score')\n",
            "    ax1.set_ylabel('Frequency')\n",
            "    ax1.set_title('Sentiment Score Distribution')\n",
            "    ax1.legend()\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    \n",
            "    # Bar chart of sentiment labels\n",
            "    if 'sentiment_label' in combined_df.columns:\n",
            "        sentiment_counts = combined_df['sentiment_label'].value_counts()\n",
            "        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}\n",
            "        bar_colors = [colors.get(label, 'blue') for label in sentiment_counts.index]\n",
            "        \n",
            "        bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, alpha=0.7)\n",
            "        ax2.set_xlabel('Sentiment Label')\n",
            "        ax2.set_ylabel('Count')\n",
            "        ax2.set_title('Sentiment Label Distribution')\n",
            "        \n",
            "        # Add percentage labels\n",
            "        total = sentiment_counts.sum()\n",
            "        for i, (label, count) in enumerate(sentiment_counts.items()):\n",
            "            percentage = (count / total) * 100\n",
            "            ax2.text(i, count + total * 0.01, f'{percentage:.1f}%', ha='center', va='bottom')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print('❌ No sentiment data available')"
        ]
    })
    
    # Sentiment over time
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Sentiment Over Time"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if not combined_df.empty and 'sentiment_compound' in combined_df.columns and 'created_datetime' in combined_df.columns:\n",
            "    # Prepare data\n",
            "    df_time = combined_df.copy()\n",
            "    df_time['created_datetime'] = pd.to_datetime(df_time['created_datetime'])\n",
            "    df_time['date'] = df_time['created_datetime'].dt.date\n",
            "    \n",
            "    # Daily sentiment averages\n",
            "    daily_sentiment = df_time.groupby('date')['sentiment_compound'].agg(['mean', 'std', 'count']).reset_index()\n",
            "    \n",
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))\n",
            "    \n",
            "    # Sentiment trend\n",
            "    ax1.plot(daily_sentiment['date'], daily_sentiment['mean'], marker='o', linewidth=2, markersize=4)\n",
            "    ax1.fill_between(daily_sentiment['date'], \n",
            "                    daily_sentiment['mean'] - daily_sentiment['std'], \n",
            "                    daily_sentiment['mean'] + daily_sentiment['std'], \n",
            "                    alpha=0.3)\n",
            "    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)\n",
            "    ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)\n",
            "    ax1.axhline(y=-0.05, color='orange', linestyle='--', alpha=0.5)\n",
            "    ax1.set_ylabel('Average Sentiment Score')\n",
            "    ax1.set_title('Daily Sentiment Trend')\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    ax1.tick_params(axis='x', rotation=45)\n",
            "    \n",
            "    # Volume over time\n",
            "    ax2.bar(daily_sentiment['date'], daily_sentiment['count'], alpha=0.7, color='lightblue')\n",
            "    ax2.set_xlabel('Date')\n",
            "    ax2.set_ylabel('Number of Posts/Comments')\n",
            "    ax2.set_title('Daily Volume')\n",
            "    ax2.grid(True, alpha=0.3)\n",
            "    ax2.tick_params(axis='x', rotation=45)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print('❌ No temporal sentiment data available')"
        ]
    })
    
    # N-gram analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Text Analysis\n",
            "\n",
            "### Most Frequent Words and Phrases"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load n-gram data\n",
            "report_dir = Path('reports')\n",
            "ngram_files = {\n",
            "    1: report_dir / f'{subreddit}_1grams.csv',\n",
            "    2: report_dir / f'{subreddit}_2grams.csv',\n",
            "    3: report_dir / f'{subreddit}_3grams.csv'\n",
            "}\n",
            "\n",
            "ngram_data = {}\n",
            "for n, filepath in ngram_files.items():\n",
            "    if filepath.exists():\n",
            "        ngram_data[n] = pd.read_csv(filepath)\n",
            "        print(f'📝 Loaded {len(ngram_data[n])} {n}-grams')\n",
            "\n",
            "if ngram_data:\n",
            "    # Plot n-grams\n",
            "    n_plots = len(ngram_data)\n",
            "    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6 * n_plots))\n",
            "    \n",
            "    if n_plots == 1:\n",
            "        axes = [axes]\n",
            "    \n",
            "    for i, (n, df) in enumerate(ngram_data.items()):\n",
            "        ax = axes[i]\n",
            "        top_ngrams = df.head(15)\n",
            "        \n",
            "        bars = ax.barh(range(len(top_ngrams)), top_ngrams['frequency'], alpha=0.7)\n",
            "        ax.set_yticks(range(len(top_ngrams)))\n",
            "        ax.set_yticklabels(top_ngrams['ngram'])\n",
            "        ax.set_xlabel('Frequency')\n",
            "        ax.set_title(f'Top {n}-grams')\n",
            "        ax.grid(True, alpha=0.3, axis='x')\n",
            "        ax.invert_yaxis()\n",
            "        \n",
            "        # Add frequency labels\n",
            "        for j, (bar, freq) in enumerate(zip(bars, top_ngrams['frequency'])):\n",
            "            ax.text(freq + max(top_ngrams['frequency']) * 0.01, j, str(freq), \n",
            "                   va='center', ha='left')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print('❌ No n-gram data available')"
        ]
    })
    
    # Methodology cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Methodology\n",
            "\n",
            "### Data Collection\n",
            "- **Source:** Reddit API via PRAW\n",
            "- **Authentication:** OAuth 2.0 with Tailscale Funnel\n",
            "- **Rate Limiting:** Respects Reddit API limits\n",
            "\n",
            "### Text Processing\n",
            "- **Cleaning:** URL removal, whitespace normalization, Reddit-specific formatting removal\n",
            "- **Tokenization:** spaCy English model (en_core_web_sm) or NLTK fallback\n",
            "- **Lemmatization:** spaCy lemmatizer for word normalization\n",
            "- **Stopwords:** NLTK English stopwords + Reddit-specific terms\n",
            "- **N-grams:** Unigrams, bigrams, and trigrams from processed tokens\n",
            "\n",
            "### Sentiment Analysis\n",
            "- **Primary:** VADER (Valence Aware Dictionary and sEntiment Reasoner)\n",
            "- **Scoring:** Compound score from -1 (most negative) to +1 (most positive)\n",
            "- **Thresholds:** Positive ≥ 0.05, Negative ≤ -0.05, Neutral between -0.05 and 0.05\n",
            "- **Optional:** Transformer models for comparison (when enabled)\n",
            "\n",
            "### Visualization\n",
            "- **Plots:** Matplotlib with default styling\n",
            "- **Charts:** Histograms, time series, bar charts, horizontal bar charts\n",
            "- **Export:** PNG format with 300 DPI for reports\n"
        ]
    })
    
    return cells


def generate_analysis_notebook(subreddit: str, posts_df: pd.DataFrame, comments_df: pd.DataFrame, ngram_data: Dict[int, pd.DataFrame]) -> Path:
    """Generate a Jupyter notebook for the analysis."""
    
    console.print(f"📓 Generating analysis notebook for r/{subreddit}...")
    
    # Create notebook structure
    notebook = {
        "cells": create_notebook_cells(subreddit, posts_df, comments_df, ngram_data),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    notebook_path = notebooks_dir / f"{subreddit}_analysis.ipynb"
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    console.print(f"✅ Notebook saved: {notebook_path}")
    
    # Try to execute notebook and export to HTML
    try:
        import nbconvert
        from nbconvert import HTMLExporter
        import nbformat
        
        # Read the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        
        # Export to HTML
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        # Save HTML report
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        html_path = reports_dir / f"{subreddit}_report.html"
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(body)
        
        console.print(f"📄 HTML report saved: {html_path}")
        
    except Exception as e:
        console.print(f"⚠️  Could not export to HTML: {e}")
    
    return notebook_path
