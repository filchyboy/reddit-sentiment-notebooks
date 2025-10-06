"""Command-line interface for Reddit sentiment analysis tool."""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .auth import get_reddit_auth
from .client import RedditClient, create_dataframes
from .config import load_config, get_logs_dir
from .io_utils import DataManager
from .text_clean import process_dataframe_text
from .sentiment import analyze_sentiment
from .analyze import analyze_subreddit_data
from .plotting import RedditPlotter

app = typer.Typer(help="Reddit sentiment analysis toolchain")
console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    logs_dir = get_logs_dir()
    log_file = logs_dir / "app.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RichHandler(console=console, show_time=False, show_path=False),
            logging.FileHandler(log_file)
        ]
    )


@app.command()
def auth():
    """Authenticate with Reddit OAuth."""
    console.print("🔐 Starting Reddit OAuth authentication...")
    
    try:
        reddit_auth = get_reddit_auth()
        
        # Check if we already have valid tokens
        access_token = reddit_auth.get_valid_access_token()
        if access_token:
            console.print("✅ Valid access token already exists!")
            
            # Test the connection
            from .client import RedditClient
            client = RedditClient()
            if client.test_connection():
                console.print("🎉 Authentication successful!")
                return
        
        # Start OAuth flow
        console.print("🚀 Starting OAuth flow...")
        success = reddit_auth.start_oauth_flow()
        
        if success:
            console.print("🎉 Authentication completed successfully!")
        else:
            console.print("❌ Authentication failed!")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"❌ Authentication error: {e}")
        raise typer.Exit(1)


@app.command()
def fetch(
    subreddit: str = typer.Option(..., "--subreddit", "-s", help="Subreddit name (without r/)"),
    limit: int = typer.Option(200, "--limit", "-l", help="Number of posts to fetch"),
    include_comments: bool = typer.Option(False, "--include-comments", help="Include comments"),
    listing_type: str = typer.Option("new", "--listing", help="Listing type: new, hot, top, rising"),
    time_filter: str = typer.Option("all", "--time-filter", help="Time filter for 'top': hour, day, week, month, year, all"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging")
):
    """Fetch Reddit data from a subreddit."""
    setup_logging(verbose)
    
    console.print(f"📡 Fetching data from r/{subreddit}...")
    console.print(f"   Limit: {limit}, Include comments: {include_comments}")
    console.print(f"   Listing: {listing_type}, Time filter: {time_filter}")
    
    try:
        # Initialize client and data manager
        client = RedditClient()
        data_manager = DataManager()
        
        # Fetch submissions
        console.print("📥 Fetching submissions...")
        submissions = client.fetch_submissions(
            subreddit=subreddit,
            limit=limit,
            listing_type=listing_type,
            time_filter=time_filter
        )
        
        # Fetch comments if requested
        comments = []
        if include_comments:
            console.print("💬 Fetching comments...")
            comments = client.fetch_comments(
                subreddit=subreddit,
                limit=limit,
                submission_limit=min(limit, 50)
            )
        
        # Save raw data
        console.print("💾 Saving raw data...")
        saved_files = data_manager.save_raw_data(subreddit, submissions, comments)
        
        # Create DataFrames
        posts_df, comments_df = create_dataframes(submissions, comments)
        
        # Save processed data
        console.print("💾 Saving processed data...")
        processed_files = data_manager.save_processed_data(subreddit, posts_df, comments_df)
        
        # Summary
        console.print("\n✅ Fetch completed!")
        table = Table(title="Fetched Data Summary")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Files", style="yellow")
        
        table.add_row("Submissions", str(len(submissions)), str(len([f for f in saved_files.keys() if 'submission' in f])))
        table.add_row("Comments", str(len(comments)), str(len([f for f in saved_files.keys() if 'comment' in f])))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Fetch failed: {e}")
        raise typer.Exit(1)


@app.command()
def analyze(
    subreddit: str = typer.Option(..., "--subreddit", "-s", help="Subreddit name to analyze"),
    ngrams: str = typer.Option("1,2", "--ngrams", help="N-gram sizes (comma-separated)"),
    use_transformer: bool = typer.Option(False, "--use-transformer", help="Use transformer for sentiment analysis"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging")
):
    """Analyze fetched Reddit data."""
    setup_logging(verbose)
    
    console.print(f"🔍 Analyzing data for r/{subreddit}...")
    
    try:
        # Parse n-gram sizes
        ngram_sizes = [int(n.strip()) for n in ngrams.split(",")]
        console.print(f"   N-grams: {ngram_sizes}")
        console.print(f"   Use transformer: {use_transformer}")
        
        # Load data
        data_manager = DataManager()
        posts_df, comments_df = data_manager.load_processed_data(subreddit)
        
        if posts_df.empty and comments_df.empty:
            console.print(f"❌ No data found for r/{subreddit}. Run fetch first.")
            raise typer.Exit(1)
        
        # Process text and calculate n-grams
        ngram_results = {}
        
        if not posts_df.empty:
            console.print("🔤 Processing post text...")
            posts_df, post_ngrams = process_dataframe_text(posts_df, ngram_sizes)
            
            console.print("🎭 Analyzing post sentiment...")
            posts_df = analyze_sentiment(posts_df, use_transformer)
            
            # Merge n-gram results
            for n, df in post_ngrams.items():
                if n not in ngram_results:
                    ngram_results[n] = df
                else:
                    ngram_results[n] = pd.concat([ngram_results[n], df]).groupby('ngram')['frequency'].sum().reset_index()
        
        if not comments_df.empty:
            console.print("🔤 Processing comment text...")
            comments_df, comment_ngrams = process_dataframe_text(comments_df, ngram_sizes)
            
            console.print("🎭 Analyzing comment sentiment...")
            comments_df = analyze_sentiment(comments_df, use_transformer)
            
            # Merge n-gram results
            for n, df in comment_ngrams.items():
                if n not in ngram_results:
                    ngram_results[n] = df
                else:
                    combined = pd.concat([ngram_results[n], df])
                    ngram_results[n] = combined.groupby('ngram')['frequency'].sum().reset_index().sort_values('frequency', ascending=False).reset_index(drop=True)
                    ngram_results[n]['rank'] = range(1, len(ngram_results[n]) + 1)
                    ngram_results[n]['ngram_size'] = n
        
        # Save updated processed data
        console.print("💾 Saving analyzed data...")
        data_manager.save_processed_data(subreddit, posts_df, comments_df)
        
        # Save n-gram results
        for n, df in ngram_results.items():
            ngram_path = data_manager.get_report_path(subreddit, f"{n}grams")
            data_manager.save_dataframe(df, ngram_path, "csv")
        
        # Create analysis summary
        console.print("📊 Creating analysis summary...")
        summary = analyze_subreddit_data(posts_df, comments_df)
        
        console.print("\n✅ Analysis completed!")
        
        # Display summary
        if 'combined' in summary and 'basic_stats' in summary['combined']:
            stats = summary['combined']['basic_stats']
            table = Table(title="Analysis Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Records", str(stats.get('total_records', 0)))
            table.add_row("Unique Authors", str(stats.get('unique_authors', 0)))
            if 'sentiment_trends' in summary['combined']:
                sentiment = summary['combined']['sentiment_trends'].get('overall_sentiment', {})
                table.add_row("Avg Sentiment", f"{sentiment.get('mean_compound', 0):.3f}")
                table.add_row("Positive Ratio", f"{sentiment.get('positive_ratio', 0):.1%}")
            
            console.print(table)
        
    except Exception as e:
        console.print(f"❌ Analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def report(
    subreddit: str = typer.Option(..., "--subreddit", "-s", help="Subreddit name to create report for"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging")
):
    """Generate analysis report and visualizations."""
    setup_logging(verbose)
    
    console.print(f"📊 Generating report for r/{subreddit}...")
    
    try:
        # Load data
        data_manager = DataManager()
        posts_df, comments_df = data_manager.load_processed_data(subreddit)
        
        if posts_df.empty and comments_df.empty:
            console.print(f"❌ No analyzed data found for r/{subreddit}. Run analyze first.")
            raise typer.Exit(1)
        
        # Load n-gram data
        ngram_data = {}
        for n in [1, 2, 3]:
            ngram_path = data_manager.get_report_path(subreddit, f"{n}grams")
            if ngram_path.exists():
                ngram_data[n] = data_manager.load_dataframe(ngram_path)
        
        # Create visualizations
        console.print("📈 Creating visualizations...")
        plotter = RedditPlotter(data_manager.report_dir)
        plot_files = plotter.create_all_plots(posts_df, comments_df, ngram_data, subreddit)
        
        # Generate notebook (placeholder for now)
        console.print("📓 Generating notebook...")
        from .notebook_gen import generate_analysis_notebook
        notebook_path = generate_analysis_notebook(subreddit, posts_df, comments_df, ngram_data)
        
        console.print("\n✅ Report generation completed!")
        console.print(f"📊 Plots saved: {len(plot_files)} files")
        console.print(f"📓 Notebook: {notebook_path}")
        
    except Exception as e:
        console.print(f"❌ Report generation failed: {e}")
        raise typer.Exit(1)


@app.command()
def list_data():
    """List available subreddit data."""
    data_manager = DataManager()
    subreddits = data_manager.list_available_subreddits()
    
    if not subreddits:
        console.print("📭 No processed data found.")
        return
    
    table = Table(title="Available Subreddit Data")
    table.add_column("Subreddit", style="cyan")
    table.add_column("Posts File", style="green")
    table.add_column("Comments File", style="yellow")
    
    for subreddit in subreddits:
        posts_path = data_manager.get_processed_data_path(subreddit, "posts", "parquet")
        comments_path = data_manager.get_processed_data_path(subreddit, "comments", "parquet")
        
        posts_info = data_manager.get_file_info(posts_path)
        comments_info = data_manager.get_file_info(comments_path)
        
        posts_status = f"✅ ({posts_info['size_mb']} MB)" if posts_info['exists'] else "❌"
        comments_status = f"✅ ({comments_info['size_mb']} MB)" if comments_info['exists'] else "❌"
        
        table.add_row(subreddit, posts_status, comments_status)
    
    console.print(table)


if __name__ == "__main__":
    app()
