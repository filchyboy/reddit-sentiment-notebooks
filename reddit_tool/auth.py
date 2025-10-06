"""OAuth authentication for Reddit API with Tailscale Funnel support."""

import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlencode

import requests
from flask import Flask, request
from rich.console import Console

console = Console()


class RedditAuth:
    """Handles Reddit OAuth authentication and token management."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.user_agent = user_agent
        self.token_file = Path("secrets/token_store.json")
        
        # Ensure secrets directory exists
        self.token_file.parent.mkdir(exist_ok=True)
    
    def get_auth_url(self, state: str = "reddit-sentiment-tool") -> str:
        """Generate Reddit OAuth authorization URL."""
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "state": state,
            "redirect_uri": self.redirect_uri,
            "duration": "permanent",
            "scope": "read"
        }
        return f"https://www.reddit.com/api/v1/authorize?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str) -> Dict[str, str]:
        """Exchange authorization code for access token."""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri
        }
        
        headers = {
            "User-Agent": self.user_agent
        }
        
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            data=data,
            headers=headers,
            auth=(self.client_id, self.client_secret)
        )
        
        if response.status_code != 200:
            raise Exception(f"Token exchange failed: {response.text}")
        
        return response.json()
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token."""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }
        
        headers = {
            "User-Agent": self.user_agent
        }
        
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            data=data,
            headers=headers,
            auth=(self.client_id, self.client_secret)
        )
        
        if response.status_code != 200:
            raise Exception(f"Token refresh failed: {response.text}")
        
        return response.json()
    
    def save_tokens(self, token_data: Dict[str, str]) -> None:
        """Save tokens to file."""
        # Add timestamp for expiration tracking
        token_data["obtained_at"] = int(time.time())
        
        with open(self.token_file, "w") as f:
            json.dump(token_data, f, indent=2)
        
        console.print(f"✅ Tokens saved to {self.token_file}")
    
    def load_tokens(self) -> Optional[Dict[str, str]]:
        """Load tokens from file."""
        if not self.token_file.exists():
            return None
        
        try:
            with open(self.token_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def is_token_expired(self, token_data: Dict[str, str]) -> bool:
        """Check if access token is expired."""
        if "obtained_at" not in token_data or "expires_in" not in token_data:
            return True
        
        obtained_at = token_data["obtained_at"]
        expires_in = token_data["expires_in"]
        current_time = int(time.time())
        
        # Add 60 second buffer
        return (current_time - obtained_at) >= (expires_in - 60)
    
    def get_valid_access_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        token_data = self.load_tokens()
        if not token_data:
            return None
        
        # If token is not expired, return it
        if not self.is_token_expired(token_data):
            return token_data.get("access_token")
        
        # Try to refresh the token
        if "refresh_token" in token_data:
            try:
                new_token_data = self.refresh_access_token(token_data["refresh_token"])
                # Preserve refresh token if not provided in response
                if "refresh_token" not in new_token_data and "refresh_token" in token_data:
                    new_token_data["refresh_token"] = token_data["refresh_token"]
                
                self.save_tokens(new_token_data)
                return new_token_data.get("access_token")
            except Exception as e:
                console.print(f"❌ Token refresh failed: {e}")
                return None
        
        return None
    
    def start_oauth_flow(self) -> bool:
        """Start OAuth flow with minimal Flask server."""
        auth_url = self.get_auth_url()
        
        console.print(f"🔗 Opening browser to: {auth_url}")
        webbrowser.open(auth_url)
        
        # Start minimal Flask server to handle callback
        app = Flask(__name__)
        received_code = {"code": None, "error": None}
        
        @app.route("/callback")
        def callback():
            code = request.args.get("code")
            error = request.args.get("error")
            
            if error:
                received_code["error"] = error
                return f"❌ Authorization failed: {error}"
            
            if code:
                received_code["code"] = code
                return "✅ Authorization successful! You can close this window."
            
            return "❌ No authorization code received"
        
        # Extract port from redirect URI
        try:
            port = 5000  # Default fallback
            if ":" in self.redirect_uri:
                port_part = self.redirect_uri.split(":")[-1]
                if "/" in port_part:
                    port = int(port_part.split("/")[0])
                else:
                    port = int(port_part)
        except ValueError:
            port = 5000
        
        console.print(f"🚀 Starting OAuth callback server on port {port}")
        
        try:
            app.run(host="127.0.0.1", port=port, debug=False)
        except KeyboardInterrupt:
            console.print("❌ OAuth flow cancelled")
            return False
        
        if received_code["error"]:
            console.print(f"❌ OAuth error: {received_code['error']}")
            return False
        
        if not received_code["code"]:
            console.print("❌ No authorization code received")
            return False
        
        try:
            token_data = self.exchange_code_for_token(received_code["code"])
            self.save_tokens(token_data)
            console.print("✅ OAuth flow completed successfully!")
            return True
        except Exception as e:
            console.print(f"❌ Token exchange failed: {e}")
            return False


def get_reddit_auth() -> RedditAuth:
    """Create RedditAuth instance from environment variables."""
    from .config import load_config
    
    config = load_config()
    return RedditAuth(
        client_id=config["reddit_client_id"],
        client_secret=config["reddit_client_secret"],
        redirect_uri=config["redirect_uri"],
        user_agent=config["user_agent"]
    )
