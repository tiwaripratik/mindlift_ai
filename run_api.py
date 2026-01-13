"""
Standalone script to run the PHQ-8 FastAPI application
"""
import os
import sys
import argparse
import uvicorn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phq8_api.config import settings
from phq8_api.ngrok_manager import ngrok_manager


def main():
    parser = argparse.ArgumentParser(
        description="Run PHQ-8 & Empathy System FastAPI Server"
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for API server (default: 8000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default="0.0.0.0",
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--no-ngrok',
        action='store_true',
        help='Disable ngrok tunnel'
    )
    
    parser.add_argument(
        '--ngrok-token',
        type=str,
        default=None,
        help='Ngrok authtoken (or set NGROK_AUTHTOKEN env var)'
    )
    
    parser.add_argument(
        '--ngrok-region',
        type=str,
        default="us",
        choices=["us", "eu", "ap", "au", "sa", "jp", "in"],
        help='Ngrok region (default: us)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload (development mode)'
    )
    
    parser.add_argument(
        '--detect-path',
        type=str,
        default=None,
        help='Path to detection model'
    )
    
    parser.add_argument(
        '--empathy-path',
        type=str,
        default=None,
        help='Path to empathy model'
    )
    
    parser.add_argument(
        '--roberta-path',
        type=str,
        default=None,
        help='Path to RoBERTa model'
    )
    
    args = parser.parse_args()
    
    # Update settings from arguments
    settings.port = args.port
    settings.host = args.host
    settings.debug = args.reload
    
    if args.detect_path:
        settings.detect_model_path = args.detect_path
    if args.empathy_path:
        settings.empathy_model_path = args.empathy_path
    if args.roberta_path:
        settings.roberta_model_path = args.roberta_path
    
    # Setup ngrok if enabled
    ngrok_url = None
    if not args.no_ngrok:
        ngrok_token = args.ngrok_token or os.getenv("NGROK_AUTHTOKEN")
        if ngrok_token:
            ngrok_manager.authtoken = ngrok_token
        ngrok_manager.region = args.ngrok_region
        
        ngrok_url = ngrok_manager.start(port=args.port)
    
    # Print startup information
    print("\n" + "="*80)
    print("🚀 PHQ-8 & Empathy System API Server")
    print("="*80)
    print(f"  🏠 Local URL: http://localhost:{args.port}")
    if ngrok_url:
        print(f"  🌐 Public URL: {ngrok_url}")
    else:
        print(f"  ⚠️  No ngrok tunnel (use --ngrok-token or set NGROK_AUTHTOKEN)")
    print(f"\n  📚 API Docs: http://localhost:{args.port}/docs")
    print(f"  💬 CBT Chat: POST http://localhost:{args.port}/chat")
    print(f"  🧠 PHQ-8 Process: POST http://localhost:{args.port}/api/process")
    print(f"  ❤️  Health Check: GET http://localhost:{args.port}/api/health")
    print(f"\n  Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    # Run server
    try:
        uvicorn.run(
            "phq8_api.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=args.reload
        )
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        if ngrok_manager.is_running:
            ngrok_manager.stop()
        print("✓ Server stopped")


if __name__ == "__main__":
    main()




