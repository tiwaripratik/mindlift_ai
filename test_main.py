#!/usr/bin/env python
"""
Test script to check if app.main runs correctly
"""
import sys
import traceback

print("Testing app.main import and initialization...")
print("=" * 60)

try:
    print("\n1. Importing app.main...")
    from app.main import app
    print("   ✓ Import successful")
    
    print("\n2. Checking app instance...")
    print(f"   ✓ App title: {app.title}")
    print(f"   ✓ App version: {app.version}")
    
    print("\n3. Checking routes...")
    routes = [route.path for route in app.routes]
    print(f"   ✓ Found {len(routes)} routes")
    for route in routes[:5]:
        print(f"     - {route}")
    if len(routes) > 5:
        print(f"     ... and {len(routes) - 5} more")
    
    print("\n4. Testing health check endpoint...")
    from app.api.models import HealthCheckResponse
    print("   ✓ Health check models imported")
    
    print("\n" + "=" * 60)
    print("✓ All checks passed! Application is ready to run.")
    print("\nTo start the server, run:")
    print("  python -m app.main")
    print("  or")
    print("  uvicorn app.main:app --reload")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nTraceback:")
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTraceback:")
    traceback.print_exc()
    sys.exit(1)

