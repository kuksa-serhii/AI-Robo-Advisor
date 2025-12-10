#!/usr/bin/env python3
"""
Quick test script to verify the installation and basic functionality.
Run this before starting the main application.
"""

import sys

def check_imports():
    """Check if all required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required = [
        'streamlit',
        'pandas',
        'numpy',
        'cvxpy',
        'openai',
        'yfinance',
        'dotenv'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_env():
    """Check if .env file exists and contains API key."""
    print("\n🔍 Checking environment configuration...")
    
    import os
    from pathlib import Path
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("  ❌ .env file not found")
        print("  Create it from .env.example and add your OpenAI API key")
        return False
    
    print("  ✅ .env file exists")
    
    # Try loading
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("  ❌ OPENAI_API_KEY not set in .env")
            return False
        
        if api_key == "your_openai_api_key_here":
            print("  ⚠️  OPENAI_API_KEY still has placeholder value")
            print("  Replace it with your actual API key")
            return False
        
        print("  ✅ OPENAI_API_KEY is set")
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading .env: {e}")
        return False


def check_data_files():
    """Check if data files exist."""
    print("\n🔍 Checking data files...")
    
    from pathlib import Path
    
    files = {
        'universe.csv': 'Universe definition',
        'prices.csv': 'Historical prices'
    }
    
    all_exist = True
    for filename, description in files.items():
        if Path(filename).exists():
            print(f"  ✅ {filename} - {description}")
        else:
            print(f"  ❌ {filename} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n  Run: python download_data.py")
    
    return all_exist


def test_imports():
    """Test importing main modules."""
    print("\n🔍 Testing module imports...")
    
    try:
        from config import RISK_FREE_RATE, W_MAX
        print(f"  ✅ config.py (RISK_FREE_RATE={RISK_FREE_RATE}, W_MAX={W_MAX})")
    except Exception as e:
        print(f"  ❌ config.py: {e}")
        return False
    
    try:
        from data_layer import load_universe, load_prices
        print("  ✅ data_layer.py")
    except Exception as e:
        print(f"  ❌ data_layer.py: {e}")
        return False
    
    try:
        from optimization_core import compute_portfolio_stats
        print("  ✅ optimization_core.py")
    except Exception as e:
        print(f"  ❌ optimization_core.py: {e}")
        return False
    
    try:
        from strategies import strat_equal_weight
        print("  ✅ strategies.py")
    except Exception as e:
        print(f"  ❌ strategies.py: {e}")
        return False
    
    try:
        from portfolio_tool import compute_three_portfolios_and_frontier
        print("  ✅ portfolio_tool.py")
    except Exception as e:
        print(f"  ❌ portfolio_tool.py: {e}")
        return False
    
    try:
        from agent import llm_parse_user_message
        print("  ✅ agent.py")
    except Exception as e:
        print(f"  ❌ agent.py: {e}")
        return False
    
    return True


def main():
    """Run all checks."""
    print("="*60)
    print("  AI ROBO-ADVISOR - SYSTEM CHECK")
    print("="*60)
    
    checks = [
        check_imports(),
        check_env(),
        check_data_files(),
        test_imports()
    ]
    
    print("\n" + "="*60)
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("\nYou can now run the application:")
        print("  streamlit run app.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running the app.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
