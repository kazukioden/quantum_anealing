import os
from pathlib import Path

def load_env_file(env_file=".env"):
    """Load environment variables from .env file"""
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"⚠️  {env_file} not found. Create it from .env.template")
        return False
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    
    return True

def get_dwave_config():
    """Get D-Wave configuration from environment variables"""
    # Load .env file if it exists
    load_env_file()
    
    config = {
        'token': os.getenv('DWAVE_API_TOKEN'),
        'solver': os.getenv('DWAVE_SOLVER', 'Advantage_system4.1'),
        'endpoint': os.getenv('DWAVE_ENDPOINT', 'https://cloud.dwavesys.com/sapi/')
    }
    
    return config

def check_dwave_setup():
    """Check if D-Wave is properly configured"""
    config = get_dwave_config()
    
    if not config['token']:
        print("❌ D-Wave API token not configured")
        print("📝 Please:")
        print("   1. Copy .env.template to .env")
        print("   2. Edit .env with your API token")
        print("   3. Get token from: https://cloud.dwavesys.com/")
        return False
    
    if config['token'] == 'DEV-your-token-here':
        print("❌ Please replace the placeholder token in .env")
        return False
    
    print("✅ D-Wave configuration found")
    print(f"   Token: {config['token'][:10]}...")
    print(f"   Solver: {config['solver']}")
    return True

if __name__ == "__main__":
    check_dwave_setup()