import subprocess
import sys
import os
import time
from datetime import datetime

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

def run_script(script_name, description):
    print(f"\n{'='*50}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*50}")
    
    script_path = os.path.join(PROJECT_ROOT, 'scripts', script_name)
    
    if not os.path.exists(script_path):
        print(f"...Error: Script not found at {script_path}")
        return False
        
    start_time = time.time()
    try:
        # We don't capture output so it streams directly to the terminal
        result = subprocess.run(
            [PYTHON_EXE, script_path],
            cwd=PROJECT_ROOT
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n...SUCCESS: {description} (Took {elapsed:.1f}s)")
            return True
        else:
            print(f"\n...FAILED: {description} (Exit Code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n...EXCEPTION running {script_name}: {e}")
        return False

def main():
    print("STARTING DAILY OPERATIONS PIPELINE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. Pipeline Execution Order
    # IMPORTANT: Independent fetchers (FRED, COT, Sentiment) must run BEFORE 
    # data_fetcher_v2.py, because data_fetcher_v2 handles the final left-join 
    # and NaN filling.
    
    steps = [
        ('fred_fetcher.py', 'FRED Macro Indicators'),
        ('cot_fetcher.py', 'COT Smart Money Data'),
        ('google_trends_fetcher.py', 'Google Trends Data'),
        ('sentiment_fetcher_v2.py', 'News Sentiment Analysis'),
        ('data_fetcher_v2.py', 'YFinance & Final Data Merge'),
        ('model_monitor.py', 'Model Health Diagnostics')
    ]
    
    success_count = 0
    failed_steps = []
    
    for script, desc in steps:
        if run_script(script, desc):
            success_count += 1
        else:
            failed_steps.append(desc)
            print(f"\n...WARNING: Step '{desc}' failed. Pipeline will continue, but data may be incomplete.")
            
            # If data_fetcher_v2 fails, the whole downstream is compromised
            if script == 'data_fetcher_v2.py':
                print("...CRITICAL FAILURE in final merge. Aborting remaining steps.")
                break
    
    print("\n" + "="*60)
    print(f"...PIPELINE COMPLETED")
    print(f"...Successful: {success_count}/{len(steps)}")
    if failed_steps:
        print(f"...Failed: {', '.join(failed_steps)}")
    print("="*60)

if __name__ == "__main__":
    main()
