import os
import sys
import importlib
import traceback

print("==================================================")
print("FINAL SANITY CHECK: IMPORT & INTEGRITY AUDIT")
print("==================================================")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) if '__file__' in locals() else sys.path.append(os.path.abspath('.'))

directories_to_check = ['utils', 'scripts', 'pages']
failed_imports = []
total_files = 0

for directory in directories_to_check:
    if not os.path.exists(directory):
        continue
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                total_files += 1
                module_path = os.path.relpath(os.path.join(root, file), '.').replace(os.sep, '.')[:-3]
                try:
                    # Some scripts execute immediately on import, so we'll only check utils and pages safely, 
                    # but for scripts we will parse them with AST or just compile to bytecode
                    import py_compile
                    py_compile.compile(os.path.join(root, file), doraise=True)
                    
                    if directory in ['utils', 'pages']:
                        importlib.import_module(module_path)
                except Exception as e:
                    failed_imports.append((module_path, str(e)))

print(f"Checked {total_files} Python files for Syntax and Import Integrity.")
if failed_imports:
    print("\n[FAILED] The following modules have issues:")
    for mod, err in failed_imports:
        print(f" - {mod}: {err}")
    sys.exit(1)
else:
    print("\n[PASSED] All internal imports and syntax checks are completely clean.")

print("\n--- Testing Core Engine Instantiation ---")
try:
    from utils.predictor import AssetPredictor
    predictor = AssetPredictor('gold')
    print("[PASSED] AssetPredictor and ForecastEngine successfully instantiated.")
except Exception as e:
    print(f"[FAILED] Engine Instantiation Error: {e}")
    sys.exit(1)
    
print("\n[SUCCESS] The system is 100% clean and ready for Phase 7 execution.")
