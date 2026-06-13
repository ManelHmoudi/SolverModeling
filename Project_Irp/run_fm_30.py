import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from FunctionMerge.main import run_function_merge

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "instance_30_clients.json")
result = run_function_merge(data_path=DATA)
print("DONE" if result else "NO SOLUTION FOUND")
