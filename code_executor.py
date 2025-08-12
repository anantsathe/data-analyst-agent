# code_executor.py
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
import subprocess
import os

def install_required_packages():
    """Install required packages if not available."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 
        'requests', 'beautifulsoup4', 'duckdb', 'lxml', 'html5lib'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except subprocess.CalledProcessError:
                print(f"Warning: Could not install {package}")

def execute_user_code(code: str, globals_dict: dict = None) -> dict:
    """
    Execute user-provided Python code safely.
    
    Args:
        code: Python code to execute
        globals_dict: Global variables to make available to the code
    
    Returns:
        dict: Execution result with 'success', 'output', 'error', and 'globals' keys
    """
    
    if globals_dict is None:
        globals_dict = {}
    
    # Add common imports and utilities
    exec_globals = {
        '__builtins__': __builtins__,
        'print': print,
        **globals_dict
    }
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Install packages if needed (for local development)
        try:
            install_required_packages()
        except:
            pass  # Continue even if package installation fails
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code
            exec(code, exec_globals)
        
        return {
            'success': True,
            'output': stdout_capture.getvalue(),
            'error': stderr_capture.getvalue() if stderr_capture.getvalue() else None,
            'globals': exec_globals
        }
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        stderr_output = stderr_capture.getvalue()
        
        full_error = f"Execution Error: {str(e)}\n"
        if stderr_output:
            full_error += f"Stderr: {stderr_output}\n"
        full_error += f"Traceback:\n{error_traceback}"
        
        return {
            'success': False,
            'output': stdout_capture.getvalue(),
            'error': full_error,
            'globals': exec_globals
        }
