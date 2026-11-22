# tools.py
# Built-in tools using LangChain @tool decorator

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluates math expressions like '2+2' or '10*5'. Use for any arithmetic."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output. Input must be valid Python."""
    from langchain_experimental.utilities import PythonREPL
    try:
        return PythonREPL().run(code)
    except Exception as e:
        return f"Error: {e}"


@tool
def echo(text: str) -> str:
    """Echo back the input text. Useful for testing."""
    return text


def get_builtin_tools():
    """Returns all built-in tools as a list."""
    return [calculator, python_repl, echo]