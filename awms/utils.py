import io
import logging
import multiprocessing
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from math import isclose
from typing import Any, Dict, List, Optional

from sympy import Interval, Matrix, N, Rational, Union, oo, simplify
from sympy.parsing.latex import parse_latex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _convert_decimals_to_fractions(string):
    """Convert all decimal numbers in a string to fractions."""
    # Match patterns for decimal numbers
    decimal_pattern = re.compile(r"([-+]?\d+\.\d+)")

    def decimal_to_fraction(match):
        decimal_value = match.group(1)
        fraction = Rational(decimal_value)  # Convert to Rational
        return str(fraction)

    # Substitute decimals with their fraction equivalent
    return decimal_pattern.sub(decimal_to_fraction, string)


def _strip_single_item_parens(string):
    """Remove parentheses and a trailing comma if the content inside
    the parentheses is a single item, such as an integer, decimal, or fraction.
    """
    if string.startswith("(") and string.endswith(",)"):
        # Remove the outermost parentheses and trailing comma
        string = string[1:-2].strip()
    return string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    # Replace (number,) with number
    # string = re.sub(r"\(\s*(-?\d+\.?\d*)\s*,\s*\)", r"\1", string)
    string = _strip_single_item_parens(string)

    # {{pmatrix} --> {pmatrix}
    string = string.replace("{{pmatrix}}", "{pmatrix}")

    # remove \in (x \in [a, b])
    string = re.sub(r"^[a-zA-Z]*\\in", "", string)

    # Convert all decimal numbers in the string to fractions
    string = _convert_decimals_to_fractions(string)

    return string


def latex_to_sympy_matrix(latex_str):
    # Replace fractions in LaTeX format with their Rational representations
    fraction_pattern = re.compile(r"\\frac\{(-?\d+)\}\{(-?\d+)\}")

    def fraction_to_rational(match):
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        return f"{Rational(numerator, denominator)}"  # Remove parentheses

    latex_str = fraction_pattern.sub(fraction_to_rational, latex_str)

    # Extract matrix rows based on LaTeX matrix structure
    row_pattern = re.compile(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", re.DOTALL)
    rows = row_pattern.findall(latex_str)
    if rows:
        rows = rows[0].split(r"\\")
        elements = []
        row_count = len(rows)
        # Updated regex to include fractions in the form a/b
        col_count = max(len(re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", row)) for row in rows)

        for row in rows:
            # Updated regex to include fractions in the form a/b
            row_elements = [
                Rational(num) if "/" in num else Rational(num)  # Parse both fractions and decimals as Rational
                for num in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+/\d+|[-+]?\d+", row)
            ]
            # Pad rows with missing columns with zeros as Rational
            while len(row_elements) < col_count:
                row_elements.append(Rational(0))
            elements.extend(row_elements)

        # Create matrix with Rational elements to ensure consistent comparisons
        return Matrix(row_count, col_count, elements)
    else:
        print(f"Failed to extract matrix rows from LaTeX string: {latex_str}")
        return None


def extract_boxed(text):
    start_index = text.rfind(r"oxed{")
    if start_index == -1:
        return None
    index = start_index + len(r"oxed{")
    brace_level = 1
    content = ""
    while index < len(text) and brace_level > 0:
        char = text[index]
        if char == "{":
            brace_level += 1
        elif char == "}":
            brace_level -= 1
        if brace_level > 0:
            content += char
        index += 1
    content = _strip_string(content)
    return content.strip()


def parse_intervals(s):
    """Parse a string containing intervals and unions into SymPy Interval and Union objects."""
    if not s:
        return None

    # Ensure `s` is a string
    s = str(s)

    # Normalize infinity and union symbols
    s = s.replace(r"\infty", "oo").replace(r"-\infty", "-oo").replace(r"\cup", "U").replace("u", "U")
    s = s.replace("-∞", "-oo").replace("∞", "oo")

    # Convert LaTeX-style fractions (e.g., -\frac{3}{2}) to rational notation
    fraction_pattern = re.compile(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}")

    def fraction_to_rational(match):
        sign = "-" if match.group(0).startswith("-") else ""
        numerator = match.group(1)
        denominator = match.group(2)
        return f"{sign}{Rational(numerator, denominator)}"

    s = fraction_pattern.sub(fraction_to_rational, s)

    # Pattern to find intervals like (-oo, -1/2) or [5, oo)
    interval_pattern = re.compile(r"([\(\[])(-?oo|[-+]?\d+/?\d*),\s*(-?oo|[-+]?\d+/?\d*)([\)\]])")
    intervals = []

    for match in interval_pattern.finditer(s):
        left_bracket, start, end, right_bracket = match.groups()

        # Convert start and end to SymPy-compatible values
        start = oo if start == "oo" else -oo if start == "-oo" else Rational(start)
        end = oo if end == "oo" else -oo if end == "-oo" else Rational(end)

        # Determine if interval is open or closed
        left_open = left_bracket == "("
        right_open = right_bracket == ")"

        # Append interval to list
        intervals.append(Interval(start, end, left_open, right_open))

    # Use Union for multiple intervals if 'U' is present
    if "U" in s:
        return Union(*intervals)
    elif intervals:
        return intervals[0]  # Single interval
    else:
        return None


def symbolic_equal(prediction, reference):
    def _parse(s):
        if isinstance(s, str):
            # Normalize union and infinity symbols
            s = s.replace(r"\infty", "oo").replace(r"-\infty", "-oo").replace(r"\cup", "U").replace("u", "U")
            s = s.replace("-∞", "-oo").replace("∞", "oo")

            # Parse matrix and interval elements using SymPy
            if "pmatrix" in s:
                return latex_to_sympy_matrix(s)
            try:
                return parse_latex(s)
            except:
                pass
        return s

    # Convert prediction and reference to SymPy expressions
    a_sym = _parse(prediction)
    b_sym = _parse(reference)

    if isinstance(a_sym, str) or isinstance(b_sym, str):
        return a_sym == b_sym  # Fallback to string comparison if parsing failed

    try:
        result = simplify(a_sym - b_sym)
        if result == 0 or (hasattr(result, "norm") and result.norm() < 1e-3):
            return True
    except:
        pass

    try:
        if isclose(N(a_sym), N(b_sym), rel_tol=1e-3):
            return True
    except:
        pass

    return False


def clean_latex_input(latex_str):
    """
    Clean LaTeX input to remove unintended escape sequences and invalid syntax.
    """
    # Remove unintended backslashes followed by digits or non-LaTeX valid characters
    latex_str = re.sub(r"\\(?=\d)", "", latex_str)  # Remove backslashes before numbers (e.g., \4)

    # Remove other invalid LaTeX escapes (e.g., \ followed by a character that's not a valid command)
    latex_str = re.sub(r"\\(?![a-zA-Z])", "", latex_str)

    return latex_str


def parse_elements(s):
    """Parse a string of elements in a tuple format like (a, b, c) to a list of SymPy objects."""
    # Clean the LaTeX string first
    s = clean_latex_input(s)

    # Remove outer parentheses
    s = s.strip("()")

    # Split by commas, ensuring fractions and decimals are handled correctly
    elements = s.split(",")
    parsed_elements = []

    for elem in elements:
        elem = elem.strip()  # Remove any whitespace around the element

        # Convert LaTeX representation to SymPy expression
        try:
            sympy_expr = parse_latex(elem)
            parsed_elements.append(sympy_expr)
        except:
            try:
                # Fallback to parsing as a Rational
                parsed_elements.append(Rational(elem))
            except:
                parsed_elements.append(elem)

    return parsed_elements


def compare_elements(a, b):
    """Compare two parsed elements using SymPy simplification and numerical closeness."""
    try:
        # Simplify the difference and check if it is zero
        if simplify(a - b) == 0:
            return True
        # If simplify fails or does not yield exact equality, try comparing numerically with isclose
        elif isclose(N(a), N(b), rel_tol=1e-4, abs_tol=1e-8):
            return True
        # If elements are symbolic and can be directly compared using equals method
        elif hasattr(a, "equals") and a.equals(b):
            return True
    except:
        pass

    # Fallback to string comparison if SymPy comparison fails
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isclose(float(a), float(b), rel_tol=1e-4, abs_tol=1e-8):
            return True
    return False


def compare_tuples(prediction, reference):
    """Compare two tuples element-wise after parsing each element into SymPy objects."""
    pred_elements = parse_elements(prediction)
    ref_elements = parse_elements(reference)

    if len(pred_elements) != len(ref_elements):
        return False

    # Compare each element individually
    for pred, ref in zip(pred_elements, ref_elements):
        if not compare_elements(pred, ref):
            return False

    return True


def math_equal(prediction, reference, include_percentage=True, is_close=True, timeout=False):
    if prediction is None or reference is None:
        return False

    # remove $
    prediction = prediction.replace("$", "").strip()
    reference = reference.replace("$", "").strip()

    if prediction == reference:
        return True

    # Check if input is a tuple format
    if prediction.startswith("(") and reference.startswith("(") and "cup" not in prediction and "cup" not in reference:
        return compare_tuples(prediction, reference)

    # Check if both are numeric
    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4, abs_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    # Check for intervals and unions
    pred_interval = parse_intervals(prediction)
    ref_interval = parse_intervals(reference)
    if pred_interval is not None and ref_interval is not None:
        # return simplify(pred_interval) == simplify(ref_interval)
        return pred_interval == ref_interval

    # Fallback to symbolic comparison
    if timeout:
        return call_with_timeout(symbolic_equal_process, prediction, reference)
    else:
        return symbolic_equal(prediction, reference)


def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def symbolic_equal_process(a, b, output_queue):
    output_queue.put(symbolic_equal(a, b))


def call_with_timeout(func, *args, timeout=10, **kwargs):
    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=func, args=(*args, output_queue), kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


class TimeoutError(Exception):
    pass


def run_with_timeout(func, args=(), timeout=50):
    """Run a function with timeout using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            result = future.result(timeout=timeout)
            return result
        except FutureTimeoutError:
            return "Timeout"
        except Exception as e:
            return f"Error: {str(e)}"


def _exec_code(code):
    """Execute code and capture output"""
    output_capture = io.StringIO()
    try:
        with redirect_stdout(output_capture):
            exec(code, globals())
        return output_capture.getvalue()
    except Exception as e:
        return f"Error: {str(e)}"


def execute_python_code(text, timeout=10):
    """Extract and execute Python code from text"""
    pattern = r"```(?:python|py)\s(.*?)```"
    code = re.findall(pattern, text, re.DOTALL)
    if code:
        code = code[0]
    else:
        code = text
    result = run_with_timeout(_exec_code, (code,), timeout)
    return result.strip() if result else "Error: No output"


@dataclass
class Message:
    """A generic message class for agent communication."""

    sender_id: str
    receiver_id: str
    content: Dict[str, Any]
    depth: int = 0
    hierarchy: List[str] = field(default_factory=list)


class ProblemState:
    def __init__(
        self,
        problem_id: str,
        problem: str,
        depth: int,
        hierarchy: List[str],
        prior_results: List[Dict[str, str]],
        attempted_problems: List[str],
        parent_id: str,
        root_problem_id: str,
    ):
        self.problem_id = problem_id
        self.problem = problem
        self.depth = depth
        self.hierarchy = hierarchy
        self.prior_results = prior_results
        self.attempted_problems = attempted_problems
        self.parent_id = parent_id
        self.root_problem_id = root_problem_id
        self.subproblems: List[str] = []
        self.current_subproblem_index = 0
        self.subproblem_solutions: List[Dict[str, str]] = []
        self.selected_tool = None
        self.attempted_tools: List[str] = []
        self.problem_type = None
        self.solution = None
        self.final_solution = None
        self.retry_count: int = 0


@dataclass
class SubproblemResult:
    question: str
    answer: str
    success: bool

    def to_dict(self):
        return asdict(self)
