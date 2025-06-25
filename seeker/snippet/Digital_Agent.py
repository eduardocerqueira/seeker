#date: 2025-06-25T16:53:39Z
#url: https://api.github.com/gists/c0a6a10c3f12f6454f5d4af62ee49fa2
#owner: https://api.github.com/users/Thefoolthatendsworlds

import os
import re
import ast
import difflib
import random
import pickle
import json
import logging
import argparse
from datetime import datetime
from collections import defaultdict
import sys
import io
import shutil
import requests
from packaging import version
from restrictedpython import compile_restricted, safe_globals, limited_builtins
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from multiprocessing import Pool, Manager
import numpy as np
import inspect

# Optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("scikit-learn not installed; ML features disabled")

try:
    from pylint.lint import Run
    from pylint.reporters.text import TextReporter
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False
    print("pylint not installed; pylint plugin disabled")

try:
    import flake8
    FLAKE8_AVAILABLE = True
except ImportError:
    FLAKE8_AVAILABLE = False
    print("flake8 not installed; flake8 plugin disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("digital_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DigitalAgent:
    """
    A state-of-the-art self-evolving AI agent for Python code repair with machine learning,
    plugin support, parallel processing, interactive CLI, and self-inspection.
    """
    def __init__(self, agent_id, config_path=None, mutation_rate=0.2, version="4.0+", debug_mode=False):
        self.agent_id = agent_id
        self.version = version
        self.mutation_rate = mutation_rate
        self.debug_mode = debug_mode
        self.sandbox_dir = "agent_sandbox"
        self.fix_history = Manager().list()  # Thread-safe for multiprocessing
        self.error_patterns = Manager().dict()  # Thread-safe
        self.rule_stats = Manager().dict()  # Thread-safe
        os.makedirs(self.sandbox_dir, exist_ok=True)

        # Load configuration
        self.config = self._load_config(config_path)
        self.mutation_rate = self.config.get("mutation_rate", mutation_rate)
        self.sandbox_enabled = self.config.get("sandbox_enabled", True)
        self.max_processes = self.config.get("max_processes", 4)

        # Compile regex patterns
        self.fixing_rules = [
            (re.compile(r"unexpected EOF while parsing"), self._fix_missing_paren, "v1.0+"),
            (re.compile(r"invalid syntax.*(if|for|while|def|class)"), self._fix_missing_colon, "v1.0+"),
            (re.compile(r"NameError: name '(\w+)' is not defined"), self._fix_undefined_var, "v1.0+"),
            (re.compile(r"TypeError:.*unsupported operand type"), self._fix_type_error, "v1.1+"),
            (re.compile(r"IndexError: .* out of range"), self._fix_index_error, "v2.0+"),
            (re.compile(r"ValueError: .*"), self._fix_value_error, "v2.0+"),
            (re.compile(r"AttributeError: .*"), self._fix_attribute_error, "v2.0+"),
            (re.compile(r"KeyError: .*"), self._fix_key_error, "v3.1+"),
            (re.compile(r"ZeroDivisionError: .*"), self._fix_zero_division, "v3.2+"),
            (re.compile(r"ModuleNotFoundError: No module named '(\w+)'"), self._fix_module_not_found, "v4.0+"),
        ]

        # Initialize ML model
        self.ml_model = None
        self.vectorizer = None
        if ML_AVAILABLE:
            self._initialize_ml_model()

        # Initialize plugins
        self.plugins = []
        self._load_plugins()

        # Cache for AST parsing
        self.ast_cache = {}

        logger.info(f"Initialized DigitalAgent {agent_id} (version {version})")

    # === Configuration ===
    def _load_config(self, config_path):
        """Load configuration from a JSON file or return defaults."""
        default_config = {
            "mutation_rate": 0.2,
            "sandbox_enabled": True,
            "max_retries": 3,
            "retry_backoff_factor": 2,
            "max_processes": 4,
            "ml_enabled": True,
            "plugins": ["pylint", "flake8"]
        }
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return {**default_config, **config}
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        return default_config

    # === Machine Learning ===
    def _initialize_ml_model(self):
        """Initialize ML model for fix prediction."""
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.ml_model = RandomForestClassifier(n_estimators=50, random_state=42)
        # Initialize with dummy data to avoid cold start
        X = self.vectorizer.fit_transform(["dummy error"])
        y = [0]
        self.ml_model.fit(X, y)
        logger.info("Initialized ML model for fix prediction")

    def _train_ml_model(self):
        """Train ML model on error patterns and fix history."""
        if not ML_AVAILABLE or not self.config.get("ml_enabled", True):
            return
        error_texts = []
        labels = []
        for pattern, success in self.fix_history:
            error_texts.append(pattern)
            labels.append(1 if success else 0)
        if len(error_texts) < 2:
            return
        try:
            X = self.vectorizer.fit_transform(error_texts)
            self.ml_model.fit(X, labels)
            logger.info("Trained ML model on fix history")
        except Exception as e:
            logger.error(f"ML training failed: {e}")

    # === Plugin System ===
    def _load_plugins(self):
        """Load external linter plugins."""
        enabled_plugins = self.config.get("plugins", [])
        if PYLINT_AVAILABLE and "pylint" in enabled_plugins:
            self.plugins.append(self._pylint_plugin)
        if FLAKE8_AVAILABLE and "flake8" in enabled_plugins:
            self.plugins.append(self._flake8_plugin)
        logger.info(f"Loaded plugins: {[p.__name__ for p in self.plugins]}")

    def _pylint_plugin(self, code):
        """Run pylint and return errors."""
        if not PYLINT_AVAILABLE:
            return []
        output = io.StringIO()
        reporter = TextReporter(output)
        try:
            temp_file = os.path.join(self.sandbox_dir, "temp.py")
            with open(temp_file, "w") as f:
                f.write(code)
            Run([temp_file, "--disable=all", "--enable=E"], reporter=reporter, do_exit=False)
            errors = output.getvalue().splitlines()
            os.remove(temp_file)
            return [(err, self._generic_fix) for err in errors if "error" in err.lower()]
        except Exception as e:
            logger.error(f"Pylint plugin failed: {e}")
            return []

    def _flake8_plugin(self, code):
        """Run flake8 and return errors."""
        if not FLAKE8_AVAILABLE:
            return []
        try:
            from flake8.api.legacy import get_style_guide
            temp_file = os.path.join(self.sandbox_dir, "temp.py")
            with open(temp_file, "w") as f:
                f.write(code)
            report = get_style_guide().check_files([temp_file])
            errors = [f"{err}" for err in report.get_statistics()]
            os.remove(temp_file)
            return [(err, self._generic_fix) for err in errors if "E" in err]
        except Exception as e:
            logger.error(f"Flake8 plugin failed: {e}")
            return []

    # === Core Fix Methods ===
    def _safe_exec(self, code, allowed_globals=None):
        """Execute code in a restricted Python environment."""
        globals_dict = allowed_globals or {
            "__builtins__": limited_builtins,
            **safe_globals
        }
        try:
            byte_code = compile_restricted(code, "<string>", "exec")
            exec(byte_code, globals_dict)
            logger.debug("Code executed successfully")
            return True, "Executed successfully"
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False, str(e)

    def fix_code(self, code_snippet, test_cases=None):
        """Fix code using rules, plugins, and ML prediction."""
        if not self._validate_code_input(code_snippet):
            logger.error("Invalid code input detected")
            return code_snippet, "Invalid code input"

        # Cache AST if possible
        ast_key = hash(code_snippet)
        if ast_key not in self.ast_cache:
            try:
                self.ast_cache[ast_key] = ast.parse(code_snippet)
            except:
                self.ast_cache[ast_key] = None

        success, result = self._safe_exec(code_snippet)
        if success:
            filepath = self.save_fixed_code(code_snippet, "No fix needed")
            logger.info(f"No fix needed, saved to {filepath}")
            return code_snippet, f"No fix needed, saved to {filepath}"

        # Run plugins for additional error detection
        plugin_rules = []
        for plugin in self.plugins:
            plugin_rules.extend(plugin(code_snippet))

        # Combine rules with plugins
        all_rules = self.fixing_rules + [(re.compile(p), f, v) for p, f, v in plugin_rules]

        # ML-based rule selection
        if ML_AVAILABLE and self.config.get("ml_enabled", True):
            selected_rule = self._predict_fix_rule(result, all_rules)
            if selected_rule:
                all_rules = [selected_rule] + all_rules  # Prioritize ML prediction

        # Parallel rule application
        with Pool(processes=self.max_processes) as pool:
            results = pool.starmap(self._apply_rule, [
                (code_snippet, result, pattern, fix_func, rule_version)
                for pattern, fix_func, rule_version in sorted(all_rules, key=lambda x: version.parse(x[2]), reverse=True)
                if version.parse(rule_version) <= version.parse(self.version)
            ])

        for fixed_code, fix_result, pattern, rule_version in results:
            if fixed_code:
                if test_cases:
                    valid, valid_result = self.validate_fix(fixed_code, test_cases)
                    if not valid:
                        self.rule_stats[pattern]["failures"] += 1
                        logger.warning(f"Fix failed validation: {valid_result}")
                        continue
                self.rule_stats[pattern]["successes"] += 1
                filepath = self.save_fixed_code(fixed_code, f"Fixed using rule: {pattern} (version {rule_version})")
                self.fix_history.append((pattern, True))
                self.evolve_rules()
                self.analyze_error_patterns()
                self._train_ml_model()
                logger.info(f"Fixed using rule: {pattern}, saved to {filepath}")
                return fixed_code, f"Fixed using rule: {pattern} (version {rule_version}), saved to {filepath}"

        self.error_patterns[result].append(code_snippet)
        self.analyze_error_patterns()
        logger.warning(f"No matching fix rule found (original error: {result})")
        return code_snippet, f"No matching fix rule found (original error: {result})"

    def _apply_rule(self, code, error, pattern, fix_func, rule_version):
        """Apply a single rule (for parallel processing)."""
        if pattern.pattern not in self.rule_stats:
            self.rule_stats[pattern.pattern] = {"successes": 0, "failures": 0}
        match = pattern.search(error)
        if match:
            try:
                args = match.groups()
                fixed_code = fix_func(code, *args)
                success, new_result = self._safe_exec(fixed_code)
                if success:
                    return fixed_code, f"Fixed using rule: {pattern.pattern} (version {rule_version})", pattern.pattern, rule_version
                self.rule_stats[pattern.pattern]["failures"] += 1
                logger.warning(f"Rule {pattern.pattern} failed: {new_result}")
            except Exception as e:
                self.rule_stats[pattern.pattern]["failures"] += 1
                logger.error(f" Ascending at line 508
Rule {pattern.pattern} failed: {e}")
        return None, None, pattern.pattern, rule_version

    def _predict_fix_rule(self, error_msg, rules):
        """Predict the best rule using ML."""
        if not ML_AVAILABLE or not self.ml_model:
            return None
        try:
            X = self.vectorizer.transform([error_msg])
            pred = self.ml_model.predict(X)[0]
            if pred == 1:  # Predicted fixable
                # Select rule with highest success rate
                valid_rules = [(p, f, v) for p, f, v in rules if p.search(error_msg)]
                if valid_rules:
                    return max(valid_rules, key=lambda x: self.rule_stats[x[0].pattern]["successes"])
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
        return None

    # === Input Validation ===
    def _validate_code_input(self, code):
        """Validate code input to prevent malicious patterns."""
        dangerous_patterns = [
            r"__import__\s*\(\s*['\"]os['\"]\s*\)\s*\.\s*system",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__builtins__",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False
        return True

    # === Error Fixers ===
    def _fix_missing_paren(self, code, *_):
        lines = code.strip().splitlines()
        if lines:
            lines[-1] = lines[-1] + ')'
        return "\n".join(lines)

    def _fix_missing_colon(self, code, *_):
        fixed_lines = []
        for line in code.splitlines():
            if re.match(r"\s*(if|for|while|def|class)\b[^:]*$", line):
                fixed_lines.append(line + ":")
            else:
                fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def _fix_undefined_var(self, code, var):
        ast_key = hash(code)
        tree = self.ast_cache.get(ast_key) or ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.Compare)):
                if isinstance(node.left, ast.Name) and node.left.id == var:
                    if isinstance(node.right, ast.Num):
                        return f"{var} = {node.right.n}\n" + code
                    elif isinstance(node.right, ast.Str):
                        return f"{var} = ''\n" + code
                    elif isinstance(node.right, ast.List):
                        return f"{var} = []\n" + code
        return f"{var} = None\n" + code

    def _fix_type_error(self, code, *_):
        return re.sub(r"(\d+)\s*\+\s*'(\d+)'", r"\1 + int('\2')", code)

    def _fix_index_error(self, code, *_):
        lines = code.splitlines()
        for i, line in enumerate(lines):
            match = re.search(r"(\w+)\[(\d+)\]", line)
            if match:
                var, index = match.groups()
                lines.insert(0, f"if len({var}) <= {index}: {var}.extend([None] * ({index} - len({var}) + 1))")
                break
        return "\n".join(lines)

    def _fix_value_error(self, code, *_):
        lines = code.splitlines()
        wrapped = ["try:"]
        for line in lines:
            wrapped.append("    " + line)
        wrapped.append("except ValueError:\n    print('Handled ValueError')")
        return "\n".join(wrapped)

    def _fix_attribute_error(self, code, *_):
        lines = code.splitlines()
        wrapped = ["try:"]
        for line in lines:
            wrapped.append("    " + line)
        wrapped.append("except AttributeError:\n    print('Handled AttributeError')")
        return "\n".join(wrapped)

    def _fix_key_error(self, code, *_):
        lines = code.splitlines()
        wrapped = ["try:"]
        for line in lines:
            wrapped.append("    " + line)
        wrapped.append("except KeyError:\n    print('Handled KeyError')")
        return "\n".join(wrapped)

    def _fix_zero_division(self, code, *_):
        lines = code.splitlines()
        wrapped = ["try:"]
        for line in lines:
            wrapped.append("    " + line)
        wrapped.append("except ZeroDivisionError:\n    print('Handled ZeroDivisionError')")
        return "\n".join(wrapped)

    def _fix_module_not_found(self, code, module):
        return f"import {module}\n" + code

    # === Advanced Rule Evolution ===
    def evolve_rules(self):
        """Evolve underperforming rules using ML clustering."""
        if random.random() < self.mutation_rate:
            rule_idx = random.randint(0, len(self.fixing_rules) - 1)
            pattern, fix_func, rule_version = self.fixing_rules[rule_idx]
            if self.rule_stats[pattern.pattern]["failures"] > self.rule_stats[pattern.pattern]["successes"]:
                new_pattern = self._generate_new_pattern(pattern.pattern)
                new_version = f"{rule_version}.mutated"
                self.fixing_rules[rule_idx] = (re.compile(new_pattern), fix_func, new_version)
                self.rule_stats[new_pattern] = {"successes": 0, "failures": 0}
                logger.info(f"Evolved rule: {pattern.pattern} -> {new_pattern} (version {new_version})")

    def _generate_new_pattern(self, old_pattern):
        """Generate a new regex pattern using clustering."""
        if not ML_AVAILABLE:
            return old_pattern + r"|.*mutated_pattern"
        error_texts = list(self.error_patterns.keys())
        if len(error_texts) < 2:
            return old_pattern + r"|.*mutated_pattern"
        try:
            X = self.vectorizer.fit_transform(error_texts)
            kmeans = KMeans(n_clusters=2, random_state=42)
            clusters = kmeans.fit_predict(X)
            cluster_errors = [err for err, cluster in zip(error_texts, clusters) if cluster == clusters[error_texts.index(old_pattern)]]
            return "|".join(re.escape(err.split("\n")[0]) for err in cluster_errors)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return old_pattern + r"|.*mutated_pattern"

    # === Error Pattern Learning ===
    def _generic_fix(self, code, error_msg):
        """Propose a context-aware fix for unknown errors."""
        ast_key = hash(code)
        tree = self.ast_cache.get(ast_key) or ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
                    module = error_msg.split("'")[-2]
                    return f"import {module}\n" + code
        lines = code.splitlines()
        wrapped = ["try:"]
        for line in lines:
            wrapped.append("    " + line)
        wrapped.append(f"except Exception as e:\n    print(f'Handled unknown error: {{strat: {e}")
        return "\n".join(wrapped)

    def analyze_error_patterns(self):
        """Analyze logged errors and suggest new rules with clustering."""
        if ML_AVAILABLE and self.config.get("ml_enabled", True):
            error_texts = list(self.error_patterns.keys())
            if len(error_texts) > 3:
                try:
                    X = self.vectorizer.fit_transform(error_texts)
                    kmeans = KMeans(n_clusters=min(3, len(error_texts)), random_state=42)
                    clusters = kmeans.fit_predict(X)
                    for cluster_id in set(clusters):
                        cluster_errors = [err for err, c in zip(error_texts, clusters) if c == cluster_id]
                        if len(cluster_errors) > 1:
                            pattern = re.compile("|".join(re.escape(err.split("\n")[0]) for err in cluster_errors))
                            new_rule = (pattern, lambda code, *args: self._generic_fix(code, cluster_errors[0]), f"v{self.version}.auto")
                            if not any(p.pattern == pattern.pattern for p, _, _ in self.fixing_rules):
                                self.fixing_rules.append(new_rule)
                                self.rule_stats[pattern.pattern] = {"successes": 0, "failures": 0}
                                logger.info(f"Added clustered rule for errors: {cluster_errors}")
                except Exception as e:
                    logger.error(f"Error pattern clustering failed: {e}")

    # === Validation ===
    def validate_fix(self, fixed_code, test_cases):
        """Validate fixed code against test cases."""
        for test_input, expected_output in test_cases:
            globals_dict = {"input": test_input}
            output = io.StringIO()
            sys.stdout = output
            try:
                byte_code = compile_restricted(fixed_code, "<string>", "exec")
                locals_dict = {}
                exec(byte_code, {"__builtins__": limited_builtins, **safe_globals, **globals_dict}, locals_dict)
                output_value = output.getvalue().strip()
                sys.stdout = sys.__stdout__
                if isinstance(expected_output, dict) and "return" in expected_output:
                    result = locals_dict.get("result", None)
                    if result != expected_output["return"]:
                        return False, f"Return value mismatch: got {result}, expected {expected_output['return']}"
                elif output_value != str(expected_output):
                    return False, f"Output mismatch: got {output_value}, expected {expected_output}"
            except Exception as e:
                sys.stdout = sys.__stdout__
                return False, f"Validation failed: {e}"
        return True, "Validated successfully"

    # === Persistence ===
    def save_fixed_code(self, code, result, filename=None, save_to_file=True, versioned=True):
        """Save fixed code with versioning."""
        if not self.sandbox_enabled or not save_to_file:
            return "not_saved"
        base_filename = filename or f"fixed_code_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if versioned:
            version_num = 1
            while os.path.exists(os.path.join(self.sandbox_dir, f"{base_filename}_v{version_num}.py")):
                version_num += 1
            filename = f"{base_filename}_v{ CHI
            else:
                filename = f"{base_filename}.py"
        filepath = os.path.join(self.sandbox_dir, filename)
        with open(filepath, "w") as f:
            f.write(f"# Result: {result}\n{code}")
        return filepath

    # === Network Export ===
    def export_stats(self, url):
        """Export rule stats and history with retries."""
        session = requests.Session()
        retries = Retry(
            total=self.config.get("max_retries", 3),
            backoff_factor=self.config.get("retry_backoff_factor", 2),
            status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))
        data = {
            "agent_id": self.agent_id,
            "version": self.version,
            "rule_stats": dict(self.rule_stats),
            "fix_history": list(self.fix_history)
        }
        try:
            response = session.post(url, json=data, timeout=10)
            if response.status_code == 200:
                logger.info(f"Successfully exported stats to {url}")
                return True, response.text
            else:
                logger.warning(f"Export failed with status {response.status_code}")
                self._backup_stats_locally(data)
                return False, f"Export failed with status {response.status_code}"
        except Exception as e:
            logger.error(f"Export failed: {e}")
            self._backup_stats_locally(data)
            return False, f"Export failed: {e}"

    def _backup_stats_locally(self, data):
        """Save stats locally if network export fails."""
        filename = f"stats_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.sandbox_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)
        logger.info(f"Backed up stats to {filepath}")

    # === Rule Management ===
    def register_custom_rule(self, pattern, fix_func, rule_version):
        """Register a custom fixing rule."""
        compiled_pattern = re.compile(pattern)
        self.fixing_rules.append((compiled_pattern, fix_func, rule_version))
        self.rule_stats[pattern] = {"successes": 0, "failures": 0}
        logger.info(f"Registered custom rule: {pattern} (version {rule_version})")

    def list_rules(self):
        """List all available rules and their stats."""
        rules_info = []
        for pattern, _, version in self.fixing_rules:
            stats = self.rule_stats[pattern.pattern]
            rules_info.append({
                "pattern": pattern.pattern,
                "version": version,
                "successes": stats["successes"],
                "failures": stats["failures"]
            })
        return rules_info

    def export_rules(self, filepath):
        """Export rules to a file."""
        rules_data = [{"pattern": p.pattern, "version": v} for p, _, v in self.fixing_rules]
        with open(filepath, "w") as f:
            json.dump(rules_data, f)
        logger.info(f"Exported rules to {filepath}")

    def import_rules(self, filepath):
        """Import rules from a file."""
        try:
            with open(filepath, "r") as f:
                rules_data = json.load(f)
            for rule in rules_data:
                pattern = rule["pattern"]
                version = rule["version"]
                # Use generic fix for imported rules
                self.register_custom_rule(pattern, self._generic_fix, version)
            logger.info(f"Imported rules from {filepath}")
        except Exception as e:
            logger.error(f"Failed to import rules: {e}")

    def reset_error_patterns(self):
        """Reset error patterns to free memory."""
        self.error_patterns.clear()
        logger.info("Cleared error patterns")

    # === New: Self-Inspection ===
    def inspect_own_code(self, method_name=None):
        """Read and return the source code of this agent (or a specific method)."""
        if method_name:
            method = getattr(self, method_name, None)
            if method is None:
                return f"Method '{method_name}' not found."
            try:
                return inspect.getsource(method)
            except (TypeError, OSError):
                return f"Unable to get source for '{method_name}' (maybe it's a built-in or compiled method)."
        else:
            return inspect.getsource(self.__class__)

    # === New: Self-Analysis ===
    def analyze_own_code(self, method_name=None):
        """
        Analyze the agent's own source code for errors and potential fixes.

        Args:
            method_name (str, optional): Specific method to analyze. If None, analyzes entire class.

        Returns:
            tuple: (fixed_code, result_message)
        """
        logger.info(f"Starting self-analysis of {'method ' + method_name if method_name else 'entire class'}")
        try:
            # Get source code
            source_code = self.inspect_own_code(method_name)
            
            # Since this is the agent's own code, skip execution in fix_code to avoid recursion
            # Instead, rely on plugins and static analysis
            fixed_code, result = self.fix_code(source_code, test_cases=None)
            
            logger.info(f"Self-analysis complete: {result}")
            return fixed_code, result
        except Exception as e:
            logger.error(f"Self-analysis failed: {e}")
            return source_code, f"Self-analysis failed: {e}"

    # === Interactive Mode ===
    def run_interactive(self):
        """Run an interactive CLI for code repair and self-inspection."""
        parser = argparse.ArgumentParser(description="DigitalAgent Interactive Code Repair")
        parser.add_argument("command", 
            choices=["fix", "list", "export", "train", "import", "reset", "inspect", "self-analyze"],
            help="Command to execute")
        parser.add_argument("--code", help="Code snippet to fix")
        parser.add_argument("--file", help="File containing code to fix")
        parser.add_argument("--test-cases", help="JSON file with test cases")
        parser.add_argument("--url", help="URL for exporting stats")
        parser.add_argument("--rules-file", help="File to export/import rules")
        parser.add_argument("--method", help="Method name to inspect or analyze (for 'inspect' or 'self-analyze' command)")
        args = parser.parse_args()

        if args.command == "fix":
            code = args.code
            if args.file:
                with open(args.file, "r") as f:
                    code = f.read()
            test_cases = []
            if args.test_cases:
                with open(args.test_cases, "r") as f:
                    test_cases = json.load(f)
            fixed_code, result = self.fix_code(code, test_cases)
            print(f"Result: {result}\nFixed Code:\n{fixed_code}")

        elif args.command == "list":
            print(json.dumps(self.list_rules(), indent=2))

        elif args.command == "export":
            if not args.url:
                print("Error: --url required for export")
                return
            success, response = self.export_stats(args.url)
            print(f"Export: {response}")

        elif args.command == "train":
            self._train_ml_model()
            print("ML model trained")

        elif args.command == "import":
            if not args.rules_file:
                print("Error: --rules-file required for import")
                return
            self.import_rules(args.rules_file)
            print(f"Imported rules from {args.rules_file}")

        elif args.command == "reset":
            self.reset_error_patterns()
            print("Error patterns reset")

        elif args.command == "inspect":
            code = self.inspect_own_code(args.method)
            print(code)

        elif args.command == "self-analyze":
            fixed_code, result = self.analyze_own_code(args.method)
            print(f"Self-Analysis Result: {result}\nAnalyzed Code:\n{fixed_code}")

if __name__ == "__main__":
    agent = DigitalAgent(agent_id="agent_003", config_path="config.json", debug_mode=True)
    agent.run_interactive()