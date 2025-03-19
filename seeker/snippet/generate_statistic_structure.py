#date: 2025-03-19T17:04:05Z
#url: https://api.github.com/gists/c05e0988d6216034317be7078ebf5964
#owner: https://api.github.com/users/PaperthinJr

"""
Project Structure Generator

This module scans a directory structure and creates a text file representation
of that structure, including Python classes and functions found in Python files.
"""

# Author: Randy Wilson
# GitHub: PaperthinJr
# Date: 3/18/2025

import ast
import os
import sys
from typing import Optional, Protocol

# Directories to exclude from scanning
IGNORE_DIRS = {
    # IDE and editor directories
    ".idea", ".vscode", ".vs", ".atom", ".eclipse",

    # Python cache directories
    "__pycache__", ".mypy_cache", ".ruff_cache", ".black_cache", ".pytest_cache",
    ".coverage", ".hypothesis",

    # Version control
    ".git", ".svn", ".hg",

    # Virtual environments
    "venv", ".venv", "env", ".env", "virtualenv",

    # Build and distribution directories
    "build", "dist", "*.egg-info", "*.eggs",

    # Package management
    "node_modules",

    # Documentation
    "_build", "site", "docs/build", "htmlcov",

    # Temporary and cache directories
    "tmp", "temp", ".direnv", ".tox", ".ipynb_checkpoints", ".DS_Store", "__MACOSX",
    "logs", "debug", "out", ".cache",

    # CI/CD directories
    ".github", ".gitlab", ".circleci",

    # Dependency management
    "bower_components", ".npm", ".yarn",
}

# Files to exclude from scanning
IGNORE_FILES = {
    # Project files
    "code_quality.py", "setup.py", "setup.cfg", "pyproject.toml",

    # Configuration files
    ".gitignore", ".gitattributes", ".env", "*.ini", "*.cfg",
    ".flake8", ".pre-commit-config.yaml",

    # Package files
    "requirements.txt", "Pipfile", "Pipfile.lock", "poetry.lock",

    # Build artifacts
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.exe",

    # Generated files
    "*.log", "*.sqlite", "*.db",

    # OS specific files
    ".DS_Store", "Thumbs.db", "desktop.ini",

    # Editor backup/swap files
    "*~", "*.bak", "*.swp", "*.swo",

    # Container/deployment files
    "Dockerfile", "docker-compose.yml", ".dockerignore",
    ".travis.yml", "azure-pipelines.yml",

    # Documentation files (if not needed in structure)
    "LICENSE", "README.md", "CHANGELOG.md", "CONTRIBUTING.md",

    # Other configuration files
    ".editorconfig", ".eslintrc", ".prettier*",
    "*.ipynb", "*.lock", ".env.example",

    # Media and binary files
    "*.jpg", "*.png", "*.gif", "*.ico",
    "*.pdf", "*.zip", "*.tar.gz", "*.rar"
}


class MethodVisitorProtocol(Protocol):
    method_calls: set[str]

    def visit(self, node): ...


class DirectoryStructure:
    """
    Generates a text representation of a directory structure.

    This class scans directories and Python files to create a hierarchical
    text representation that includes Python classes and functions.
    """

    def __init__(self, root_dir: str):
        """
        Initialize the DirectoryStructure generator.

        Args:
            root_dir: The root directory path to scan
        """
        self.root_dir: str = root_dir
        self.project_name: str = os.path.basename(root_dir)
        self.structure: str = ""
        self.comment_position: int = 40
        self.entries: list[tuple[str, Optional[str]]] = []
        self.error_codes_found: bool = False  # Track if any error codes were found
        self.stats: dict[str, int] = {
            "directories": 0,
            "py_files": 0,
            "classes": 0,
            "functions": 0,
            "methods": 0,
            "total_lines": 0,
        }

    def scan_directory(self, current_dir: str, prefix: str = "") -> None:
        """
        Recursively scans directories and extracts Python class/function names.

        Args:
            current_dir: The directory path to scan
            prefix: The prefix to use for indentation in the output
        """
        # First scan to collect entries and determine longest line
        self.entries = []
        self._collect_entries(current_dir, "")

        # Calculate the position for comments
        if self.entries:
            self.comment_position = max(len(entry[0]) for entry in self.entries) + 2

        # Second pass: generate the structure with aligned comments
        self.structure = ""
        self._generate_structure(current_dir, "")

    @staticmethod
    def _should_process_item(item: str, is_dir: bool) -> bool:
        """
        Determine if an item should be processed based on ignore lists.

        Args:
            item: The item name to check
            is_dir: Whether the item is a directory

        Returns:
            True if the item should be processed, False otherwise
        """
        if is_dir:
            return item not in IGNORE_DIRS
        return (
            item.endswith(".py") and item != os.path.basename(__file__) and item not in IGNORE_FILES
        )

    def _collect_entries(self, current_dir: str, prefix: str) -> None:
        """
        First pass to collect all entries and their comments.

        Args:
            current_dir: Directory to scan
            prefix: Current prefix for this level
        """
        try:
            directory_items = sorted(os.listdir(current_dir))
            filtered_items = []

            for item in directory_items:
                item_path = os.path.join(current_dir, item)
                is_dir = os.path.isdir(item_path)
                if self._should_process_item(item, is_dir):
                    filtered_items.append((item, item_path, is_dir))

            for index, (item, item_path, is_dir) in enumerate(filtered_items):
                is_last_item = index == len(filtered_items) - 1
                connector = "└── " if is_last_item else "├── "
                entry_prefix = prefix + connector
                next_prefix = prefix + ("    " if is_last_item else "│   ")

                if is_dir:
                    # Add directory
                    self.stats["directories"] += 1
                    self.entries.append((f"{entry_prefix}{item}/", None))
                    self._collect_entries(item_path, next_prefix)
                elif item.endswith(".py"):
                    # Add Python file
                    self.stats["py_files"] += 1
                    self.entries.append((f"{entry_prefix}{item}", None))
                    self._collect_python_entries(item_path, next_prefix)
        except PermissionError:
            self.entries.append((f"{prefix}[Permission denied]", None))
        except FileNotFoundError:
            self.entries.append((f"{prefix}[Directory not found]", None))
        except OSError as e:
            self.entries.append((f"{prefix}[OS Error: {str(e)}]", None))

    def _collect_python_entries(self, file_path: str, prefix: str) -> None:
        """
        Collect Python class and function entries with their comments.

        Args:
            file_path: Path to the Python file
            prefix: Current prefix for this level
        """
        try:
            with open(file_path, encoding="utf-8") as file:
                source_lines = file.readlines()
                self.stats["total_lines"] += len(source_lines)
                source_code = "".join(source_lines)
                tree = ast.parse(source_code, filename=file_path)

            class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            function_nodes = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name != "__init__"
            ]

            self.stats["classes"] += len(class_nodes)
            self.stats["functions"] += len(function_nodes)

            # Process class definitions
            for class_index, class_node in enumerate(class_nodes):
                is_last_class = class_index == len(class_nodes) - 1 and len(function_nodes) == 0
                class_connector = "└── " if is_last_class else "├── "
                class_entry = f"{prefix}{class_connector}class {class_node.name}"

                self.entries.append((class_entry, None))

                # Calculate prefix for class methods
                class_prefix = prefix + ("    " if is_last_class else "│   ")

                # Extract methods inside class but exclude __init__
                method_nodes = [
                    n
                    for n in class_node.body
                    if isinstance(n, ast.FunctionDef) and n.name != "__init__"
                ]
                self.stats["methods"] += len(method_nodes)

                for method_index, method_node in enumerate(method_nodes):
                    is_last_method = method_index == len(method_nodes) - 1
                    method_connector = "└── " if is_last_method else "├── "

                    # Get docstring or comment for the method
                    comment = self._get_function_comment(method_node, source_lines)
                    method_entry = f"{class_prefix}{method_connector}def {method_node.name}"

                    self.entries.append((method_entry, comment))

            # Process standalone functions
            for func_index, function_node in enumerate(function_nodes):
                is_last_func = func_index == len(function_nodes) - 1
                func_connector = "└── " if is_last_func else "├── "

                # Get docstring or comment for the function
                comment = self._get_function_comment(function_node, source_lines)
                func_entry = f"{prefix}{func_connector}def {function_node.name}"

                self.entries.append((func_entry, comment))

        except UnicodeDecodeError:
            self.entries.append((f"{prefix}[File encoding not supported]", None))
        except SyntaxError:
            self.entries.append((f"{prefix}[Python syntax error]", None))
        except PermissionError:
            self.entries.append((f"{prefix}[Permission denied]", None))
        except OSError as e:
            self.entries.append((f"{prefix}[I/O Error: {str(e)}]", None))

    def _generate_structure(self, current_dir: str, prefix: str) -> None:
        """
        Second pass to generate actual structure with aligned comments.

        Args:
            current_dir: Directory to scan
            prefix: Current prefix for this level
        """
        try:
            directory_items = sorted(os.listdir(current_dir))
            filtered_items = []

            for item in directory_items:
                item_path = os.path.join(current_dir, item)
                is_dir = os.path.isdir(item_path)
                if self._should_process_item(item, is_dir):
                    filtered_items.append((item, item_path, is_dir))

            for index, (item, item_path, is_dir) in enumerate(filtered_items):
                is_last_item = index == len(filtered_items) - 1
                connector = "└── " if is_last_item else "├── "
                next_prefix = prefix + ("    " if is_last_item else "│   ")

                if is_dir:
                    self.structure += f"{prefix}{connector}{item}/\n"
                    self._generate_structure(item_path, next_prefix)
                elif item.endswith(".py"):
                    self.structure += f"{prefix}{connector}{item}\n"
                    self._generate_python_structure(item_path, next_prefix)
        except PermissionError:
            self.structure += f"{prefix}[Permission denied]\n"
        except FileNotFoundError:
            self.structure += f"{prefix}[Directory not found]\n"
        except OSError as e:
            self.structure += f"{prefix}[OS Error: {str(e)}]\n"

    def _calculate_method_complexity(self, method_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity with differentiation between calls and constants."""
        complexity = 1  # Base complexity

        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.score = 0
                self.nesting_level = 0
                self.max_nesting_level = 0
                self.function_calls = 0
                self.recursive_calls = 0
                self.recursive_call_depths = []
                self.self_calls = 0
                self.self_constant_accesses = 0  # Track constant accesses separately
                self.unique_called_functions = set()

            def visit_Call(self, node):
                self.generic_visit(node)

            def visit_Attribute(self, node):
                # Track self.CONSTANT pattern (all caps = constant by convention)
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and hasattr(node, "attr")
                    and node.attr.isupper()
                ):
                    self.self_constant_accesses += 1
                self.generic_visit(node)

        visitor = ComplexityVisitor()
        visitor.visit(method_node)

        # Store metrics on the node for warning detection
        method_node.max_nesting_level = visitor.max_nesting_level
        method_node.excessive_nesting = visitor.max_nesting_level >= 4
        method_node.constant_access_ratio = (
            visitor.self_constant_accesses / max(1, visitor.self_calls)
            if visitor.self_calls > 0
            else 0
        )

        # Calculate complexity with existing factors
        complexity += visitor.score

        # Add recursive call complexity (weighted by nesting depth)
        for depth in visitor.recursive_call_depths:
            complexity += max(2.0, 1.5 + depth * 0.5)

        # Add reduced weight for constant accesses (less complex than method calls)
        complexity += visitor.self_constant_accesses * 0.3

        # Add normal weight for self method calls
        complexity += visitor.self_calls * 0.7

        # Add factors for unique function calls and call volume
        complexity += min(len(visitor.unique_called_functions) * 0.5, 5)
        complexity += min(visitor.function_calls * 0.2, 3)

        return int(complexity)

    @staticmethod
    def _analyze_method_cohesion(methods: list[ast.FunctionDef]) -> float:
        """
        Calculate cohesion score based on method name similarity.

        Groups methods by common prefixes or verbs to determine logical groupings.

        Args:
            methods: List of method AST nodes

        Returns:
            Cohesion score (0-1, higher is more cohesive)
        """
        if not methods:
            return 1.0

        # Extract common prefixes or verbs from method names
        method_names = [m.name for m in methods if not m.name.startswith("__")]

        # Skip private methods for this analysis
        public_methods = [name for name in method_names if not name.startswith("_")]
        if not public_methods:
            return 1.0

        # Group by common prefixes (get_, set_, is_, has_, etc.)
        prefix_groups = {}
        for name in public_methods:
            prefix = None
            for candidate in [
                "get_",
                "set_",
                "is_",
                "has_",
                "create_",
                "build_",
                "update_",
                "delete_",
                "handle_",
                "process_",
                "validate_",
            ]:
                if name.startswith(candidate):
                    prefix = candidate
                    break

            # If no standard prefix, use first word until underscore
            prefix = name.split("_")[0] + "_" if prefix is None and "_" in name else name

            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(name)

        # Calculate cohesion score: 1 - (number of groups / number of methods)
        # More groups = less cohesion, fewer groups = more cohesion
        cohesion_score = 1.0 - (len(prefix_groups) / len(public_methods))
        return cohesion_score

    def _analyze_class_methods(self, class_node: ast.ClassDef, source_lines: list[str]):
        """Analyze methods in a class to determine its role and quality."""
        method_nodes = [n for n in class_node.body if isinstance(n, ast.FunctionDef)]

        # Calculate class lines (from first to last line)
        class_lines = source_lines[class_node.lineno - 1 : class_node.end_lineno].count("\n") + 1

        # Separate private, public and special methods
        private_methods = [
            m for m in method_nodes if m.name.startswith("_") and not m.name.startswith("__")
        ]
        public_methods = [m for m in method_nodes if not m.name.startswith("_")]
        dunder_methods = [m for m in method_nodes if m.name.startswith("__")]

        # Count methods
        private_method_count = len(private_methods)
        total_method_count = len(method_nodes) - (
            len(dunder_methods) - 1
            if any(m.name == "__init__" for m in dunder_methods)
            else len(dunder_methods)
        )
        public_method_count = len(public_methods)
        dunder_method_count = len(dunder_methods)

        # Calculate method length
        method_lengths = (
            [m.end_lineno - m.lineno + 1 for m in public_methods] if public_methods else [0]
        )
        avg_method_length = sum(method_lengths) / len(method_lengths) if method_lengths else 0

        # Calculate complexity scores
        public_complexities = [self._calculate_method_complexity(m) for m in public_methods]
        avg_complexity = (
            sum(public_complexities) / len(public_complexities) if public_complexities else 0
        )
        max_complexity = max(public_complexities) if public_complexities else 0

        # Determine the class type based on naming conventions
        class_name = class_node.name
        utility_patterns = ["Utils", "Helper", "Factory", "Manager", "Builder", "Service"]
        data_patterns = ["Model", "DTO", "Entity", "Record", "Struct", "Data"]

        is_utility_class = any(class_name.endswith(pattern) for pattern in utility_patterns)
        is_data_class = any(class_name.endswith(pattern) for pattern in data_patterns)

        # Enhanced accessor counting
        accessor_count = 0
        getter_count = 0
        setter_count = 0
        for method in public_methods:
            method_name = method.name
            if (
                method_name.startswith("get_")
                or method_name.startswith("is_")
                or method_name.startswith("has_")
            ):
                accessor_count += 1
                getter_count += 1
            elif method_name.startswith("set_"):
                accessor_count += 1
                setter_count += 1

        # Count property decorators
        property_count = 0
        static_method_count = 0
        static_methods = []
        for method in method_nodes:
            for decorator in method.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "property":
                    property_count += 1
                if isinstance(decorator, ast.Name) and decorator.id == "staticmethod":
                    static_method_count += 1
                    static_methods.append(method)

        # Check if class is abstract
        is_abstract = any(
            isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
            for node in method_nodes
            for decorator in node.decorator_list
        )

        # Calculate if class has inheritance
        has_inheritance = len(class_node.bases) > 0

        # Check for ABC usage
        is_abc = any(isinstance(base, ast.Name) and base.id == "ABC" for base in class_node.bases)

        # Calculate cohesion
        name_cohesion = self._analyze_method_cohesion(method_nodes)
        structural_cohesion = self._analyze_class_cohesion(class_node)

        # Weighted cohesion score
        cohesion_score = (name_cohesion * 0.4) + (structural_cohesion * 0.6)

        # Calculate property ratio
        property_ratio = (
            (accessor_count + property_count) / public_method_count
            if public_method_count > 0
            else 0
        )

        # Define adaptive property threshold based on class size
        property_threshold = 0.7 if public_method_count <= 5 else 0.6

        # Define warnings with weights and error codes
        warnings = []

        # Use avg_complexity to detect classes with consistently complex methods
        if avg_complexity > 7 and len(public_methods) >= 3:
            warnings.append((75, "E75"))

        # Use property_ratio with property_threshold for better data class detection
        if public_method_count > 0:
            accessor_ratio = (accessor_count + property_count) / public_method_count
            if accessor_ratio >= property_threshold and public_method_count >= 3:
                warnings.append((70, "E70A"))

        # Use class_lines for bloated class detection
        if class_lines > 200 and not is_utility_class:
            warnings.append((65, "E65A"))

        # Use avg_method_length for method length warnings
        if avg_method_length > 15 and public_method_count > 2:
            warnings.append((60, "E60A"))

        # Check for single-method static utility classes
        if total_method_count == 1 and static_method_count == 1:
            warnings.append((80, "E80"))

        # Check for classes dominated by static methods
        if (
            static_method_count > 0
            and static_method_count / total_method_count > 0.8
            and total_method_count > 2
        ):
            warnings.append((75, "E75A"))

        # Check for data-centric classes (excessive getters/setters)
        if public_method_count > 0:
            if property_count > 0 and property_count / public_method_count > 0.8:
                warnings.append((65, "E65B"))
            elif getter_count > 0 and getter_count / public_method_count > 0.7:
                warnings.append((70, "E70B"))
            elif setter_count > 0 and setter_count / public_method_count > 0.7:
                warnings.append((70, "E70C"))
            elif accessor_count > 0 and accessor_count / public_method_count > 0.9:
                warnings.append((50, "E50"))

        # God class detection with high priority
        if is_utility_class:
            if public_method_count > 15 and cohesion_score < 0.3:
                warnings.append((90, "E90"))
        else:
            if public_method_count > 10 and max_complexity > 10 and cohesion_score < 0.4:
                warnings.append((95, "E95"))

        # Handle anemic class detection (lower priority)
        if public_method_count < 2 and not is_utility_class and not has_inheritance:
            warnings.append((60, "E60B"))

        # Check for data classes with excessive methods
        if is_data_class and public_method_count > 2 * dunder_method_count:
            warnings.append((85, "E85"))

        if warnings:
            self.error_codes_found = True

        # Return the highest-weight warning, or empty string if none
        warnings.sort(reverse=True)  # Sort by weight descending
        if not warnings:
            return ""
        warnings.sort(reverse=True)  # Sort by weight descending
        return ", ".join([code for _, code in warnings])

    def _detect_feature_envy(self, method_node: ast.FunctionDef) -> tuple[bool, float, str]:
        """
        Detect if method shows signs of feature envy with graduated severity levels.

        Args:
            method_node: The method AST node to analyze

        Returns:
            Tuple of (has_feature_envy, envy_ratio, envied_object)
        """

        class FeatureEnvyVisitor(ast.NodeVisitor):
            def __init__(self):
                self.self_calls = 0
                self.external_obj_calls = {}  # Dictionary of external_obj -> call_count

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    obj_name = node.func.value.id
                    if obj_name == "self":
                        self.self_calls += 1
                    else:
                        if obj_name not in self.external_obj_calls:
                            self.external_obj_calls[obj_name] = 0
                        self.external_obj_calls[obj_name] += 1
                self.generic_visit(node)

        visitor = FeatureEnvyVisitor()
        visitor.visit(method_node)

        # Get the object with most calls
        max_external_obj = None
        max_external_calls = 0
        for obj, calls in visitor.external_obj_calls.items():
            if calls > max_external_calls:
                max_external_calls = calls
                max_external_obj = obj

        # No self calls case
        if visitor.self_calls == 0:
            if max_external_calls > 4:  # Increased threshold for no self calls
                return True, max_external_calls, max_external_obj or "unknown"
            return False, 0, ""

        # Calculate ratio and determine severity
        if max_external_calls > 0:
            envy_ratio = max_external_calls / max(1, visitor.self_calls)

            # Graduated thresholds for different severity levels
            if envy_ratio >= 5.0 and max_external_calls >= 5:
                return True, envy_ratio, max_external_obj or "unknown"  # Critical
            elif envy_ratio >= 3.0 and max_external_calls >= 3:
                return True, envy_ratio, max_external_obj or "unknown"  # High

        return False, 0, ""

    def _analyze_class_cohesion(self, class_node: ast.ClassDef) -> float:
        """
        Analyze class cohesion based on internal method dependencies.

        Examines method calls between class methods to determine coupling level.

        Args:
            class_node: The class AST node

        Returns:
            Cohesion score (0-1, higher means more cohesive)
        """
        if not isinstance(class_node, ast.ClassDef):
            raise TypeError(f"Expected ast.ClassDef, got {type(class_node).__name__}")

        # Get all methods in the class with explicit type checking
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append(node)

        method_names = {method.name for method in methods}

        if len(methods) <= 1:
            return 1.0  # Single method classes are considered cohesive

        # Create a dependency graph between methods
        call_graph = {}
        for method in methods:
            if not isinstance(method, ast.FunctionDef):
                continue  # Extra safety check

            call_graph[method.name] = set()
            visitor = self._create_method_call_visitor(method_names)
            visitor.visit(method)
            call_graph[method.name] = visitor.method_calls

        # Calculate cohesion metrics
        return self._calculate_cohesion_score(methods, call_graph)

    def _create_method_call_visitor(self, method_names: set) -> MethodVisitorProtocol:
        """
        Create a visitor that finds method calls within a method.

        Args:
            method_names: Set of method names to check for

        Returns:
            An AST visitor object
        """

        class MethodCallVisitor(ast.NodeVisitor):
            def __init__(self, containing_class_methods: set):
                self.method_calls = set()
                self.class_methods = containing_class_methods

            def visit_Call(self, node: ast.Call) -> None:
                # Check for method calls like self.method_name()
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and hasattr(node.func, "attr")
                    and node.func.attr in self.class_methods
                ):
                    self.method_calls.add(node.func.attr)
                self.generic_visit(node)

        return MethodCallVisitor(method_names)

    @staticmethod
    def _calculate_cohesion_score(methods: list[ast.FunctionDef], call_graph: dict) -> float:
        """
        Calculate cohesion metrics based on method dependencies.

        Args:
            methods: List of method AST nodes
            call_graph: Dictionary mapping method names to their called methods

        Returns:
            Cohesion score between 0 and 1
        """
        total_methods = len(methods)
        interconnections = 0
        isolated_methods = 0

        for method_name, called_methods in call_graph.items():
            if not called_methods and not any(
                method_name in calls for _, calls in call_graph.items()
            ):
                isolated_methods += 1
            interconnections += len(called_methods)

        # Maximum possible interconnections in a fully connected graph of n nodes is n(n-1)
        max_connections = total_methods * (total_methods - 1)

        # Calculate connectivity density (0-1)
        connectivity = interconnections / max_connections if max_connections > 0 else 0

        # Calculate isolation ratio (0-1, lower is better)
        isolation_factor = isolated_methods / total_methods if total_methods > 0 else 0

        # Final cohesion score (higher is more cohesive)
        return (connectivity * 0.7) + ((1 - isolation_factor) * 0.3)

    def _generate_python_structure(self, file_path: str, prefix: str) -> None:
        """
        Generates a text representation of Python file structure with aligned comments.

        Args:
            file_path: Path to the Python file
            prefix: Current prefix for this level
        """
        try:
            with open(file_path, encoding="utf-8") as file:
                source_lines = file.readlines()
                source_code = "".join(source_lines)
                tree = ast.parse(source_code, filename=file_path)

            class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            function_nodes = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name != "__init__"
            ]

            # Process class definitions
            for class_index, class_node in enumerate(class_nodes):
                is_last_class = class_index == len(class_nodes) - 1 and len(function_nodes) == 0
                class_connector = "└── " if is_last_class else "├── "

                # Count methods in class (excluding __init__)
                method_nodes = [n for n in class_node.body if isinstance(n, ast.FunctionDef)]

                # Separate dunder methods from regular methods
                non_dunder_methods = [
                    n
                    for n in method_nodes
                    if not (n.name.startswith("__") and n.name.endswith("__"))
                ]
                non_init_methods = [n for n in non_dunder_methods if n.name != "__init__"]

                # Count total non-dunder methods (excluding __init__)
                method_count = len(non_init_methods)

                # Check for properties (both @property decorated and naming convention)
                accessor_count = 0
                property_decorators = 0
                abstract_methods = 0
                method_body_lines = 0

                for method in non_init_methods:
                    # Count method body lines (excluding decorators and definition)
                    method_body_lines += method.end_lineno - method.lineno

                    # Check naming conventions
                    method_name = method.name.lower()
                    if method_name.startswith(("get_", "set_", "is_", "has_")):
                        accessor_count += 1

                    # Check for @property decorator and @abstractmethod
                    for decorator in method.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if decorator.id == "property":
                                property_decorators += 1
                                accessor_count += 1
                            elif decorator.id == "abstractmethod":
                                abstract_methods += 1
                        elif isinstance(decorator, ast.Attribute):
                            if hasattr(decorator, "attr") and decorator.attr == "abstractmethod":
                                abstract_methods += 1

                # Count total lines in class body
                class_lines = class_node.end_lineno - class_node.lineno

                # Calculate property ratio (excluding dunder methods)
                property_ratio = accessor_count / method_count if method_count > 0 else 0

                # Dynamic property threshold based on class size
                property_threshold = max(
                    0.7, 0.9 - (0.02 * method_count)
                )  # Relaxes as methods increase

                # Calculate average method body length
                avg_method_length = method_body_lines / method_count if method_count > 0 else 0

                # Check for base classes to consider inheritance depth
                has_inheritance = len(class_node.bases) > 0  # noqa: F841

                # Check if class is likely an ABC
                for base in class_node.bases:
                    if (
                        isinstance(base, ast.Attribute)
                        and hasattr(base, "attr")
                        and base.attr == "ABC"
                        or isinstance(base, ast.Name)
                        and base.id == "ABC"
                    ):
                        pass

                # God Class and structure analysis - improved with cohesion metrics
                class_comment = self._analyze_class_methods(class_node, source_lines)

                # Add class entry with possible warning
                entry = f"{prefix}{class_connector}class {class_node.name}"
                if class_comment:
                    padding = " " * (self.comment_position - len(entry))
                    entry += f"{padding}# {class_comment}"
                self.structure += f"{entry}\n"

                # Calculate prefix for class methods
                class_prefix = prefix + ("    " if is_last_class else "│   ")

                # Process method nodes (excluding __init__ but including other dunder methods)
                methods_to_display = [n for n in method_nodes if n.name != "__init__"]
                self._add_function_entries(
                    methods_to_display, class_prefix, source_lines, to_structure=True
                )

            # Process standalone functions
            self._add_function_entries(function_nodes, prefix, source_lines, to_structure=True)

        except UnicodeDecodeError:
            self.structure += f"{prefix}└── [File encoding not supported]\n"
        except SyntaxError:
            self.structure += f"{prefix}└── [Python syntax error]\n"
        except PermissionError:
            self.structure += f"{prefix}└── [Permission denied]\n"
        except OSError as e:
            self.structure += f"{prefix}└── [I/O Error: {str(e)}]\n"

    def _add_function_entries(
        self,
        nodes: list[ast.FunctionDef],
        prefix: str,
        source_lines: list[str],
        to_structure: bool = False,
    ) -> None:
        """Process function nodes and add them to structure or entries with scaled warnings."""
        if not nodes:
            return

        for index, node in enumerate(nodes):
            is_last = index == len(nodes) - 1
            connector = "└── " if is_last else "├── "

            # Get base comment
            comment = self._get_function_comment(node, source_lines)
            warnings = []

            # Check for overly nested methods with severity scaling
            if hasattr(node, "max_nesting_level"):
                nesting_level = node.max_nesting_level
                if nesting_level >= 6:
                    warnings.append(f"CRITICAL NESTING (DEPTH {nesting_level})")
                elif nesting_level >= 4:
                    warnings.append(f"HIGH NESTING (DEPTH {nesting_level})")
                elif nesting_level == 3:
                    warnings.append(f"MODERATE NESTING (DEPTH {nesting_level})")

            # Check for feature envy with severity indication
            has_feature_envy, envy_ratio, envied_obj = self._detect_feature_envy(node)
            if has_feature_envy:
                severity = "CRITICAL" if envy_ratio >= 5.0 else "HIGH"
                warnings.append(f"{severity} FEATURE ENVY OF '{envied_obj}' ({envy_ratio:.1f}x)")

            # Combine warnings with existing comment
            if warnings:
                warnings_str = "; ".join(warnings)
                comment = warnings_str if not comment else f"{comment}; {warnings_str}"

            entry = f"{prefix}{connector}def {node.name}"

            if to_structure:
                comment_suffix = f"# {comment}" if comment else ""
                padding = " " * max(1, self.comment_position - len(entry))
                self.structure += (
                    f"{entry}{padding}{comment_suffix}\n" if comment_suffix else f"{entry}\n"
                )
            else:
                self.entries.append((entry, comment))

    @staticmethod
    def _get_function_comment(node: ast.FunctionDef, source_lines: list[str]) -> str:
        """
        Extract comments above a function or from its docstring.

        Args:
            node: The AST node for the function
            source_lines: All source lines from the file

        Returns:
            Comment text or empty string if no comment found
        """
        # Check for a docstring
        if node.body and isinstance(node.body[0], ast.Expr) and hasattr(node.body[0], "value"):
            expr_value = node.body[0].value
            # Handle Python 3.8+ (ast.Constant)
            if isinstance(expr_value, ast.Constant) and isinstance(expr_value.value, str):
                docstring = expr_value.value.strip().split("\n")[0]
                return docstring
            # Handle Python 3.7 and earlier (ast.Str)
            elif hasattr(expr_value, "s") and isinstance(
                expr_value, getattr(ast, "Str", type(None))
            ):
                docstring = expr_value.s.strip().split("\n")[0]
                return docstring

        # Check for line comment above the function
        if node.lineno > 1:
            line_above = source_lines[node.lineno - 2].strip()
            if line_above.startswith("#"):
                return line_above.lstrip("#").strip()

        return ""

    def save_structure(self) -> None:
        """
        Saves the directory structure to a text file.

        The file is named based on the project name with statistics at the top.
        """
        output_filename = f"{self.project_name} Structure.txt"
        try:
            with open(output_filename, "w", encoding="utf-8") as file:
                # Add statistics at the top
                statistics = self.add_statistics()
                file.write(statistics)
                # Then add the project structure
                file.write(f"{self.project_name}/\n{self.structure.strip()}\n")
            print(f"Structure saved to '{output_filename}'.")
        except OSError as e:
            print(f"Error saving structure to file: {e}", file=sys.stderr)

    def _generate_error_code_table(self) -> str:
        """Generate a reference table of error codes, weights, and messages."""
        error_codes = {
            "E95": "POSSIBLE GOD CLASS",
            "E90": "LARGE UTILITY CLASS - CONSIDER SPLITTING",
            "E85": "DATA CLASS WITH BEHAVIOR - CONSIDER SPLITTING",
            "E80": "SINGLE STATIC METHOD CLASS - CONSIDER USING MODULE FUNCTION",
            "E75": "HIGH AVERAGE COMPLEXITY - CONSIDER REFACTORING",
            "E75A": "NAMESPACE CLASS - CONSIDER USING MODULE FUNCTIONS",
            "E70A": "DATA-CENTRIC CLASS - CONSIDER ADDING BEHAVIOR",
            "E70B": "DATA-CENTRIC CLASS - MOSTLY GETTERS",
            "E70C": "DATA-CENTRIC CLASS - MOSTLY SETTERS",
            "E65A": "LARGE CLASS - CONSIDER SPLITTING RESPONSIBILITIES",
            "E65B": "DATA-CENTRIC CLASS - EXCESSIVE ACCESSORS",
            "E60A": "HIGH AVERAGE METHOD LENGTH - CONSIDER REFACTORING",
            "E60B": "POSSIBLE ANEMIC CLASS",
            "E50": "HIGH ACCESSOR RATIO CLASS - CONSIDER DATA CLASS",
        }

        # Find the maximum length of code + colon for alignment
        max_code_length = max(len(code) for code in error_codes) + 2  # +2 for colon and space

        table = "Error Code Reference\n------------------\n"
        # Sort by error code to group by weight
        for code in sorted(error_codes.keys(), reverse=True):
            # Format with left-aligned padding for vertical alignment
            table += f"{code + ':':<{max_code_length}} {error_codes[code]}\n"

        return f"{table}\n"

    def add_statistics(self) -> str:
        """
        Generate project statistics with the project name as heading.

        Returns:
            String containing project statistics
        """
        # Use project name as the heading
        stats_text = f"{self.project_name} Statistics\n"
        stats_text += "-" * len(f"{self.project_name} Statistics") + "\n"

        # Include the statistics
        stats_text += f"Directories: {self.stats['directories']}\n"
        stats_text += f"Python files: {self.stats['py_files']}\n"
        stats_text += f"Classes: {self.stats['classes']}\n"
        stats_text += f"Functions: {self.stats['functions']}\n"
        stats_text += f"Methods: {self.stats['methods']}\n"
        stats_text += f"Total lines: {self.stats['total_lines']}\n"
        stats_text += "-" * len(f"{self.project_name} Statistics") + "\n\n"

        # Only add error code reference table if error codes were found
        if self.error_codes_found:
            stats_text += self._generate_error_code_table()

        return stats_text


if __name__ == "__main__":
    root_directory = os.path.dirname(os.path.abspath(__file__))  # Current directory
    directory_scanner = DirectoryStructure(root_directory)
    directory_scanner.scan_directory(root_directory)
    directory_scanner.save_structure()
