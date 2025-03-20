#date: 2025-03-20T17:03:23Z
#url: https://api.github.com/gists/53d98dc1e3fc9d60fa075ec7e3b479cf
#owner: https://api.github.com/users/TGudz

import os
import json
import argparse
from pathlib import Path
import re
from typing import Dict, Set

class DependencyGraph:
    def __init__(self):
        self.graph: Dict[str, Set[str]] = {}

    def add_dependency(self, source: str, target: str):
        if source not in self.graph:
            self.graph[source] = set()
        self.graph[source].add(target)

    def get_subgraph(self, filename: str) -> Dict[str, Set[str]]:
        if filename not in self.graph:
            return {}
        
        subgraph = {}
        to_visit = {filename}
        visited = set()

        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                if current in self.graph:
                    subgraph[current] = self.graph[current]
                    to_visit.update(self.graph[current] - visited)
        
        return subgraph

def scan_directory(root_dir: str) -> DependencyGraph:
    graph = DependencyGraph()
    extensions = {'.js', '.jsx', '.ts', '.tsx'}
    
    import_patterns = [
        r"import\s+.*?\s+from\s+['\"](\./|\../)?[a-zA-Z0-9_/.-]+['\"]",
        r"import\s+['\"](\./|\../)?[a-zA-Z0-9_/.-]+['\"]",
        r"require\s*\(\s*['\"](\./|\../)?[a-zA-Z0-9_/.-]+['\"]\s*\)"
    ]

    for root, _, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue
                
                for pattern in import_patterns:
                    for match in re.finditer(pattern, content):
                        import_path = match.group(1) or match.group(2) or match.group(3)
                        full_import_path = os.path.normpath(
                            os.path.join(os.path.dirname(relative_path), import_path)
                        )
                        if not Path(full_import_path).suffix:
                            full_import_path += '.js'
                        graph.add_dependency(relative_path, full_import_path)
    
    return graph

def pretty_print_graph(graph: Dict[str, Set[str]], indent: int = 0):
    for node, dependencies in graph.items():
        print('  ' * indent + node)
        pretty_print_graph({dep: graph.get(dep, set()) for dep in dependencies}, indent + 1)

def save_graph_to_json(graph: DependencyGraph, output_file: str):
    with open(output_file, 'w') as f:
        json.dump(graph.graph, f, indent=2)

def load_graph_from_json(input_file: str) -> DependencyGraph:
    graph = DependencyGraph()
    with open(input_file, 'r') as f:
        graph.graph = {k: set(v) for k, v in json.load(f).items()}
    return graph

def print_subgraph(graph: DependencyGraph, filename: str, root_dir: str, 
                  print_content: bool = False, output_file: str = None):
    subgraph = graph.get_subgraph(filename)
    
    if print_content:
        output = []
        for file in subgraph:
            full_path = os.path.join(root_dir, file)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                output.append(f"{file}\n```\n{content}\n```")
            except Exception as e:
                output.append(f"{file}\n```\nError reading file: {e}\n```")
        result = '\n'.join(output)
    else:
        result = ''
        for node in subgraph:
            result += f"{node}\n"
            for dep in subgraph[node]:
                result += f"  -> {dep}\n"

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
    else:
        print(result)

def main():
    description = """
Node.js Project Dependency Analyzer

This script analyzes a Node.js project directory to create a dependency graph of 
JavaScript/TypeScript files (.js, .jsx, .ts, .tsx). It can scan import/require 
statements, save/load the dependency graph, and display it in various formats.

Usage:
    python script.py <root_dir> [options]

Examples:
    # Scan and save dependency graph to JSON
    python script.py /path/to/project --save graph.json
    
    # Load and print full dependency graph
    python script.py /path/to/project --load graph.json --print
    
    # Show subgraph for a specific file
    python script.py /path/to/project --subgraph src/App.jsx
    
    # Show subgraph with file contents
    python script.py /path/to/project --subgraph src/App.jsx -c
    
    # Save subgraph with contents to file
    python script.py /path/to/project --subgraph src/App.jsx -c --output deps.md
    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('root_dir', help='Root directory of the Node.js project')
    parser.add_argument('--save', help='Save graph to JSON file')
    parser.add_argument('--load', help='Load graph from JSON file')
    parser.add_argument('--print', action='store_true', help='Print dependency graph')
    parser.add_argument('--subgraph', help='Print subgraph for specific file')
    parser.add_argument('-c', '--print-content', action='store_true', 
                       help='Include file contents when printing subgraph')
    parser.add_argument('--output', help='Output file for subgraph')
    
    args = parser.parse_args()

    if args.load:
        graph = load_graph_from_json(args.load)
    else:
        graph = scan_directory(args.root_dir)

    if args.save:
        save_graph_to_json(graph, args.save)
    
    if args.print:
        pretty_print_graph(graph.graph)
    
    if args.subgraph:
        print_subgraph(graph, args.subgraph, args.root_dir, 
                      print_content=args.print_content, 
                      output_file=args.output)

if __name__ == '__main__':
    main()