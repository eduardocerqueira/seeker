#date: 2026-01-30T17:29:28Z
#url: https://api.github.com/gists/88a30601cb02174f455215285fadcb38
#owner: https://api.github.com/users/fernandesi2244

#!/usr/bin/env python3
"""
C# Project Dependency Mapper - Interactive Version
Analyzes .csproj files and creates an interactive HTML visualization with hover effects.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import json

def find_csproj_files(root_dir="."):
    """
    Recursively find all .csproj files in the directory tree.
    
    Args:
        root_dir: Root directory to start searching from
        
    Returns:
        List of Path objects for all .csproj files found
    """
    csproj_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csproj'):
                csproj_files.append(Path(root) / file)
    return csproj_files

def extract_project_name(csproj_path):
    """
    Extract the project name from a .csproj file path.
    
    Args:
        csproj_path: Path to the .csproj file
        
    Returns:
        Project name (filename without .csproj extension)
    """
    return csproj_path.stem

def parse_project_references(csproj_path):
    """
    Parse a .csproj file and extract ProjectReference dependencies.
    
    Args:
        csproj_path: Path to the .csproj file
        
    Returns:
        List of project names that this project depends on
    """
    dependencies = []
    
    try:
        tree = ET.parse(csproj_path)
        root = tree.getroot()
        
        # Find all ProjectReference elements (namespace-agnostic)
        for project_ref in root.findall(".//ProjectReference"):
            include_path = project_ref.get('Include')
            if include_path:
                # Resolve the relative path to get the absolute path
                ref_path = (csproj_path.parent / include_path).resolve()
                # Extract just the project name
                project_name = ref_path.stem
                dependencies.append(project_name)
                
    except ET.ParseError as e:
        print(f"Warning: Could not parse {csproj_path}: {e}")
    except Exception as e:
        print(f"Warning: Error processing {csproj_path}: {e}")
    
    return dependencies

def build_dependency_graph(root_dir="."):
    """
    Build a complete dependency graph of all C# projects.
    
    Args:
        root_dir: Root directory to start searching from
        
    Returns:
        Dictionary mapping project names to lists of their dependencies
    """
    csproj_files = find_csproj_files(root_dir)
    
    if not csproj_files:
        print(f"No .csproj files found in {root_dir}")
        return {}
    
    print(f"Found {len(csproj_files)} .csproj files")
    
    dependency_graph = {}
    test_project_count = 0
    asz_project_count = 0
    
    for csproj_path in csproj_files:
        project_name = extract_project_name(csproj_path)
        
        # Mark if this is a test project
        is_test = project_name.startswith("Test") or project_name.endswith(".Tests")
        if is_test:
            test_project_count += 1
        
        # Mark if this is an Asz project
        is_asz = project_name.endswith(".Asz")
        if is_asz:
            asz_project_count += 1
        
        dependencies = parse_project_references(csproj_path)
        
        dependency_graph[project_name] = {
            'dependencies': dependencies,
            'is_test': is_test,
            'is_asz': is_asz
        }
        
        if dependencies:
            print(f"{project_name} -> {', '.join(dependencies)}")
        else:
            print(f"{project_name} (no dependencies)")
    
    if test_project_count > 0:
        print(f"\nFound {test_project_count} test project(s) (starting with 'Test' or ending with '.Tests')")
    if asz_project_count > 0:
        print(f"Found {asz_project_count} .Asz project(s)")
    
    return dependency_graph

def calculate_reverse_dependencies(dependency_graph):
    """
    Calculate reverse dependencies (who depends on each project).
    
    Args:
        dependency_graph: Dictionary mapping project names to their data (dependencies and is_test flag)
        
    Returns:
        Dictionary mapping project names to lists of projects that depend on them
    """
    reverse_deps = defaultdict(list)
    
    for project, data in dependency_graph.items():
        for dep in data['dependencies']:
            reverse_deps[dep].append(project)
    
    # Ensure all projects are in the reverse_deps dict
    for project in dependency_graph.keys():
        if project not in reverse_deps:
            reverse_deps[project] = []
    
    return dict(reverse_deps)

def create_interactive_visualization(dependency_graph, output_file="dependency_graph_interactive.html"):
    """
    Create an interactive HTML visualization using D3.js.
    
    Args:
        dependency_graph: Dictionary mapping project names to their dependencies
        output_file: Filename for the output HTML file
    """
    if not dependency_graph:
        print("No dependencies to visualize")
        return
    
    # Calculate reverse dependencies
    reverse_deps = calculate_reverse_dependencies(dependency_graph)
    
    # Prepare data for D3.js
    nodes_data = []
    edges_data = []
    
    # Create nodes
    for project, data in dependency_graph.items():
        nodes_data.append({
            "id": project,
            "dependencies": data['dependencies'],
            "dependents": reverse_deps.get(project, []),
            "is_test": data['is_test'],
            "is_asz": data['is_asz']
        })
    
    # Create edges
    for project, data in dependency_graph.items():
        for dep in data['dependencies']:
            edges_data.append({
                "source": project,
                "target": dep
            })
    
    # Create HTML with embedded D3.js visualization
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>C# Project Dependency Graph - Interactive</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
        }}
        
        #container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-top: 0;
        }}
        
        #graph {{
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }}
        
        .node rect {{
            fill: lightblue;
            stroke: darkblue;
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .node.highlighted rect {{
            fill: #ffeb3b;
            stroke: #f57c00;
            stroke-width: 3px;
        }}
        
        .node.dependent rect {{
            fill: #81c784;
            stroke: #388e3c;
        }}
        
        .node.dependency rect {{
            fill: #ff8a65;
            stroke: #d84315;
        }}
        
        .node text {{
            font-size: 16px;
            font-weight: bold;
            pointer-events: none;
            user-select: none;
        }}
        
        .link {{
            fill: none;
            stroke: #999;
            stroke-width: 2.5;
            stroke-opacity: 0.4;
            marker-end: url(#arrowhead);
        }}
        
        .link.highlighted {{
            stroke: #f57c00;
            stroke-width: 3.5;
            stroke-opacity: 0.9;
        }}
        
        .link.from-highlighted {{
            stroke: #d84315;
            stroke-width: 3.5;
            stroke-opacity: 0.9;
        }}
        
        .link.to-highlighted {{
            stroke: #388e3c;
            stroke-width: 3.5;
            stroke-opacity: 0.9;
        }}
        
        #info-panel {{
            margin-top: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 4px;
            border: 1px solid #ddd;
            min-height: 80px;
        }}
        
        .info-section {{
            margin: 10px 0;
        }}
        
        .info-section h3 {{
            margin: 5px 0;
            color: #333;
            font-size: 14px;
        }}
        
        .info-section ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
        
        .info-section li {{
            margin: 3px 0;
            color: #666;
        }}
        
        .legend {{
            margin-top: 15px;
            padding: 10px;
            background: #fff;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            font-size: 14px;
        }}
        
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            vertical-align: middle;
            margin-right: 5px;
            border: 1px solid #333;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <h1>C# Project Dependency Graph - Interactive</h1>
        <p style="text-align: center; color: #666;">
            Hover over a project to see its dependencies and dependents<br>
            <strong>Controls:</strong> Scroll to zoom • Drag to pan • Double-click to reset view
        </p>
        
        <div style="text-align: center; margin: 15px 0; padding: 15px; background: #f9f9f9; border-radius: 4px; border: 1px solid #ddd;">
            <div style="display: inline-block; margin: 0 20px;">
                <label style="font-size: 14px; cursor: pointer;">
                    <input type="checkbox" id="staggerToggle" style="margin-right: 5px; cursor: pointer;">
                    Enable alternating vertical offsets
                </label>
            </div>
            <div style="display: inline-block; margin: 0 20px;">
                <label style="font-size: 14px; cursor: pointer;">
                    <input type="checkbox" id="includeTestsToggle" style="margin-right: 5px; cursor: pointer;">
                    Include test projects
                </label>
            </div>
            <div style="display: inline-block; margin: 0 20px;">
                <label style="font-size: 14px; cursor: pointer;">
                    <input type="checkbox" id="includeAszToggle" style="margin-right: 5px; cursor: pointer;">
                    Include .Asz projects
                </label>
            </div>
            <div style="display: inline-block; margin: 0 20px;">
                <label style="font-size: 14px; cursor: pointer;">
                    <input type="checkbox" id="includeNoDepsToggle" style="margin-right: 5px; cursor: pointer;" checked>
                    Include projects with no dependencies
                </label>
            </div>
            <br><br>
            <div style="display: inline-block; margin: 0 20px; text-align: left; vertical-align: top;">
                <label style="font-size: 14px; display: block; margin-bottom: 5px;">
                    <strong>Custom Filters:</strong> Hide projects that match (starts AND ends):
                </label>
                <div style="margin-top: 8px;">
                    <label style="font-size: 13px; margin-right: 5px;">
                        Starts with: 
                        <input type="text" id="filterPrefix" placeholder="e.g. Test" 
                               style="padding: 4px 8px; margin-left: 5px; width: 100px; border: 1px solid #ccc; border-radius: 3px;">
                    </label>
                    <label style="font-size: 13px; margin-right: 5px;">
                        Ends with: 
                        <input type="text" id="filterSuffix" placeholder="e.g. .Tests" 
                               style="padding: 4px 8px; margin-left: 5px; width: 100px; border: 1px solid #ccc; border-radius: 3px;">
                    </label>
                    <button id="addFilter" style="padding: 5px 12px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 13px;">
                        Add Filter
                    </button>
                </div>
                <div id="activeFilters" style="margin-top: 10px; min-height: 20px;">
                    <!-- Active filters will appear here -->
                </div>
            </div>
            <br><br>
            <div style="display: inline-block; margin: 0 20px; text-align: left;">
                <label for="horizontalSpacing" style="font-size: 14px; display: block; margin-bottom: 5px;">
                    Horizontal spacing between cells: <strong><span id="horizontalValue">350</span>px</strong>
                </label>
                <input type="range" id="horizontalSpacing" min="150" max="600" value="350" step="10" 
                       style="width: 250px; cursor: pointer;">
            </div>
            <div style="display: inline-block; margin: 0 20px; text-align: left;">
                <label for="verticalSpacing" style="font-size: 14px; display: block; margin-bottom: 5px;">
                    Vertical spacing between layers: <strong><span id="verticalValue">120</span>px</strong>
                </label>
                <input type="range" id="verticalSpacing" min="60" max="300" value="120" step="10" 
                       style="width: 250px; cursor: pointer;">
            </div>
        </div>
        
        <svg id="graph"></svg>
        
        <div id="info-panel">
            <p style="color: #999; text-align: center;">Hover over a project to see details (info stays visible until you hover over another project)</p>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background: #ffeb3b; border-color: #f57c00;"></span>
                <span>Hovered Project</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #ff8a65; border-color: #d84315;"></span>
                <span>Dependencies (this project depends on)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #81c784; border-color: #388e3c;"></span>
                <span>Dependents (projects that depend on this)</span>
            </div>
        </div>
    </div>
    
    <script>
        const allNodesData = {json.dumps(nodes_data, indent=8)};
        const allEdgesData = {json.dumps(edges_data, indent=8)};
        
        let includeTests = false;
        let includeAsz = false;
        let includeNoDeps = true;  // Start with this checked/enabled
        let customFilters = [];  // Array of {{prefix, suffix}} objects
        
        function renderActiveFilters() {{
            const container = document.getElementById('activeFilters');
            if (customFilters.length === 0) {{
                container.innerHTML = '<span style="color: #999; font-size: 12px;">No active filters</span>';
                return;
            }}
            
            container.innerHTML = customFilters.map((filter, index) => {{
                const prefixText = filter.prefix || '(any)';
                const suffixText = filter.suffix || '(any)';
                return `
                    <div style="display: inline-block; margin: 3px; padding: 5px 10px; background: #e3f2fd; border: 1px solid #2196F3; border-radius: 3px; font-size: 12px;">
                        <strong>Filter ${{index + 1}}:</strong> Starts: <em>${{prefixText}}</em>, Ends: <em>${{suffixText}}</em>
                        <button onclick="removeFilter(${{index}})" style="margin-left: 8px; padding: 2px 6px; background: #f44336; color: white; border: none; border-radius: 2px; cursor: pointer; font-size: 11px;">
                            ✕
                        </button>
                    </div>
                `;
            }}).join('');
        }}
        
        window.removeFilter = function(index) {{
            customFilters.splice(index, 1);
            renderActiveFilters();
            applyFilters();
            initializeGraph();
        }};
        
        function applyFilters() {{
            let filteredNodes = allNodesData;
            
            // Filter out test projects if needed
            if (!includeTests) {{
                filteredNodes = filteredNodes.filter(n => !n.is_test);
            }}
            
            // Filter out .Asz projects if needed
            if (!includeAsz) {{
                filteredNodes = filteredNodes.filter(n => !n.is_asz);
            }}
            
            // Filter out projects with no dependencies if needed
            if (!includeNoDeps) {{
                filteredNodes = filteredNodes.filter(n => n.dependencies.length > 0);
            }}
            
            // Apply all custom filters
            customFilters.forEach(filter => {{
                filteredNodes = filteredNodes.filter(n => {{
                    const matchesPrefix = filter.prefix ? n.id.startsWith(filter.prefix) : true;
                    const matchesSuffix = filter.suffix ? n.id.endsWith(filter.suffix) : true;
                    // Exclude nodes that match BOTH conditions
                    if (filter.prefix && filter.suffix) {{
                        return !(matchesPrefix && matchesSuffix);
                    }} else if (filter.prefix) {{
                        return !matchesPrefix;
                    }} else if (filter.suffix) {{
                        return !matchesSuffix;
                    }}
                    return true;
                }});
            }});
            
            nodesData = filteredNodes;
            
            // Filter edges to only include those between visible nodes
            const visibleNodeIds = new Set(filteredNodes.map(n => n.id));
            edgesData = allEdgesData.filter(e => 
                visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target)
            );
        }}
        
        // Apply initial filters
        applyFilters();
        renderActiveFilters();
        
        // Set up SVG with much larger canvas
        const containerWidth = document.getElementById('graph').clientWidth;
        const containerHeight = 800;
        
        const svg = d3.select("#graph")
            .attr("width", containerWidth)
            .attr("height", containerHeight);
        
        // Create a group for all content that will be zoomed/panned
        let g = svg.append("g");
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Variables for layout
        let horizontalSpacing = 350;
        let verticalSpacing = 120;
        let enableStagger = false;
        let nodes, links, positions, layers, layerGroups, maxNodesInLayer, maxLayer, canvasWidth, canvasHeight;
        
        function initializeGraph() {{
            // Clear existing graph
            g.selectAll("*").remove();
            
            // First, calculate layers to get accurate max nodes per layer
            const nodeMap = new Map(nodesData.map(n => [n.id, n]));
        
            // Improved topological sort and layering using longest path
            function calculateLayers() {{
            const layers = new Map();
            const inDegree = new Map();
            const adjList = new Map();
            
            // Initialize
            nodesData.forEach(node => {{
                layers.set(node.id, 0);
                inDegree.set(node.id, 0);
                adjList.set(node.id, []);
            }});
            
            // Build adjacency list and calculate in-degrees
            // Note: edges go FROM dependent TO dependency
            edgesData.forEach(edge => {{
                if (adjList.has(edge.target)) {{
                    adjList.get(edge.target).push(edge.source);
                }}
                if (inDegree.has(edge.source)) {{
                    inDegree.set(edge.source, inDegree.get(edge.source) + 1);
                }}
            }});
            
            // Use a queue for topological sort (Kahn's algorithm with longest path)
            const queue = [];
            
            // Start with nodes that have no dependencies (in-degree 0)
            inDegree.forEach((degree, nodeId) => {{
                if (degree === 0) {{
                    queue.push(nodeId);
                    layers.set(nodeId, 0);
                }}
            }});
            
            // Process nodes level by level
            while (queue.length > 0) {{
                const current = queue.shift();
                const currentLayer = layers.get(current);
                
                // Update all nodes that depend on this node
                const dependents = adjList.get(current) || [];
                dependents.forEach(dependent => {{
                    // The dependent should be at least one layer higher
                    const newLayer = currentLayer + 1;
                    if (newLayer > layers.get(dependent)) {{
                        layers.set(dependent, newLayer);
                    }}
                    
                    // Decrease in-degree
                    const newDegree = inDegree.get(dependent) - 1;
                    inDegree.set(dependent, newDegree);
                    
                    // If all dependencies processed, add to queue
                    if (newDegree === 0) {{
                        queue.push(dependent);
                    }}
                }});
            }}
            
            return layers;
        }}
        
        layers = calculateLayers();
        
        // Group nodes by layer to find max nodes in any layer
        layerGroups = new Map();
        layers.forEach((layer, nodeId) => {{
            if (!layerGroups.has(layer)) layerGroups.set(layer, []);
            layerGroups.get(layer).push(nodeId);
        }});
        
        maxNodesInLayer = Math.max(...Array.from(layerGroups.values()).map(arr => arr.length));
        maxLayer = Math.max(...layers.values());
        
        // Calculate required canvas size based on actual layout
        canvasWidth = Math.max(containerWidth, maxNodesInLayer * horizontalSpacing + 400);
        canvasHeight = Math.max(2000, (maxLayer + 2) * verticalSpacing);
        
        // Create arrow marker
        g.append("defs").append("marker")
        g.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 15)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");
        
        // Position nodes with more spacing
        positions = new Map();
        let enableStagger = false;  // Toggle for alternating vertical offsets
        
        layerGroups.forEach((nodeIds, layer) => {{
            nodeIds.forEach((nodeId, i) => {{
                const x = (i - nodeIds.length / 2 + 0.5) * horizontalSpacing + canvasWidth / 2;
                const y = canvasHeight - (layer + 1) * verticalSpacing;
                const stagger = enableStagger ? ((i % 2 === 0) ? 20 : -20) : 0;
                positions.set(nodeId, {{ x, y: y + stagger }});
            }});
        }});
        
        // Create links
        links = g.append("g").selectAll("path")
            .data(edgesData)
            .enter().append("path")
            .attr("class", "link")
            .attr("d", d => {{
                const source = positions.get(d.source);
                const target = positions.get(d.target);
                if (!source || !target) return "";
                return `M${{source.x}},${{source.y}} L${{target.x}},${{target.y}}`;
            }});
        
        // Create nodes
        nodes = g.append("g").selectAll("g")
            .data(nodesData)
            .enter().append("g")
            .attr("class", "node")
            .attr("transform", d => {{
                const pos = positions.get(d.id);
                return pos ? `translate(${{pos.x}},${{pos.y}})` : "";
            }});
        
        // Add rectangles and text for each node
        nodes.each(function(d) {{
            const node = d3.select(this);
            
            // Create temporary text to measure size
            const tempText = svg.append("text")
                .style("font-size", "16px")
                .style("font-weight", "bold")
                .text(d.id);
            const bbox = tempText.node().getBBox();
            tempText.remove();
            
            const padding = 20;
            const width = bbox.width + padding;
            const height = bbox.height + padding;
            
            node.append("rect")
                .attr("x", -width / 2)
                .attr("y", -height / 2)
                .attr("width", width)
                .attr("height", height)
                .attr("rx", 4);
            
            node.append("text")
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .text(d.id);
        }});
        
        // Add hover interactions
        nodes
            .on("mouseenter", function(event, d) {{
                // Highlight current node
                d3.select(this).classed("highlighted", true);
                
                // Highlight dependencies
                d.dependencies.forEach(dep => {{
                    nodes.filter(n => n.id === dep).classed("dependency", true);
                }});
                
                // Highlight dependents
                d.dependents.forEach(dep => {{
                    nodes.filter(n => n.id === dep).classed("dependent", true);
                }});
                
                // Highlight links
                links.classed("from-highlighted", link => link.source === d.id);
                links.classed("to-highlighted", link => link.target === d.id);
                
                // Update info panel (and keep it)
                updateInfoPanel(d);
            }})
            .on("mouseleave", function(event, d) {{
                // Remove all highlights
                nodes.classed("highlighted", false)
                     .classed("dependency", false)
                     .classed("dependent", false);
                
                links.classed("from-highlighted", false)
                     .classed("to-highlighted", false);
                
                // DON'T clear info panel - keep the last hovered project info
            }});
        
        // Initial zoom to fit all content - wait a bit for rendering
        setTimeout(() => {{
            const bounds = g.node().getBBox();
            const fullWidth = bounds.width;
            const fullHeight = bounds.height;
            const midX = bounds.x + fullWidth / 2;
            const midY = bounds.y + fullHeight / 2;
            
            const scale = 0.9 / Math.max(fullWidth / containerWidth, fullHeight / containerHeight);
            const translate = [containerWidth / 2 - scale * midX, containerHeight / 2 - scale * midY];
            
            svg.call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
            
            // Double-click to reset zoom
            svg.on("dblclick.zoom", function() {{
                svg.transition()
                    .duration(750)
                    .call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
            }});
        }}, 100);
        }}  // End of initializeGraph function
        
        // Initialize the graph for the first time
        initializeGraph();
        
        // Add toggle event handler for including test projects
        document.getElementById('includeTestsToggle').addEventListener('change', function(e) {{
            includeTests = e.target.checked;
            applyFilters();
            initializeGraph();
        }});
        
        // Add toggle event handler for including .Asz projects
        document.getElementById('includeAszToggle').addEventListener('change', function(e) {{
            includeAsz = e.target.checked;
            applyFilters();
            initializeGraph();
        }});
        
        // Add toggle event handler for including projects with no dependencies
        document.getElementById('includeNoDepsToggle').addEventListener('change', function(e) {{
            includeNoDeps = e.target.checked;
            applyFilters();
            initializeGraph();
        }});
        
        // Add custom filter handlers
        document.getElementById('addFilter').addEventListener('click', function() {{
            const prefix = document.getElementById('filterPrefix').value.trim();
            const suffix = document.getElementById('filterSuffix').value.trim();
            
            if (prefix || suffix) {{
                customFilters.push({{ prefix, suffix }});
                document.getElementById('filterPrefix').value = '';
                document.getElementById('filterSuffix').value = '';
                renderActiveFilters();
                applyFilters();
                initializeGraph();
            }}
        }});
        
        // Allow Enter key to add filter
        document.getElementById('filterPrefix').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                document.getElementById('addFilter').click();
            }}
        }});
        
        document.getElementById('filterSuffix').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                document.getElementById('addFilter').click();
            }}
        }});
        
        // Helper function to recalculate positions
        function recalculatePositions() {{
            // Recalculate canvas size
            canvasWidth = Math.max(containerWidth, maxNodesInLayer * horizontalSpacing + 400);
            canvasHeight = Math.max(2000, (maxLayer + 2) * verticalSpacing);
            
            // Recalculate positions
            layerGroups.forEach((nodeIds, layer) => {{
                nodeIds.forEach((nodeId, i) => {{
                    const x = (i - nodeIds.length / 2 + 0.5) * horizontalSpacing + canvasWidth / 2;
                    const y = canvasHeight - (layer + 1) * verticalSpacing;
                    const stagger = enableStagger ? ((i % 2 === 0) ? 20 : -20) : 0;
                    positions.set(nodeId, {{ x, y: y + stagger }});
                }});
            }});
            
            // Update node positions with animation
            nodes.transition()
                .duration(300)
                .attr("transform", d => {{
                    const pos = positions.get(d.id);
                    return pos ? `translate(${{pos.x}},${{pos.y}})` : "";
                }});
            
            // Update link positions with animation
            links.transition()
                .duration(300)
                .attr("d", d => {{
                    const source = positions.get(d.source);
                    const target = positions.get(d.target);
                    if (!source || !target) return "";
                    return `M${{source.x}},${{source.y}} L${{target.x}},${{target.y}}`;
                }});
        }}
        
        // Add toggle event handler for stagger
        document.getElementById('staggerToggle').addEventListener('change', function(e) {{
            enableStagger = e.target.checked;
            recalculatePositions();
        }});
        
        // Add slider event handlers
        document.getElementById('horizontalSpacing').addEventListener('input', function(e) {{
            horizontalSpacing = parseInt(e.target.value);
            document.getElementById('horizontalValue').textContent = horizontalSpacing;
            recalculatePositions();
        }});
        
        document.getElementById('verticalSpacing').addEventListener('input', function(e) {{
            verticalSpacing = parseInt(e.target.value);
            document.getElementById('verticalValue').textContent = verticalSpacing;
            recalculatePositions();
        }});
        
        function updateInfoPanel(node) {{
            const panel = document.getElementById('info-panel');
            
            let html = `<h2 style="margin-top: 0; color: #333;">${{node.id}}</h2>`;
            
            if (node.dependencies.length > 0) {{
                html += `<div class="info-section">
                    <h3>📦 Dependencies (this project depends on):</h3>
                    <ul>${{node.dependencies.map(d => `<li>${{d}}</li>`).join('')}}</ul>
                </div>`;
            }} else {{
                html += `<div class="info-section">
                    <h3>📦 Dependencies:</h3>
                    <p style="color: #999; margin: 5px 0;">No dependencies</p>
                </div>`;
            }}
            
            if (node.dependents.length > 0) {{
                html += `<div class="info-section">
                    <h3>⬆️ Dependents (projects that depend on this):</h3>
                    <ul>${{node.dependents.map(d => `<li>${{d}}</li>`).join('')}}</ul>
                </div>`;
            }} else {{
                html += `<div class="info-section">
                    <h3>⬆️ Dependents:</h3>
                    <p style="color: #999; margin: 5px 0;">No dependents</p>
                </div>`;
            }}
            
            panel.innerHTML = html;
        }}
        
        function clearInfoPanel() {{
            const panel = document.getElementById('info-panel');
            panel.innerHTML = '<p style="color: #999; text-align: center;">Hover over a project to see details</p>';
        }}
    </script>
</body>
</html>"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nInteractive dependency graph saved to {output_file}")
    print("Open this file in a web browser to view the interactive visualization.")

def main():
    """Main function to execute the dependency analysis."""
    print("C# Project Dependency Mapper - Interactive Version")
    print("=" * 50)
    
    # Get the current working directory
    root_dir = os.getcwd()
    print(f"Scanning directory: {root_dir}\n")
    
    # Build the dependency graph
    dependency_graph = build_dependency_graph(root_dir)
    
    if dependency_graph:
        print("\n" + "=" * 50)
        # Create interactive visualization
        create_interactive_visualization(dependency_graph)
    else:
        print("No C# projects found to analyze.")

if __name__ == "__main__":
    main()