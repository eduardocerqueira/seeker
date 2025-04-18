#date: 2025-04-18T16:31:46Z
#url: https://api.github.com/gists/fd9e30bb9f19875c8e6822b21589c2be
#owner: https://api.github.com/users/0ndrec

#!/usr/bin/env python3
"""
Swagger OpenAPI to Markdown Converter

This tool analyzes Swagger OpenAPI documentation and converts it
into a structured Markdown document optimized for AI agent recognition.

Install the required libraries using: pip install requests pyyaml
Example usage: python thisfile.py <source> [-o <output>] [-v]

"""

import json
import yaml
import argparse
import requests
from urllib.parse import urlparse
import os
import sys
import re
from typing import Dict, List, Any, Union, Optional

class SwaggerToMarkdown:
    def __init__(self, verbose: bool = False):
        """
        Initialize the converter
        
        Args:
            verbose: Flag for detailed output
        """
        self.verbose = verbose
        
    def log(self, message: str) -> None:
        """Output a message if verbose mode is enabled"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def load_spec(self, source: str) -> Dict[str, Any]:
        """
        Loads OpenAPI specification from URL or file
        
        Args:
            source: URL or path to the specification file
        
        Returns:
            Dictionary with specification data
        """
        self.log(f"Loading specification from {source}")
        
        # Check if source is a URL
        parsed_url = urlparse(source)
        is_url = bool(parsed_url.scheme and parsed_url.netloc)
        
        if is_url:
            # Load from URL
            try:
                response = requests.get(source)
                response.raise_for_status()
                content = response.text
            except requests.exceptions.RequestException as e:
                print(f"Error loading URL {source}: {e}")
                sys.exit(1)
        else:
            # Load from local file
            if not os.path.exists(source):
                print(f"File not found: {source}")
                sys.exit(1)
                
            with open(source, 'r', encoding='utf-8') as file:
                content = file.read()
        
        # Determine format (JSON or YAML) and load data
        try:
            if source.lower().endswith(('.yaml', '.yml')) or content.strip().startswith(('swagger:', 'openapi:')):
                spec = yaml.safe_load(content)
            else:
                spec = json.loads(content)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            print(f"Error parsing specification: {e}")
            sys.exit(1)
            
        self.log(f"Specification successfully loaded")
        return spec
    
    def get_description_markdown(self, description: Optional[str]) -> str:
        if not description:
            return ""
        
        # Remove extra spaces and line breaks
        description = re.sub(r'\n\s*\n+', '\n\n', description.strip())
        
        # If description is short, return as is
        if len(description.split('\n')) == 1 and len(description) < 80:
            return description
            
        # Otherwise format as a description block
        return f"\n{description}\n"
    
    def format_schema(self, schema: Dict[str, Any], indent: int = 0) -> str:
        """
        Formats data schema into readable Markdown
        
        Args:
            schema: Schema dictionary
            indent: Indentation level
            
        Returns:
            Formatted string representation of the schema
        """
        if not schema:
            return ""
            
        result = []
        padding = "  " * indent
            
        # Process basic types
        schema_type = schema.get("type", "object")
        
        # Special handling for references
        if "$ref" in schema:
            ref_name = schema["$ref"].split('/')[-1]
            return f"{padding}- Type: Reference to `{ref_name}`"
            
        # Process arrays
        if schema_type == "array" and "items" in schema:
            items_schema = schema["items"]
            items_type = items_schema.get("type", "object")
            
            if "$ref" in items_schema:
                ref_name = items_schema["$ref"].split('/')[-1]
                result.append(f"{padding}- Type: Array of `{ref_name}`")
            else:
                result.append(f"{padding}- Type: Array of {items_type}")
                if items_type == "object" and "properties" in items_schema:
                    result.append(f"{padding}  Properties:")
                    result.append(self.format_schema(items_schema, indent + 1))
                    
        # Process objects
        elif schema_type == "object":
            properties = schema.get("properties", {})
            if properties:
                result.append(f"{padding}Properties:")
                for prop_name, prop_schema in properties.items():
                    prop_type = prop_schema.get("type", "string")
                    required = "required" in schema and prop_name in schema["required"]
                    req_marker = " (required)" if required else ""
                    
                    if "$ref" in prop_schema:
                        ref_name = prop_schema["$ref"].split('/')[-1]
                        result.append(f"{padding}  - **{prop_name}**{req_marker}: Reference to `{ref_name}`")
                    else:
                        result.append(f"{padding}  - **{prop_name}**{req_marker}: {prop_type}")
                    
                    # Add property description if available
                    if "description" in prop_schema:
                        desc = prop_schema["description"].replace("\n", " ")
                        if len(desc) > 100:  # If description is long, truncate it
                            result.append(f"{padding}    {desc[:100]}...")
                        else:
                            result.append(f"{padding}    {desc}")
                    
                    # If this is a nested object, process it recursively
                    if prop_type == "object" and "properties" in prop_schema:
                        result.append(self.format_schema(prop_schema, indent + 2))
                    # If this is an array of objects
                    elif prop_type == "array" and "items" in prop_schema:
                        items = prop_schema["items"]
                        if "properties" in items:
                            result.append(self.format_schema(items, indent + 2))
        else:
            # Primitive data types
            type_info = [f"Type: {schema_type}"]
            
            # Add format if specified
            if "format" in schema:
                type_info.append(f"Format: {schema['format']}")
                
            # Add constraints if specified
            constraints = []
            if "minimum" in schema:
                constraints.append(f"min: {schema['minimum']}")
            if "maximum" in schema:
                constraints.append(f"max: {schema['maximum']}")
            if "minLength" in schema:
                constraints.append(f"minLength: {schema['minLength']}")
            if "maxLength" in schema:
                constraints.append(f"maxLength: {schema['maxLength']}")
            if "pattern" in schema:
                constraints.append(f"pattern: {schema['pattern']}")
            if "enum" in schema:
                enum_values = ", ".join([f"`{v}`" for v in schema["enum"]])
                constraints.append(f"enum: [{enum_values}]")
                
            if constraints:
                type_info.append(f"Constraints: {', '.join(constraints)}")
                
            result.append(f"{padding}{', '.join(type_info)}")
                
        return "\n".join(result)
        
    def convert_spec_to_markdown(self, spec: Dict[str, Any]) -> str:
        output = []
        
        # Title and basic information
        api_title = spec.get('info', {}).get('title', 'API Documentation')
        api_version = spec.get('info', {}).get('version', '')
        api_description = spec.get('info', {}).get('description', '')
        
        output.append(f"# {api_title} v{api_version}")
        output.append("")
        
        if api_description:
            output.append(self.get_description_markdown(api_description))
            output.append("")
        
        # Server information
        servers = spec.get('servers', [])
        if servers:
            output.append("## Servers")
            for server in servers:
                output.append(f"- {server.get('url')}")
                if 'description' in server:
                    output.append(f"  - {server['description']}")
            output.append("")
        
        # Tags (endpoint groups)
        tags = spec.get('tags', [])
        tag_descriptions = {}
        if tags:
            output.append("## API Tags")
            for tag in tags:
                tag_name = tag.get('name', '')
                tag_description = tag.get('description', '')
                output.append(f"- **{tag_name}**: {tag_description}")
                tag_descriptions[tag_name] = tag_description
            output.append("")
            
        # Data models section
        components = spec.get('components', {})
        schemas = components.get('schemas', {})
        
        if schemas:
            output.append("## Data Models")
            output.append("")
            
            for schema_name, schema_def in schemas.items():
                output.append(f"### {schema_name}")
                
                if "description" in schema_def:
                    output.append(self.get_description_markdown(schema_def["description"]))
                
                output.append(self.format_schema(schema_def))
                output.append("")
        
        # Endpoints (paths)
        paths = spec.get('paths', {})
        
        if paths:
            output.append("## API Endpoints")
            output.append("")
            
            endpoints_by_tag = {}
            
            for path, path_item in paths.items():
                for method, operation in path_item.items():
                    if method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                        op_tags = operation.get('tags', ['default'])
                        
                        for tag in op_tags:
                            if tag not in endpoints_by_tag:
                                endpoints_by_tag[tag] = []
                                
                            endpoints_by_tag[tag].append((path, method, operation))
            

            for tag, endpoints in endpoints_by_tag.items():
                output.append(f"### {tag}")
                
                if tag in tag_descriptions:
                    output.append(self.get_description_markdown(tag_descriptions[tag]))
                
                for path, method, operation in endpoints:
                    operation_id = operation.get('operationId', f"{method} {path}")
                    summary = operation.get('summary', operation_id)
                    
                    # Endpoint header
                    output.append(f"#### {method.upper()} {path}")
                    output.append(f"**{summary}**")
                    
                    # Operation description
                    if 'description' in operation:
                        output.append(self.get_description_markdown(operation['description']))
                    
                    # Request parameters
                    parameters = operation.get('parameters', [])
                    if parameters:
                        output.append("**Parameters:**")
                        
                        # Group parameters by type (path, query, header, cookie)
                        param_groups = {}
                        for param in parameters:
                            param_in = param.get('in', 'query')
                            if param_in not in param_groups:
                                param_groups[param_in] = []
                            param_groups[param_in].append(param)
                        
                        # Output parameters by groups
                        for param_type, params in param_groups.items():
                            output.append(f"- **{param_type.capitalize()} Parameters:**")
                            
                            for param in params:
                                name = param.get('name', '')
                                required = " (required)" if param.get('required', False) else ""
                                schema = param.get('schema', {})
                                schema_type = schema.get('type', 'string')
                                
                                output.append(f"  - **{name}**{required}: {schema_type}")
                                
                                # Parameter description
                                if 'description' in param:
                                    desc = param['description'].replace("\n", " ")
                                    output.append(f"    - {desc}")
                                
                                # Additional information about parameter schema
                                if 'enum' in schema:
                                    enum_values = ", ".join([f"`{v}`" for v in schema["enum"]])
                                    output.append(f"    - Allowed values: {enum_values}")
                        
                        output.append("")
                    
                    # Request body
                    request_body = operation.get('requestBody', {})
                    if request_body:
                        output.append("**Request Body:**")
                        
                        if 'description' in request_body:
                            output.append(self.get_description_markdown(request_body['description']))
                        
                        content = request_body.get('content', {})
                        for content_type, content_schema in content.items():
                            output.append(f"- Content-Type: `{content_type}`")
                            
                            schema = content_schema.get('schema', {})
                            if "$ref" in schema:
                                ref_name = schema["$ref"].split('/')[-1]
                                output.append(f"  - Schema: Reference to `{ref_name}`")
                            else:
                                output.append(self.format_schema(schema, indent=1))
                        
                        output.append("")
                    
                    # Responses
                    responses = operation.get('responses', {})
                    if responses:
                        output.append("**Responses:**")
                        
                        for status_code, response in responses.items():
                            output.append(f"- **{status_code}**: {response.get('description', '')}")
                            
                            # Схема ответа
                            content = response.get('content', {})
                            if content:
                                for content_type, content_schema in content.items():
                                    output.append(f"  - Content-Type: `{content_type}`")
                                    
                                    schema = content_schema.get('schema', {})
                                    if "$ref" in schema:
                                        ref_name = schema["$ref"].split('/')[-1]
                                        output.append(f"    - Schema: Reference to `{ref_name}`")
                                    else:
                                        output.append(self.format_schema(schema, indent=2))
                        
                        output.append("")
                    
                    if 'x-code-samples' in operation:
                        output.append("**Code Examples:**")
                        
                        for sample in operation['x-code-samples']:
                            lang = sample.get('lang', '')
                            source = sample.get('source', '')
                            
                            output.append(f"```{lang}")
                            output.append(source)
                            output.append("```")
                        
                        output.append("")
                    
                    output.append("---")
                    output.append("")
        
        security_schemes = components.get('securitySchemes', {})
        if security_schemes:
            output.append("## Authentication")
            output.append("")
            
            for scheme_name, scheme in security_schemes.items():
                output.append(f"### {scheme_name}")
                
                scheme_type = scheme.get('type', '')
                scheme_desc = scheme.get('description', '')
                
                output.append(f"**Type**: {scheme_type}")
                
                if scheme_desc:
                    output.append(self.get_description_markdown(scheme_desc))
                
                if scheme_type == 'apiKey':
                    output.append(f"**Name**: {scheme.get('name', '')}")
                    output.append(f"**In**: {scheme.get('in', '')}")
                elif scheme_type == 'http':
                    output.append(f"**Scheme**: {scheme.get('scheme', '')}")
                elif scheme_type == 'oauth2':
                    output.append("**Flows:**")
                    flows = scheme.get('flows', {})
                    for flow_name, flow in flows.items():
                        output.append(f"- **{flow_name}**")
                        if 'authorizationUrl' in flow:
                            output.append(f"  - Authorization URL: {flow['authorizationUrl']}")
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"' "**********"t "**********"o "**********"k "**********"e "**********"n "**********"U "**********"r "**********"l "**********"' "**********"  "**********"i "**********"n "**********"  "**********"f "**********"l "**********"o "**********"w "**********": "**********"
                            output.append(f"  - Token URL: "**********"
                        if 'scopes' in flow:
                            output.append("  - Scopes:")
                            for scope, desc in flow['scopes'].items():
                                output.append(f"    - `{scope}`: {desc}")
                
                output.append("")
            
        return "\n".join(output)
        
    def convert(self, source: str, output_file: Optional[str] = None) -> str:
        spec = self.load_spec(source)
        self.log("Converting specification to Markdown")
        markdown_content = self.convert_spec_to_markdown(spec)
        
        if output_file:
            self.log(f"Saving result to {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
        else:
            return markdown_content

def main():

    parser = argparse.ArgumentParser(
        description="Convert Swagger OpenAPI specification to Markdown format."
    )
    
    parser.add_argument(
        "source", 
        help="URL or path to the OpenAPI specification file"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output file path for the Markdown document"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    converter = SwaggerToMarkdown(verbose=args.verbose)
    result = converter.convert(args.source, args.output)
    
    if not args.output:
        print(result)

if __name__ == "__main__":
    main()