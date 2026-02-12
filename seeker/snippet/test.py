#date: 2026-02-12T17:47:35Z
#url: https://api.github.com/gists/105f83ad335a6f565f288b19f7c2fac4
#owner: https://api.github.com/users/vishwast-gep

import json
import copy

def add_thoughts_and_reasoning_to_tools(json_data):
    """
    Add 'thoughts' and 'reasoning' parameters to all tools in the workflow JSON.
    This function is idempotent - it will not override existing parameters.
    
    Args:
        json_data: The workflow JSON object
    
    Returns:
        Modified JSON object with thoughts and reasoning added to all tools,
        along with counts of added vs skipped parameters
    """
    # Create a deep copy to avoid modifying the original
    modified_data = copy.deepcopy(json_data)
    
    # Track statistics
    stats = {
        'thoughts_added': 0,
        'thoughts_skipped': 0,
        'reasoning_added': 0,
        'reasoning_skipped': 0
    }
    
    # Find all agent nodes
    for node in modified_data.get('nodes', []):
        if node.get('type') == 'agent':
            tools = node.get('config', {}).get('tools', [])
            
            for tool in tools:
                # Get the existing schema
                schema = tool.get('config', {}).get('schema', {})
                
                # Ensure properties object exists
                if 'properties' not in schema:
                    schema['properties'] = {}
                
                # Add thoughts parameter only if not already present
                if 'thoughts' not in schema['properties']:
                    schema['properties']['thoughts'] = {
                        "type": "string",
                        "description": "Non-technical explanation for end users about what this agent is doing at this stage. Explain the purpose and outcome in simple, business-friendly language that helps users understand the process without technical jargon."
                    }
                    stats['thoughts_added'] += 1
                else:
                    stats['thoughts_skipped'] += 1
                
                # Add reasoning parameter only if not already present
                if 'reasoning' not in schema['properties']:
                    schema['properties']['reasoning'] = {
                        "type": "string",
                        "description": "Technical reason why this specific tool/agent is being called. Explain the decision logic, conditions met, or context that led to selecting this tool for the current task."
                    }
                    stats['reasoning_added'] += 1
                else:
                    stats['reasoning_skipped'] += 1
                
                # Ensure required array exists
                if 'required' not in schema:
                    schema['required'] = []
                
                # Add thoughts and reasoning to required fields if not already present
                if 'thoughts' not in schema['required']:
                    schema['required'].append('thoughts')
                if 'reasoning' not in schema['required']:
                    schema['required'].append('reasoning')
                
                # Update the schema back
                tool['config']['schema'] = schema
    
    return modified_data, stats

def main():
    # Read the input JSON file
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = '/Users/vishwas.tak/Downloads/amendment.json'
    
    with open(input_path, 'r') as f:
        workflow_data = json.load(f)
    
    # Add thoughts and reasoning to all tools
    updated_workflow, stats = add_thoughts_and_reasoning_to_tools(workflow_data)
    
    # Save the modified JSON
    output_path = '/Users/vishwas.tak/Downloads/amendment-updated.json'
    with open(output_path, 'w') as f:
        json.dump(updated_workflow, f, indent=2)
    
    # Print summary
    agent_count = 0
    tool_count = 0
    for node in updated_workflow.get('nodes', []):
        if node.get('type') == 'agent':
            agent_count += 1
            tool_count += len(node.get('config', {}).get('tools', []))
    
    print(f"✓ Successfully processed workflow JSON")
    print(f"✓ Found {agent_count} agent node(s) with {tool_count} tool(s)")
    print(f"\nIdempotent update summary:")
    print(f"  - 'thoughts' added: {stats['thoughts_added']}, skipped (already present): {stats['thoughts_skipped']}")
    print(f"  - 'reasoning' added: {stats['reasoning_added']}, skipped (already present): {stats['reasoning_skipped']}")
    print(f"✓ Both parameters marked as required")
    print(f"\nOutput saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    output_file = main()