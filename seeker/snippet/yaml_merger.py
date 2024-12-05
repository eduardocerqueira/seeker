#date: 2024-12-05T17:03:51Z
#url: https://api.github.com/gists/2d21dab285a18e838bc74ce3a35fa593
#owner: https://api.github.com/users/vlad-dh

# TODO: Move to proper place and add to git workflow

import yaml
import argparse
from typing import Dict, Any
from copy import deepcopy
from pathlib import Path


def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a nested dictionary structure with another dictionary.
    Args:
        base_dict: The original dictionary to be updated
        update_dict: The dictionary containing override values
    Returns:
        Updated dictionary with merged values
    """
    result = deepcopy(base_dict)

    for key, value in update_dict.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # If both values are dictionaries, recurse
            result[key] = deep_update(result[key], value)
        else:
            # Otherwise, override the value
            result[key] = deepcopy(value)

    return result


def merge_yaml_files(base_file: str, override_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Merge two YAML files where the override file takes precedence.
    Args:
        base_file: Path to the base YAML file with the complete structure
        override_file: Path to the YAML file containing override values
        output_file: Optional path to save the merged YAML
    Returns:
        Merged dictionary
    """
    try:
        # Load base file
        with open(base_file, 'r') as f:
            base_config = yaml.safe_load(f)

        # Load override file
        with open(override_file, 'r') as f:
            override_config = yaml.safe_load(f)

        if not isinstance(base_config, dict):
            raise ValueError("Base YAML must be a dictionary")
        if not isinstance(override_config, dict):
            raise ValueError("Override YAML must be a dictionary")

        # Merge the configurations
        merged_config = deep_update(base_config, override_config)

        # Save to output file if specified
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

        return merged_config

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing files: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge two YAML files, where the second file overrides values from the first file.'
    )
    parser.add_argument('base_file', type=str, help='Path to the base YAML file with the complete structure')
    parser.add_argument('override_file', type=str, help='Path to the YAML file containing override values')
    parser.add_argument('-o', '--output', type=str, help='Output file path (optional, defaults to merged_output.yaml)',
                        default='merged_output.yaml')
    parser.add_argument('--print', action='store_true', help='Print the merged YAML to stdout')

    args = parser.parse_args()

    # Validate file paths
    if not Path(args.base_file).exists():
        print(f"Error: Base file '{args.base_file}' does not exist")
        return 1
    if not Path(args.override_file).exists():
        print(f"Error: Override file '{args.override_file}' does not exist")
        return 1

    try:
        merged = merge_yaml_files(args.base_file, args.override_file, args.output)
        print(f"Successfully merged YAML files. Result saved to {args.output}")

        if args.print:
            print("\nMerged YAML content:")
            print(yaml.dump(merged, default_flow_style=False, sort_keys=False))

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())