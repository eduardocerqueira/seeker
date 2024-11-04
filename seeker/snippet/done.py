#date: 2024-11-04T17:06:24Z
#url: https://api.github.com/gists/f31293ccd66129d3dc38637f661a8cbe
#owner: https://api.github.com/users/RajChowdhury240

from rich.console import Console
import subprocess

def execute_aws_command():
    """Execute the AWS command to get the caller identity and display the result in a styled key-value format."""
    try:
        # Run the AWS command and capture the output
        result = subprocess.run("aws sts get-caller-identity", shell=True, check=True, text=True, capture_output=True)
        # Convert output from JSON string to Python dictionary
        identity_info = json.loads(result.stdout)
        console = Console()
        # Print each key-value pair with styling
        for key, value in identity_info.items():
            console.print(f"[green bold]{key}[/green bold]: [green]{value}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"Failed to execute AWS command: [red]{e}[/red]", style="bold red")

