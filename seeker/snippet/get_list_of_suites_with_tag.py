#date: 2024-12-03T17:02:09Z
#url: https://api.github.com/gists/d66348748e9720ad595700ce83a084c3
#owner: https://api.github.com/users/adiralashiva8

import os
from robot.api.parsing import get_model, ModelVisitor, Token

class RobotParser(ModelVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.has_smoke_tag = False

    def visit_SettingSection(self, node):
        # Check for 'Force Tags' in the settings section
        for child in node.body:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"c "**********"h "**********"i "**********"l "**********"d "**********". "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"F "**********"O "**********"R "**********"C "**********"E "**********"_ "**********"T "**********"A "**********"G "**********"S "**********": "**********"
                suite_tags = "**********"
                if any("smoke" in tag.lower() for tag in suite_tags):
                    self.has_smoke_tag = True
                    return  # Exit early if 'smoke' is found

    def visit_TestCase(self, node):
        # Check for 'Tags' in the test cases
        for section in node.body:
            if hasattr(section, "get_value"):
                tags = "**********"
                if tags and any("smoke" in tag.lower() for tag in tags):
                    self.has_smoke_tag = True
                    return  # Exit early if 'smoke' is found

    def suite_contains_smoke(self):
        return self.has_smoke_tag


def get_robot_metadata(filepath):
    """Checks if a Robot Framework suite file contains the 'smoke' tag."""
    model = get_model(filepath)
    robot_parser = RobotParser(filepath)
    robot_parser.visit(model)
    return robot_parser.suite_contains_smoke()


def fetch_suites_with_smoke_tag(folder):
    """Fetches all Robot Framework suite files containing the 'smoke' tag."""
    suites = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".robot"):
                file_path = os.path.join(root, file)
                if get_robot_metadata(file_path):
                    suites.append(file_path)
    return suites


# Example usage
if __name__ == "__main__":
    folder = "SNOW"
    suites_with_smoke_tag = fetch_suites_with_smoke_tag(folder)

    for suite in suites_with_smoke_tag:
        print(suite)
