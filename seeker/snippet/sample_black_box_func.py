#date: 2024-12-13T17:07:48Z
#url: https://api.github.com/gists/2f2b5b2a34a0ef39e6e114f20c28ad88
#owner: https://api.github.com/users/pradhyumna85

from model_utils_example import modpredict
from preprocess import preprocess_example
from pathlib import Path
from typing import Union


def black_box_function(param1: int, param2: str, param3: bool, 
                       input_file1_path: str, input_file2_path: str, 
                       output_file1_path: str, output_file2_path: str) -> tuple[str, str]:
    """
    A black-box function to demonstrate parameter usage, file input/output handling, 
    and integration with imported utilities.

    Args:
        param1 (int): Example integer parameter to control some logic (e.g., iterations).
        param2 (str): Example string parameter (e.g., model name or operation type).
        param3 (bool): Example boolean parameter to toggle a condition.
        input_file1_path (str): Path to the first input file.
        input_file2_path (str): Path to the second input file.
        output_file1_path (str): Path where the first output file will be written.
        output_file2_path (str): Path where the second output file will be written.

    Returns:
        tuple[str, str]: The paths of the output files.
    """
    try:
        # Step 0: Any Input validations
        if not isinstance(param1, int) or param1 <= 0:
            raise ValueError("param1 must be a positive integer.")
        pass

        # Step 1: Any Preprocessing the input files etc
        print("Starting preprocessing...")
        preprocessed_data1 = preprocess_example(input_file1_path)
        preprocessed_data2 = preprocess_example(input_file2_path)
        print("Preprocessing completed.")

        # Step 2: Applying the model or main operation logic
        print(f"Running the model with param1={param1}, param2='{param2}', param3={param3}...")
        result1 = modpredict(preprocessed_data1, param1, param2)
        result2 = modpredict(preprocessed_data2, param1, param2)
        print("Model prediction completed.")

        # Step 3: Any misc. steps
        if param3:
            print("Performing additional processing due to param3 being True...")
            result1 = result1.upper()  # Example modification
            result2 = result2.lower()  # Example modification

        # Step 4: Writing results to the output files
        print("Writing results to output files...")
        with open(output_file1_path, 'w') as file1:
            file1.write(result1)
        with open(output_file2_path, 'w') as file2:
            file2.write(result2)
        print("Output files written successfully.")

        return output_file1_path, output_file2_path

    except Exception as e:
        print(f"An error occurred: {e}")
