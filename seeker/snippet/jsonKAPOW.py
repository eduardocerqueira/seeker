#date: 2025-05-09T16:59:17Z
#url: https://api.github.com/gists/3baa905955269d8b4f6f284811c4ecce
#owner: https://api.github.com/users/SWORDIntel

#!/usr/bin/env python3
"""
JSON File Partitioner
---------------------
Splits a large JSON file into smaller chunks (10MB by default)
while preserving the integrity of individual records.
"""

import os
import json
import glob
import npyscreen
import sys


class PartitionerApp(npyscreen.NPSAppManaged):
    def main(self):
        form = self.addForm("MAIN", PartitionerForm, name="JSON File Partitioner")
        self.run()


class PartitionerForm(npyscreen.Form):
    def create(self):
        self.add(npyscreen.TitleText, name="Looking for JSON files...", editable=False)
        
        # Find JSON files in current directory
        json_files = glob.glob("*.json")
        
        if not json_files:
            self.add(npyscreen.TitleText, name="Error: No JSON files found in current directory", 
                     editable=False)
            self.add(npyscreen.ButtonPress, name="Exit", when_pressed_function=self.exit_app)
            return
        
        if len(json_files) > 1:
            self.add(npyscreen.TitleText, name="Warning: Multiple JSON files found. Using first one.", 
                     editable=False)
        
        self.json_file = json_files[0]
        self.add(npyscreen.TitleText, name=f"Found JSON file: {self.json_file}", editable=False)
        
        # Get file size
        file_size = os.path.getsize(self.json_file) / (1024 * 1024)
        self.add(npyscreen.TitleText, name=f"File size: {file_size:.2f} MB", editable=False)
        
        # Options
        self.chunk_size = self.add(npyscreen.TitleSlider, name="Chunk Size (MB):", 
                             out_of=100, value=10, step=1, lowest=1)
        
        self.output_prefix = self.add(npyscreen.TitleText, name="Output Prefix:", 
                                value=os.path.splitext(self.json_file)[0])
        
        # Add buttons
        self.add(npyscreen.ButtonPress, name="Start Processing", 
                 when_pressed_function=self.start_processing)
        self.add(npyscreen.ButtonPress, name="Exit", 
                 when_pressed_function=self.exit_app)
    
    def start_processing(self):
        self.parentApp.setNextForm(None)
        self.editing = False
        
        # Create results form to show progress
        results_form = self.parentApp.addForm("RESULTS", ResultsForm, name="Processing Results")
        results_form.file_path = self.json_file
        results_form.chunk_size_mb = self.chunk_size.value
        results_form.output_prefix = self.output_prefix.value
        
        self.parentApp.switchForm("RESULTS")
    
    def exit_app(self):
        self.parentApp.setNextForm(None)
        self.editing = False


class ResultsForm(npyscreen.Form):
    def create(self):
        self.progress_text = self.add(npyscreen.TitleText, name="Progress:", 
                               value="Starting...", editable=False)
        self.progress = self.add(npyscreen.TitleSlider, name="", 
                         out_of=100, value=0, editable=False)
        self.results = self.add(npyscreen.BufferPager, name="Results:", height=10)
        self.add(npyscreen.ButtonPress, name="Exit", when_pressed_function=self.exit_app)
    
    def afterEditing(self):
        self.process_file()
    
    def exit_app(self):
        self.parentApp.setNextForm(None)
        self.editing = False
    
    def update_progress(self, value, message):
        self.progress.value = value
        self.progress_text.value = message
        self.display()
    
    def add_result(self, message):
        self.results.buffer.append(message)
        self.results.display()
    
    def process_file(self):
        """Process the JSON file and split it into chunks."""
        try:
            file_path = self.file_path
            chunk_size_bytes = self.chunk_size_mb * 1024 * 1024
            output_prefix = self.output_prefix
            
            self.update_progress(5, f"Analyzing {file_path}...")
            
            # Determine if the file is an array of JSON objects or newline-delimited
            with open(file_path, 'r') as f:
                first_char = f.read(1).strip()
                is_json_array = first_char == '['
            
            self.update_progress(10, f"Detected format: {'JSON array' if is_json_array else 'Newline-delimited JSON'}")
            
            if is_json_array:
                self.process_json_array(file_path, chunk_size_bytes, output_prefix)
            else:
                self.process_newline_json(file_path, chunk_size_bytes, output_prefix)
                
        except Exception as e:
            self.update_progress(100, f"Error: {str(e)}")
            self.add_result(f"An error occurred: {str(e)}")
    
    def process_json_array(self, file_path, chunk_size_bytes, output_prefix):
        """Process a JSON file that contains an array of objects."""
        self.update_progress(15, "Loading JSON array (this may take a while for large files)...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                self.update_progress(100, "Error: JSON file is not an array")
                self.add_result("The JSON file does not contain an array at the top level.")
                return
                
            total_records = len(data)
            self.update_progress(30, f"Loaded {total_records} records from JSON array")
            
            # Split into chunks
            current_chunk = []
            current_size = 2  # Account for [ and ] characters
            chunk_num = 1
            chunks_created = 0
            
            self.update_progress(40, "Starting to create chunks...")
            
            for i, record in enumerate(data):
                # Calculate the size this record would add
                record_json = json.dumps(record)
                record_size = len(record_json.encode('utf-8'))
                
                # Add comma if not first record in chunk
                if current_chunk:
                    current_size += 1  # for the comma
                
                # If adding this record would exceed chunk size and we have records, write chunk
                if current_chunk and (current_size + record_size > chunk_size_bytes):
                    output_file = f"{output_prefix}_part{chunk_num:03d}.json"
                    with open(output_file, 'w') as f:
                        f.write("[\n")
                        f.write(",\n".join(json.dumps(obj) for obj in current_chunk))
                        f.write("\n]")
                    
                    chunks_created += 1
                    self.add_result(f"Created {output_file} with {len(current_chunk)} records")
                    
                    # Reset for next chunk
                    current_chunk = []
                    current_size = 2  # Reset size ([ and ])
                    chunk_num += 1
                
                # Add record to current chunk
                current_chunk.append(record)
                current_size += record_size
                
                # Update progress periodically
                if i % max(1, total_records // 100) == 0:
                    progress_pct = min(40 + int(50 * i / total_records), 90)
                    self.update_progress(progress_pct, f"Processing record {i+1}/{total_records}...")
            
            # Write the last chunk if there's anything left
            if current_chunk:
                output_file = f"{output_prefix}_part{chunk_num:03d}.json"
                with open(output_file, 'w') as f:
                    f.write("[\n")
                    f.write(",\n".join(json.dumps(obj) for obj in current_chunk))
                    f.write("\n]")
                
                chunks_created += 1
                self.add_result(f"Created {output_file} with {len(current_chunk)} records")
            
            self.update_progress(100, f"Complete! Created {chunks_created} chunk files.")
            self.add_result(f"Successfully split {total_records} records into {chunks_created} files.")
            
        except json.JSONDecodeError as e:
            self.update_progress(100, f"JSON parsing error: {str(e)}")
            self.add_result(f"Failed to parse JSON: {str(e)}")
        except Exception as e:
            self.update_progress(100, f"Error: {str(e)}")
            self.add_result(f"An error occurred: {str(e)}")
    
    def process_newline_json(self, file_path, chunk_size_bytes, output_prefix):
        """Process a newline-delimited JSON file."""
        try:
            # First, count total lines for progress reporting
            self.update_progress(15, "Counting records in file...")
            total_lines = 0
            with open(file_path, 'r') as f:
                for _ in f:
                    total_lines += 1
            
            self.update_progress(20, f"Found {total_lines} records")
            
            # Now process line by line
            current_chunk = []
            current_size = 0
            chunk_num = 1
            chunks_created = 0
            processed_lines = 0
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Calculate size this record would add
                    line_size = len(line.encode('utf-8'))
                    
                    # If adding this line would exceed chunk size and we have records, write chunk
                    if current_chunk and (current_size + line_size > chunk_size_bytes):
                        output_file = f"{output_prefix}_part{chunk_num:03d}.json"
                        with open(output_file, 'w') as out_f:
                            out_f.write("\n".join(current_chunk))
                        
                        chunks_created += 1
                        self.add_result(f"Created {output_file} with {len(current_chunk)} records")
                        
                        # Reset for next chunk
                        current_chunk = []
                        current_size = 0
                        chunk_num += 1
                    
                    # Add line to current chunk
                    current_chunk.append(line)
                    current_size += line_size
                    
                    # Update progress periodically
                    processed_lines += 1
                    if processed_lines % max(1, total_lines // 100) == 0:
                        progress_pct = min(20 + int(70 * processed_lines / total_lines), 90)
                        self.update_progress(progress_pct, f"Processing record {processed_lines}/{total_lines}...")
            
            # Write the last chunk if there's anything left
            if current_chunk:
                output_file = f"{output_prefix}_part{chunk_num:03d}.json"
                with open(output_file, 'w') as f:
                    f.write("\n".join(current_chunk))
                
                chunks_created += 1
                self.add_result(f"Created {output_file} with {len(current_chunk)} records")
            
            self.update_progress(100, f"Complete! Created {chunks_created} chunk files.")
            self.add_result(f"Successfully split {processed_lines} records into {chunks_created} files.")
            
        except Exception as e:
            self.update_progress(100, f"Error: {str(e)}")
            self.add_result(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    app = PartitionerApp()
    app.run()