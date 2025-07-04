#date: 2025-07-04T16:55:14Z
#url: https://api.github.com/gists/27cef40a6be3f6bb725aeafdf8410b51
#owner: https://api.github.com/users/GitFather47

import re
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Union
import threading
import os


class LogAnalyzer:
    def __init__(self): 
        self.log_entries = []
        self.parse_errors = []
        self.ip_counter = Counter()
        self.endpoint_counter = Counter()
        self.status_counter = Counter()
        self.total_response_size = 0
        self.total_requests = 0
        self.unique_ips = set()
        self.timestamps = []
        
    def validate_file(self, filepath: str) -> bool:
        """Input Validation--if the file is a valid log file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found.")
        
        # Check file extension - only .txt files are supported
        valid_extensions = ['.txt']
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension not in valid_extensions:
            raise ValueError(f"Invalid file type. Only .txt files are supported. Got: {file_extension}")
        
        # Check if file is readable and contains text
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Try to read first few lines to validate it's a text file
                sample = f.read(1024)
                if not sample:
                    raise ValueError("File is empty or unreadable.")
                
                # Check for binary content
                if '\x00' in sample:
                    raise ValueError("File appears to contain binary data, not text.")
                    
        except UnicodeDecodeError:
            raise ValueError("File is not a valid text file (encoding issues).")
        except Exception as e:
            raise ValueError(f"Cannot read file: {str(e)}")
        
        return True
    
    def validate_log_format(self, filepath: str) -> bool:
        """Validate if the file contains valid log entries"""
        valid_log_lines = 0
        total_lines = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  
                        continue
                    
                    total_lines += 1
                    if self.parse_log_line(line):
                        valid_log_lines += 1
                    
                    # Check first 20 lines for quick validation
                    if total_lines >= 20:
                        break
        except Exception as e:
            raise ValueError(f"Error validating log format: {str(e)}")
        
        if total_lines == 0:
            raise ValueError("File contains no content.")
        
        # Require at least 10% of lines to be valid log entries
        validity_ratio = valid_log_lines / total_lines
        if validity_ratio < 0.1:
            raise ValueError(f"File does not appear to contain valid log entries. Only {validity_ratio:.1%} of lines are valid log format.")
        
        return True
        
    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a single log line using regex pattern matching"""
        # Apache Common Log Format pattern
        pattern = r'^(\S+) \S+ \S+ \[([^\]]+)\] "([A-Z]+) ([^"]*) HTTP/[\d\.]+" (\d+) (\d+|\-)$'
        
        match = re.match(pattern, line.strip())
        if not match:
            return None
            
        ip, timestamp_str, method, path, status_code, response_size = match.groups()
        
        try:
            timestamp = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
            
            # Handle response size
            size = int(response_size) if response_size != '-' else 0
            
            return {
                'ip': ip,
                'timestamp': timestamp,
                'method': method,
                'path': path,
                'status_code': int(status_code),
                'response_size': size
            }
        except (ValueError, TypeError) as e:
            return None
    
    def analyze_file(self, filepath: str, progress_callback=None):
        """Process entire log file and extract analytics"""
        self.validate_file(filepath)
        self.validate_log_format(filepath)
        
        try:
            # Get file size for progress tracking
            file_size = os.path.getsize(filepath)
            bytes_read = 0
            
            with open(filepath, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    bytes_read += len(line) + 1  #for newline
                    
                    if not line: 
                        continue
                        
                    parsed = self.parse_log_line(line)
                    if parsed:
                        self.log_entries.append(parsed)
                        self._update_counters(parsed)
                    else:
                        self.parse_errors.append(f"Line {line_num}: {line}")
                    
                    # Update progress every 1000 lines
                    if progress_callback and line_num % 1000 == 0:
                        progress = (bytes_read / file_size) * 100
                        progress_callback(progress)
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filepath}' not found.")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
            
        self._calculate_metrics()
        
    def _update_counters(self, entry: Dict):
        """Update internal counters with parsed log entry"""
        self.ip_counter[entry['ip']] += 1
        self.endpoint_counter[entry['path']] += 1
        self.status_counter[entry['status_code']] += 1
        self.total_response_size += entry['response_size']
        self.total_requests += 1
        self.unique_ips.add(entry['ip'])
        self.timestamps.append(entry['timestamp'])
        
    def _calculate_metrics(self):
        """Calculate derived metrics"""
        if not self.log_entries:
            return
        self.timestamps.sort()
        
    def _get_status_distribution(self) -> Dict:
        """Calculate status code distribution by category"""
        distribution = {
            '2xx': {'count': 0, 'rate': 0.0},
            '3xx': {'count': 0, 'rate': 0.0},
            '4xx': {'count': 0, 'rate': 0.0},
            '5xx': {'count': 0, 'rate': 0.0}
        }
        
        if self.total_requests == 0:
            return distribution
            
        for status_code, count in self.status_counter.items():
            if 200 <= status_code < 300:
                distribution['2xx']['count'] += count
            elif 300 <= status_code < 400:
                distribution['3xx']['count'] += count
            elif 400 <= status_code < 500:
                distribution['4xx']['count'] += count
            elif 500 <= status_code < 600:
                distribution['5xx']['count'] += count
                
        for category in distribution:
            distribution[category]['rate'] = round(
                (distribution[category]['count'] / self.total_requests) * 100, 1
            )
            
        return distribution
    
    def _get_error_rate(self) -> float:
        """Calculate error rate (4xx and 5xx responses)"""
        if self.total_requests == 0:
            return 0.0
            
        error_count = sum(count for status, count in self.status_counter.items() 
                         if status >= 400)
        return round((error_count / self.total_requests) * 100, 1)
    
    def _get_average_response_size(self) -> int:
        """Calculate average response size"""
        if self.total_requests == 0:
            return 0
        return int(self.total_response_size / self.total_requests)
    
    def _get_analysis_period(self) -> str:
        """Get analysis period string"""
        if not self.timestamps:
            return "No valid timestamps found"
            
        start_time = self.timestamps[0].strftime('%Y-%m-%d %H:%M:%S')
        end_time = self.timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')
        return f"{start_time} to {end_time}"
    
    def generate_console_report(self) -> str:
        """Generate human-readable console report as string"""
        report = []
        report.append("=== Log Analysis Report ===")
        report.append(f"Analysis Period: {self._get_analysis_period()}")
        report.append("")
        
        report.append("SUMMARY:")
        report.append(f"- Total Requests: {self.total_requests:,}")
        report.append(f"- Unique IP Addresses: {len(self.unique_ips)}")
        report.append(f"- Average Response Size: {self._get_average_response_size():,} bytes")
        report.append(f"- Error Rate: {self._get_error_rate()}%")
        report.append("")
        
        report.append("TOP IP ADDRESSES:")
        for i, (ip, count) in enumerate(self.ip_counter.most_common(5), 1):
            report.append(f"{i}. {ip} ({count} requests)")
        report.append("")
        
        report.append("STATUS CODE DISTRIBUTION:")
        status_dist = self._get_status_distribution()
        for category, data in status_dist.items():
            status_name = {'2xx': 'Success', '3xx': 'Redirect', '4xx': 'Client Error', '5xx': 'Server Error'}[category]
            report.append(f"- {category} {status_name}: {data['count']:,} ({data['rate']}%)")
        report.append("")
        
        report.append("TOP ENDPOINTS:")
        for i, (endpoint, count) in enumerate(self.endpoint_counter.most_common(10), 1):
            report.append(f"{i}. {endpoint} ({count} requests)")
        
        # Show parse errors (if any)
        if self.parse_errors:
            report.append(f"\nPARSE ERRORS ({len(self.parse_errors)} lines):")
            for error in self.parse_errors[:5]:  # Show first 5 errors
                report.append(f"- {error}")
            if len(self.parse_errors) > 5:
                report.append(f"... and {len(self.parse_errors) - 5} more")
        
        return "\n".join(report)
    
    def generate_json_report(self) -> str:
        """Generate machine-readable JSON report as string"""
        top_ips = [{"ip": ip, "requests": count} 
                   for ip, count in self.ip_counter.most_common(5)]
        
        top_endpoints = [{"endpoint": endpoint, "count": count} 
                        for endpoint, count in self.endpoint_counter.most_common(10)]
        
        status_distribution = self._get_status_distribution()
        
        report = {
            "total_requests": self.total_requests,
            "unique_ip": len(self.unique_ips),
            "average_response_size": self._get_average_response_size(),
            "error_rate": self._get_error_rate(),
            "analysis_period": self._get_analysis_period(),
            "top_ips": top_ips,
            "status_distribution": status_distribution,
            "top_endpoints": top_endpoints,
            "parse_errors": len(self.parse_errors)
        }
        
        return json.dumps(report, indent=2)


class LogAnalyzerGUI:
    def __init__(self, root): 
        self.root = root
        self.root.title("Log Analyzer - Internship Task for Omukk")
        self.root.geometry("900x700")
        
        # Initialize analyzer
        self.analyzer = LogAnalyzer()
        self.current_file = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Log File:").grid(row=0, column=0, sticky=tk.W)
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_var, state="readonly")
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(0, 5))
        self.analyze_btn = ttk.Button(file_frame, text="Analyze", command=self.analyze_file, state="disabled")
        self.analyze_btn.grid(row=0, column=3)

        self.validation_label = ttk.Label(file_frame, text="Supported formats: .txt files only", 
                                        foreground="gray")
        self.validation_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="5")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text widget with scrollbar
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=80, height=30)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        export_frame = ttk.Frame(main_frame)
        export_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        self.export_txt_btn = ttk.Button(export_frame, text="Export as TXT", command=self.export_txt, state="disabled")
        self.export_txt_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.export_json_btn = ttk.Button(export_frame, text="Export as JSON", command=self.export_json, state="disabled")
        self.export_json_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(export_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=(5, 0))
        
    def browse_file(self):
        """Open file dialog to select log file"""
        filename = filedialog.askopenfilename(
            title="Select Log File",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            # Validate file immediately upon selection
            try:
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension not in ['.txt']:
                    messagebox.showerror("Invalid File Type", 
                                       f"Please select a .txt file.\n"
                                       f"Selected file has extension: {file_extension}")
                    return
                
                self.file_var.set(filename)
                self.current_file = filename
                self.analyze_btn.config(state="normal")
                self.validation_label.config(text="✓ Valid file selected", foreground="green")
                
            except Exception as e:
                messagebox.showerror("File Selection Error", f"Error selecting file:\n{str(e)}")
                self.validation_label.config(text="✗ Invalid file selected", foreground="red")
                
    def analyze_file(self):
        """Analyze the selected log file"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select a log file first.")
            return
            
        # Reset progress bar
        self.progress['value'] = 0
        self.root.update_idletasks()
        
        # Disable analyze button during processing
        self.analyze_btn.config(state="disabled")
        
        # Run analysis in a separate thread to prevent GUI freezing
        threading.Thread(target=self._analyze_thread, daemon=True).start()
        
    def _analyze_thread(self):
        """Thread function for analyzing log file"""
        try:
            # Reset analyzer
            self.analyzer = LogAnalyzer()
            
            # Analyze file with progress callback
            self.analyzer.analyze_file(self.current_file, self.update_progress)
            
            # Generate report
            report = self.analyzer.generate_console_report()
            
            self.root.after(0, self._analysis_complete, report)
            
        except ValueError as e:
            self.root.after(0, self._validation_error, str(e))
        except Exception as e:
            self.root.after(0, self._analysis_error, str(e))
            
    def update_progress(self, value):
        """Update progress bar"""
        self.progress['value'] = value
        self.root.update_idletasks()
        
    def _analysis_complete(self, report):
        """Called when analysis is complete"""
        self.progress['value'] = 100
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, report)
        
        # Buttons
        self.analyze_btn.config(state="normal")
        self.export_txt_btn.config(state="normal")
        self.export_json_btn.config(state="normal")
        
        messagebox.showinfo("Success", f"Analysis complete!\nProcessed {self.analyzer.total_requests:,} requests")
        
    def _validation_error(self, error_msg):
        """Called when file validation fails"""
        self.progress['value'] = 0
        self.analyze_btn.config(state="normal")
        self.validation_label.config(text="✗ File validation failed", foreground="red")
        messagebox.showerror("File Validation Error", 
                           f"The selected file is not valid:\n\n{error_msg}\n\n"
                           f"Please ensure you select a valid .txt file containing "
                           f"Apache Common Log Format entries.")
        
    def _analysis_error(self, error_msg):
        """Called when analysis encounters an error"""
        self.progress['value'] = 0
        self.analyze_btn.config(state="normal")
        messagebox.showerror("Analysis Error", f"Analysis failed:\n{error_msg}")
        
    def export_txt(self):
        """Export results as text file"""
        if not self.analyzer.log_entries:
            messagebox.showwarning("Warning", "No analysis results to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Text Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                report = self.analyzer.generate_console_report()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Text report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save text report:\n{e}")
                
    def export_json(self):
        """Export results as JSON file"""
        if not self.analyzer.log_entries:
            messagebox.showwarning("Warning", "No analysis results to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save JSON Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                report = self.analyzer.generate_json_report()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"JSON report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save JSON report:\n{e}")
                
    def clear_results(self):
        """Clear the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.export_txt_btn.config(state="disabled")
        self.export_json_btn.config(state="disabled")
        self.validation_label.config(text="Supported formats: .txt files only", foreground="gray")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = LogAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":  
    main()