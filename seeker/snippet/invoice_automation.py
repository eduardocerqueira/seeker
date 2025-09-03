#date: 2025-09-03T16:58:40Z
#url: https://api.github.com/gists/da0936e6afdc2e12c9bc6dd492c5ec06
#owner: https://api.github.com/users/phyrk

import os
from PyPDF2 import PdfReader, PdfWriter
import zipfile
import re
 
# Define paths for input and output
# input_pdf_path = r'C:\.... Invoice Report.pdf'
output_folder = r'W:\...\Invoices\\'


os.makedirs(output_folder, exist_ok=True)
 
# Step 1: Extract the 8th line from the text
def extract_8th_line(text):
    # Only keep the first 8 lines and skip full-page extraction
    lines = text.split("\n")[:5]
    if len(lines) >= 5:
        return lines[3].strip()  # 8th line (index 7)
    return "Invoice_Unknown"
 
# Step 2: Extract anything between "-" and "AR#" from the 8th line
def extract_between_dash_and_ar(text):
    match = re.search(r'-(.*?)AR#', text)
    if match:
        return match.group(1).strip()
    return None  # Return None if no match is found
 
# Sanitize file names to avoid invalid characters
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
 
# Step 3: Split and combine PDF by extracted name
def split_and_combine_pdf_by_8th_line(input_pdf_path, output_folder):
    reader = PdfReader(input_pdf_path)
    files_written = {}
    # Loop through each page of the PDF
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()[:1000]  # Extract only the first 1000 characters for faster processing
        eighth_line = extract_8th_line(text)
        file_name = extract_between_dash_and_ar(eighth_line)
 
        # If we can't extract the file name, create a fallback name based on the page number
        if not file_name:
            file_name = f"Invoice_Page_{page_num + 1}"
 
        file_name = sanitize_filename(file_name)  # Sanitize the file name
 
        # Collect all pages for the same file name
        if file_name not in files_written:
            files_written[file_name] = []
        files_written[file_name].append(page)
 
    # Write all files, combining pages for same-named files
    written_files = []
    for file_name, pages in files_written.items():
        output_path = f"{output_folder}{file_name}.pdf"
        writer = PdfWriter()
        for page in pages:
            writer.add_page(page)
        # Write the combined pages at once
        with open(output_path, "wb") as output_pdf:
            writer.write(output_pdf)
            written_files.append(output_path)
        print(f"Written file: {output_path}")  # Debugging statement
    return written_files
 
# Execute the function to split and combine the PDF
written_files_combined = split_and_combine_pdf_by_8th_line(input_pdf_path, output_folder)
 
# Optional: Zip the files if you want to package them for transfer or storage

output_zip_path = r'W:\...combined_invoices.zip'

with zipfile.ZipFile(output_zip_path, 'w') as zipf:
    for file_path in written_files_combined:
        file_name = os.path.basename(file_path)
        zipf.write(file_path, arcname=file_name)
 
print(f"PDF files written to: {output_folder}")


## Emailing PDFs

import os
import pandas as pd
import win32com.client as win32
 
# Load the Excel file containing email data (only from the 'Normal' sheet)

excel_path = r'W:\...\List.xlsx'
df = pd.read_excel(excel_path, sheet_name='Normal')
 
# Set the directory containing the PDF files

folder_path = r'W:\...Invoices\\'

outlook = win32.Dispatch('outlook.application')

no_matching_recipient = []
 
# Loop through each PDF in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        # Extract the file name without the ".pdf" extension
        pdf_name = filename.replace('.pdf', '')
        matched_row = df[df['Column1'].str.contains(pdf_name, case=False, na=False)]
        if not matched_row.empty:
            recipient_email = matched_row['Email'].values[0]
     
            if not str(recipient_email) == 'nan':
                subject = f'Invoice: {matched_row["Column1"].values[0]}'
                body = f'Please find the attached invoice for {matched_row["Column1"].values[0]}.'
                # Create a new email
                mail = outlook.CreateItem(0)
                mail.Subject = subject
                mail.Body = body
                mail.To = recipient_email
                # Attach the PDF file
                attachment_path = os.path.join(folder_path, filename)
                mail.Attachments.Add(attachment_path)
                # Send the email
                mail.Send()
                print(f"Email sent to {recipient_email} with attachment: {filename}")
            else:
                print(f"***BLANK EMAIL for {filename}. Ignored.")    
                no_matching_recipient.append('+++NO EMAIL: ' + filename )  # Add unmatched PDFs to the list
        else:
            print(f"No matching recipient for {filename}. Ignored.")
            no_matching_recipient.append('---NOT IN THE LIST: ' + filename)  # Add unmatched PDFs to the list
           
# If there are unmatched files, save them to an Excel file
if no_matching_recipient:
    warning_df = pd.DataFrame(no_matching_recipient, columns=["Filename"])
   
    output_path = r'W:\...\WARNING No Matching Recipient.xlsx'
    warning_df.to_excel(output_path, index=False)
    print(f"Unmatched PDFs saved to: {output_path}")
else:
    print("All PDFs matched and sent successfully.")
