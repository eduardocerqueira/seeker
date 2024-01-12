#date: 2024-01-12T16:42:27Z
#url: https://api.github.com/gists/cb97cc07a30ce63b10784076b9e8117e
#owner: https://api.github.com/users/PatrickMTonne

import pandas as pd

# Run STEP 1 or STEP 2 individually (i.e. comment out STEP 2 when running STEP 1)

# STEP 1
# read in Sections and Courses CSVs
df1 = pd.read_csv('~/Desktop/sections_prod_01_11.csv', low_memory=False) # Sections
df2 = pd.read_csv('~/Desktop/courses_prod_01_11.csv', low_memory=False) # Courses

merged = df1.merge(df2, on='canvas_course_id', how='inner')

# Remove rows that are not empty in the 'section_id' column
merged = merged[~merged['section_id'].notna()]

# Keep rows where 'canvas_term_id' column is greater than 312 or equals 80.
# canvas_term_ids > 312 are associated with 2021 or later (canvas_term_id 312 == term_id 2020-16; canvas_term_id 313 == term_id 2021-4)
# canvas_term_id 80 == Ongoing
merged = merged[(merged['canvas_term_id'] > 312) | (merged['canvas_term_id'] == 80)]

# merge and create output file
merged.to_csv('~/Desktop/sections_course_prod_filtered_01_11.csv', index=False)

# prints size of resulting data set
print(merged.shape[0])


# STEP 2
# read in resulting Sections/Course merged file from above and Enrollments CSV
df1 = pd.read_csv('~/Desktop/sections_course_prod_filtered_01_11.csv', low_memory=False) # Output file from above
df2 = pd.read_csv('~/Desktop/enrollments_prod_01_11.csv', low_memory=False) # Enrollments

merged = df1.merge(df2, on='canvas_course_id', how='inner')

# merge and create output file
merged.to_csv('~/Desktop/sections_course_enrollments_filtered_prod_01_11.csv', index=False)
print(merged.shape[0])