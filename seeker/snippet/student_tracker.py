#date: 2026-02-25T17:43:46Z
#url: https://api.github.com/gists/7234b7c629354890cc49d6bdda1bb3c0
#owner: https://api.github.com/users/scovia433

students = [
    ("Scovia", 15),
    ("Jenniffer", 18),
    ("Chinelo", 12)
]

# 2. Additional info stored in a dictionary
student_info = {
    "Scovia": {"age": 13, "grade": "8"},
    "Jenniffer": {"age": 14, "grade": "8"},
    "Chinelo": {"age": 13, "grade": "7"}
}

# --- Print all students and their scores ---
print("All Students:")
for name, score in students:
    print(f"{name} - {score}")
print()  


name_input = input("Enter student name to view details: ").strip()

# Find the student in the list and get score
score_found = None
for name, score in students:
    if name.lower() == name_input.lower():  
        score_found = score
        student_name = name  
        break

if score_found is not None and student_name in student_info:
    info = student_info[student_name]
    print(f"Name: {student_name}")
    print(f"Score: {score_found}")
    print(f"Age: {info['age']}")
    print(f"Grade: {info['grade']}")
else:
    print("Student not found.")
print()

scores = [score for _, score in students]
average = sum(scores) / len(scores)
print(f"Class Average Score: {average}")


max_score = max(scores)
min_score = min(scores)

# Find which student(s) have the max and min
max_students = [name for name, score in students if score == max_score]
min_students = [name for name, score in students if score == min_score]

# Format output (if multiple students tie, they are all shown)
print(f"Highest Score: {max_score} ({', '.join(max_students)})")
print(f"Lowest Score: {min_score} ({', '.join(min_students)})")