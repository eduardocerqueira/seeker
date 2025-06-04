#date: 2025-06-04T16:56:38Z
#url: https://api.github.com/gists/5884590e937f2cbb39d195b5564937b7
#owner: https://api.github.com/users/logickoder

import requests
import re
import pandas as pd


def fetch_student_result(matric_no):
    url = "https://student-results-db.onrender.com/"
    response = requests.post(url, data={"matric_no": matric_no})

    if response.status_code == 200:
        html_content = response.text

        name_match = re.search(r"<h3>Result for (.*?) \((.*?)\)</h3>", html_content)
        if name_match:
            student_name = name_match.group(1)
            matric_number = name_match.group(2)

            lab_score = re.search(
                r"Lab attendance and report <em>/50</em>: (\d+)", html_content
            )
            exam_score = re.search(r"Exam <em>/50</em>: (\d+)", html_content)
            total_score = re.search(r"Total <em>/100</em>: (\d+)", html_content)

            result = {
                "Name": student_name,
                "Matric No": matric_number,
                "Lab": int(lab_score.group(1)) if lab_score else None,
                "Exam": int(exam_score.group(1)) if exam_score else None,
                "Total": int(total_score.group(1)) if total_score else None,
            }

            return result

        return None
    else:
        print(f"Failed to fetch results for {matric_no}: {response.status_code}")
        return None


def main():
    print("Fetching student matric...")
    with open("matric.txt", "r") as f:
        matric_nos = [line.strip() for line in f if line.strip().isdigit()]

    results = []
    for matric_no in matric_nos:
        result = fetch_student_result(matric_no)
        if result:
            print(f"Fetched results for {result['Matric No']}: {result['Name']}")
            results.append(result)
        else:
            print(f"No results found for matric number {matric_no}.")

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Matric No")
        # df = df.sort_values(by="Total") Sort by total score if needed
        df.to_excel("student_results.xlsx", index=False)
        print("Results exported to student_results.xlsx")


if __name__ == "__main__":
    main()
