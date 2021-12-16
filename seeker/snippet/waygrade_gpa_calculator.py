#date: 2021-12-16T17:17:44Z
#url: https://api.github.com/gists/fd305822620de49ef79aa22fd79b7fb4
#owner: https://api.github.com/users/thevolt6

    def gpa_calculator(class_details):

        if not class_details:
            print("NO CLASS DETAILS FOR GPA CALCULATOR!")
            return {"Weighted": "", "Unweighted": ""}

        LETTER_GPA = {
            "COLLEGE": {
                "A": 4.0,
                "A-": 3.7,
                "B+": 3.3,
                "B": 3.0,
                "B-": 2.7,
                "C+": 2.3,
                "C": 2.0,
                "C-": 1.7,
                "D+": 1.3,
                "D": 1.0,
                "D-": 0.7,
                "F": 0.0
            },
            "HONORS": {
                "A": 4.5,
                "A-": 4.2,
                "B+": 3.8,
                "B": 3.5,
                "B-": 3.2,
                "C+": 2.8,
                "C": 2.5,
                "C-": 2.2,
                "D+": 1.8,
                "D": 1.5,
                "D-": 1.2,
                "F": 0.5
            },
            "AP": {
                "A": 5.0,
                "A-": 4.7,
                "B+": 4.3,
                "B": 4.0,
                "B-": 3.7,
                "C+": 3.3,
                "C": 3.0,
                "C-": 2.7,
                "D+": 2.3,
                "D": 2.0,
                "D-": 1.7,
                "F": 1.0
            }
        }

        classes = class_details["CLASSES"]
        classCount = len(classes)
        unweightedTotal = 0
        weightedTotal = 0

        for _class in classes:
            className = _class["CLASS_NAME"]
            classGrade = _class["LETTER_GRADE"]
            if classGrade == "TBA":
                classCount -= 1
                continue

            AP = re.findall("AP", className)
            AP2 = re.findall("Advanced Placement", className)
            HONORS = re.findall("Honors", className)
            COLLEGE = re.findall("College", className)

            # Unweighted GPA Calc
            unweightedTotal += LETTER_GPA["COLLEGE"][classGrade]

            # Weighted GPA Calc
            if COLLEGE:
                weightedTotal += LETTER_GPA["COLLEGE"][classGrade]
            elif HONORS:
                weightedTotal += LETTER_GPA["HONORS"][classGrade]
            elif AP or AP2:
                weightedTotal += LETTER_GPA["AP"][classGrade]
            else:
                weightedTotal += LETTER_GPA["COLLEGE"][classGrade]
        if not classCount:
            weightedGPA = 0
            unweightedGPA = 0
        else:
            weightedGPA = round(weightedTotal / classCount, 2)
            unweightedGPA = round(unweightedTotal / classCount, 2)

        return {"Weighted": weightedGPA,
                "Unweighted": unweightedGPA}