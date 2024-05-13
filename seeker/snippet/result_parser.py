#date: 2024-05-13T17:07:34Z
#url: https://api.github.com/gists/6dc1c6c36a0f7d3fbade67eb64387ecc
#owner: https://api.github.com/users/T-Dynamos

from pprint import pprint
import traceback
import sys

file = sys.argv[1]


SUBJECT_CODES = {
    "001": "ENGLISH ELECTIVE-N",
    "002": "HINDI ELECTIVE",
    "003": "URDU ELECTIVE",
    "004": "PUNJABI",
    "022": "SANSKRIT ELECTIVE",
    "027": "HISTORY",
    "028": "POLITICAL SCIENCE",
    "029": "GEOGRAPHY",
    "030": "ECONOMICS",
    "031": "CARNATIC MUSIC VOC",
    "034": "HIND.MUSIC VOCAL",
    "035": "HIND.MUSIC MEL.INS",
    "036": "HIND MUSIC.INS.PER",
    "037": "PSYCHOLOGY",
    "039": "SOCIOLOGY",
    "040": "PHILOSOPHY",
    "041": "MATHEMATICS",
    "042": "PHYSICS",
    "043": "CHEMISTRY",
    "044": "BIOLOGY",
    "045": "BIOTECHNOLOGY",
    "046": "ENGG. GRAPHICS",
    "048": "PHYSICAL EDUCATION",
    "049": "PAINTING",
    "050": "GRAPHICS",
    "051": "SCULPTURE",
    "052": "APP/COMMERCIAL ART",
    "053": "FASHION STUDIES",
    "054": "BUSINESS STUDIES",
    "055": "ACCOUNTANCY",
    "056": "DANCE-KATHAK",
    "057": "DANCE-BHARATNATYAM",
    "059": "DANCE-ODISSI",
    "061": "DANCE-KATHAKALI",
    "064": "HOME SCIENCE",
    "065": "INFORMATICS PRAC.",
    "066": "ENTREPRENEURSHIP",
    "067": "MULTIMEDIA & WEB T",
    "068": "AGRICULTURE",
    "069": "CR WRTNG TR STUDY",
    "070": "HERITAGE CRAFTS",
    "071": "GRAPHIC DESIGN",
    "072": "MASS MEDIA STUDIES",
    "SUBJECTWISE": "NO. OF CANDIDATES(Reg) FOR",
    "073": "KNOW TRAD & PRAC.",
    "074": "LEGAL STUDIES",
    "075": "HUMAN RIGHTS & G S",
    "076": "NAT. CADET CORPS",
    "078": "THEATRE STUDIES",
    "079": "LIBRARY & INFO SC.",
    "083": "COMPUTER SCIENCE",
    "086": "SCIENCE",
    "087": "SOCIAL SCIENCE",
    "085": "HINDI COURSE-B",
    "101": "ENGLISH ELECTIVE-C",
    "104": "PUNJABI",
    "105": "BENGALI",
    "106": "TAMIL",
    "107": "TELUGU",
    "108": "SINDHI",
    "109": "MARATHI",
    "110": "GUJARATI",
    "111": "MANIPURI",
    "112": "MALAYALAM",
    "113": "ODIA",
    "114": "ASSAMESE",
    "115": "KANNADA",
    "116": "ARABIC",
    "117": "TIBETAN",
    "118": "FRENCH",
    "120": "GERMAN",
    "121": "RUSSIAN",
    "123": "PERSIAN",
    "124": "NEPALI",
    "125": "LIMBOO",
    "126": "LEPCHA",
    "189": "TELUGU - TELANGANA",
    "184": "ENGLISH",
    "193": "TANGKHUL",
    "194": "JAPANESE",
    "195": "BHUTIA",
    "196": "SPANISH",
    "198": "MIZO",
    "205": "BENGALI W/O PR.",
    "241": "APPLIED MATHEMATICS",
    "301": "ENGLISH CORE",
    "302": "HINDI CORE",
    "303": "URDU CORE",
    "322": "SANSKRIT CORE",
    "402": "INFORMATION TECHNOLOGY",
    "604": "OFFCE PROC.& PRAC.",
    "605": "SECY.PRAC & ACCNTG",
    "606": "OFF. COMMUNICATION",
    "607": "TYPOGRAPHY &CA ENG",
    "608": "SHORTHAND ENGLISH",
    "609": "TYPOGRAPHY &CA HIN",
    "610": "SHORTHAND HINDI",
    "622": "ENGINEERING SCI.",
    "625": "APPLIED PHYSICS",
    "626": "MECH. ENGINEERING",
    "627": "AUTO ENGG. - II",
    "628": "AUTOSHOP RPR&PR-II",
    "632": "AC & REFRGTN-III",
    "633": "AC & REFRGTN-IV",
    "657": "BIO-OPTHALMIC - II",
    "658": "OPTICS-II",
    "659": "OPHTHALMIC TECH.",
    "660": "LAB MEDCN-II (MLT)",
    "661": "CLNCL BIOCHEM(MLT)",
    "662": "MICROBIOLOGY (MLT)",
    "666": "RADIATION PHYSICS",
    "667": "RADIOGRAPHY-I (GN)",
    "668": "RADIOGRAPHY-II(SP)",
    "728": "HLT ED,C & PR & PH",
    "729": "B CONCEPT OF HD&MT",
    "730": "FA & EMER MED CARE",
    "731": "CHILD HLTH NURSING",
    "732": "MIDWIFERY",
    "733": "HEALTH CENTRE MGMT",
    "734": "FOOD PROD-III",
    "735": "FOOD PRODUCTION-IV",
    "736": "FOOD SERVICES-II",
    "737": "FOOD BEV CST & CTR",
    "738": "EVOL&FORM OF MM-II",
    "739": "CR&CM PR IN MM-II",
    "740": "GEOSPATIAL TECH",
    "741": "LAB MEDICINE - II",
    "742": "CL BIOCHEM & MB-II",
    "743": "RETAIL OPER - II",
    "744": "RETAIL SERVICES-II",
    "745": "BEAUTY & HAIR - II",
    "746": "HOLISTIC HEALTH-II",
    "747": "LIB SYS & RES MGMT",
    "748": "INFO STORAGE & RET",
    "749": "INTGRTD TRANS OPRN",
    "750": "LOG OP & SCMGMT-II",
    "751": "BAKERY - II",
    "752": "CONFECTIONERY",
    "753": "FRONT OFFICE OPRNS",
    "754": "ADV FRONT OFF OPRN",
    "756": "INTRO TO HOSP MGMT",
    "757": "TR AGN & TOUR OP B",
    "762": "B HORTICULTURE -II",
    "763": "OLERICULTURE - II",
    "765": "FLORICULTURE",
    "766": "BUS.OP & ADMN - II",
    "774": "FABRIC STUDY",
    "775": "BASIC PATTERN DEV.",
    "776": "GARMENT CONST.-II",
    "777": "TRAD IND TEXTILE",
    "778": "PRINTED TEXTILE",
    "779": "TEXTILE CHEM PROC",
    "780": "FIN ACCOUNT - II",
    "781": "COST ACCOUNTING",
    "782": "TAXATION - II",
    "783": "MARKETING-II",
    "784": "SALESMANSHIP-II",
    "785": "BANKING-II",
    "786": "INSURANCE-II",
    "787": "ELECTRICAL MACHINE",
    "788": "ELECTRICAL APPLIAN",
    "789": "OP & MNT OF COM DV",
    "790": "TR SHOOTING & MEE",
    "793": "CAPITAL MKT OPERNS",
    "794": "DERIVATIVE MKT OPR",
    "795": "DATABASE MGMT APP",
    "796": "WEB APPLICATION-II",
}

with open(file, "r") as file_descp:
    file_data = file_descp.read()
    file_descp.close()

RESULT = {"STUDENTS": {}}


def isint(word):
    try:
        int(word)
        return True
    except:
        return False


def parse_line(main, marks):
    _parsed = {
        "name": None,
        "gender": None,
        "marks": {},
        "main_sub_grades": [],
        "result": None,
        "%age":"",
        "%age (Best 5)":"",
    }

    # Parse main line frist
    main_split = [_.strip() for _ in main.split(" ")]
    while "" in main_split:
        main_split.remove("")

    _subject_codes = []
    _current_name = None
    for word in main_split:
        if not _parsed["gender"]:
            if len(word) == 1:
                # Add extras per your need
                _parsed["gender"] = {"M": "Male", "F": "Female"}[word]
        # Parse name (kinda complex)
        elif not _parsed["name"] and not isint(word) and len(_subject_codes) == 0:
            if not _current_name:
                _current_name = word
            else:
                _parsed["name"] = _current_name + " " + word
        # Subjects code
        elif isint(word):
            _subject_codes.append(word)
        # Main subject grade
        elif word in ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2", "E"]:
            _parsed["main_sub_grades"].append(word)
        else:
            _parsed["result"] = word

    if not _parsed["name"]:
        _parsed["name"] = _current_name

    # Parse Marks line
    # Parse main line frist
    marks_split = [_.strip() for _ in marks.split(" ")]
    while "" in marks_split:
        marks_split.remove("")
    
    _marks_list = []
    _cur_subj_index = 0
    for mark in marks_split:
        if isint(mark):
            _marks_list.append(int(mark))
            _parsed["marks"][SUBJECT_CODES[_subject_codes[_cur_subj_index]]] = {
                "marks": int(mark),
                "grade": "",
            }
        else:
            _parsed["marks"][SUBJECT_CODES[_subject_codes[_cur_subj_index]]][
                "grade"
            ] = mark
            _cur_subj_index += 1
    
    # Process percentage
    _parsed["%age"] = "{:.2f}%".format(sum(_marks_list) / len(_marks_list))
    _marks_list.sort(reverse=True)
    _best_5 = _marks_list[:5]
    _parsed["%age (Best 5)"] = "{:.2f}%".format(sum(_best_5) / len(_best_5))
    return _parsed


MAIN_SPLIT = file_data.split("\n")

i = 0
while i + 1 != len(MAIN_SPLIT):
    try:
        roll_no = int(MAIN_SPLIT[i].split(" ")[0].strip())
        try:
            RESULT["STUDENTS"][roll_no] = parse_line(MAIN_SPLIT[i], MAIN_SPLIT[i + 1])
        except:
            print(traceback.format_exc())
        print()
        pprint(RESULT["STUDENTS"][roll_no])
        i += 2
    except Exception as e:
        #print(e)
        i += 1

print("TOTAL: ", len(list(RESULT["STUDENTS"].keys())))