#date: 2025-04-04T16:57:45Z
#url: https://api.github.com/gists/3b7c1e4107eca98b0bc8f5aa968db876
#owner: https://api.github.com/users/Wileync

import random #imports random
from datetime import datetime #imports datetime to give us an accurate date to use for finding the persons age

coin = random.randint(1, 2) #acts as a coin being flipped to give us some diversity in people and their lives
college_coin = random.randint(1, 2) #acts as a coin flip to tell us whether the person went to college or not
attending_college_coin = random.randint(1, 2) #flips a coin to tell us if the person is attending college or just working after highschool
do_it = 1 #acts as a do it to fullfill some of the else statements in an understandable way


first_name = "" #creates a first name variable as an empty string to be filled
middle_name = "" #creates a middle name variable as an empty string to be filled
last_name = "" #creates a last name variable as an empty string to be filled
area_code = "" #creates a area code variable as an empty string to be filled
location_codes = "" #creates a location code variable as an empty string to be filled
email_addy = "" #creates a email address variable as an empty string to be filled
educational_experience = "" #creates a education experience variable as an empty string to be filled
young_work = "" #creates a young work variable as an empty string to be filled


def name(): #defines our name function to give us a random name
    if coin == 1: #if coin is 1 or heads, its a male
        male_name_list = ["Chris", "Samual", "Justin", "Derek", "Brody", "Joe", "Charlie", "Joey", "Ben", "Vince", "Liam", #male first name list
            "Ethan", "Bill", "Allen", "Andrew", "Trystin", "Simon", "Toby", "Zach", "Jordan", "James", "Jackson", #male first name list
            "Connor", "Noah", "Fred", "George", "Harry", "Issac", "Kyle", "Mason", "Nathan", "Oscar", "Patrick", #male first name list
            "Josh", "Bradley", "Nick", "Xavier", "Shawn", "Saun", "Richard", "Wesley", "Kent", "Erick", #male first name list
            "Micheal", "Roger", "Jacob", "Wiley", "Adam", "Dustin", "Henry", "Chuck", "Quinn", "Peter", "Dylan", "Lebron", "Tom", "Drake", "Jake"] #male first name list
        first_name = random.choice(male_name_list) #picks a random first name and assigns it to the first name variable
        print(f"First name: {first_name}") #prints the first name
    elif coin == 2: #if coin is 2 or tails, its a female
        female_name_list = ["Malia", "Kate", "Kara", "Ainsley", "Presley", "Jennifer", "Katie", "Lisa", "Debra", "Carly", "Megan", "Taylor", #female first name list
            "Madison", "Sarah", "Lindsey", "Lizzy", "Elizabeth", "Anna", "Sydney", "Abigal", "Scarlet", "Lillian", "Lilly", "Riley",  #female first name list
            "Kacey", "Mia", "Gabriella", "Chloe", "Lilah", "Sherry", "Charolette", "Charol", "Ashley", "Kathy", "Violet", "Ava",  #female first name list
            "Trinity", "Alison", "Oliva", "Emma", "Ella", "Marge", "Irene", "Sally", "Claire", "Sophia", "Sophie", "Claudia",  #female first name list
            "Aniela", "Siena", "Sierra", "Geniveve", "Gertrude", "Alexa", "Callah", "Jenna", "Brandy", "Hazel", "Gianna", "Amelia", "Aubrey"] #female first name list
        first_name = random.choice(female_name_list) #picks a random first name and assigns it to the first name variable
        print(f"First name: {first_name}") #prints first name
    if do_it == 1: #executes no matter what
        middle_name_list = ["James", "Marie", "Ann", "Joseph", "Elizabeth", "Rose", "Louise", "Skyler", "Jaxon", "Everly", "Zane", "Aria", "Phoenix", "River", "Asher", #middle name list
            "Theodore", "Alexander", "Nichols", "Daniel", "Indigo", "Orion", "Echo", "Sable", "Finn", "Cassius", "Faith", "Gabriel", "Malachi", "Forrest", "Rain",  #middle name list
            "Stone", "Rock", "Eleanor", "Arthur", "Benjamin", "Oliver", "Patrick", "Marvin", "Edward", "Robert", "Micheal", "Jace", "Paisley", "Julian", "Atlas",  #middle name list
            "Cassian", "Cosette", "Jett", "Soleil", "Elijah", "Ezekiel", "Levi", "Ruth", "Abel", "Reed", "Juniper", "Guinevere", "Serenity", "Christopher", "Gregory"] #middle name list
        middle_name = random.choice(middle_name_list) #picks a random middle name
        print(f"Middle name: {middle_name}") #prints the middle name
    if do_it == 1: #executes no matter what
        last_name_list = ["Carlin", "Williams", "Rogers", "Hoffmeier", "Hampleman", "Montgomery", "Bartowski", "Smith", "Keck", "Greenlaw", "Johnson", "Miller",  #last name list
            "McDougal", "Flower", "Hertz", "Cavin", "Hill", "Kellogg", "Keller", "Carr", "Hart", "McGregor", "Burrow", "Jefferson", "Kelce", "Brady", "Hufford", "Manning", #last name list
            "Walters", "Pickens", "James", "Needham", "Calderon", "Caldes", "Novello", "Thompson", "Cook", "Baum", "Butler", "Penn", "McGinnis", "Jordan", "Fichser", "Desoto", #last name list
            "Fraiser", "Berkman", "West", "Parker", "Kelly", "King", "Lavin", "Burns", "Buruse", "Ingro", "Martin", "Bishop"] #last name list
        last_name = random.choice(last_name_list) #picks a random last name
        print(f"Last name: {last_name}") #prints last name
    return(first_name, middle_name, last_name) #returns all three names to later generate an email


def random_age():  #defines age function
    age = random.randint(18, 80) #picks a random integer 18 - 80 

    today = datetime.today() #grabs todays date

    birth_year = today.year - age #Birth year is todays year minus the age
    birth_date = datetime(birth_year, random.randint(1, 12), random.randint(1, 28)) #Gets random day and month for the persons birthday

    if (birth_date.month, birth_date.day) > (today.month, today.day): #if the age is more because of the day being past todays date in their birth year, 
        age -= 1  #minus 1 year of age

    dob = birth_date.strftime("%B %d, %Y") #sets date of birth variable

    print(f"Age: {age}") #prints age is
    print(f"Date of Birth: {dob}") #prints the date of birth
    return(age) #returns age to be used for education and ocupation

def location(): #defines location function
    location_area_codes = {
        "Manhattan, New York, United States": "212", "Honolulu, Hawaii, United States": "808", #random places and their area code.
        "Seattle, Washington, United States": "206", "Anchorage, Alaska, United States": "907",  #random places and their area code.
        "Portland, Oregon, United States": "503", "San Francisco, California, United States": "415", #random places and their area code.
        "Los Angeles, California, United States": "213", "Santa Cruz, California, United States": "831",  #random places and their area code.
        "San Diego, California, United States": "619", "San Jose, California, United States": "408",  #random places and their area code.
        "Sacramento, California, United States": "916", "Missoula, Montana, United States": "406", #random places and their area code.
        "Boise, Idaho, United States": "208", "Las Vegas, Nevada, United States": "702",  #random places and their area code.
        "Phoenix, Arizona, United States": "602", "Billings, Montana, United States": "406",  #random places and their area code.
        "Cheyenne, Wyoming, United States": "307", "Denver, Colorado, United States": "303", #random places and their area code.
        "Golden, Colorado, United States": "303", "Arvada, Colorado, United States": "720",  #random places and their area code.
        "Albuquerque, New Mexico, United States": "505", "Austin, Texas, United States": "512",  #random places and their area code.
        "Dallas, Texas, United States": "214", "Houston, Texas, United States": "713", #random places and their area code.
        "Bismarck, North Dakota, United States": "701", "Wichita, Kansas, United States": "316",  #random places and their area code.
        "Oklahoma City, Oklahoma, United States": "405", "Kansas City, Missouri, United States": "816",  #random places and their area code.
        "Des Moines, Iowa, United States": "515", "Conway, Arkansas, United States": "501", #random places and their area code.
        "New Orleans, Louisiana, United States": "504", "Minneapolis, Minnesota, United States": "612",  #random places and their area code.
        "Milwaukee, Wisconsin, United States": "414", "Saint Louis, Missouri, United States": "314",  #random places and their area code.
        "Chicago, Illinois, United States": "312", "Indianapolis, Indiana, United States": "317", #random places and their area code.
        "Detroit, Michigan, United States": "313", "Cincinnati, Ohio, United States": "513",  #random places and their area code.
        "Columbus, Ohio, United States": "614", "Memphis, Tennessee, United States": "901",  #random places and their area code.
        "Nashville, Tennessee, United States": "615", "Jackson, Mississippi, United States": "601", #random places and their area code.
        "Huntsville, Alabama, United States": "256", "Montgomery, Alabama, United States": "334",  #random places and their area code.
        "Pensacola, Florida, United States": "850", "Lexington, Kentucky, United States": "859",  #random places and their area code.
        "Tampa, Florida, United States": "813", "Miami, Florida, United States": "305", #random places and their area code.
        "Orlando, Florida, United States": "407", "Charleston, West Virginia, United States": "304",  #random places and their area code.
        "Charlotte, North Carolina, United States": "704", "Myrtle Beach, South Carolina, United States": "843",  #random places and their area code.
        "Charleston, South Carolina, United States": "843", "Augusta, Georgia, United States": "706", #random places and their area code.
        "Atlanta, Georgia, United States": "404", "Richmond, Virginia, United States": "804",  #random places and their area code.
        "Georgetown, Delaware, United States": "302", "Baltimore, Maryland, United States": "410",  #random places and their area code.
        "Pittsburgh, Pennsylvania, United States": "412", "Philadelphia, Pennsylvania, United States": "215", #random places and their area code.
        "Atlantic City, New Jersey, United States": "609", "Brooklyn, New York, United States": "718",  #random places and their area code.
        "Long Island, New York, United States": "516", "Hartford, Connecticut, United States": "860",  #random places and their area code.
        "Providence, Rhode Island, United States": "401", "Boston, Massachusetts, United States": "617", #random places and their area code.
        "Burlington, Vermont, United States": "802", "Concord, New Hampshire, United States": "603",  #random places and their area code.
        "Portland, Maine, United States": "207", "Buffalo, New York, United States": "716",  #random places and their area code.
        "Jacksonville, Florida, United States": "904", "Washington D.C., District of Columbia, United States": "202", #random places and their area code.
        "Savannah, Georgia, United States": "912", "Mobile, Alabama, United States": "251",  #random places and their area code.
        "Omaha, Nebraska, United States": "402", "Lincoln, Nebraska, United States": "402",  #random places and their area code.
        "Aberdeen, South Dakota, United States": "605"
    }

     # Select a random location and its area code
    selected_location, area_code = random.choice(list(location_area_codes.items())) #selects a random location

    print(f"Location: {selected_location} (Area Code: {area_code})") #prints location and area code
    return area_code  #Return the area code for use in phone_number()


def phone_number(area_code): #defines phone number
    random_phone_number_3digit = random.randint(100, 999)  #picks three random digits for the phone number
    random_phone_number_4digit = f"{random.randint(0, 9999):04d}"  #picks four random digits for the phone number
    
    print(f"Phone Number: {area_code}-{random_phone_number_3digit}-{random_phone_number_4digit}") #prints the phone number with the correct area code.



def hobbies(): #defines hobbies
    hobbies_list = ["Fishing", "Hunting", "Cars", "Bowling", "Tennis", "Baseball Games", "Martial Arts", "Cooking", "Coding", "Drawing", "Video Games", "Painting", #random hobbies
        "Gardening", "Lawn Care", "Football", "Soccer", "Fortnite", "Running", "Model Building", "Fly Fishing", "Hiking", "Photography", "Costmetics", "Sculpting", "Biking",  #random hobbies
        "Sewing", "Birdwatching", "Kyaking", "Boating", "Driving", "Racing", "Archery", "Camping", "Weight Lifting", "Badmiton", "Golf", "Singing", "Comedy", "Music", "Animation", "Politics", #random hobbies
        "Drone Flying", "Podcasts", "Yoga", "Calisthenics", "Body Building", "Virtual Reality", "Weaving", "Metal Working", "Social Media", "Baking", "Pottery", "Soap Making", "Linguistics"] #random hobbies
    selected_hobbies = random.sample(hobbies_list, 3) #picks three random hobbies
    
    print(f"Hobbies: {selected_hobbies[0]}, {selected_hobbies[1]}, and {selected_hobbies[2]}") #prints the three hobbies


def email(first_name, middle_name, last_name): #defines email function
    email_addy = f"{first_name.lower()}{middle_name[0].lower()}{last_name.lower()}@gmail.com" #sets the email to the first name, middle initial, and last name, followed by @gmail.com
    print(f"Email: {email_addy}") #prints the email address



def education(age): #defines education function
    if age >= 24: #if age is greater than or equal to 24
        if college_coin == 1: #if college coin is 1, or head, they went to college
            college_education_list = ["Ohio State University (OSU)", "University of Colorado Boulder (CU)", "Auburn University (AU)", "University of Oregon (UO)", "University of Alabama (UA)", "University of Florida (UF)", "University of Georgia (UGA)", "University of Michigan (UM)",
                "Michigan State University (MSU)", "University of Texas at Austin (UT)", "Texas A&M University (A&M)", "University of California, Los Angeles (UCLA)", "University of California, Berkeley (UC Berkeley)", "Pennsylvania State University (Penn State)", "Louisiana State University (LSU)", "University of South Carolina",
                "Florida State University (FSU)", "University of Kentucky (UK)", "University of North Carolina at Chapel Hill (UNC)", "University of Virginia (UVA)", "University of Arizona (UA)", "Arizona State University (ASU)", "University of Washington (UW)", "University of Missouri (MU)",
                "University of Nebraska–Lincoln (UNL)", "University of Oklahoma (OU)", "University of Iowa (UI)", "Iowa State University (ISU)", "West Virginia University (WVU)", "Clemson University (Clemson)", "University of Mississippi (Ole Miss)", "Mississippi State University (MSU)",
                "University of Kansas (KU)"]
            educational_experience = random.choice(college_education_list) #picks a random college
            print(f"Attended: {educational_experience}") #prints that the person attended the college
        else:
            educational_experience = "High School Diploma" #otherwise, they have a high school diploma
            print(educational_experience) #prints that they have the diploma
    else:
        if attending_college_coin == 1: #if they are attending college, picks a college
            college_education_list = ["Ohio State University (OSU)", "University of Colorado Boulder (CU)", "Auburn University (AU)", "University of Oregon (UO)", "University of Alabama (UA)", "University of Florida (UF)", "University of Georgia (UGA)", "University of Michigan (UM)",
                "Michigan State University (MSU)", "University of Texas at Austin (UT)", "Texas A&M University (A&M)", "University of California, Los Angeles (UCLA)", "University of California, Berkeley (UC Berkeley)", "Pennsylvania State University (Penn State)", "Louisiana State University (LSU)", "University of South Carolina",
                "Florida State University (FSU)", "University of Kentucky (UK)", "University of North Carolina at Chapel Hill (UNC)", "University of Virginia (UVA)", "University of Arizona (UA)", "Arizona State University (ASU)", "University of Washington (UW)", "University of Missouri (MU)",
                "University of Nebraska–Lincoln (UNL)", "University of Oklahoma (OU)", "University of Iowa (UI)", "Iowa State University (ISU)", "West Virginia University (WVU)", "Clemson University (Clemson)", "University of Mississippi (Ole Miss)", "Mississippi State University (MSU)",
                "University of Kansas (KU)"]
            attending_college_this = random.choice(college_education_list) #picks the random college that they are attending
            print(f"Currently attending {attending_college_this}") #prints that they are attending that college
        else:
            
            print(f"Education: High School Diploma") #Prints their education as highschool diploma

def ocupation(age): #defines our occupation function
    if age >= 24: #if age is over 23
        if college_coin == 1: #and if they went to college
            college_occupation_list = ["Doctor", "IT Developer", "Actuary", "Mechanical Engineer", "Financial Analyst", #random college ocupation
                                        "Marketing Manager", "Civil Engineer", "Data Scientist", "Software Engineer", #random college ocupation
                                          "Architect", "Pharmacist", "Teacher", "Nurse Practitioner", "Accountant", #random college ocupation
                                            "Lawyer", "Biomedical Researcher", "Environmental Scientist", #random college ocupation
                                              "Human Resources Specialist", "Journalist", "Graphic Designer"] #random college ocupation

            college_work = random.choice(college_occupation_list) #picks a random college ocupation
            if age >= 62: #if age is over 61
                print(f"Retired {college_work}") #print retired from the ocupation
            else:
                print(f"Ocupation: {college_work}") #otherwise print their ocupation
        else:
            no_college_ocupation_list = ["Construction Site Manager", "Real Estate Agent", "Electrician", "Plumber", #random non college ocupation
                                          "HVAC Technician", "Carpenter", "Commercial Pilot", "Police Officer", #random non college ocupation
                                            "Firefighter", "Automotive Technician", "Welder", "Truck Driver", #random non college ocupation
                                              "Solar Panel Installer", "Insurance Sales Agent", "Massage Therapist", #random non college ocupation
                                                "Entrepreneur", "Freelance Photographer", "Personal Trainer", #random non college ocupation
                                                 "Private Investigator", "Aircraft Mechanic"] #random non college ocupation

            no_college_work = random.choice(no_college_ocupation_list) #picks a random no college ocupation
            if age >= 62: #if age is over 61
                print(f"Retired {no_college_work}") #prints retired from the job
            else:
                print(f"Ocupation: {no_college_work}") #otherwise print that job as their ocupation
    else:
        young_work_list = ["Bookstore", "Coffee shop", "City Art gallery", "Farmer's market", "Pet grooming salon", "Music store", "Toy shop", "Gym & fitness center", "Bakery", "Classic Antique shop", "Local Electronics repair shop", #not in college, young work list
                               "Plant nursery", "City Ice cream parlor", "Tattoo studio", "Bicycle repair shop", "Thrift store", "Photography studio", "Comic book store", "Dance studio", "Home decor boutique"] #not in college, young work list
        young_work = random.choice(young_work_list) #picks a job
        print(f"Ocupation: Currently working at a {young_work}") #prints that as their job


first_name, middle_name, last_name = name() #calls name function and takes out first, middle and last names
age = random_age() #calls random age function and takes out age
hobbies() #calls hobbies function
area_code = location()  #calls location function and takes out the area code
phone_number(area_code)  #calls the phone number function and brings in area code
email(first_name, middle_name, last_name) #calls email function and brings in first, middle and last names
education(age) #calls eductation function and brings in age
ocupation(age) #call ocupation function and brings in age
