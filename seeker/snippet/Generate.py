#date: 2025-04-04T16:55:49Z
#url: https://api.github.com/gists/fa41f6ee62093231453b01d6ece4afa7
#owner: https://api.github.com/users/Brodym340

import math # Import math module
import random # Import random so we can get random values
from datetime import datetime, timedelta # Import datetime to get dates

def name(coin, do_it): # We used a coin flip to decide male or female
    if coin == 1: # If the coin flip equals 1 then it will be a male
        male_name_list = ["Chris", "Samual", "Justin", "Derek", "Brody", "Joe", "Charlie", "Joey", "Ben", "Vince", "Liam",
            "Ethan", "Bill", "Allen", "Andrew", "Trystin", "Simon", "Toby", "Zach", "Jordan", "James", "Jackson",
            "Connor", "Noah", "Fred", "George", "Harry", "Issac", "Kyle", "Mason", "Nathan", "Oscar", "Patrick",
            "Josh", "Bradley", "Nick", "Xavier", "Shawn", "Saun", "Richard", "Wesley", "Kent", "Erick",
            "Micheal", "Roger", "Jacob", "Wiley", "Adam", "Dustin", "Henry", "Chuck", "Quinn", "Peter", "Dylan", "Lebron", "Tom", "Drake", "Jake"]
        first_name = random.choice(male_name_list)
    else: # If the coin is 2 then it will be a female
        female_name_list = ["Malia", "Kate", "Kara", "Ainsley", "Presley", "Jennifer", "Katie", "Lisa", "Debra", "Carly", "Megan", "Taylor", 
            "Madison", "Sarah", "Lindsey", "Lizzy", "Elizabeth", "Anna", "Sydney", "Abigal", "Scarlet", "Lillian", "Lilly", "Riley", 
            "Kacey", "Mia", "Gabriella", "Chloe", "Lilah", "Sherry", "Charolette", "Charol", "Ashley", "Kathy", "Violet", "Ava", 
            "Trinity", "Alison", "Oliva", "Emma", "Ella", "Marge", "Irene", "Sally", "Claire", "Sophia", "Sophie", "Claudia", 
            "Aniela", "Siena", "Sierra", "Geniveve", "Gertrude", "Alexa", "Callah", "Jenna", "Brandy", "Hazel", "Gianna", "Amelia", "Aubrey"]
        first_name = random.choice(female_name_list)

    middle_name = ""
    last_name = ""
            # Now we add a random middle name and last name to our person
    if do_it:
        middle_name_list = ["James", "Marie", "Ann", "Joseph", "Elizabeth", "Rose", "Louise", "Skyler", "Jaxon", "Everly", "Zane", "Aria", "Phoenix", "River", "Asher", 
            "Theodore", "Alexander", "Nichols", "Daniel", "Indigo", "Orion", "Echo", "Sable", "Finn", "Cassius", "Faith", "Gabriel", "Malachi", "Forrest", "Rain", 
            "Stone", "Rock", "Eleanor", "Arthur", "Benjamin", "Oliver", "Patrick", "Marvin", "Edward", "Robert", "Micheal", "Jace", "Paisley", "Julian", "Atlas", 
            "Cassian", "Cosette", "Jett", "Soleil", "Elijah", "Ezekiel", "Levi", "Ruth", "Abel", "Reed", "Juniper", "Guinevere", "Serenity", "Christopher", "Gregory"]
        middle_name = random.choice(middle_name_list)

        last_name_list = ["Carlin", "Williams", "Rogers", "Hoffmeier", "Hampleman", "Montgomery", "Bartowski", "Smith", "Keck", "Greenlaw", "Johnson", "Miller", 
            "McDougal", "Flower", "Hertz", "Cavin", "Hill", "Kellogg", "Keller", "Carr", "Hart", "McGregor", "Burrow", "Jefferson", "Kelce", "Brady", "Hufford", "Manning", 
            "Walters", "Pickens", "James", "Needham", "Calderon", "Caldes", "Novello", "Thompson", "Cook", "Baum", "Butler", "Penn", "McGinnis", "Jordan", "Fichser", "Desoto", 
            "Fraiser", "Berkman", "West", "Parker", "Kelly", "King", "Lavin", "Burns", "Buruse", "Ingro", "Martin", "Bishop"]
        last_name = random.choice(last_name_list)
# Print the generated first middle and last names
    print(first_name)
    if middle_name: print(middle_name)
    if last_name: print(last_name)
# Return the generated names
    return first_name, middle_name, last_name
# Function to generate a random age and date of birth
def random_age():
    age = random.randint(18, 80)
    today = datetime.today() # Gets todays date
    birth_year = today.year - age # Calculate the birth year based on the chosen age
    birth_date = datetime(birth_year, random.randint(1, 12), random.randint(1, 28)) # Generate a random birth date
    if (birth_date.month, birth_date.day) > (today.month, today.day): # If there birthday hasnt came up yet then we subtract 1 from there age
        age -= 1
    dob = birth_date.strftime("%B %d, %Y") # Format
    print(f"Age: {age}")
    print(f"Date of Birth: {dob}")
    return age
# Function to determine education but based off of their age
def education(age):
    popular_colleges = [
        "Harvard University", "Stanford University", "Massachusetts Institute of Technology", "University of California, Berkeley",
        "Princeton University", "Yale University", "Columbia University", "University of Chicago", "University of Pennsylvania",
        "California Institute of Technology", "Cornell University", "Duke University", "University of Michigan", "Johns Hopkins University",
        "Northwestern University", "University of California, Los Angeles", "University of Southern California", "New York University",
        "University of Texas at Austin", "University of Washington", "Georgia Institute of Technology", "University of Wisconsin–Madison",
        "University of North Carolina at Chapel Hill", "University of Florida", "Boston University", "University of Illinois Urbana-Champaign",
        "Penn State University", "University of Virginia", "Purdue University", "University of Minnesota", "Ohio State University",
        "Michigan State University", "Indiana University", "Texas A&M University", "University of Arizona", "University of Georgia",
        "University of Maryland", "University of Colorado Boulder", "University of Pittsburgh", "University of Miami",
        "University of Utah", "University of Oregon", "University of Iowa", "University of Kansas", "University of Oklahoma",
        "University of Missouri", "University of Kentucky", "University of Alabama", "University of Tennessee", "Arizona State University"
    ]

    if age < 18: # If there age is below 18 then they will currently be in high school
        print("Education: Currently in high school")
    elif age < 22: # If there age is lower than 22 they will just haev a high school diploma
        print("Education: High School Diploma")
    else: # If they are over 22 than we add the college they went to
        college = random.choice(popular_colleges)
        print(f"Education: Bachelor's degree from {college}")
# Function for there job
def job(age):
    jobs_list = [
        "Teacher", "Mechanic", "Librarian", "Accountant", "Barista", "Chef", "Delivery Driver", "Receptionist", "Security Guard", "Construction Worker",
        "Cashier", "Retail Worker", "Waiter", "Bartender", "Paramedic", "Nurse", "Doctor", "Dentist", "Electrician", "Plumber",
        "Janitor", "Bank Teller", "Bus Driver", "Taxi Driver", "Police Officer", "Firefighter", "IT Technician", "Web Developer", "Software Engineer", "Data Analyst",
        "Fitness Trainer", "Landscaper", "Warehouse Worker", "Flight Attendant", "Journalist", "Photographer", "Social Worker", "Pharmacist", "Veterinarian", "Mail Carrier"
    ]
    if age < 21: # Under 21 no job
        print("Occupation: None")
    elif age > 65: # Over 65 then they are retired
        print(f"Occupation: Retired from being a {random.choice(jobs_list)}")
    else: # Other ages just get a random occupation
        print(f"Occupation: {random.choice(jobs_list)}")
  # Random list of hobbies
def hobbies():
    hobbies_list = ["Fishing", "Hunting", "Cars", "Bowling", "Tennis", "Baseball Games", "Martial Arts", "Cooking", "Coding", "Drawing", "Video Games", "Painting", 
        "Gardening", "Lawn Care", "Football", "Soccer", "Fortnite", "Running", "Model Building", "Fly Fishing", "Hiking", "Photography", "Costmetics", "Sculpting", "Biking", 
        "Sewing", "Birdwatching", "Kyaking", "Boating", "Driving", "Racing", "Archery", "Camping", "Weight Lifting", "Badmiton", "Golf", "Singing", "Comedy", "Music", "Animation", "Politics",
        "Drone Flying", "Podcasts", "Yoga", "Calisthenics", "Body Building", "Virtual Reality", "Weaving", "Metal Working", "Social Media", "Baking", "Pottery", "Soap Making", "Linguistics"]
    chosen = random.sample(hobbies_list, 3)
    print(f"Hobbies: {', '.join(chosen)}")
 # Random list of locations with the area codes
def location():
    location_area_codes = {
        "Manhattan, New York, United States": "212", "Honolulu, Hawaii, United States": "808", 
        "Seattle, Washington, United States": "206", "Anchorage, Alaska, United States": "907", 
        "Portland, Oregon, United States": "503", "San Francisco, California, United States": "415",
        "Los Angeles, California, United States": "213", "Santa Cruz, California, United States": "831", 
        "San Diego, California, United States": "619", "San Jose, California, United States": "408", 
        "Sacramento, California, United States": "916", "Missoula, Montana, United States": "406",
        "Boise, Idaho, United States": "208", "Las Vegas, Nevada, United States": "702", 
        "Phoenix, Arizona, United States": "602", "Billings, Montana, United States": "406", 
        "Cheyenne, Wyoming, United States": "307", "Denver, Colorado, United States": "303",
        "Golden, Colorado, United States": "303", "Arvada, Colorado, United States": "720", 
        "Albuquerque, New Mexico, United States": "505", "Austin, Texas, United States": "512", 
        "Dallas, Texas, United States": "214", "Houston, Texas, United States": "713",
        "Bismarck, North Dakota, United States": "701", "Wichita, Kansas, United States": "316", 
        "Oklahoma City, Oklahoma, United States": "405", "Kansas City, Missouri, United States": "816", 
        "Des Moines, Iowa, United States": "515", "Conway, Arkansas, United States": "501",
        "New Orleans, Louisiana, United States": "504", "Minneapolis, Minnesota, United States": "612", 
        "Milwaukee, Wisconsin, United States": "414", "Saint Louis, Missouri, United States": "314", 
        "Chicago, Illinois, United States": "312", "Indianapolis, Indiana, United States": "317",
        "Detroit, Michigan, United States": "313", "Cincinnati, Ohio, United States": "513", 
        "Columbus, Ohio, United States": "614", "Memphis, Tennessee, United States": "901", 
        "Nashville, Tennessee, United States": "615", "Jackson, Mississippi, United States": "601",
        "Huntsville, Alabama, United States": "256", "Montgomery, Alabama, United States": "334", 
        "Pensacola, Florida, United States": "850", "Lexington, Kentucky, United States": "859", 
        "Tampa, Florida, United States": "813", "Miami, Florida, United States": "305",
        "Orlando, Florida, United States": "407", "Charleston, West Virginia, United States": "304", 
        "Charlotte, North Carolina, United States": "704", "Myrtle Beach, South Carolina, United States": "843", 
        "Charleston, South Carolina, United States": "843", "Augusta, Georgia, United States": "706",
        "Atlanta, Georgia, United States": "404", "Richmond, Virginia, United States": "804", 
        "Georgetown, Delaware, United States": "302", "Baltimore, Maryland, United States": "410", 
        "Pittsburgh, Pennsylvania, United States": "412", "Philadelphia, Pennsylvania, United States": "215",
        "Atlantic City, New Jersey, United States": "609", "Brooklyn, New York, United States": "718", 
        "Long Island, New York, United States": "516", "Hartford, Connecticut, United States": "860", 
        "Providence, Rhode Island, United States": "401", "Boston, Massachusetts, United States": "617",
        "Burlington, Vermont, United States": "802", "Concord, New Hampshire, United States": "603", 
        "Portland, Maine, United States": "207", "Buffalo, New York, United States": "716", 
        "Jacksonville, Florida, United States": "904", "Washington D.C., District of Columbia, United States": "202",
        "Savannah, Georgia, United States": "912", "Mobile, Alabama, United States": "251", 
        "Omaha, Nebraska, United States": "402", "Lincoln, Nebraska, United States": "402", 
        "Aberdeen, South Dakota, United States": "605"
    }
    place, code = random.choice(list(location_area_codes.items()))
    print(f"Location: {place} (Area Code: {code})")
    return code
    # Function for phone number
def phone_number(area_code):
    print(f"Phone Number: {area_code}-{random.randint(100,999)}-{random.randint(0, 9999):04d}")
    # Function for email
def email(first_name, last_name):
    print(f"Email: {first_name.lower()}{last_name.lower()}@example.com")
    # Function to put everything together in format to generate the person
def generate_person():
    coin = random.randint(1, 2)
    do_it = 1
    first_name, middle_name, last_name = name(coin, do_it)
    age = random_age()
    education(age)
    job(age)
    hobbies()
    area_code = location()
    phone_number(area_code)
    email(first_name, last_name)

# Generate the person
generate_person()
