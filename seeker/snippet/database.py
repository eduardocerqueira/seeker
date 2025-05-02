#date: 2025-05-02T16:53:56Z
#url: https://api.github.com/gists/7164714cb67f99877b1c884ee0f7a13a
#owner: https://api.github.com/users/NontobekoNcube

import firebase_admin
from firebase_admin import credentials, firestore

#from database import db  # ✅ Import Firestore client-database.py already defines db after
#  Firestore is initialized, so there's no need to import it within the same file.

#Circular Import Issue: If database.py tries to import itself before it fully loads, Python gets confused.

# Load Firebase credentials (Use your correct path!)
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Initialize Firestore database
db = firestore.client()

def add_sample_mentors():
    mentors = [
        {"name": "Alice Johnson", "expertise": "Software Engineering"},
        {"name": "Michael Smith", "expertise": "Data Science"},
        {"name": "Jane Doe", "expertise": "Cybersecurity"}
    ]
    for mentor in mentors:
        db.collection("mentors").add(mentor)  # ✅ Adds mentors to Firestore
        print(f"✅ Mentor '{mentor['name']}' added!")

def add_job(title, company, required_skills):
    job_data = {
        "title": title,
        "company": company,
        "required_skills": required_skills
    }
    db.collection("jobs").add(job_data)
    print(f"✅ Job '{title}' added successfully!")

def add_mentor(name, expertise, available):
    mentor_data = {
        "name": name,
        "expertise": expertise,
        "available": available
    }
    db.collection("mentors").add(mentor_data)
    print(f"✅ Mentor '{name}' added successfully!")

#Fetch Job Listings
def get_jobs():
    jobs = db.collection("jobs").stream()
    job_list = [{"title": job.to_dict()["title"], "company": job.to_dict()["company"], "required_skills": job.to_dict()["required_skills"]} for job in jobs]
    return job_list

#Fetch Mentor Profiles
def get_mentors():
    mentors = db.collection("mentors").stream()
    mentor_list = [{"name": mentor.to_dict()["name"], "expertise": mentor.to_dict()["expertise"], "available": mentor.to_dict()["available"]} for mentor in mentors]
    return mentor_list

#store job applications
def track_application(user_id, job_title, status):
    application_data = {
        "user_id": user_id,
        "job_title": job_title,
        "status": status  # Example: "Applied", "Interviewing", "Offer Received"
    }
    db.collection("applications").add(application_data)
    print(f"✅ Job '{job_title}' application status updated: {status}")

#Enable company-driven career tracks
def add_career_track(company, role, steps):
    track_data = {
        "company": company,
        "role": role,
        "steps": steps  # Example: ["Complete coding challenge", "Join internship", "Apply for full-time role"]
    }
    db.collection("career_tracks").add(track_data)
    print(f"✅ Career track for '{role}' at '{company}' added.")


def match_mentor(user_interest):
    mentors = db.collection("mentors").where("expertise", "==", user_interest).stream()
    mentor_list = [{"name": mentor.to_dict()["name"], "expertise": mentor.to_dict()["expertise"]} for mentor in mentors]
    return mentor_list if mentor_list else [{"message": "No mentors found for this career path"}]





#Test_code
if __name__ == "__main__":
    add_job("Software Engineer", "TechCorp", ["Python", "Django", "SQL"])
    add_mentor("Nontobeko", "Career Coaching", True)
    add_sample_mentors() 
