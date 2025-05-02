#date: 2025-05-02T16:56:29Z
#url: https://api.github.com/gists/c460ea5a962fec465628c750c7e27bbe
#owner: https://api.github.com/users/NontobekoNcube

# Import Flask & Firebase libraries  
from flask import Flask, jsonify, request 
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS


from database import get_jobs, get_mentors
from matching import match_jobs, match_mentors, analyze_applications
from database import track_application
from database import db  # ✅ Import Firestore database instance
from matching import match_mentor  # ✅ Import mentor matching function


# Initialize the Flask application  
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])  # Enable CORS for all routes

# Define an API route called "/career-match"  and "user profiles"
@app.route("/career-match", methods=["GET", "POST"])
def match():
    return jsonify({"message": "Career matching in progress!"})

# Define an API route for adding user profiles to Firebase
@app.route("/add-user", methods=["GET","POST"])
@app.route("/add-user", methods=["POST"])
def add_user():
    data = request.get_json()
    add_user_profile(data)  # Call function from database.py
    return jsonify({"message": "User added successfully!"})

def add_user_profile(user_data):
    db.collection("users").add(user_data)
    print(f"✅ User '{user_data['name']}' added successfully!")

@app.route("/get-user-profile", methods=["GET"])
def get_user_profile():
    # Get the user ID from the request query parameters
    user_id = request.args.get("user_id")

    # Fetch the user document from Firestore using the provided user ID
    user_ref = db.collection("users").document(user_id).get()

    # Check if the user exists in the database
    if user_ref.exists:
        user_data = user_ref.to_dict()  # Convert Firestore document to dictionary
        return jsonify(user_data)  # Send user profile data as a JSON response
    else:
        # If user does not exist, return an error message with status code 404
        return jsonify({"error": "User not found"}), 404



##API Endpoint to Fetch Jobs
@app.route("/jobs", methods=["GET"])
def get_all_jobs():
    jobs = get_jobs()
    return jsonify(jobs)

##API Endpoint to Fetch Mentors
@app.route("/mentors", methods=["GET"])
def get_all_mentors():
    mentors = get_mentors()
    return jsonify(mentors)

# API Endpoint for Career Matching
@app.route("/career-match", methods=["POST"])
def career_match():
    data = request.json
    user_skills = data.get("skills", [])
    jobs = get_jobs()
    mentors = get_mentors()
    
    matched_jobs = match_jobs(user_skills, jobs)
    matched_mentors = match_mentors(user_skills, mentors)
    
    return jsonify({"jobs": matched_jobs, "mentors": matched_mentors})

#API Endpoint to add job applications will store jobs and status for users allowing tracking progress through stages
@app.route("/track-application", methods=["POST"])
def track_application_api():
    data = request.json
    track_application(data["user_id"], data["job_title"], data["status"])
    return jsonify({"message": "Application tracked successfully!"})

#API Endpoint to provide feedback Identifies why a user isn’t getting interviews 
# Provides AI-driven feedback to improve applications
@app.route("/application-feedback", methods=["GET"])
def application_feedback():
    user_id = request.args.get("user_id")
    feedback = analyze_applications(user_id)
    return jsonify({"feedback": feedback})

# API Endpoint to Track Application Progress & Provide AI Insights
@app.route("/get-application-status", methods=["GET"])
def get_application_status():
    user_id = request.args.get("user_id")
    applications = db.collection("applications").where("user_id", "==", user_id).stream()

    insights = []
    statuses = []

    for app in applications:
        data = app.to_dict()
        statuses.append(data["status"])
        insights.append(f"📌 {data['job_title']} → Status: {data['status']}")

    # AI Analysis: Identify Bottlenecks
    if "Applied" in statuses and "Interviewing" not in statuses:
        insights.append("❌ You’ve applied but haven’t received interviews yet. Consider refining your resume or applying to more relevant roles.")
    elif "Interviewing" in statuses and "Offer Received" not in statuses:
        insights.append("🎤 You’re interviewing but haven’t received an offer yet. Practice mock interviews and refine your negotiation skills.")

    # Mentor Recommendation
    if "Applied" in statuses and "Interviewing" not in statuses:
        insights.append("👥 Suggested Mentor: Resume Writing Specialist")
    elif "Interviewing" in statuses and "Offer Received" not in statuses:
        insights.append("👥 Suggested Mentor: Interview Coach")

    return jsonify(insights)


#Allows companies to create career pathways 
#  Guides users on structured career progression
@app.route("/career-tracks", methods=["GET"])
def get_career_tracks():
    tracks = db.collection("career_tracks").stream()
    track_list = [{"company": track.to_dict()["company"], "role": track.to_dict()["role"], "steps": track.to_dict()["steps"]} for track in tracks]
    return jsonify(track_list)

@app.route("/get-mentor", methods=["GET"])
def get_mentor_api():
    career_interest = request.args.get("career_interest")
    mentors = match_mentor(career_interest)
    return jsonify(mentors)




# Run the Flask app  
if __name__ == "__main__":
    app.run(debug=True, port=5001)
