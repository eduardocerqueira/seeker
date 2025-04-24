#date: 2025-04-24T16:48:50Z
#url: https://api.github.com/gists/9efecf450535f17ccdbce7f942d18d99
#owner: https://api.github.com/users/lukelittle

import os
import logging
import json
from datetime import datetime
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Environment variables
WEBEX_BEARER_TOKEN = "**********"
HUBSPOT_PRIVATE_APP_TOKEN = "**********"

# API endpoints
HUBSPOT_API_BASE = 'https://api.hubapi.com'
HUBSPOT_SEARCH_ENDPOINT = f'{HUBSPOT_API_BASE}/crm/v3/objects/contacts/search'
HUBSPOT_CREATE_CONTACT_ENDPOINT = f'{HUBSPOT_API_BASE}/crm/v3/objects/contacts'
HUBSPOT_ENGAGEMENTS_ENDPOINT = f'{HUBSPOT_API_BASE}/crm/v3/objects/notes'

# Constants
DEFAULT_NAME = "Unknown Caller"

def search_contact_by_phone(phone_number):
    """
    Search for a contact in HubSpot by phone number
    
    Args:
        phone_number (str): Phone number to search for
        
    Returns:
        dict or None: Contact data if found, None otherwise
    """
    headers = {
        'Authorization': "**********"
        'Content-Type': 'application/json'
    }
    
    # Format the phone number to remove special characters for search
    formatted_phone = ''.join(filter(str.isdigit, phone_number))
    if formatted_phone.startswith('1') and len(formatted_phone) == 11:
        # Handle US numbers with country code
        formatted_phone = formatted_phone[1:]
    
    # Create the search payload
    payload = {
        "filterGroups": [{
            "filters": [{
                "propertyName": "phone",
                "operator": "**********"
                "value": formatted_phone
            }]
        }],
        "properties": ["firstname", "lastname", "phone", "email"],
        "limit": 1
    }
    
    try:
        logger.info(f"Searching for contact with phone: {phone_number}")
        response = requests.post(
            HUBSPOT_SEARCH_ENDPOINT,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        results = response.json()
        if results.get('total', 0) > 0:
            contact = results['results'][0]
            logger.info(f"Contact found: {contact['properties'].get('firstname', '')} {contact['properties'].get('lastname', '')}")
            return contact
        else:
            logger.info(f"No contact found for phone: {phone_number}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching for contact: {str(e)}")
        return None

def create_contact(phone_number):
    """
    Create a new contact in HubSpot
    
    Args:
        phone_number (str): Phone number for the new contact
        
    Returns:
        dict or None: Created contact data if successful, None otherwise
    """
    headers = {
        'Authorization': "**********"
        'Content-Type': 'application/json'
    }
    
    payload = {
        "properties": {
            "firstname": DEFAULT_NAME,
            "phone": phone_number
        }
    }
    
    try:
        logger.info(f"Creating new contact with phone: {phone_number}")
        response = requests.post(
            HUBSPOT_CREATE_CONTACT_ENDPOINT,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        contact = response.json()
        logger.info(f"Contact created with ID: {contact['id']}")
        return contact
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating contact: {str(e)}")
        return None

def log_call_activity(contact_id, call_data):
    """
    Log a call activity as a note for a contact in HubSpot
    
    Args:
        contact_id (str): HubSpot contact ID
        call_data (dict): Call data from Webex
        
    Returns:
        dict or None: Created note data if successful, None otherwise
    """
    headers = {
        'Authorization': "**********"
        'Content-Type': 'application/json'
    }
    
    # Extract call details
    call_direction = call_data.get('direction', 'Unknown')
    call_timestamp = call_data.get('timestamp', datetime.now().isoformat())
    call_duration = call_data.get('duration', 'N/A')
    
    # Format the note content
    note_content = (
        f"Call Type: {call_direction}\n"
        f"Timestamp: {call_timestamp}\n"
        f"Duration: {call_duration} seconds\n"
    )
    
    if call_data.get('status'):
        note_content += f"Status: {call_data.get('status')}\n"
        
    payload = {
        "properties": {
            "hs_note_body": note_content,
            "hs_timestamp": datetime.now().timestamp() * 1000,  # HubSpot uses milliseconds
            "hubspot_owner_id": "1",  # Default owner ID, update as needed
        },
        "associations": [
            {
                "to": {
                    "id": contact_id
                },
                "types": [
                    {
                        "associationCategory": "HUBSPOT_DEFINED",
                        "associationTypeId": 202  # Contact to Note association
                    }
                ]
            }
        ]
    }
    
    try:
        logger.info(f"Logging call activity for contact ID: {contact_id}")
        response = requests.post(
            HUBSPOT_ENGAGEMENTS_ENDPOINT,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        note = response.json()
        logger.info(f"Call activity logged with ID: {note['id']}")
        return note
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error logging call activity: {str(e)}")
        return None

def validate_webex_request(request):
    """
    Validate if the request is coming from Webex
    
    Args:
        request: Flask request object
        
    Returns:
        bool: True if valid, False otherwise
    """
    auth_header = request.headers.get('Authorization')
    
    # Simple validation - check if token matches
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"a "**********"u "**********"t "**********"h "**********"_ "**********"h "**********"e "**********"a "**********"d "**********"e "**********"r "**********"  "**********"a "**********"n "**********"d "**********"  "**********"a "**********"u "**********"t "**********"h "**********"_ "**********"h "**********"e "**********"a "**********"d "**********"e "**********"r "**********"  "**********"= "**********"= "**********"  "**********"f "**********"' "**********"B "**********"e "**********"a "**********"r "**********"e "**********"r "**********"  "**********"{ "**********"W "**********"E "**********"B "**********"E "**********"X "**********"_ "**********"B "**********"E "**********"A "**********"R "**********"E "**********"R "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"} "**********"' "**********": "**********"
        return True
    return False

def extract_phone_number(call_data):
    """
    Extract phone number from Webex Calling event data
    
    Args:
        call_data (dict): Call data from Webex
        
    Returns:
        str or None: Phone number if found, None otherwise
    """
    # The actual structure depends on Webex Calling webhook format
    # This is a simplified example - adjust based on actual payload structure
    
    if call_data.get('from') and call_data['from'].get('phoneNumber'):
        return call_data['from']['phoneNumber']
    
    if call_data.get('callerNumber'):
        return call_data['callerNumber']
    
    if call_data.get('origin') and call_data['origin'].get('address'):
        return call_data['origin']['address']
        
    return None

@app.route('/webhook/webex-calling', methods=['POST'])
def webex_calling_webhook():
    """
    Webhook endpoint for Webex Calling events
    """
    # Validate request
    if not validate_webex_request(request):
        logger.warning("Invalid authentication")
        return jsonify({"status": "error", "message": "Invalid authentication"}), 401
    
    # Parse the incoming JSON
    try:
        call_data = request.json
        logger.info(f"Received webhook: {json.dumps(call_data, indent=2)}")
        
        # Extract phone number from the call data
        phone_number = extract_phone_number(call_data)
        
        if not phone_number:
            logger.warning("No phone number found in call data")
            return jsonify({"status": "error", "message": "No phone number found"}), 400
        
        # Search for the contact in HubSpot
        contact = search_contact_by_phone(phone_number)
        
        # If contact doesn't exist, create one
        if not contact:
            contact = create_contact(phone_number)
            if not contact:
                return jsonify({"status": "error", "message": "Failed to create contact"}), 500
        
        # Log the call activity
        note = log_call_activity(contact['id'], call_data)
        if not note:
            return jsonify({"status": "error", "message": "Failed to log call activity"}), 500
        
        return jsonify({
            "status": "success",
            "message": "Call processed successfully",
            "contact_id": contact['id']
        })
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)