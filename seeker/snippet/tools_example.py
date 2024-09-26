#date: 2024-09-26T17:13:40Z
#url: https://api.github.com/gists/1ce04d4169a638a4e267384100affad5
#owner: https://api.github.com/users/labeveryday

import boto3
import random
from datetime import datetime, timedelta, timezone


bedrock_client = boto3.client('bedrock-runtime')

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

def get_date_time():
    # Define the Eastern Time Zone offset
    # Note: This doesn't account for daylight saving time changes
    eastern_offset = timedelta(hours=-5)
    eastern_tz = timezone(eastern_offset, name="EST")

    # Get the current time and convert it to Eastern Time
    now = datetime.now(timezone.utc).astimezone(eastern_tz)

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "Eastern Time (EST)"
    }

def get_weather(location):
    weather_data = {
        "New York": {"temperature": 22, "condition": "Sunny"},
        "London": {"temperature": 15, "condition": "Cloudy"},
        "Tokyo": {"temperature": 28, "condition": "Rainy"}
    }
    return weather_data.get(location, {"temperature": 20, "condition": "Unknown"})

def get_historical_fact():
    facts =[
        "Jesse Owens was born on this day of September 12, in 1913 in Oakville, Alabama, U.S.—died March 31, 1980, Tucson, Arizona. He was an American track-and-field athlete who became one of the sport’s most legendary competitors after winning four gold medals at the 1936 Olympic Games in Berlin.",
        "On this day of Sept. 12th 1992, astronaut Mae Jemison became the first African American woman to fly in space, part of the STS-47 Spacelab J mission.",
        "On this day of Sept. 12th 1959 the TV series Bonanza premiered on NBC, and it became one of the longest-running westerns in broadcast history."
    ]
    return {"09-12-2024": random.choice(facts)}

def conversation(model_id, messages):
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=[{"text": """You are a helpful AI assistant named Claude, designed to provide accurate information about the current time and date, weather conditions, and historical facts. Your primary function is to assist users with queries related to these specific topics. Please adhere to the following guidelines:

        1. Tool Usage:
        - Only use the provided tools (get_date_time, get_weather, get_historical_fact) when directly asked about time, date, weather, or historical events.
        - Do not attempt to use these tools for unrelated queries.

        2. Response Scope:
        - Limit your responses to information that can be obtained through the available tools.
        - For questions about time and date, always use the get_date_time tool to ensure accuracy.
        - For weather inquiries, use the get_weather tool and ask for the location if not provided.
        - For historical facts, use the get_historical_fact tool, which provides information about events on the current date.

        3. Boundaries:
        - If asked about topics unrelated to time, date, weather, or historical facts, politely explain that you're specialized in these areas and cannot assist with other topics.
        - Do not make up information or use external knowledge beyond what the tools provide.

        4. Clarity and Conciseness:
        - Provide clear, concise answers that directly address the user's query.
        - If a user's question is ambiguous, ask for clarification before using any tools.

        5. Privacy and Security:
        - Do not ask for or store any personal information from users.
        - If a user provides personal information, do not repeat or acknowledge it.

        6. Ethical Considerations:
        - If asked to use the tools in a way that could be harmful or unethical, refuse and explain why.

        Remember, your purpose is to be a helpful, accurate, and responsible assistant focused on providing information about time, date, weather, and historical facts using the specified tools."""}],
        inferenceConfig={
            "temperature": 0.5
            },
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_date_time",
                        "description": "Get the current date and time in Eastern Time Zone.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }
                },
                {
                    "toolSpec": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string", "description": "The city to get weather for"}
                                },
                                "required": ["location"]
                            }
                        }
                    }
                },
                {
                    "toolSpec": {
                        "name": "get_historical_fact",
                        "description": "Get a historical fact that occurred on the current date",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    }
                }
            ],
            "toolChoice": {"auto": {}}
        }
    )
    return response['output']['message']


def process_response(ai_response, messages):
    for content in ai_response['content']:
        if 'text' in content:
            print(f"AI: {content['text']}")
        elif 'toolUse' in content:
            tool_use = content['toolUse']
            tool_name = tool_use['name']
            print(f"# Using the {tool_use['name']} tool...")
            
            tool_functions = {
                'get_date_time': get_date_time,
                'get_weather': lambda: get_weather(tool_use['input']['location']),
                'get_historical_fact': get_historical_fact
            }
            
            if tool_name in tool_functions:
                try:
                    tool_result = tool_functions[tool_name]()
                    messages.append({
                        "role": "user",
                        "content": [{
                            "toolResult": {
                                "toolUseId": tool_use['toolUseId'],
                                "content": [{"json": tool_result}],
                                "status": "success"
                            }
                        }]
                    })
                    result = conversation(model_id, messages)
                    messages.append(result)
                    print(f"AI: {result['content'][0]['text']}")
                    return messages
                except Exception as e:
                    print(f"Error using {tool_name} tool: {str(e)}")
                    return messages
    return messages


messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() == '/exit':
        break
    if user_input.lower() == '/history':
       print("Here is the message history:")
       for message in messages:
           print(message)
       user_input = input("You: ")
    
    messages.append({
        "role": "user",
        "content": [{"text": user_input}]
    })
    
    ai_response = conversation(model_id, messages)
    messages.append(ai_response)
    messages = process_response(ai_response, messages)