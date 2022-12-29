#date: 2022-12-29T16:36:53Z
#url: https://api.github.com/gists/2e8a9a993e1b652919a34e7121356b2b
#owner: https://api.github.com/users/eddie-knight

import hazelcast
import openai

# Start the Hazelcast Client and connect to an already running Hazelcast Cluster on 127.0.0.1
client = hazelcast.HazelcastClient()
# Get a Distributed Map called "inputs"
inputs_map = client.get_map("inputs").blocking()

# Function to handle incoming user input
def handle_input(input_string):
    # Add the input to the map
    inputs_map.put(input_string)

# Set up a topic to listen for user input
input_topic = client.get_topic("user-inputs").blocking()
input_topic.add_listener(handle_input)

# Set up a topic to listen for ChatGPT responses
response_topic = client.get_topic("chatgpt-responses").blocking()

# Function to send a message to ChatGPT and get a response
def get_response(prompt):
    # Use the openai API to get a response from ChatGPT
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.5,
        max_tokens= "**********"
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    ).get("choices")[0]["text"]
    # Publish the response to the response topic
    response_topic.publish(response)

# Function to check for new input and send it to ChatGPT
def process_inputs():
    # Check for new input in the map
    input_strings = inputs_map.values()
    # If there is new input, send it to ChatGPT and get a response
    if input_strings:
        prompt = " ".join(input_strings)
        get_response(prompt)
        # Clear the map
        inputs_map.clear()

# Run the input processing loop indefinitely
while True:
    process_inputs()
