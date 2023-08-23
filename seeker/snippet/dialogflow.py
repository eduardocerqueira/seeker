#date: 2023-08-23T16:59:46Z
#url: https://api.github.com/gists/eb03071f5437b6bce8c95043381d7b53
#owner: https://api.github.com/users/anjalichimnanig

# Import the required Methods
import agent_operations

# Create Agent and specify the project, location, agent_name, language, and time zone parameters
agent_response = agent_operations.create_agent("ccai-platform-project", "global", "cai_agent_library", "en", "Asia/Calcutta")
print(agent_response)

# The system generated agent name is the name key in the agent response returned
agent_name = agent_response.name

# List all the agent in the specified project and location
agent_list = agent_operations.list_agents("ccai-platform-project", "global")
print(agent_list)

# Export the agent specified by its system generated agent name
agent_export_response = agent_operations.export_agent(agent_name)
print(agent_export_response)

# Delete the agent specified by its system generated agent name
agent_operations.delete_agent(agent_name)