#date: 2024-05-03T17:05:47Z
#url: https://api.github.com/gists/d6b5c3b7302db1e0cacde981152a9160
#owner: https://api.github.com/users/nbonamy

import os
import requests
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.agents import initialize_agent, load_tools

os.environ['OPENAI_API_KEY']='YOUR_API_KEY'
os.environ['TAVILY_API_KEY']='YOUR_API_KEY'

# erase all files in the output directory
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)
for file in os.listdir(output_dir):
  file_path = os.path.join(output_dir, file)

# Get the prompt to use - you can modify this!
prompt = hub.pull('hwchase17/openai-tools-agent')
prompt.messages[0].prompt.template = """
You are a helpful assistant that help people creating static websites by generating HTML files and images.
You have access to a set of tools that allow you to interact with local files, query the Internet, Wikipedia, generate images with DALL-E and more.
When asked to create a website, you will search the Internet for the most relevant information instead of using your owm knowledge.
The websites you will create contains an index page and then a detailed page for each item of the index page.
The index page should have a title, a subtitle and a list of items with an image that points to the illustration that you created with the size of a small thumbnail. Clicking on a park should link to the detailed page.
You need to save locally all the HTML files (index page and detailed pages) and all images generated.
For images use the full URL provided by DALL-E including request parameters.
"""

write_file = FileManagementToolkit(
  root_dir='./output',
  selected_tools=['write_file'],
).get_tools()[0]

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

dalle = load_tools(['dalle-image-generator'])[0]

@tool
def write_image(url: str, filename: str):
  """Tool that saves an image to disk given a url and a filename"""
  response = requests.get(url)
  with open(f'{output_dir}/{filename}', 'wb') as file:
    file.write(response.content)
  return f'Image written successfully to {filename}'

tools = [TavilySearchResults(), wikipedia, write_image, write_file, dalle]

# Choose the LLM that will drive the agent
# Only certain models support this
model = ChatOpenAI(model='gpt-4-turbo', temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(model, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

park_count = 2

agent_executor.invoke({
  'input': f"""
    Create a website listing {park_count} US national parks randomly selected out of the list of all national parks in the US.
    Each national park should be accompanied by an illustration in the style of Rob Decker without any text..
    Before creating each detailed page, you will determine what information a visitor of the site will want to see about a national park (for example: location, area...).
    The HTML detailed page of each park should have a title with the name of the park, an image pointing to the image of the park that you created in reasonable size, an extensive text of 5 paragraphs about the park and the information you determined previously.
    Each page should have a color theme reminding of the park.
  """
})
