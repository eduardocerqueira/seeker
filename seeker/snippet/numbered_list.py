#date: 2024-05-14T17:04:04Z
#url: https://api.github.com/gists/37f6ae47115cfddae94fd6422f12554b
#owner: https://api.github.com/users/willbakst

from mirascope.openai import OpenAICall
from pydantic import BaseModel, Field


class NumberedListOutput(OpenAICall):
    prompt_template = """
    SYSTEM:
    Your response should be a numbered list with each item on a new line.
    For example: \n\n1. foo\n\n2. bar\n\n3. baz

    USER:
    {query}
    """

    query: str
  
  
response = NumberedListOutput(query="...").call()
parsed_numbered_list = re.findall(r"\d+\.\s([^\n]+)", text)
print(parsed_numbered_list)


class NumberedList(BaseModel):
    ordered_items: list[str] = Field(..., description="An ordered list of items")

      
class ListOutputExtractor(OpenAIExtractor[NumberedList]):
    extract_schema: type[NumberedList] = NumberedList
    
    prompt_template = "{query}"
    
    query: str
    
output = ListOutputExtractor(query="...").extract()
print(output.ordered_items)