#date: 2023-07-06T17:00:24Z
#url: https://api.github.com/gists/ee669194f2e8cdf15c6a0ef8318ddc65
#owner: https://api.github.com/users/sln-dns

import asyncio
from aiohttp import ClientSession
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import tiktoken 

app = FastAPI()

openai_api_url = "https://api.openai.com/v1/engines/davinci-codex/completions"
openai_api_key = "OPENAI_API_KEY"

# Определение модели данных AnswerToApp 
# Здесь определяется формат выходных данных, какой именно json будет на выходе

class AnswerToApp(BaseModel):
    question: str = Field(description="question")
    answer: str = Field(description="answer")

# Инициализация объектов OpenAI и PromptTemplate

model_name = "text-davinci-003"
temperature = 0.0 # не очень понятно какое именно приложение будет отдавать нам вопрос, 
                    #температуру можно и нужно регулировать в соответствии с этим.

model = OpenAI(model_name=model_name, temperature=temperature)

#определяем обработку входных данных
# можем задать роль, либо добавить к вопросу пояснения

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n", #непосредственно препромт
    input_variables=["query"],
    partial_variables={
        "format_instructions": "Answer to this question with details." # также место для добавления инструкций
    },
)

# не смотря на отсутствие явных и заявленных в документации ограничений на использование api openai
# выполнил ограничение на количество одновременных запросов и времени между запросами

semaphore = asyncio.Semaphore(5)  # Ограничение на количество одновременных запросов
request_delay = 0.5  # Задержка между запросами в секундах

async def make_request(url, headers, data):
    async with semaphore:
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                return await response.json()

#определяем эндпойнт который обрабатывает POST-запросы.
@app.post("/generate-text") #Он принимает текстовый параметр text, который представляет собой входной текст для генерации.
async def generate_text(text: str): 
    query = text

# Подсчет количества токенов в запросе
    
    token_count = "**********"

    # проверка на длину запроса, зависит от модели и от наших желаний тратить токены и деньги.
    if token_count > 4000: "**********"
        raise HTTPException(status_code= "**********"="Too many tokens in the request")


    _input = prompt.format_prompt(query=query)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "prompt": _input.to_string(),
        "max_tokens": "**********"
    }

# не смотря на то, что проверку на ошибки выполняет PydanticOutputParser
# в случае необходимости дозапрашивая апи, здесь всё равно организована
# простая обработка 500 ошибки, лишь для демонстрации возможности  
    try:
        response = await make_request(openai_api_url, headers, data)
        parser = PydanticOutputParser(pydantic_object=AnswerToApp)
        parsed_output = parser.parse(response)
        return parsed_output.dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate text") from erequest(openai_api_url, headers, data)
        parser = PydanticOutputParser(pydantic_object=AnswerToApp)
        parsed_output = parser.parse(response)
        return parsed_output.dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate text") from e