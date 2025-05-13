#date: 2025-05-13T17:10:32Z
#url: https://api.github.com/gists/2eaa0ec815b29ce83384f385402cb206
#owner: https://api.github.com/users/maietta

import os
import asyncio
import random
from dotenv import load_dotenv
from loguru import logger
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# Sample questions to ask
QUESTIONS = [
    "What's the most interesting fact about quantum physics?",
    "How do plants communicate with each other?",
    "What's the history behind the invention of the internet?",
    "How do birds navigate during migration?",
    "What makes a rainbow appear?",
    "How do submarines work?",
    "What's the science behind dreams?",
    "How do vaccines work?",
    "What causes the northern lights?",
    "How do bees make honey?"
]

async def main():
    # Load environment variables
    load_dotenv()
    
    # Debug: Print current working directory and check if .env exists
    current_dir = os.getcwd()
    env_path = os.path.join(current_dir, '.env')
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Looking for .env file at: {env_path}")
    logger.info(f".env file exists: {os.path.exists(env_path)}")
    
    # Get API key from environment variable
    api_key = os.getenv("OPENROUTER_API_KEY")
    logger.info(f"API key found: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    # Initialize LLM service
    llm = OpenRouterLLMService(
        api_key=api_key,
        model="qwen/qwen3-0.6b-04-28:free"  # Using GPT-4 for better responses
    )
    
    try:
        # Select a random question
        question = random.choice(QUESTIONS)
        print(f"\nAsking: {question}")
        
        # Create context and messages
        context = OpenAILLMContext()
        messages = [{"role": "user", "content": question}]
        
        # Get streaming response
        async for chunk in await llm.get_chat_completions(context, messages):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())