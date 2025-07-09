#date: 2025-07-09T16:48:16Z
#url: https://api.github.com/gists/2fcc5914b86e80506be4fbfcf2ffa4f1
#owner: https://api.github.com/users/djb4ai

from groq import Groq

# Initialize the client
client = Groq()

# Example 1: Code Generation with Prefill
print("=== Example 1: Code Generation ===")
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function to calculate the factorial of a number."
        },
        {
            "role": "assistant",
            "content": "```python"  # Prefill to start code block
        }
    ],
    stop="```",  # Stop at closing code block
    stream=True,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")

print("\n\n" + "="*50 + "\n")

# Example 2: JSON Data Extraction with Prefill
print("=== Example 2: JSON Data Extraction ===")
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": """Extract the name, age, and occupation from this text as JSON:
            
            "John Smith is a 28-year-old software engineer who works at a tech startup in San Francisco."
            """
        },
        {
            "role": "assistant",
            "content": "{\n"  # Prefill to start JSON with newline
        }
    ],
    stop="}",  # Stop at closing brace
    stream=True,
)

print("{", end="")  # Print the opening brace since we stopped before it
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
print("}")

print("\n\n" + "="*50 + "\n")

# Example 3: Structured Response without Prefill (for comparison)
print("=== Example 3: Same request WITHOUT prefill ===")
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Write a Python function to calculate the factorial of a number."
        }
    ],
    stream=True,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")

print("\n\n" + "="*50 + "\n")

# Example 4: XML Format with Prefill
print("=== Example 4: XML Format ===")
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Convert this information to XML: Product name is 'Laptop', price is $999, category is 'Electronics'"
        },
        {
            "role": "assistant",
            "content": "<product>\n"  # Prefill to start XML
        }
    ],
    stop="</product>",
    stream=True,
)

print("<product>", end="")
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
print("</product>")

print("\n\n" + "="*50 + "\n")

# Example 5: List Format with Prefill
print("=== Example 5: Numbered List ===")
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Give me 5 tips for better sleep"
        },
        {
            "role": "assistant",
            "content": "1. "  # Prefill to start numbered list
        }
    ],
    stream=True,
)

print("1. ", end="")
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")