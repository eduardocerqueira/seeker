#date: 2024-09-25T16:45:00Z
#url: https://api.github.com/gists/9cd37fa34b1150e5c9d35a4b1cd87857
#owner: https://api.github.com/users/lovemycodesnippets

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_user_profile",
            "description": "Create a user profile with name, age, and email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "age", "email"]
            }
        }
    }
]