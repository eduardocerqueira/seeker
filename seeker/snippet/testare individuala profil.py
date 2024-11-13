#date: 2024-11-13T17:10:49Z
#url: https://api.github.com/gists/a6da488dd250950423c7656da30ef2eb
#owner: https://api.github.com/users/me-suzy

import openai
import os
import json

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Cheia API OpenAI nu a fost setată în variabilele de mediu.")
openai.api_key = OPENAI_API_KEY

def test_analyze_character(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Analizează textul și extrage detalii despre personajul principal.
                    Format răspuns: {
                        "gender": "male/female",
                        "age_group": "young/middle-aged/elderly",
                        "main_activity": "ex: runner/athlete/etc",
                        "physical_description": "descriere concisă",
                        "key_characteristics": ["trăsături importante"]
                    }
                    Asigură-te că toate câmpurile sunt completate corect.
                    """
                },
                {
                    "role": "user",
                    "content": f"Analizează acest text și extrage profilul personajului principal: {text}"
                }
            ]
        )
        response_content = response.choices[0].message.content
        print(f"Răspuns API ChatCompletion: {response_content}")  # Debug print

        # Extrage JSON din răspuns
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("Răspunsul API nu conține un obiect JSON valid.")

        json_str = response_content[json_start:json_end]
        character_profile = json.loads(json_str)

        print("\nProfil personaj identificat:")
        print(f"Gen: {character_profile.get('gender', 'unknown')}")
        print(f"Grup de vârstă: {character_profile.get('age_group', 'unknown')}")
        print(f"Activitate principală: {character_profile.get('main_activity', 'unknown')}")
        print(f"Descriere fizică: {character_profile.get('physical_description', 'unknown')}")
        return character_profile
    except Exception as e:
        print(f"Eroare la analiza personajului: {str(e)}")
        return None

# Text de test
test_text = "Ion este un atlet de 35 de ani, care se antrenează intens pentru competițiile de maraton. Este înalt și musculos, cu păr scurt și ochi căprui."

profile = test_analyze_character(test_text)
