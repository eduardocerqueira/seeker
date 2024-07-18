#date: 2024-07-18T16:43:42Z
#url: https://api.github.com/gists/3ac1d71ab06ef1ccad27d2275af9af02
#owner: https://api.github.com/users/me-suzy

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests
from PIL import Image
import io

 "**********"d "**********"e "**********"f "**********"  "**********"l "**********"o "**********"g "**********"i "**********"n "**********"_ "**********"t "**********"o "**********"_ "**********"c "**********"h "**********"a "**********"t "**********"g "**********"p "**********"t "**********"( "**********"d "**********"r "**********"i "**********"v "**********"e "**********"r "**********", "**********"  "**********"e "**********"m "**********"a "**********"i "**********"l "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    driver.get("https://chat.openai.com/auth/login")
    
    # Așteptăm și completăm emailul
    email_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "username"))
    )
    email_input.send_keys(email)
    email_input.send_keys(Keys.RETURN)
    
    # Așteptăm și completăm parola
    password_input = "**********"
        EC.presence_of_element_located((By.NAME, "password"))
    )
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)
    
    # Așteptăm să se încarce pagina principală
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[data-id='root']"))
    )

def send_prompt_and_save_image(driver, prompt, save_path):
    # Găsim și completăm câmpul de text
    text_area = driver.find_element(By.CSS_SELECTOR, "textarea[data-id='root']")
    text_area.send_keys(prompt)
    text_area.send_keys(Keys.RETURN)
    
    # Așteptăm generarea imaginii
    image_element = WebDriverWait(driver, 180).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "img[alt='Generated image']"))
    )
    
    # Salvăm imaginea
    image_url = image_element.get_attribute('src')
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))
    img.save(save_path)
    
    print(f"Imagine salvată: {save_path}")

def main():
    email = "your_email@example.com"
    password = "**********"
    prompts = [
        "O imagine abstractă reprezentând leadership",
        "Un peisaj futurist cu elemente de tehnologie avansată",
        "O metaforă vizuală pentru creștere personală și dezvoltare",
        # Adăugați mai multe prompt-uri aici
    ]
    
    driver = webdriver.Chrome()  # Asigurați-vă că aveți ChromeDriver instalat
    
    try:
        login_to_chatgpt(driver, email, password)
        
        for i, prompt in enumerate(prompts):
            save_path = f"imagine_generata_{i+1}.png"
            send_prompt_and_save_image(driver, prompt, save_path)
            
            if i < len(prompts) - 1:
                time.sleep(120)  # Așteptăm 2 minute între prompt-uri
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()