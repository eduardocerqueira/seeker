#date: 2026-02-18T17:28:19Z
#url: https://api.github.com/gists/6fa6ebf88a8a402de7eeed450a2dfdf7
#owner: https://api.github.com/users/Thecookieinthejar

import streamlit as st 

# Alfabet
lower = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","æ","ø","å"]
upper = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","Æ","Ø","Å"]

# Gentag nøgle
def repeat_key(key, text):
    repeated = ""
    key_index = 0

    for char in text:
        repeated += key[key_index]
        key_index += 1

        if key_index == len(key):
            key_index = 0

    return repeated

# Kryptering
def encrypt(key, text):
    key = key.lower()
    repeated = repeat_key(key, text)
    result = ""

    for t, k in zip(text, repeated):

        was_upper = t in upper
        t_lower = t.lower()

        if t_lower not in lower:
            result += t
            continue

        t_num = lower.index(t_lower)
        k_num = lower.index(k)
        encrypted_num = (t_num + k_num) % len(lower)

        if was_upper:
            encrypted_letter = upper[encrypted_num]
        else:
            encrypted_letter = lower[encrypted_num]

        result += encrypted_letter

    return result

# Dekryptering
def decrypt(key, text):
    key = key.lower()
    repeated = repeat_key(key, text)
    result = ""

    for t, k in zip(text, repeated):

        was_upper = t in upper
        t_lower = t.lower()

        if t_lower not in lower:
            result += t
            continue

        t_num = lower.index(t_lower)
        k_num = lower.index(k)
        decrypted_num = (t_num - k_num) % len(lower)

        if was_upper:
            decrypted_letter = upper[decrypted_num]
        else:
            decrypted_letter = lower[decrypted_num]

        result += decrypted_letter

    return result

# Beregn nøgle
def calculated_key(text, encrypted_text):
    result = ""

    for t_char, e_char in zip(text, encrypted_text):

        t_lower = t_char.lower()
        e_lower = e_char.lower()

        if t_lower not in lower:
            result += t_char
            continue

        t_num = lower.index(t_lower)
        e_num = lower.index(e_lower)

        shift = (e_num - t_num) % len(lower)
        key_letter = lower[shift]

        result += key_letter

    return result


#layout with streamlit

st.title("Vigenere Kryptering og dekryptering.")

col1, col2 = st.columns([2, 2])

# VENSTRE KOLONNE – KRYPT / DEKRYPT
with col1:
    st.header("Kryptering / Dekryptering")

    text = st.text_area("Indtast tekst her:", key="text_encrypt")
    key = st.text_input("Indtast nøgle her:", key="key_encrypt")
    action = st.selectbox("Vælg handling:", ("Krypter", "Dekrypter"), key="action_encrypt")

    if st.button("Udfør", key="run_encrypt"):
        if action == "Krypter":
            encrypted_text = encrypt(key, text)
            st.write("Krypteret tekst:")
            st.code(encrypted_text)
        else:
            decrypted_text = decrypt(key, text)
            st.write("Dekrypteret tekst:")
            st.code(decrypted_text)

    st.caption('Prøv fx: "dog" + "taq", "cow" + "kav", "pig" + "bqzw"')
    st.caption('Tal og symboler ændres ikke i denne metode.')

# Find nøglen 
with col2:
    st.header("Prøv og find på en sjov nøgle :)")

    text2 = st.text_area("Original tekst:", key="text_key")
    encrypted_text2 = st.text_area("Krypteret tekst:", key="encrypted_key")

    if st.button("Find nøgle", key="run_key"):
        calculated = calculated_key(text2, encrypted_text2)
        st.write("Beregnet nøgle:")
        st.code(calculated)