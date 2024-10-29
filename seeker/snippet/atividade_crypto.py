#date: 2024-10-29T17:01:04Z
#url: https://api.github.com/gists/d739fd7bf58f623517689933fe9228a0
#owner: https://api.github.com/users/jbohme

from cryptography.fernet import Fernet

# Função para gerar uma chave
def generate_key():
    return Fernet.generate_key()

# Função para criptografar uma mensagem
def encrypt_message(key, message):
    fernet = Fernet(key)
    encrypted_message = fernet.encrypt(message.encode())
    return encrypted_message

# Função para descriptografar uma mensagem
def decrypt_message(key, encrypted_message):
    fernet = Fernet(key)
    decrypted_message = fernet.decrypt(encrypted_message).decode()
    return decrypted_message

# Função principal
def main():
    # Gera uma chave
    key = generate_key()
    

    while True:
        print("\nEscolha uma ação:")
        print("1. Criptografar uma mensagem")
        print("2. Descriptografar uma mensagem")
        print("3. Sair")
        choice = input("Digite o número da opção: ")

        if choice == '1':
            message = input("Digite a mensagem a ser criptografada: ")
            encrypted_message = encrypt_message(key, message)
            print("Mensagem criptografada:", encrypted_message.decode())
            print("Chave usada para criptografar:", key.decode())  # Mostra a chave imediatamente após a criptografia
        elif choice == '2':
            encrypted_message = input("Digite a mensagem criptografada: ")
            encrypted_message_bytes = encrypted_message.encode()  # Converte de string para bytes
            
            # Solicita a chave para descriptografar
            key_input = input("Digite a chave usada para criptografar a mensagem: ").encode()
            try:
                decrypted_message = decrypt_message(key_input, encrypted_message_bytes)
                print("Mensagem descriptografada:", decrypted_message)
            except Exception as e:
                print("Erro na descriptografia:", e)
        elif choice == '3':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()