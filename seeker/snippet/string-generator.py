#date: 2025-03-12T17:09:06Z
#url: https://api.github.com/gists/e640a6a797d5f5eeba0396799f7e8442
#owner: https://api.github.com/users/TechHoruser

#!/usr/bin/env python3
import argparse
import random
import string
from datetime import datetime, timedelta

def generate_dni():
  """Generate a random Spanish DNI (Documento Nacional de Identidad)"""
  # DNI consists of 8 digits and a letter
  numbers = ''.join(random.choices('0123456789', k=8))
  # Calculate the letter based on the DNI algorithm
  letter_index = int(numbers) % 23
  letters = "TRWAGMYFPDXBNJZSQVHLCKE"
  letter = letters[letter_index]
  return f"{numbers}{letter}"

def generate_nie():
  """Generate a random Spanish NIE (Número de Identidad de Extranjero)"""
  # NIE starts with X, Y, or Z, followed by 7 digits and a letter
  first_letter = random.choice(['X', 'Y', 'Z'])
  
  # Convert the first letter to its numeric value for the algorithm
  if first_letter == 'X':
    first_num = 0
  elif first_letter == 'Y':
    first_num = 1
  else:  # 'Z'
    first_num = 2
  
  numbers = ''.join(random.choices('0123456789', k=7))
  # Calculate the letter based on the NIE algorithm
  check_number = str(first_num) + numbers
  letter_index = int(check_number) % 23
  letters = "TRWAGMYFPDXBNJZSQVHLCKE"
  letter = letters[letter_index]
  return f"{first_letter}{numbers}{letter}"

def generate_passport():
  """Generate a random passport number"""
  # For simplicity, let's assume a Spanish passport
  # It typically consists of 3 letters followed by 6 digits
  letters = ''.join(random.choices(string.ascii_uppercase, k=3))
  numbers = ''.join(random.choices('0123456789', k=6))
  return f"{letters}{numbers}"

def generate_iban():
    """Generate a valid Spanish IBAN (International Bank Account Number)"""
    # Spanish IBAN structure: ES + 2 check digits + 20 digits (bank code + account number)
    # First, generate the basic account number components
    bank_code = ''.join(random.choices('0123456789', k=4))        # 4-digit bank code
    branch_code = ''.join(random.choices('0123456789', k=4))      # 4-digit branch code
    check_digits = ''.join(random.choices('0123456789', k=2))     # 2-digit account control
    account_number = ''.join(random.choices('0123456789', k=10))  # 10-digit account number
    
    # Combine to form BBAN (Basic Bank Account Number)
    bban = f"{bank_code}{branch_code}{check_digits}{account_number}"
    
    # Calculate IBAN check digits
    # 1. Move country code to the end and append '00'
    rearranged = f"{bban}ES00"
    # 2. Convert letters to digits (E=14, S=28)
    numeric_iban = ""
    for char in rearranged:
        if char.isalpha():
            # Convert A=10, B=11, ..., Z=35
            numeric_iban += str(ord(char) - ord('A') + 10)
        else:
            numeric_iban += char
    
    # 3. Calculate mod 97 and determine check digits
    check = 98 - (int(numeric_iban) % 97)
    iban_check = str(check).zfill(2)
    
    # Return the complete valid IBAN
    return f"ES{iban_check}{bban}"

def generate_credit_card():
  """Generate a random credit card number with expiration date and CVV"""
  # Generate a prefix based on card brand
  card_types = {
      'Visa': ['4'],
      'Mastercard': ['51', '52', '53', '54', '55'],
      'American Express': ['34', '37'],
      'Discover': ['6011', '644', '645', '646', '647', '648', '649', '65']
  }
  
  brand = random.choice(list(card_types.keys()))
  prefix = random.choice(card_types[brand])
  
  # Generate the remaining digits
  remaining_length = 16 - len(prefix)  # Most cards have 16 digits
  if brand == 'American Express':
      remaining_length = 15 - len(prefix)  # Amex has 15 digits
  
  digits = prefix + ''.join(random.choices('0123456789', k=remaining_length - 1))
  
  # Apply Luhn algorithm to generate the check digit
  total = 0
  for i, digit in enumerate(reversed(digits)):
      d = int(digit)
      if i % 2 == 1:  # odd position (from right)
          d *= 2
          if d > 9:
              d -= 9
      total += d
  
  check_digit = (10 - (total % 10)) % 10
  full_number = digits + str(check_digit)
  
  # Generate expiration date (1-5 years in the future)
  today = datetime.now()
  future_years = random.randint(1, 5)
  future_months = random.randint(0, 11)
  expiration_date = today + timedelta(days=(future_years * 365) + (future_months * 30))
  expiry = expiration_date.strftime("%m/%y")
  
  # Generate CVV (3 digits, 4 for Amex)
  cvv_length = 4 if brand == 'American Express' else 3
  cvv = ''.join(random.choices('0123456789', k=cvv_length))
  
  return {
      'number': full_number,
      'expiry': expiry,
      'cvv': cvv,
      'brand': brand
  }

def main():
  parser = argparse.ArgumentParser(description='Generate accreditation documents')
  parser.add_argument('--type', '-T', choices=['dni', 'nie', 'passport', 'iban', 'credit_card', 'full'], required=True,
            help='Type of document to generate (dni, nie, passport, iban, credit_card, full)')
  parser.add_argument('--number', '-N', type=int, default=1,
            help='Number of documents to generate (default: 1)')
  
  args = parser.parse_args()
  
  digits = len(str(args.number))

  for i in range(args.number):
    if args.type == 'dni':
      doc_id = generate_dni()
    elif args.type == 'nie':
      doc_id = generate_nie()
    elif args.type == 'passport':
      doc_id = generate_passport()
    elif args.type == 'iban':
      raw_iban = generate_iban()
      formatted_iban = f"{raw_iban[:4]} {raw_iban[4:8]} {raw_iban[8:12]} {raw_iban[12:14]} {raw_iban[14:]}"
      doc_id = formatted_iban
    elif args.type == 'credit_card':
      card_data = generate_credit_card()
      number = card_data['number']
      if card_data['brand'] == 'American Express':
          formatted_number = f"{number[:4]} {number[4:10]} {number[10:]}"
      else:
          formatted_number = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:16]}"
      
      doc_id = f"{formatted_number.ljust(19)} | Exp: {card_data['expiry']} | CVV: {card_data['cvv'].ljust(4)} | {card_data['brand']}"
    else:
      raise ValueError(f"Unknown document type: {args.type}")
      
    formatted_number = str(i+1).zfill(digits)
    print(f"Document {formatted_number}: {args.type.upper()} - {doc_id}")

if __name__ == "__main__":
  main()
