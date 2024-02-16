#date: 2024-02-16T16:48:30Z
#url: https://api.github.com/gists/fd2fe47d753e363c1de8ecc7bf029a69
#owner: https://api.github.com/users/SoulFireMage

#how to make a silly image like thing for use in datagrids like DevExpress. Friday snap special
from PIL import Image

image = Image.open('C:\Downloadstmp\LeftAktarian75 - Copy.bmp').convert('L')  # Convert to grayscale

image = image.resize((64, 64))

# Threshold value to distinguish between '1' and ' '
threshold = 128

ascii_art = []
for y in range(64):
    row = ''
    for x in range(64):
        pixel = image.getpixel((x, y))
        if pixel < threshold:
            row += '1'  # Dark pixel
        else:
            row += ' ' 
    ascii_art.append(row)

# Print the ASCII art
for row in ascii_art:
    print(row)

def generate_sql_from_ascii(ascii_art):
    sql_statements = []
    for row in ascii_art:
        row_query = "SELECT "
        for i, char in enumerate(row, start=1):
            if char == '1':
                row_query += "'1' AS col{}, ".format(i)
            else:
                row_query += "' ' AS col{}, ".format(i)
        row_query = row_query.rstrip(", ")
        sql_statements.append(row_query)

    combined_sql = " UNION ALL ".join(sql_statements)
    return combined_sql


# Generate SQL
sql_query = generate_sql_from_ascii(ascii_art)

print(sql_query)