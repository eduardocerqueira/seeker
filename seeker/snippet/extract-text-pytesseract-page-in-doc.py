#date: 2022-02-23T17:00:57Z
#url: https://api.github.com/gists/3b20557d532a3e5189bb4b68a63adc0c
#owner: https://api.github.com/users/ecdedios

doc = fitz.open(SCANNED_FILE)

print("Generated pages: ")
for page in doc:
    pix = page.get_pixmap(matrix=mat)
    png = '..\\data\\out\\input-' + SCANNED_FILE.split('\\')[-1].split('.')[0] + 'page-%i.png' % page.number
    print(png)
    pix.save(png)