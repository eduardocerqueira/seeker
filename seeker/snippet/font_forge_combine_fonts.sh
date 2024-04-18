#date: 2024-04-18T17:00:13Z
#url: https://api.github.com/gists/7e0d42b5a58c07c16884fad5ebf74498
#owner: https://api.github.com/users/anthonyheckmann

#!/usr/bin/env bash


# example of using fontforge from the CLI to merge fonts into a single TTC file
# also removes extra spaces from metadata and can remove a specific string from metadata

fontforge -lang=py -c "$(cat <<EOF
from os import remove
import sys
import glob
import re


fonts = glob.glob('*.ttf') + glob.glob('*.otf')
main_font = None
if len(fonts) > 2:
    print('Merging the following fonts:')
    for font in fonts:
        print(font)
else:
    sys.exit('Please provide at least 3 fonts to merge')

remove_this = re.compile(r'GARBAGE\s?DESCRIPTION', re.IGNORECASE)

def remove_extra_spaces(text):
    return ' '.join(text.split())

opened_fonts = []

font_families = {}
for font in fonts:
    my_font = fontforge.open(font)
    my_font_name = my_font.fontname
    my_family_name = my_font.familyname
    print(f'Processing {my_font_name} of family {my_family_name}')



    property_names = ['fontname', 'fullname', 'familyname']

    for property_name in property_names:
        value = getattr(my_font, property_name)
        if value and remove_this.search(value):
            new_value = remove_this.sub('', value)
            new_value = remove_extra_spaces(new_value)
            setattr(my_font, property_name, new_value)
    sfnt_string = my_font.sfnt_names
    new_sfnt_string = []

    for lang, strid, my_string in sfnt_string:
        if my_string and remove_this.search(my_string):
            my_string = remove_this.sub('', my_string)
            my_string = remove_extra_spaces(my_string)
        new_sfnt_string.append((lang, strid, my_string))

    my_font.sfnt_names = new_sfnt_string
    if not my_family_name in font_families.keys():
        font_families[my_family_name] = { 'main': my_font, 'others': [] }
    else:
         font_families[my_family_name]['others'].append(my_font)


#make family name into Camelcase

for family_name, fonts in font_families.items():
    main_font = fonts['main']
    opened_fonts = fonts['others']
    output_file_name = ''.join([word.capitalize() for word in main_font.familyname.split(' ')])
    main_font.generateTtc(f'{output_file_name}.ttc',opened_fonts)

EOF
)"
