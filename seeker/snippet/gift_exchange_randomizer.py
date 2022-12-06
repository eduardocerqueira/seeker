#date: 2022-12-06T16:59:34Z
#url: https://api.github.com/gists/f8357a7e88b5ac1d286964b0485713f4
#owner: https://api.github.com/users/MBarrows20

example_input = {
    'familyA':[
        'A_person1',
        'A_person2',
        'A_person3',
        'A_person4',
        'A_person5'
    ],
    'familyB':[
        'B_person6',
        'B_person7'
    ],
    'familyC':[
        'C_person8',
        'C_person9',
        'C_person10',
        'C_person11'
    ],
    'familyD':[
        'D_person12'
    ]
}

# ^^^ Example input Family ^^^
# ---------------------------
# vvv Executable Code vvv

import random
import logging

def get_giftee_options(family_structure:dict,current_family:str,output:dict)->list:
    """Returns the giftees who are not in the same family that don't have an assigned gifter yet"""

    options = []
    for family, people in family_structure.items():
        if family is not current_family: 
            for person in people: 
                if person not in output.get('Giftees'):
                    options.append(person)
    return options

def assign_pairings(family_structure:dict) -> dict:
    """
    Select a random giftee for each gifter. Ensure the pair are not in the same family. 

    example: 
    >>> family_structure = {
        'familyA':[
            'A_p1',
            'A_p2',
        ],
        'familyB:[
            'B_p3',
            'B_p4'
        ]

    >>> assign_pairings(family_structure)
    {
        'Gifters': ['A_p1','A_p2','B_p3','B_p4'], 
        'Giftees':['B_p3','B_p4','A_p1','A_p2']
    }
    """
    output = {
        'Gifters':[],
        'Giftees':[]
    }
    for family,people in family_structure.items():
        for person in people:
            output.get('Gifters').append(person)
            try: 
                output.get('Giftees').append(random.choice(get_giftee_options(family_structure,family,output)))
            except IndexError:
                logging.warning('Could not generate even pairings. Re-running...')
    return output

def print_pairings(output:dict) -> None:
    """ 
    Prints pairings in a human-readable format

    example: 
    >>> print_pairings({'Gifters': ['P1','P2','P3'], 'Giftees':['P2','P3','P1']})
    P1 gives to P2
    P2 gives to P3
    P3 gives to P1
    """
    pairings = list()
    for i,x in enumerate(output.get('Gifters')):
        pairings.append((x,output.get('Giftees')[i]))
    print('\n🔀 Matches Generated! 🔀\n')
    for pairing in pairings:
        print(f'{" gives to ".join(pairing)}')
    print('\n🎁 Happy Gifting! 🎁')
    print('Tool developed with ❤ by https://github.com/MBarrows20')

if __name__ == '__main__': 
    output = assign_pairings(example_input)
    while len(output.get('Giftees')) != len(output.get('Gifters')):
        output = assign_pairings(example_input)
    print_pairings(output)