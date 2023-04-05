#date: 2023-04-05T16:50:57Z
#url: https://api.github.com/gists/6b507913ce77360ae919650150645e01
#owner: https://api.github.com/users/Gennaro-Farina

"""
    This W.I.P. script convert the amazon massive dataset into a multijson BIO tagging for sequence/sentence NLP task.
    This script isn't complete yet, but It should already run.

    Link to the dataset: https://github.com/alexa/massive
        [MASSIVE](https://github.com/alexa/massive) [paper](https://arxiv.org/abs/2204.08582)


    Original Sequence tagging field is annot_utt: 
    citing the previously linked resource: the text from utt with slot annotations formatted as [{label} : {entity}]
    The output BIO-tagging formatted field is 'labels'.
    
"""
import os
import re
import json
import functools
import sys

FILE_PATH = '/link/to/folder/amazon-massive-dataset-1.0/1.0/data'
FILE_NAME = 'it-IT.jsonl' # for the italian file

OUTFILE_PATH = '/link/to/the/path'
OUT_FNAME = 'it-IT_converted.json' # choose an output name for your file

def search_for_entities(input_str: "**********":list= [], out_annot:list= []):
    """

    :param input_str: the input string in [{label} : {entity}] format
    :type input_str:
    : "**********": this is the input tokens array. You can pass an empty array if the resulting one should contain
    only the parsed input
    : "**********": list
    :param out_annot: this is the input annotation array. You can pass an empty array if the resulting one should contain
    only the parsed input
    :type out_annot: list
    :return:
            two lists with tokens and BIO tags
    :rtype: (list, list)
    """
    """
        This method search for annotated utterances with entities inside the AMAZON Massive dataset format
        The current regular expression is the following one:

    res = re.search(r'[\[+\S+]+[\s]*:[\s*[A-Za-z0-9]*\]+',
                     'portalo [time: questa mattina] cortesemente, capito [person: Giovanni]?')
    if len(res.regs) > 0:
        print('portalo [time: questa mattina] cortesemente, capito [person: Giovanni]?'[res.regs[0][0]:res.regs[0][1]])

        It should be tested with the following cases:

    # test set (to be integrated)
    # It shouldn't work with [person
    # It shouldn't work with []
    # It shouldn't work with person: Pippo]
    # It shouldn't work with [person trial: Pippo]
    # It shouldn't work with [person: ]
    # It should work with [entity_name: entity]
    # It should work with [entity_name: "**********"
    # It should work with [entity_name : "**********"
    # It should work with [entity_name : "**********"
    """

    curr_str = input_str

    if len(curr_str) < 1:
        return (out_tokens, out_annot)
    else:
        # res = re.search(r'[\[+\S+]+[\s]*:[\s*][\s*[A-Za-z0-9]*\]+', curr_str)
        res = re.search(r'[\[+\S+]+[\s]*:[\s*][\s*\w]*\]+', curr_str)


        if res is not None:
            if len(res.regs) > 0:
                patt = curr_str[res.regs[0][0]:res.regs[0][1]]
                parts = curr_str.split(patt)

                # taking the part before the match
                if len(parts)>0:
                    tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********"  "**********"> "**********"  "**********"0 "**********": "**********"
                            out_tokens.append(token)
                            out_annot.append('O')

                # managing the current match
                entity, tokens = patt.split(': "**********"

                    # the first character shuould always be '[', taking chars from the second one.
                    # the detected entity may span over multiple tokens, removing the initial whitespace and last ']' then split
                for idx_s, sub_t in enumerate(tokens.lstrip()[: "**********":
                    out_tokens.append(sub_t)
                    prefix = 'B-' if idx_s == 0 else 'I-'
                    out_annot.append(f'{prefix}{entity[1:]}')

                # managing the part after the match
                if len(parts) > 0:
                    new_string = parts[1]
                else:
                    new_string = ''

                return search_for_entities(new_string, out_tokens, out_annot)
        else:
            # assuming not annotated string, token are assumed to be tagged as 'O'
            if len(curr_str) > 0:
                tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********"  "**********"> "**********"  "**********"0 "**********": "**********"
                        out_tokens.append(token)
                        out_annot.append('O')
            return out_tokens, out_annot



def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    # https://stackoverflow.com/a/34482761/5404074
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
    
def convert_file(complete_input_filename, complete_output_filename, encoding= 'utf-8'):
    assert os.path.exists(complete_input_filename), f'File {complete_input_filename} doens\'t exists...'
    assert os.path.isfile(complete_input_filename), f'File {complete_input_filename} isn\'t a valid file...'

    # preparing the output file
    with open(complete_output_filename, 'w', encoding=encoding):
        pass

    # reading the input file
    with open(complete_input_filename, 'r', encoding=encoding) as f:
        # getting file lines as a list
        lines = f.readlines()

    # checking lines and make sure all data have same fields
    json_ref_structure = list(json.loads(lines[0]).keys())

    # avoiding the first element (that gives the reference file structure)
    for line in lines[1:]:
        json_curr_line_struc = list(json.loads(line).keys())
        assert functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, json_ref_structure, json_curr_line_struc), True), f"entry: {line} has no valid format..."

    # processing lines
    for i, line in zip(progressbar(lines, "Converting... : ", 40), lines):
        example = json.loads(line)

        # searching and converting entities into BIO tagging
        search_string = example['annot_utt']
        out_tokens, out_entities = "**********"
        (out_tokens, out_entities) = "**********"

        # create the single j-son line
        out_example = {}
        out_example['id'] = example['id']
        out_example['sentence'] = example['utt']
        out_example['category'] = example['intent']
        out_example['features'] = "**********"
        out_example['labels'] = out_entities

        # appending the just created line to the output file
        with open(complete_output_filename, 'a', encoding=encoding) as of:
            json.dump(out_example, of)
            of.write('\n')


if __name__ == '__main__':
    complete_input_name = os.path.join(FILE_PATH, FILE_NAME)
    complete_output_fname = os.path.join(OUTFILE_PATH, OUT_FNAME)
    convert_file(complete_input_name, complete_output_fname)

