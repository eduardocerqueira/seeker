#date: 2022-03-08T17:02:22Z
#url: https://api.github.com/gists/dbaf67f08216fe425af544ec9d7e93f7
#owner: https://api.github.com/users/J-hoplin1

import sys,os,json
from typing import MutableSequence
try:
    import click
except ModuleNotFoundError:
    print("\033[91m" + "Module 'click' not found. Please install 'click' with command 'pip3 install click' or 'python3 -m pip install click" + '\033[0m')
    sys.exit()

class TextColor:
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class Configuration(object):
    '''
    Class document : Configuration

    Declare class variable(static) for

    search_extension - Save extension list
    '''
    search_extension = ['.INFO','.ERROR','.FATAL','.WARNING']


def searching(directory : str,file_list : MutableSequence,json_former:dict):
    print(f"{TextColor.OKBLUE}Checking Directory : {directory}{TextColor.ENDC}")
    for i in file_list:
        checking = os.path.join(directory,i)
        if os.path.isdir(checking):
            searching(checking,os.listdir(checking),json_former)
        else:
            if os.path.splitext(checking)[-1] in Configuration.search_extension:
                json_former['total_found'] += 1
                try:
                    json_former[directory].append(i)
                except KeyError as e:
                    json_former[directory] = list()
                    json_former[directory].append(i)

@click.command()
@click.option('--dir',type=click.STRING,required=False,help="Directory Required : Use linux command 'pwd' value. If this flag is not expressed, executed directory will be set as default.",default=os.getcwd())
def main(dir):
    json_former = {
        "total_found" : 0,
    }
    searching(dir,os.listdir(dir),json_former)
    with open('search_result.json','w') as j:
        json.dump(json_former,j,indent=4)

if __name__ == "__main__":
    main()