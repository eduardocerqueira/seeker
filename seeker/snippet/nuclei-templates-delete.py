#date: 2022-07-05T16:45:52Z
#url: https://api.github.com/gists/f932b53a0e8468b3c911be379fee47ec
#owner: https://api.github.com/users/m4ll0k


import os,sys,yaml,argparse

# example
# python nuclei-templates-delete.py -p <nuclei-templates-dir> -d <id,id1,id2> or <ids.txt file>


ids = []
nuclei_template_path = ''

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',help="nuclei tempaltes path",action="store",default="")
parser.add_argument('-d','--ids',help="list of ids or ids file",action="store",default="")
args = parser.parse_args()

if args.path == '' or os.path.exists(args.path) is False:
    sys.exit(print('empty path arg or dir doesn\'t exist!'))
else:
    nuclei_template_path = args.path

if args.ids == '':
    sys.exit(print('ids arg required! e.g: id,id2,.. or ids.txt file'))

if ',' in args.ids:
    ids = args.ids.split(',')
elif '.txt' in args.ids:
    ids = [x.strip()for x in open(args.ids)]
else:
    ids = [args.ids]

def get_all_template_paths()->list:
    templates = []
    for root, dirs, files in os.walk(nuclei_template_path):
        for file in files:
            if file.endswith(".yaml"):
                path = os.path.join(root, file)
                if path not in templates:
                    templates.append(path)
    return templates

def remove_template(path:str)->None:
    os.remove(path)

def main():
    templates = get_all_template_paths()
    for _id in ids:
        for template in templates:
            try:
                content = yaml.safe_load(open(template)) or {}
            except Exception as err:
                content = {}
            if content.get('id') == _id:
                remove_template(template)

if __name__ == '__main__':
    main()