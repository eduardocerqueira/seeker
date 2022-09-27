#date: 2022-09-27T17:19:54Z
#url: https://api.github.com/gists/2687b9e33d068a716d010ce4d8cd0b26
#owner: https://api.github.com/users/dermitzos

#!/usr/bin/python3

import sys, getopt, csv

def main(argv):
    inputfile = ''
    outputfile = ''
    client = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:c:", ["ifile=", "ofile=", "client="])
    except getopt.GetoptError:
        print('transform.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('transform.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-c", "--client"):
            client = arg
    print('Input file is "'+inputfile+'"')
    print('Output file is "'+outputfile+'"')
    print('Client is "'+client+'"')

    with open(inputfile, newline='') as csvfile:
        inputreader = csv.DictReader(csvfile)
        with open(outputfile, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(['Project', 'Task', 'Client'])
            for row in inputreader:
                task = row['Issue key'] + ' - ' + row['Summary']
                project = row['Project name']
                writer.writerow([project, task, client])


if __name__ == "__main__":
    main(sys.argv[1:])