#date: 2025-04-08T16:59:55Z
#url: https://api.github.com/gists/1984572bd48aeaf8254b9aff36972897
#owner: https://api.github.com/users/juandesant

def get_dimensions_from_image_path(path):
    import os, subprocess, plistlib #os for path expansion; subprocess for Popen; plistlib for parsing the plist format
    mdls_output = subprocess.Popen(
        # command line arguments, including "-p -" for stdout output
        ['/usr/bin/mdls', '-p', '-', os.path.expandvars(path).rstrip('\n')],
        stdout=subprocess.PIPE # pipe for stdout
    ).stdout.read().decode() # read bytes from stdout, and decode them as string
    mdls_dict = plistlib.loads(mdls_output) # load string as Plist
    return mdls_dict['kMDItemPixelWidth'], mdls_dict['kMDItemPixelHeight'] # keys for image width and height