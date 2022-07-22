#date: 2022-07-22T16:50:21Z
#url: https://api.github.com/gists/8044dbfa2e9af4211e20d9b94121b084
#owner: https://api.github.com/users/DenverN3

# By Val Neekman (val@neekware.com)

import os, urllib2, sys, re, datetime, smtplib

## HOWTO:
# Change the following settings then run this script as "python loto_robo.py"

###### Settings ########
SEND_EMAIL = True
FROMADDR = "val@neekware.com"
TOADDRS = ["val@neekware.com"]
HOST = 'localhost'
###### Settings Ends ########


def between_i(string, start='', end=''):
    """
    Iteratively find all string wrapped by start and end within a string.
    Does not include start/end.
    
    @type string: str
    @param string: The string to search within.
    @type start: str
    @param start: The starting string.
    @type end: str
    @param end: The ending string.
    @rtype: list
    @return: A list of every occurence of start...end in the string.
    """
    # Used to store the results
    result = []
    
    # Iterate until there's no string found
    while True:
        # If no start is specified, start at the begining
        if not start:
            s = 0
        # Else find the first occurence of start
        else:
            s = string.find(start)
        # If end is empty, end at the ending
        if not end:
            e = len(string)
        # Else find the first occurence of end
        else:
            e = string.find(end)
        # Base case, if can't find one of the element, stop the iteration
        if s < 0 or e < 0:
            break
        # Append the result
        result.append(string[s+len(start):e])
        # Cut the string
        string = string[e+len(end):]
    return result

def get_htm_content():
    """ return the lotto site """
    url = "http://www.calottery.com/play/draw-games/mega-millions"
    try:
        resp = urllib2.urlopen(url)
    except urllib2.URLError, e:
        print >> sys.stderr, 'Failed to fetch (%s)' % url
    else:
        try:
            content = resp.read()
        except:
            content = []
    
    return content

def remove_extra_spaces(data):
    p = re.compile(r'\s+')
    return p.sub(' ', data)
    
    
    
if __name__ == "__main__":
    winning_numbers = []
    path =  os.path.dirname(os.path.abspath(__file__))
    print path
    html = get_htm_content()
    # print html
    clean_html = remove_extra_spaces(html)
    nums = between_i(clean_html, "winning_number_sm", "daily-derby-detail")
    if nums:
        nums = between_i(nums[0], "><span>", "</span>")
        if nums and len(nums) == 6:
            for num in nums:
                winning_numbers.append(num.replace('&nbsp;', ''))
    
    if winning_numbers:
        if SEND_EMAIL:
            msg = {}
            now = datetime.datetime.now().isoformat().split('T')[0]
            SUBJECT = 'Loto Robo: winning numbers for %s' % now
            BODY = 'The CalLotto winning numbers for %s are [%s]' % (now, '-'.join(winning_numbers))
            
            msg = ("From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (FROMADDR, ", ".join(TOADDRS), SUBJECT) )
            msg += "%s\r\n" % BODY
            
            try:
                server = smtplib.SMTP('localhost')
                server.sendmail(FROMADDR, TOADDRS, msg)
                server.quit()
            except:
                print >> sys.stderr, 'Failed to email out lotto winning numbers'
            else:
                print >> sys.stderr, msg

        else:
            print >> sys.stderr, winning_numbers