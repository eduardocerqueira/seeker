#date: 2026-01-21T17:44:49Z
#url: https://api.github.com/gists/8da2c5bcc49290bd0929e4c9958426bd
#owner: https://api.github.com/users/pranshu-raj-211

def filter_konami(s:str)->dict[str,str]:
    len_seq=0
    symbols = ["","","","","",""]
    symbols_seen = set()
    # assuming a pattern like the one we want always exists
    # would have to add some length checks otherwise to prevent array out of bounds
    for i in range(len(s)):
        if len_seq==0:
            if s[i]==s[i+1]:
                symbols[0]=s[i]
                len_seq=2
                symbols_seen.add(s[i])
            else:
                continue
        elif len_seq==2:
            if s[i]not in symbols_seen and s[i]==s[i+1]:
                symbols[1]=s[i]
                len_seq=4
                symbols_seen.add(s[i])
            else:
                len_seq=0
                symbols[0]=""
                symbols_seen.clear() #TODO: check correct way of clearing a set
        elif len_seq==4:
            if s[i] not in symbols_seen and s[i+1] not in symbols_seen and s[i]==s[i+2] and s[i+1]==s[i+3]:
                symbols[2]=s[i]
                symbols[3]=s[i+1]
                len_seq=8
                symbols_seen.add(s[i])
                symbols_seen.add(s[i+1])
            else:
                len_seq=0
                symbols[0]=symbols[1]="" 
                symbols_seen.clear() 
        elif len_seq==8:
            if s[i] not in symbols_seen and s[i+1] not in symbols_seen:
                symbols[4]=s[i]
                symbols[5]=s[i+1]
                len_seq=10
                break
            else:
                len_seq=0
                symbols[0]=symbols[1]=symbols[2]=symbols[3]=""
                symbols_seen.clear() 
    if len_seq==10:
        return {symbols[0]:"U", symbols[1]:"D", :symbols[2]:"L", symbols[3]:"R", symbols[4]:"B", symbols[5]:"A"}
    return dict()
