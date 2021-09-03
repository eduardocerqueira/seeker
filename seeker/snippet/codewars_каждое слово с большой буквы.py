#date: 2021-09-03T17:13:37Z
#url: https://api.github.com/gists/f82708142ebb84d0e3b6b76456fe6047
#owner: https://api.github.com/users/Maine558

def to_jaden_case(string):
    alf_s = ["q","w","e","r","t","y","u","i","o","p","a","s","d","f","g","h","j","k","l","z","x","c","v","b","n","m"]
    alf_b = ["Q","W","E","R","T","Y","U","I","O","P","A","S","D","F","G","H","J","K","L","Z","X","C","V","B","N","M"]
    string = string.split()
    new_string = ""
    for i in range(len(string)):
        q = str(string[i])

        for j in range(len(alf_s)):
            if q[0] == alf_s[j]:
                q = q.replace(q[0],alf_b[j],1)
        if i != len(string)-1:
            new_string += q + " "
    new_string += q
    return  new_string






print(to_jaden_case("Hekk a  sdij  jdsi k jdsi jsk jd"))


def s(q):
    q = (w.capita)