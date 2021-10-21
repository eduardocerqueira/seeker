#date: 2021-10-21T17:11:25Z
#url: https://api.github.com/gists/253e9e87c4154362fc5a3817a4006924
#owner: https://api.github.com/users/kirilsSurovovs

# 1. uzdevums
def get_char_count(text):
    results_dict = {}
    for c in text:
        if c in results_dict.keys():
            results_dict[c] += 1
        else:
            results_dict[c] = 1
    return results_dict
print(get_char_count("hubba bubba"))

def get_digit_dict(num):
    text_num = str(num)
    results_dict = get_char_count(text_num)
    results_dict_new = {}
    for i in range(10):
        if str(i) in results_dict:
            results_dict_new[str(i)] = results_dict[str(i)]
        else:
            results_dict_new[str(i)] = 0
    return results_dict_new
print(get_digit_dict(599637003), "\n")


# 2. uzdevums
def replace_dict_value(d, bad_val, good_val):
    for k,v in d.items():
        if v == bad_val:
            d[k] = good_val
    return d
print(replace_dict_value({'a':5,'b':6,'c':5}, 5, 10), "\n")


# 3. uzdevums
def clean_dict_value(d, bad_val):
    d_copy = d.copy()
    for k,v in d_copy.items():
        if v == bad_val:
            if k in d.keys():
                del d[k]
    return d
print(clean_dict_value({'a':5,'b':6,'c':5}, 5))

def clean_dict_values(d, v_list):
    d_copy = d.copy()
    for k,v in d_copy.items():
        if v in v_list:
            if k in d.keys():
                del d[k]
    return d
print(clean_dict_values({'a':5,'b':6,'c':5}, [3,4,5]))