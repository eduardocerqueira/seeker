#date: 2024-09-11T17:03:33Z
#url: https://api.github.com/gists/c3928d07870d26665d3cbed167800b82
#owner: https://api.github.com/users/ohhh25

""" To download your own data, go to Instagram Settings > Meta Accounts Center > 
    Your information and permissions > Download your information. Choose 
    "Download or transfer information", "Some of your information", and search
    "Followers and following". Select the item under "Connections" and choose 
    "Download to device". Edit the Date range to be "All time", format to be 
    "JSON", and Media quality to be "High".
"""

import json

def read_json(given_file, following=False):
    with open(given_file, 'r') as f:
        data = json.load(f) if not following else json.load(f)['relationships_following']
    return data

def format(user_list):
    twib_list = []
    for user in user_list:
        data = user['string_list_data'][0]
        twib_list.append([data['value'], data['href']])
    return twib_list

def get_difference(list_to_check, dict_to_check_against):
    assert type(list_to_check) == list
    assert type(dict_to_check_against) == dict

    difference = []
    for item in list_to_check:
        if item[0] not in dict_to_check_against.keys():
            difference.append(item)

    return difference

def main():
    followers = format(read_json('followers_and_following/followers_1.json'))
    following = format(read_json('followers_and_following/following.json', following=True))

    n_followers, n_following  = len(followers), len(following)
    print(f"You have: {n_followers} followers and you are following {n_following} accounts")

    # Check with user:
    if input("If this is correct, enter 'y' to continue: ") != 'y':
        print("Exiting...")
        exit()

    followers_dict = {username: href for (username, href) in followers}
    following_dict = {username: href for (username, href) in following}

    # Sanity Check
    assert len(followers_dict.keys()) == n_followers
    assert len(following_dict.keys()) == n_following

    all_dict = {**followers_dict, **following_dict}
    print(f"Total of {len(all_dict.keys())} unique accounts")

    accounts_dont_follow_back = get_difference(following, followers_dict)
    print(f"\nThere are {len(accounts_dont_follow_back)} accounts you follow but they do not follow you back.")
    for account in accounts_dont_follow_back:
        print(account)

    accounts_you_dont_follow_back = get_difference(followers, following_dict)
    print(f"\nThere are {len(accounts_you_dont_follow_back)} accounts you do not follow but they follow you back.")
    for account in accounts_you_dont_follow_back:
        print(account)

if __name__ == '__main__':
    main()
