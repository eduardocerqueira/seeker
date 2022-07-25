#date: 2022-07-25T16:55:25Z
#url: https://api.github.com/gists/a0f326466694287202f94c650897bf7e
#owner: https://api.github.com/users/Wimukti

# Github related variables
GITHUB_ACCESS_TOKEN = os.getenv('Personal_Access_Token')
OWNER = "Your Corporation"
REPOSITORY = "Your Repository"
RELEASE_TAG = os.getenv("Release_Tag")
RELEASE_URL = "https://api.github.com/repos/{}/{}/releases/tags/{}".format(OWNER, REPOSITORY, RELEASE_TAG)

# Getting release information
release_response = requests.get(RELEASE_URL,headers={'Authorization': 'token %s' %GITHUB_ACCESS_TOKEN})

# Getting the release asset download URL
asset_download_url = release_response.json().get('assets')[0].get('url')
asset_name = release_response.json().get('assets')[0].get('name')

## Getting asset binary content
asset_response = requests.get(asset_download_url,headers={'Authorization': 'token %s' %GITHUB_ACCESS_TOKEN,'Accept': 'application/octet-stream'})

# Saving content to a file
def save_to_file(content,filename):
    open(filename,'wb').write(content)
    
if asset_binary.status_code == 200:            
  save_to_file(asset_binary.content, asset_name)            
  fileData = open(asset_name,'rb')