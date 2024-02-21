#date: 2024-02-21T17:00:31Z
#url: https://api.github.com/gists/5b3f85b8194357731a01a3a6b1885945
#owner: https://api.github.com/users/mvandermeulen

from github import Github # https://github.com/PyGithub/PyGithub
import time

new_labels_to_add = [
    ("low", "78d60c"),
    ("medium", "f4df42"),
    ("high", "d12349")
]

g = "**********"

# Just pick the first org (need more? adapt!)
org = g.get_user().get_orgs()[0]
print("Fetching " + org.name)

for repo in org.get_repos():
    
    if repo.permissions.push is False:
        print("Ignoring not-allowed repo: %s (permissions.push= %s)" % (repo.name, str(repo.permissions.push)))
        continue

    print("For %s" % repo.name)
    for label in repo.get_labels():
        if label.url:
            print("Deleting: "+ label.url)
            label.delete()
        else:
            print("cannot delete: "+ label.name)
        time.sleep(0.5)

    for newlabel in new_labels_to_add:
        print("creating: %s" % str(newlabel))
        repo.create_label(newlabel[0], newlabel[1])
        time.sleep(0.5)
        time.sleep(0.5)
