#date: 2023-12-06T16:48:44Z
#url: https://api.github.com/gists/e3d4ae8f2f3ecbc94f95ef6e63dbefc3
#owner: https://api.github.com/users/dpieski

from sist2 import Sist2Index, Sist2Document, print_progress
import sys

print_progress(0,2)
print("Instantiating the Index...")
index = Sist2Index(sys.argv[1])

def upsertTags(doc: Sist2Document, tag: str | None = None, tags: list | None = None) -> Sist2Document:
    if tag and not tags:
        tags = [tag]
    if "tag" not in doc.json_data:
        doc.json_data["tag"] = [] 
    doc.json_data["tag"] = [
        *(t for t in doc.json_data["tag"] if t not in tags),
        *tags
    ]
    return doc

print("Iterating through the short-date documents...")
short_docs = 0
try:
    for doc in index.document_iter('mtime <= 631152000'):
        short_docs += 1
        try:
            if "tag" not in doc.json_data:
                print("Found a small-date! Document: '%s' path: '%s'" %( doc.json_data['name'], doc.json_data['path']))
            else:
                print("Found a small-date! Document: '%s' path: '%s', tags: '%s' " %( doc.json_data['name'], doc.json_data['path'], *doc.json_data['tag']))
            doc._replace(mtime=631152000)

            tag = "Misc.DATE-Short.#fc9797"
            doc = upsertTags(doc=doc, tag=tag)
    
            #print(doc.json_data["tag"])
            #print("Updating document....")
            index.update_document(doc)
        except Exception as error:
            print("[WARNING] Something went wrong with setting the mtime for a small doc!")
            print(error)
except Exception as error:
    print("[ERROR] Something went wrong with the small doc loop!")
    print(error)

print("Syncing the tag table....")
index.sync_tag_table()

print("Committing changes....")
index.commit()

print("-Finished Processing %d short-date documents." % short_docs)
print_progress(1,2)

long_docs = 0
print("Iterating through the long-date documents...")

try:
    for doc in index.document_iter('mtime >= 1735689600'):
        long_docs += 1
        try:
            if "tag" not in doc.json_data:
                print("Found a long-date! Document: '%s' path: '%s'" %( doc.json_data['name'], doc.json_data['path']))
            else:
                print("Found a long-date! Document: '%s' path: '%s', tags: '%s' " %( doc.json_data['name'], doc.json_data['path'], *doc.json_data['tag']))

            doc._replace(mtime=1735689600)

            tag="Misc.DATE-Long.#fc9797"
            doc = upsertTags(doc=doc, tag=tag)

            index.update_document(doc)
        except Exception as error:
            print("[WARNING] Something went wrong with setting the mtime for a long doc!")
            print(error)
except Exception as error:
    print("[ERROR] Something went wrong with the long doc loop!")
    print(error)

print("-Finished Processing %d long-date documents." % long_docs)
print_progress(2,2)


print("Syncing the tag table....")
index.sync_tag_table()

print("Committing changes....")
index.commit()

print("Done!")
