#date: 2023-04-17T16:44:19Z
#url: https://api.github.com/gists/a3bbec6c0b77af910b8f8089584ef123
#owner: https://api.github.com/users/andy0130tw

#! /usr/bin/python3

import aptsources.sourceslist as sl
import lsb_release

codename = lsb_release.get_distro_information()['CODENAME']
sources = sl.SourcesList()

for source in sources.list:
    if source.comment.lower().find("因升級至") >= 0:
        source.comment = ''
        source.set_enabled(True)
        print(source)
sources.save()
