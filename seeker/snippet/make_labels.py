#date: 2024-02-21T17:00:50Z
#url: https://api.github.com/gists/84abd10f74e69fe4d8685b71a86cbadd
#owner: https://api.github.com/users/mvandermeulen

# -*- coding: utf-8 -*-
"""
This script creates specific labels for all repositories of an organization.

before using it do a ``pip install PyGithub``.

"""
from github import Github
import argparse

LABELS = {
    'pr wip': '0052cc',
    'pr rebase': '207de5',
    'pr review': 'fbca04',
    'pr testing': 'eb6420',
    'pr ok': '009800',
    'pr orphaned': 'e11d21',
    'pr p4.3': 'f7f7f7',
    'pr p5.0': 'f7f7f7',
}

MIGRATE = {
    'pr p4.0': 'pr p5.0',
    'Question': 'question',
}

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--token',
    required=True,
    help= "**********"
argparser.add_argument(
    '--debug-limit',
    type=int,
    help='Limit the number of repos fetched, for debugging')


def make_labels():
    args = argparser.parse_args()
    gh = "**********"
    organization = gh.get_organization('plone')
    all_labels = set()
    for idx, repo in enumerate(organization.get_repos()):
        if args.debug_limit and idx+1 > args.debug_limit:
            break
        print 'repo #{0} {1} (limit at {2} of {3})'.format(
            idx+1,
            repo.name,
            gh.rate_limiting[0],
            gh.rate_limiting[1]
        )
        current_labels = [_ for _ in repo.get_labels()]
        current_label_names = [_.name for _ in current_labels]
        all_labels.update(current_label_names)
        for clabel in current_labels:
            # migrate name
            if clabel.name in MIGRATE:
                if MIGRATE[clabel.name] in current_label_names:
                    print '-> migration for {0}" target {1} exists, ' \
                        'delete!'.format(
                            clabel.name,
                            MIGRATE[clabel.name]
                        )
                    current_label_names.remove(clabel.name)
                    clabel.delete()
                else:
                    print "-> migrate {0} to {1}".format(
                        clabel.name,
                        MIGRATE[clabel.name]
                    )
                    current_label_names.append(MIGRATE[clabel.name])
                    all_labels.update([MIGRATE[clabel.name]])
                    clabel.edit(
                        MIGRATE[clabel.name],
                        LABELS.get(MIGRATE[clabel.name], clabel.color)
                    )
                    clabel.update()
                continue

            # adjust color
            if clabel.name in LABELS and clabel.color != LABELS[clabel.name]:
                print '-> update color of {0} to {1}'.format(
                    clabel.name,
                    LABELS[clabel.name]
                )
                clabel.edit(clabel.name, LABELS[clabel.name])

        for label_name, color in LABELS.items():
            if label_name in current_label_names:
                continue
            print '-> create label {0}'.format(label_name)
            repo.create_label(label_name, color)

    print "All Labels:"
    print ", ".join(sorted(all_labels))


if __name__ == '__main__':
    make_labels()
labels()
