#date: 2023-06-19T17:02:08Z
#url: https://api.github.com/gists/09c78a7108cf8fbd85f4c437908fbde9
#owner: https://api.github.com/users/nirmalsnair

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:35:01 2023
@author: Nirmal

Find papers in a specific subject area (task) using paperswithcode API.
"""

from paperswithcode import PapersWithCodeClient


if __name__=="__main__":
    client = PapersWithCodeClient()

    # Subject area (task) that we want to search for papers.
    # IDs can be obtained from the paperswithcode URL.
    # e.g., https://paperswithcode.com/task/3d-reconstruction
    task_id = '3d-reconstruction'

    # PWC search API does not support the retrieval of all papers at once. It
    # returns search results as pages with a max limit of 500 items per page.
    # No need to change this value.
    max_items_per_page = 500

    # Full list of papers (in the form of multiple pages)
    pages = []

    # Grab the first page of search results
    # NB: Page number starts at 1, not 0
    i = 1
    print('\nRetrieving pages of \'{}\''.format(task_id))
    print('Fetching page {}...'.format(i), end='')
    papers = client.task_paper_list(task_id, items_per_page=max_items_per_page)
    pages.append(papers)
    n_papers = len(papers.results)
    print('contains {} papers'.format(n_papers))

    # Grab the remaining pages
    # NB: `.next_page` will be null if there are no more pages left
    while pages[-1].next_page:
        i = i + 1
        print('Fetching page {}...'.format(i), end='')
        papers = client.task_paper_list(task_id, page=i, items_per_page=max_items_per_page)
        pages.append(papers)
        n_papers = len(papers.results)
        print('contains {} papers'.format(n_papers))
    print('Retrieval done!')

    # Print total number of retrieved papers and pages
    n_papers = 0
    for p in pages:
        n_papers = n_papers + len(p.results)
    print('\nTotal pages retrieved  = {}'.format(len(pages)))
    print('Total papers retrieved = {}'.format(n_papers))

    # # Find which of the retrieved papers have code implementations
    # # NB: `.count` will show how many implementations each paper has
    # for i in range(len(pages)//2):
    #     for j in range(len(pages[i].results)//2):
    #         repo = client.paper_repository_list(pages[i].results[j].id)
    #         if repo.count:
    #             print(i, j, repo.count)
    #         else:
    #             print(i, j)
