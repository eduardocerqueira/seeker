#date: 2024-02-21T17:00:34Z
#url: https://api.github.com/gists/128d02e499a1b6f8d35f57ed69628fc7
#owner: https://api.github.com/users/mvandermeulen

# Define search function because pygithub does not support /search/commits API yet.
import github
def __Github_search_commits(self, word, sort=github.GithubObject.NotSet, order=github.GithubObject.NotSet, **kwargs):
    """
    :calls: `GET /search/commit`
    :param str word: string search keyword
    :param str sort: string ('author-date', 'commiter-date') (default: best match)
    :param str order: string ('desc', 'asc') (default: 'desc')
    :param dict kwargs: dictionary including query parameter set
    :rtype: :class:`github.PaginatedList.PaginatedList` of :class:`github.Commit`
    """
    #assert self._Github__requester._Requester__api_preview
    assert isinstance(word, (str, )), word    # for Python2, check if word is (str, unicode)
    url = '/search/commits'
    url_parameters = dict()
    q = '+'.join([k+':'+v for (k,v) in kwargs.items()])
    q += '+' + word
    #url_parameters["q"] = q    github.Requester encodes the url_parameters and Github reject it.
    url += '?q=' + q    # So, We directly include the parameters into URL itself
    
    if sort is not github.GithubObject.NotSet:
        #url_parameters["sort"] = sort
        url += '&sort=' + sort
    if order is not github.GithubObject.NotSet:
        #url_parameters["order"] = order
        url += '&order=' + order
    
    return github.PaginatedList.PaginatedList(
        github.Commit.Commit,
        self._Github__requester,  # access Github.__requester
        url,
        url_parameters,
        headers = {'Accept': 'application/vnd.github.cloak-preview'}
    )


# Factory of Github object supporting /search/commits
def _Github(login_or_token=None, password=None, base_url='https: "**********":
    import types
    g = "**********"=False)
    g.search_commits = types.MethodType(__Github_search_commits, g)
    return gub.Github(login_or_token, password, base_url, timeout, client_id, client_secret, user_agent, per_page, api_preview=False)
    g.search_commits = types.MethodType(__Github_search_commits, g)
    return g