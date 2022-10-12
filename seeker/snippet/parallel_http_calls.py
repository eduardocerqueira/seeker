#date: 2022-10-12T17:22:49Z
#url: https://api.github.com/gists/1af5457f81931c1bdb71d34f3ccd5c69
#owner: https://api.github.com/users/ppaska

from 'rxjs' import forkJoin, lastValueFrom, map

ids = [2, 7, 4, 9, 5]

httpRequests$ = ids
    .map(
        requestId => httpRequest$("GET", "https://jsonplaceholder.typicode.com/posts/" + requestId)
                                            .pipe(
                                                map(r => {requestId, response: r.data})
                                            )
    )

return lastValueFrom(forkJoin(httpRequests$))
