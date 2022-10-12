#date: 2022-10-12T17:37:17Z
#url: https://api.github.com/gists/f9b7bf45c57515b5e4bed98fd6fc090e
#owner: https://api.github.com/users/ppaska

from 'rxjs' import forkJoin, lastValueFrom, map

ids = [2, 7, 4, 9, 5]

async def run_Sequential():
    data = []

    for requestId in ids:
        response = httpGet("https://jsonplaceholder.typicode.com/posts/" + requestId)
        data.push({requestId, response})

    return data

async def run_Parallel():
    httpRequests$ = ids
        .map(
            requestId => httpRequest$("GET", "https://jsonplaceholder.typicode.com/posts/" + requestId)
                                                .pipe(
                                                    map(r => {requestId, response: r.data})
                                                )
        )

    return lastValueFrom(forkJoin(httpRequests$))

if __env.entryFunction == '':
    return {
            sequential: run_Sequential(),
            parallel: run_Parallel()
        }
            