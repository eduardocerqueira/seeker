#date: 2023-07-04T17:07:42Z
#url: https://api.github.com/gists/a4a5f1e037b7e4f9847ce348e50b0ed9
#owner: https://api.github.com/users/nirvana6

    kindle = Kindle(
        options.csrf_token,
        options.domain,
        options.outdir,
        options.outdedrmdir,
        options.outepubmdir,
        options.cut_length,
        session_file=options.session_file,
        device_sn=options.device_sn,
    )