#date: 2023-07-04T17:07:42Z
#url: https://api.github.com/gists/a4a5f1e037b7e4f9847ce348e50b0ed9
#owner: https://api.github.com/users/nirvana6

    if options.readme:
        # generate readme stats
        kindle.make_kindle_stats_readme()
    else:
        # check the download mode
        if options.mode == "all":
            # download all books
            kindle.download_books(
                start_index=options.index - 1, filetype=options.filetype
            )
        elif options.mode == "sel":
            # download selected books
            download_selected_books(kindle, options)
        else:
            print("mode error, please input all or sel")