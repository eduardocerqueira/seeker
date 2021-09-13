#date: 2021-09-13T17:13:43Z
#url: https://api.github.com/gists/74976706c775a21b186899f11b26d2d1
#owner: https://api.github.com/users/Hafsah2018

doc = nlp(' Last year, I spoke about the Ujjwala programme , through which, I am happy to report, 50 million free liquid-gas connections have been provided so far')
png = visualise_spacy_tree.create_png(doc)
display(Image(png))