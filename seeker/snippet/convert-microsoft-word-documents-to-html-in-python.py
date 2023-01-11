#date: 2023-01-11T17:10:27Z
#url: https://api.github.com/gists/428cc43a577379d1fd3c7647c9d897e9
#owner: https://api.github.com/users/ed-science

doc = aw.Document(dataDir + "sourceWordFile.docx")

doc.save(dataDir + "wordtoHtml.html", aw.SaveFormat.HTML)