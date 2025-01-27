#date: 2025-01-27T16:55:37Z
#url: https://api.github.com/gists/5e995fd80b89b5256e15eaf98bc1b9ba
#owner: https://api.github.com/users/tonyjurg

# This code traverses the hierarchical structure of a Text-Fabric corpus, 
# starting from books and drilling down through chapters, verses, and words. 
# At each level, it allows for processing or extracting specific data 
# related to the respective nodes (book, chapter, verse, or word).

   # Loop through all the book nodes in the corpus
   for bookNode in F.otype.s('book'):
        # Do something on the book nodes data

        # Loop through all the chapter nodes in the book
        for chapterNode in L.i(bookNode,otype='chapter'):
            # Do something on the chapter nodes data
      
            # Loop through the verse nodes in the chapter
            for verseNode in L.i(chapterNode,otype='verse'):
		# Do something on the verse nodes data			
                          
                # Loop throuh all the word nodes in the verse
                for wordNode in L.i(verseNode,otype='word'):
	              # Do something on the word nodes data