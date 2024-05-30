#date: 2024-05-30T16:47:58Z
#url: https://api.github.com/gists/01634b15f1e3888a6c88d7f975c8e095
#owner: https://api.github.com/users/codingmatheus

# You can declare multiline strings in Python using triple quotes (single or doubled) or f-strings

## using triple quotes
def multiline_strings_with_triple_quotes():
    return '''
    This is a multiline string
    
    with triple quotes
    '''
  
##using f-strings
def multline_strings_with_f_strings():
    return (
    "This is a multiline string"
        
    f"with f strings"
    )
        

if __name__ == '__main__':
    
    #note how with triple quotes any new lines, tabs, or spaces are preserved. This is the equivalent of using something like the
    #<pre> tag in HTML
    print(f"\n---------------------------------------\nMultiline strings with triple quotes:\n---------------------------------------\n")
    print(f"{multiline_strings_with_triple_quotes()}")

    #f-strings will just blindly concatenate strings. It will only preserve spaces and it's up to you to add a space at the end
    #of your sentences so it doesn't just join them up as it happens here
    print(f"\n---------------------------------------\nMultiline strings with f strings:\n---------------------------------------\n")
    print(f"{multline_strings_with_f_strings()}\n\n")
    
#BONUS TIP: Python only has single-line comments! don't use tripe quotes as comments because they are literal strings which
#will be garbage collected if not assigned. You certainly don't want something that is only meant as a comment to take up memory
#and processing time.
#as for docstrings, use them if you have an actual use for them such as, for example, you will be displaying them using the built-in
#help() function in Python. They can be cool for demos/PoCs/tutorials, but you should have a very specific need if you're deploying
#that code to production