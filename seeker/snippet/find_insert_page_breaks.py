#date: 2025-02-25T16:54:14Z
#url: https://api.github.com/gists/5f92746d24946c5aeef2a491222eb796
#owner: https://api.github.com/users/richardye101

import logging
from typing import List, Tuple
import pymupdf
from fuzzysearch import find_near_matches

logger = logging.Logger("find_insert_page_breaks", level = logging.INFO)
def map_alphanum(string: str, is_markdown: bool = False) -> Tuple[str, List[str]]:
    '''
    For a given input string, the alphanumeric characters and their positions in the original string are returned.
    If the string uses markdown format, all links and images are stripped to the text/alt-text.

    This function is used to assist in aligning text such that when searching, exact matches can be found, and the positions
    in the original string can be retrieved.

    Args:
        string (str): The original string to convert.
        is_markdown (bool, optional): Indicate whether string is in markdown format.
            If so, extra processing is performed to remove link/image paths.
            Defaults to False.

    Returns:
        Tuple[str, Union[str,List[str]]]: Tuple containing the alphanumeric version of string,
            and a list of indices mapping characters in the alphanumeric string back to the original.

    Example: 
    string = "This is a [link](http://example.com) and ![image](folder/img.png) in Markdown format."
    Output = ('ThisisalinkandimageinMarkdownformat',
              [0, 1, 2, 3, 5, 6, 8, 11, 12, 13, 14,
               37, 38, 39, 43, 44, 45, 46, 47, 66, 
               67, 69, 70, 71, 72, 73, 74, 75, 76, 
               78, 79, 80, 81, 82, 83])
    '''
    pos = list()
    alnum = list()
    i = 0
    while i < len(string):
        c = string[i]
        if is_markdown:
            if c == "(" and string[i-1] == "]":
                while string[i] != ")":
                    i += 1  
        if c.isalnum():
            pos.append(i)
            alnum.append(c)
        i += 1
    return (''.join(alnum), pos)

def find_insert_page_breaks(markdown:str, pdf_doc: pymupdf.Document,
                     prefix_sizes: List[float] = [.1, .25, .4, .6],
                     page_buffer: float = 0.1) -> str:
    '''
    For a given input `markdown` string, page breaks are identified and inserted based on it's
    equivalent PDF document `pdf_doc`. It first converts the entire markdown into only alphanumeric characters,
    removing any hyperlinks or image paths using the `map_alphanum` function. Each page in the PDF is also 
    converted to it's alphanumeric representation. The first 10% of alphanumeric characters on the page 
    (prefix substring) are searched for in the sliced alphanumeric markdown string. The 10% can be adjusted up 
    if the prefix substring exists on the PDF page more than once. We guesstimate the slice to
    contain the same amount of characters as the PDF page, plus a buffer on the head and tail. 
    Once the position in the markdown slice is determined, the starting index of the prefix substring in
    the original markdown is identified using the list from step 1, and stored. Once every page has been
    iterated through, the list of starting indices are returned. Those indices are used to edit the original
    markdown to insert page breaks.

    Args:
        markdown (str): The markdown string where page breaks will be inserted.
        pdf_doc (pymupdf.Document): The PDF version of the markdown string.
        prefix_sizes (List[float]): A preset and editable list of percentages which determine the 
            progression of the prefix substring size. The default progression goes 10%, 25%, 40%, and 60% of
            the PDF page.
        page_buffer (float): The percentage of text to capture as buffer around the markdown string slice
            to be searched. Default is 10% before and after the markdown slice.

    Returns:
        str: The resultant markdown string that includes page breaks.

    Usage:
    test_md = markdown_string
    test_pdf = pymupdf.open(pdf_file_path)
    new_md = find_insert_page_breaks(test_md, test_pdf)
    print(new_md)
    '''
    alphanum_markdown, alphanum_markdown_pos = map_alphanum(markdown, is_markdown=True)

    actual_starts = list()

    new_start = 0
    for pid in range(len(pdf_doc)):
        # extract page
        page = pdf_doc[pid]

        # extract text from page
        page_text = page.get_text()
        alphanum_page, _ = map_alphanum(page_text)

        # set the approximate slice window for the markdown string
        start_idx = int(new_start * (1-page_buffer))
        end_idx = int(new_start + len(alphanum_page) * (1+page_buffer))

        # ensure the searched for string is unique in the page
        psizeid = 0
        substr_size = int(len(alphanum_page) * prefix_sizes[psizeid])
        substr = alphanum_page[:substr_size]
        #increase size of substring until there are no duplicates
        while alphanum_page.count(substr) > 1:
            psizeid += 1
            substr_size = int(len(alphanum_page) * prefix_sizes[psizeid])
            substr = alphanum_page[:substr_size]
        
        # find matches and the position of the match
        match_results = find_near_matches(substr, alphanum_markdown[start_idx:end_idx], max_l_dist=33)
        
        if match_results:
            # store the index of where this string begins in the actual markdown string
            start_of_page = start_idx + match_results[0].start
            actual_starts.append(alphanum_markdown_pos[start_of_page])

            # set the new starting index for the next alphanum page
            new_start = start_of_page + len(alphanum_page)
        else:
            logger.warning(f'Unable to match current PDF page {pid} to a position in the markdown document.')
        
        # print(f'Word Doc: \n {alphanum_markdown[start_idx:end_idx]}\n')
        # print(f'Searching for: \n {substr}')
        # print(match_results)
        # print(f'Found new page starting at \n {alphanum_markdown[start_of_page:end_idx]}')
        # print(f'Position in actual markdown is: {alphanum_markdown_pos[start_of_page]}')
        # print(f'Actual markdown: \n {markdown[alphanum_markdown_pos[start_of_page]:alphanum_markdown_pos[start_of_page] + 100]}')

    page_indicator_offset = 0
    for i, start in enumerate(actual_starts):
        page_indicator = f"\n---Page {i+1} Start---\n"
        markdown = markdown[:start + page_indicator_offset] + page_indicator + markdown[start + page_indicator_offset:]
        page_indicator_offset += len(page_indicator)
    return markdown