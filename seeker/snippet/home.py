#date: 2023-10-20T17:07:20Z
#url: https://api.github.com/gists/61a46d9208d5b39c40fd184fd9e38b16
#owner: https://api.github.com/users/jfjensen

import json
from browser import alert, document, ajax, html

base_url = "http://127.0.0.1:8000/"

def complete_get_all_books(request):
    data = request.json
    document["canvas"].text = ""
    if len(data) > 0:
        table = html.TABLE(cellspacing=10, Class="table table-striped")
        t = html.TBODY()
        title = html.TR()
        
        title <= html.TH("Title", align="center")
        title <= html.TH("Author", align="center")
        title <= html.TH("Publisher", align="center")
        title <= html.TH("Genre", align="center")
        title <= html.TH("")
        title <= html.TH("")
        t <= title
        rows = []
        for ix, entry in enumerate(data):
            row = html.TR()
            row <= html.TD(entry['title'], align="center")
            row <= html.TD(entry['author'], align="center")
            row <= html.TD(entry['publisher'], align="center")
            row <= html.TD(entry['genre'], align="center")
            row <= html.TD(html.BUTTON("EDIT", Class="btn button-edit", Data_id=entry['id']), align="center")
            row <= html.TD(html.BUTTON("DELETE", Class="btn button-delete", Data_id=entry['id']), align="center")
            rows.append(row)
    
        t <= rows
        table <= t
        document["canvas"] <= table
        for button in document.select('.button-edit'):
            button.bind("click", click_button_edit)
        for button in document.select('.button-delete'):
            button.bind("click", click_button_delete)
    else:
        document["canvas"] <= html.DIV()
        
def complete_add_book(response):
    document["modal-add-book"].class_name = "modal"
    get_all_books()
    document["input-title"].value = ""
    document["input-author"].value = ""
    document["input-publisher"].value = ""
    document["input-genre"].value = ""
    
def complete_delete_book(response):
    get_all_books()
    
def get_all_books():
    url = base_url + "books"
    ajax.get(url, oncomplete=complete_get_all_books, mode="json")
    document["canvas"].text = "waiting..."
    
def click_get_all_books(event):
    get_all_books()
    
def click_add_book(event):
    document["modal-add-book"].class_name = "modal active"
    document["button_add_book_submit"].bind("click", click_add_book_submit)
    
def click_add_book_submit(event):
    document["button_add_book_submit"].unbind("click")
    url = base_url + "book"
    data = {
        "title": str(document["input-title"].value),
        "author": str(document["input-author"].value),
        "publisher": str(document["input-publisher"].value),
        "genre": str(document["input-genre"].value),
    }
    ajax.post(url, oncomplete=complete_add_book, data=json.dumps(data))
    
def click_modal_close(event):
    document["modal-add-book"].class_name = "modal"
    document["modal-edit-book"].class_name = "modal"
    
def click_button_edit(event):
    book_id = event.currentTarget.attrs['data-id']
    url = base_url + "book/" + str(book_id)
    ajax.get(url, oncomplete=complete_edit_get_book, mode="json")
    
def complete_edit_get_book(response):
    data = response.json
    document["edit-input-id"].value = data['id']
    document["edit-input-title"].value = data['title']
    document["edit-input-author"].value = data['author']
    document["edit-input-publisher"].value = data['publisher']
    document["edit-input-genre"].value = data['genre']
    document["modal-edit-book"].class_name = "modal active"
    document["button_edit_book_submit"].bind("click", click_edit_book_submit)
    
def click_edit_book_submit(event):
    document["button_edit_book_submit"].unbind("click")
    book_id = document["edit-input-id"].value
    url = base_url + "book/" + str(book_id)
    data = {
        "title": str(document["edit-input-title"].value),
        "author": str(document["edit-input-author"].value),
        "publisher": str(document["edit-input-publisher"].value),
        "genre": str(document["edit-input-genre"].value),
    }
    ajax.put(url, oncomplete=complete_edit_book, data=json.dumps(data))
    
def complete_edit_book(response):
    document["modal-edit-book"].class_name = "modal"
    get_all_books()
    
def click_button_delete(event):
    book_id = event.currentTarget.attrs['data-id']
    url = base_url + "book/" + str(book_id)
    ajax.delete(url, oncomplete=complete_delete_book, mode="json")
    
get_all_books()

document["button_add_book"].bind("click", click_add_book)
document["modal_add_close"].bind("click", click_modal_close)
document["modal_edit_close"].bind("click", click_modal_close)