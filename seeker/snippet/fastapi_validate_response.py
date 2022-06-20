#date: 2022-06-20T17:12:51Z
#url: https://api.github.com/gists/290a79e887e202749e22955759275106
#owner: https://api.github.com/users/akpp

# "response_model" is used for validation and generation documentation (docs, redoc)
# response_model_exclude_unset=True - omit unset params in response
@app.post("/items", response_model=Dict[str, Item], response_model_exclude_unset=True)
def create_book(item: Item):
    return {"item": item}