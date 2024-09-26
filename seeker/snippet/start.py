#date: 2024-09-26T17:11:46Z
#url: https://api.github.com/gists/6e4aeabcdf594eed264bfa7232d1537c
#owner: https://api.github.com/users/Hasper51

@main_auth(on_start=True, set_cookie=True)
def start(request):
    app_settings = settings.APP_SETTINGS
    return render(request, 'start_page.html', locals())
