#date: 2021-12-16T16:51:29Z
#url: https://api.github.com/gists/59db048646b8ad7a777b0f3d6cea9f18
#owner: https://api.github.com/users/katipogluMustafa

def edit_story(request, story_id):
    if request.method == "POST":
      
        title = request.POST["mce_0"]
        description = request.POST["mce_2"]
        content = request.POST["mce_3"]
        
        update_story_object(story_id, title, description, content)
        
        return redirect("show_story_page", story_id)
    else:
        story = get_object_or_404(Story, pk=story_id)
        
        return render(request, "edit_story.html", {'story': story})