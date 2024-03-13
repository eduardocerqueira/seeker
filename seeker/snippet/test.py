#date: 2024-03-13T17:03:37Z
#url: https://api.github.com/gists/f7bf27c973cf7fb4906260f5315654c1
#owner: https://api.github.com/users/shazogg

def on_mouse_move(event):
    # Déterminer quel widget se trouve sous la souris
    widget = event.widget.winfo_containing(event.x_root, event.y_root)
    if widget == canvas:
        # Si la souris est sur le canvas, appeler la fonction de gestion du mouvement de la souris du canvas
        on_canvas_mouse_move(event)
    else:
        # Si la souris est en dehors du canvas, appeler la fonction de gestion du mouvement de la souris en dehors du canvas
        on_non_canvas_mouse_move(event)


def on_canvas_mouse_move(event):
    print('Mouse move on canvas')
    # Implémentez ici votre logique pour le mouvement de la souris sur le canvas
    pass


def on_non_canvas_mouse_move(event):
    print('Mouse move outside canvas')
    # Implémentez ici votre logique pour le mouvement de la souris en dehors du canvas
    pass




ui.bind_all('<Motion>', on_mouse_move)