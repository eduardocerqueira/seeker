#date: 2026-01-21T17:38:10Z
#url: https://api.github.com/gists/47fc4bcae1872ef684eacab8304e9001
#owner: https://api.github.com/users/ChicagoDev

from guizero import App, Picture, Text
from PIL import Image

app = App()

Picture(app, image="card_image/club/2.png", align="left")

Picture(app, image="card_image/club/3.png", align="left")


def overlay_cards(card1_path, card2_path, save_path="hand.png", offset=30):

    card1 = Image.open(card1_path).convert("RGBA")
    card2 = Image.open(card2_path).convert("RGBA")

    # Create a new blank image that is wide enough for both cards
    width = card1.width + offset
    height = max(card1.height, card2.height)
    hand = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # transparent background

    # Paste both cards
    hand.paste(card1, (0, 0), card1)  # card1 at x=0
    hand.paste(card2, (offset, 0), card2)  # card2 shifted right

    # Save the result
    hand.save(save_path)

    return save_path

overlayed_path = overlay_cards("card_image/club/2.png", "card_image/club/3.png")

Picture(app, image="hand.png", align="left")

app.display()
