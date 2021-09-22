#date: 2021-09-22T17:07:31Z
#url: https://api.github.com/gists/cfd7979de8ab6f17f8fec144f0882a3d
#owner: https://api.github.com/users/RaphaelGoutmann

from tinyengine import *
import ion

def main():
    game = Game()
    mainScene = Scene()
    game.setCurrentScene(mainScene)

    player = Rectangle("player", 0, 0, 10, 10, color.BLACK)
    mainScene.addObject(player)

    while(1):
        speed = Vector2(0, 0)

        if keydown(KEY_LEFT):
            speed = speed.left()
        elif keydown(KEY_RIGHT):
            speed = speed.right()
        elif keydown(KEY_UP):
            speed = speed.up()
        elif keydown(KEY_DOWN):
            speed = speed.down()
        
        player.setX(player.getX() + speed.x)
        player.setY(player.getY() + speed.y)

        game.update()

main()
