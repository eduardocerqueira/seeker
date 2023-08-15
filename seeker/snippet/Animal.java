//date: 2023-08-15T16:52:28Z
//url: https://api.github.com/gists/9c685705aac7282e5e0d3f49be8cbaf7
//owner: https://api.github.com/users/speedofearth

public void move() {
        int speed = movementSpeed;
        int currentX = this.x;
        int currentY = this.y;
        int maxX = grid.getXLimit() - 1;
        int maxY = grid.getYLimit() - 1;

        for (int i = 0; i < speed; i++) {
            if (currentX == 0 && currentY < maxY) {
                currentY++;
            } else if (currentY == maxY && currentX < maxX) {
                currentX++;
            } else if (currentX == maxX && currentY > 0) {
                currentY--;
            } else if (currentY == 0 && currentX > 0) {
                currentX--;
            }
        }
        setLocation(currentX, currentY);
        System.out.println("X:" + currentX + ", " + "Y:" + currentY);
    }