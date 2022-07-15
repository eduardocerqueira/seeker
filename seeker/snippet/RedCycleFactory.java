//date: 2022-07-15T16:53:08Z
//url: https://api.github.com/gists/30894ef7941e55470c49893d400a75b0
//owner: https://api.github.com/users/ericdiazcodes

public class RedCycleFactory implements CycleFactory {
    @Override
    public Bicycle createBicycle() {
        return new RedBicycle();
    }

    @Override
    public Tricycle createTricycle() {
        return new RedTricycle();
    }
}

public class BlueCycleFactory implements CycleFactory {
    @Override
    public Bicycle createBicycle() {
        return new BlueBicycle();
    }

    @Override
    public Tricycle createTricycle() {
        return new BlueTricycle();
    }
}