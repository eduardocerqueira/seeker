//date: 2022-07-15T17:00:56Z
//url: https://api.github.com/gists/cb2bd8ceae6307c97a870ff9bec3b6ee
//owner: https://api.github.com/users/ericdiazcodes

    public static void main(String[] args) {
        var redCycleFactory = new RedCycleFactory();
        var blueCycleFactory = new BlueCycleFactory();

        var redBicycle = redCycleFactory.createBicycle();
        var blueTricycle = blueCycleFactory.createTricycle();

        // prints -> Ridding my RedBicycle on 2 wheels
        redBicycle.ride();
       
       // prints -> Ridding my BlueTricycle on 3 wheels
        blueTricycle.ride();
    }