//date: 2023-05-11T17:00:55Z
//url: https://api.github.com/gists/1196e9d235c4fc35c0964f56fde30a90
//owner: https://api.github.com/users/DonMassa84

class Garage {
    private ParkingLot parkingLot;

    public Garage(int floors, int spacesPerFloor) {
        this.parkingLot = new ParkingLot(floors, spacesPerFloor);
    }

    public boolean parkVehicle(Vehicle vehicle) {
        int[] emptySpace = parkingLot.findEmptySpace();
        if (emptySpace != null) {
           
            parkingLot.getParkingSpace(emptySpace[0], emptySpace[1]).parkVehicle(vehicle);
            return true;
        } else {
            return false;
        }
    }

    public boolean removeVehicle(String licensePlate) {
        int[] vehiclePosition = parkingLot.findVehicle(licensePlate);
        if (vehiclePosition != null) {
            parkingLot.getParkingSpace(vehiclePosition[0], vehiclePosition[1]).removeVehicle();
            return true;
        } else {
            return false;
        }
    }

    public int[] getVehiclePosition(String licensePlate) {
        return parkingLot.findVehicle(licensePlate);
    }

    public int getNumberOfFreeSpaces() {
        return parkingLot.getNumberOfFreeSpaces();
    }
}