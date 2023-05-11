//date: 2023-05-11T17:00:55Z
//url: https://api.github.com/gists/1196e9d235c4fc35c0964f56fde30a90
//owner: https://api.github.com/users/DonMassa84

class ParkingLot {
    private int floors;
    private int spacesPerFloor;
    private ParkingSpace[][] parkingSpaces;

    public ParkingLot(int floors, int spacesPerFloor) {
        this.floors = floors;
        this.spacesPerFloor = spacesPerFloor;
        this.parkingSpaces = new ParkingSpace[floors][spacesPerFloor];

        for (int i = 0; i < floors; i++) {
            for (int j = 0; j < spacesPerFloor; j++) {
                parkingSpaces[i][j] = new ParkingSpace();
            }
        }
    }

    public int getFloors() {
        return floors;
    }

    public int getSpacesPerFloor() {
        return spacesPerFloor;
    }

    public ParkingSpace getParkingSpace(int floor, int space) {
        return parkingSpaces[floor][space];
    }

    public int[] findEmptySpace() {
        for (int i = 0; i < floors; i++) {
            for (int j = 0; j < spacesPerFloor; j++) {
                if (!parkingSpaces[i][j].isOccupied()) {
                    return new int[]{i, j};
                }
            }
        }
        return null;
    }

    public int getNumberOfFreeSpaces() {
        int count = 0;
        for (int i = 0; i < floors; i++) {
            for (int j = 0; j < spacesPerFloor; j++) {
                if (!parkingSpaces[i][j].isOccupied()) {
                    count++;
                }
            }
        }
        return count;
    }

    public int[] findVehicle(String licensePlate) {
        for (int i = 0; i < floors; i++) {
            for (int j = 0; j < spacesPerFloor; j++) {
                if (parkingSpaces[i][j].isOccupied() && parkingSpaces[i][j].getVehicle().getLicensePlate().equals(licensePlate)) {
                    return new int[]{i, j};
                }
            }
        }
        return null;
    }
}