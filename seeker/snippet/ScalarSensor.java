//date: 2025-07-04T16:37:01Z
//url: https://api.github.com/gists/e24a93b9d8dc94b336881d88b148ed98
//owner: https://api.github.com/users/AbhijayS

// Represents any measurement device that provides a singular measurement value with some units.
public interface ScalarSensor {
    double getValue();
    ScalarUnits getUnits();
}