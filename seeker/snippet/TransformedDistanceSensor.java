//date: 2025-07-04T16:37:01Z
//url: https://api.github.com/gists/e24a93b9d8dc94b336881d88b148ed98
//owner: https://api.github.com/users/AbhijayS

// A sensor that applies a function transformation to the raw measurements from a sensor
// into some usable output with units specified.
public class TransformedDistanceSensor implements ScalarSensor {
    private final DoubleSupplier rawValueSupplier;
    private final Function<Double, Double> transformer;
    private final ScalarUnits transformedOutputUnits;

    public TransformedDistanceSensor(DoubleSupplier rawValueSupplier,
                                     Function<Double, Double> transformer,
                                     ScalarUnits transformedOutputUnits) {
        this.rawValueSupplier = rawValueSupplier;
        this.transformer = transformer;
        this.transformedOutputUnits = transformedOutputUnits;
    }

    @Override
    public double getValue() {
        double raw = rawValueSupplier.getAsDouble();
        return transformer.apply(raw);
    }

    @Override
    public ScalarUnits getUnits() {
        return transformedOutputUnits;
    }
}