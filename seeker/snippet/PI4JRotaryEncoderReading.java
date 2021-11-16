//date: 2021-11-16T17:05:51Z
//url: https://api.github.com/gists/0ec496d7eb6f14883f33e7639f3f2dab
//owner: https://api.github.com/users/zipCoder933

/**
 *
 * @author zipCoder933
 */
public class RotaryEncoder {

    /**
     * @return the encoder value
     */
    public final int getValue() {
        return value;
    }

    private int value;

    /**
     * callback receiving -1 or 1
     *
     * @param i
     */
    public void onRotate(int i) {
    }

    public RotaryEncoder(Pin pinA, Pin pinB) {
        init(pinA, pinB, 0);
    }

    public RotaryEncoder(Pin pinA, Pin pinB, int initialValue) {
        init(pinA, pinB, initialValue);
    }

    private void init(Pin pinA, Pin pinB, int initialValue) {
        value = initialValue;
        GpioPinDigitalInput inputA = GpioFactory.getInstance().provisionDigitalInputPin(pinA, "PinA", PinPullResistance.PULL_UP);
        GpioPinDigitalInput inputB = GpioFactory.getInstance().provisionDigitalInputPin(pinB, "PinB", PinPullResistance.PULL_UP);
        inputA.addListener(new GpioPinListenerDigital() {
            int lastA;

            @Override
            public synchronized void handleGpioPinDigitalStateChangeEvent(GpioPinDigitalStateChangeEvent arg0) {
                int a = inputA.getState().getValue();
                int b = inputB.getState().getValue();
                if (lastA != a) {
                    value += (b == a ? -1 : 1);
                    onRotate(b == a ? -1 : 1);
                    lastA = a;
                }
            }
        });
    }
}