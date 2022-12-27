//date: 2022-12-27T16:38:00Z
//url: https://api.github.com/gists/fd8291daad05e006ae2d244b4edb894a
//owner: https://api.github.com/users/Mikaeryu

package org.stepic.tasks;

public final class ComplexNumber {
    private final double re;
    private final double im;

    public ComplexNumber(double re, double im) {
        this.re = re;
        this.im = im;
    }

    public double getRe() {
        return re;
    }

    public double getIm() {
        return im;
    }

    @Override
    public boolean equals(Object anObject) {
        if (this == anObject) {
            return true;
        }
        if (anObject == null || anObject.getClass() != this.getClass()) {
            return false;
        }
        ComplexNumber complexNum = (ComplexNumber) anObject;
        return this.re == complexNum.re && this.im == complexNum.im; // равность только при равных полях экземпляров
    }

    @Override
    public int hashCode() {
        int result = 2;
        result = 31 * result + Double.hashCode(re);
        result = 31 * result + Double.hashCode(im);

        return result;
    }
}