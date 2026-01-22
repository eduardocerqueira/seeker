//date: 2026-01-22T17:17:33Z
//url: https://api.github.com/gists/718523970e00e81d395b31841ff1ff19
//owner: https://api.github.com/users/nobody5050

import edu.wpi.first.math.MathUtil;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.kinematics.ChassisSpeeds;
import java.util.function.Supplier;
import java.util.Optional;

/**
 * {@code AntiTipping} provides a proportional correction system to prevent the robot from tipping
 * over during operation.
 *
 * <p>It uses pitch and roll measurements (in degrees) to detect excessive inclination and computes
 * a correction velocity in the opposite direction of the tilt. The resulting correction can be
 * added to the robot’s translational velocity to help stabilize it.
 *
 * <h2>Usage</h2>
 * <ol>
 *   <li>Instantiate with pitch and roll suppliers and initial configuration parameters.
 *   <li>Call {@link #calculate()} periodically (e.g. once per control loop).
 *   <li>Add the correction from {@link #getVelocityAntiTipping()} to your drive command.
 * </ol>
 *
 * <h2>Configuration</h2>
 * <ul>
 *   <li>{@link #setTippingThreshold(double)} — sets the tipping detection threshold in degrees.
 *   <li>{@link #setMaxCorrectionSpeed(double)} — sets the maximum correction velocity (m/s).
 * </ul>
 *
 * <p>The correction is purely proportional: {@code correction = kP * inclinationMagnitude}, and
 * clamped to {@code maxCorrectionSpeed}.
 * 
 *
 * @since 2025
 */
public class AntiTipping {

  public double tippingThresholdDegrees;
  public double maxCorrectionSpeed; // m/s
  public double kP; // proportional gain

  private double pitch = 0.0;
  private double roll = 0.0;
  private double correctionSpeed = 0.0;
  private double inclinationMagnitude = 0.0;
  private double yawDirectionDeg = 0.0;
  
  private Rotation2d tiltDirection = new Rotation2d();

  /**
   * Creates a new {@code AntiTipping} instance.
   *
   * @param attitudeSupplier supplier providing the current robot attitude as a {@link Rotation3d}
   * @param kP proportional gain for correction
   * @param tippingThresholdDegrees tipping detection threshold (degrees)
   * @param maxCorrectionSpeed maximum correction velocity (m/s)
   */
  public AntiTipping(
      double kP,
      double tippingThresholdDegrees,
      double maxCorrectionSpeed) {

    this.kP = kP;
    this.tippingThresholdDegrees = tippingThresholdDegrees;
    this.maxCorrectionSpeed = maxCorrectionSpeed;
  }

  /**
   * Updates tipping detection and computes the proportional correction.
   *
   * <p>This method updates internal values (pitch, roll, direction, magnitude, etc.) and generates
   * a correction {@link ChassisSpeeds} vector that can be applied to stabilize the robot.
   * It should be called periodically (e.g. once per control loop).
   * 
   * @param attitudeSupplier supplier providing the current robot attitude as a {@link Rotation3d}
   * @return correction {@link ChassisSpeeds} to counteract tipping
   */
  public Optional<ChassisSpeeds> calculate(Supplier<Rotation3d> attitudeSupplier) {
    pitch = attitudeSupplier.get().getY();
    roll = attitudeSupplier.get().getX();

    boolean isTipping = Math.abs(pitch) > tippingThresholdDegrees || Math.abs(roll) > tippingThresholdDegrees;

    // Tilt direction (the direction the robot is falling towards)
    tiltDirection = new Rotation2d(Math.atan2(-roll, -pitch));
    yawDirectionDeg = tiltDirection.getDegrees();

    // Tilt magnitude (hypotenuse of pitch and roll)
    inclinationMagnitude = Math.hypot(pitch, roll);

    // Proportional correction
    correctionSpeed = kP * -inclinationMagnitude;
    correctionSpeed = MathUtil.clamp(correctionSpeed, -maxCorrectionSpeed, maxCorrectionSpeed);

    // Correction vector (field-relative)
    Translation2d correctionVector =
        new Translation2d(0, 1).rotateBy(tiltDirection).times(correctionSpeed);

    if (isTipping) {
      return Optional.of(new ChassisSpeeds(correctionVector.getX(), -correctionVector.getY(), 0));
    } else {
      return Optional.empty();
    }
  }

  /** Returns the most recent pitch value in degrees. */
  public double getPitch() {
    return pitch;
  }

  /** Returns the most recent roll value in degrees. */
  public double getRoll() {
    return roll;
  }

  /** Returns the latest tilt magnitude (hypotenuse of pitch and roll). */
  public double getLastInclinationMagnitude() {
    return inclinationMagnitude;
  }

  /** Returns the most recent tilt direction in degrees (pseudo-yaw). */
  public double getLastYawDirectionDeg() {
    return yawDirectionDeg;
  }

  /** Returns the most recent tilt direction as a {@link Rotation2d}. */
  public Rotation2d getLastTiltDirection() {
    return tiltDirection;
  }
}