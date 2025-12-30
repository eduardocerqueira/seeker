//date: 2025-12-30T16:41:21Z
//url: https://api.github.com/gists/1890091e7dac3b6203232ea66690614e
//owner: https://api.github.com/users/mgrecu35

package org.firstinspires.ftc.teamcode;
import com.qualcomm.hardware.gobilda.GoBildaPinpointDriver;
import org.firstinspires.ftc.robotcore.external.navigation.AngleUnit;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.hardware.CRServo;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorEx;
import com.qualcomm.robotcore.hardware.DistanceSensor;
import com.qualcomm.robotcore.hardware.Servo;
import com.qualcomm.robotcore.util.ElapsedTime;
import org.firstinspires.ftc.robotcore.external.JavaUtil;
import org.firstinspires.ftc.robotcore.external.navigation.DistanceUnit;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;

@TeleOp(name = "StarterBotTeleop_December29th (Blocks to Java)")
public class teleop_dec29 extends LinearOpMode {

  private DcMotor launcher;
  private Servo ramp;
  private Servo arm;
  private GoBildaPinpointDriver odo;
  private DcMotor frontright;
  private DcMotor backright;
  private DistanceSensor distancesensor;
  private DcMotor intakeone;
  private DcMotor _2ndintake;
  private CRServo leftguider;
  private CRServo rightguider;
  private DcMotor backleft;
  private DcMotor frontleft;
  private PrintWriter writer;
  String IDLE;
  String SPIN_UP;
  String LAUNCH;
  String launchState;
  int LAUNCHER_TARGET_VELOCITY;
  ElapsedTime launchTime;
  String LAUNCHING;
  int LAUNCHER_MIN_VELOCITY;
  double power_scaling;

  /**
   * Describe this function...
   */
  private void createVariables() {
    int LAUNCHER_MAX_VELOCITY;

    IDLE = "IDLE";
    SPIN_UP = "SPIN_UP";
    LAUNCH = "LAUNCH";
    LAUNCHING = "LAUNCHING";
    launchState = IDLE;
    LAUNCHER_MAX_VELOCITY = 2000;
    LAUNCHER_TARGET_VELOCITY = 1400;
    LAUNCHER_MIN_VELOCITY = 1000;
    launchTime = new ElapsedTime();
  }

  
  @Override
  public void runOpMode() {
    double ramp2;
    int toggle;
    int yoggle;
    double launcher_power;
    double ramp_calibration;
    double odoY;
    double odoX;

    launcher = hardwareMap.get(DcMotor.class, "launcher");
    ramp = hardwareMap.get(Servo.class, "ramp");
    arm = hardwareMap.get(Servo.class, "arm");
    odo = hardwareMap.get(GoBildaPinpointDriver.class, "odo");
    frontright = hardwareMap.get(DcMotor.class, "front right");
    backright = hardwareMap.get(DcMotor.class, "back right");
    distancesensor = hardwareMap.get(DistanceSensor.class, "distance sensor");
    intakeone = hardwareMap.get(DcMotor.class, "intake one");
    _2ndintake = hardwareMap.get(DcMotor.class, "2nd intake");
    leftguider = hardwareMap.get(CRServo.class, "left guider");
    rightguider = hardwareMap.get(CRServo.class, "right guider");
    backleft = hardwareMap.get(DcMotor.class, "back left");
    frontleft = hardwareMap.get(DcMotor.class, "front left");

    // Put initialization blocks here.
    createVariables();
    initMotors();
    waitForStart();
    frontright.setDirection(DcMotor.Direction.REVERSE);
    backright.setDirection(DcMotor.Direction.REVERSE);
    ramp2 = ramp.getPosition();
    toggle = 0;
    yoggle = 0;
    power_scaling = 0.67;
    ramp.setPosition(0.32);
    ramp.scaleRange(0, 1);
    arm.scaleRange(0, 1);
    odo.setPosX(0, DistanceUnit.CM);
    odo.setPosY(0, DistanceUnit.CM);
    
   // drive_y(50);
   // drive_y(-50);
   // drive_x(50);
   // drive_x(-50);
   
   int sign=1;
   drive_around_2(0, -1,  (0.3 * sign), 0.55, 1200);
   sleep(1000);
   launcher_power = 0.58;
   launch_boom(launcher_power);
   
   //** new capabilities based on odometry
   turn_around(20.0);
   
   if (opModeIsActive()) {
      // Put run blocks here.
      telemetry.setMsTransmissionInterval(1);

      while (opModeIsActive()) {
        ramp_calibration = distancesensor.getDistance(DistanceUnit.METER) * 0.06 + 0.248;
        if (gamepad2.dpadDownWasPressed()) {
          arm.setPosition(0.5);
        }
        if (gamepad2.dpadDownWasPressed()) {
          arm.setPosition(1);
        }
        if (gamepad2.bWasPressed()) {
          launcher.setPower(0);
        }
        if (gamepad2.yWasPressed()) {
          launcher.setPower(launcher_power);
        }
        if (gamepad2.leftBumperWasPressed()) {
          intakeone.setPower(-1);
        }
        if (gamepad2.leftBumperWasReleased()) {
          intakeone.setPower(0);
        }
        if (gamepad2.xWasPressed()) {
          intakeone.setPower(1);
          _2ndintake.setPower(-0.5);
        }
        if (gamepad2.xWasPressed()){
          intakeone.setPower(0);
          _2ndintake.setPower(0);
        }
        if (gamepad2.dpadUpWasPressed() && launcher_power < 0.8) {
          launcher_power += 0.01;
        }
        if (gamepad2.dpadDownWasPressed() && launcher_power > 0.1) {
          launcher_power += -0.01;
        }
        if (gamepad2.right_trigger > 0.5 && distancesensor.getDistance(DistanceUnit.METER) <= 1.5) {
          ramp.setPosition(ramp_calibration);
        } else {
          ramp.setPosition(0.32);
        }
        if (gamepad2.rightBumperWasPressed()) {
          leftguider.setPower(-0.5);
          rightguider.setPower(-0.5);
          _2ndintake.setPower(-0.91);
          intakeone.setPower(1);
        } else if (gamepad2.rightBumperWasReleased()) {
          leftguider.setPower(0);
          rightguider.setPower(0);
          _2ndintake.setPower(0);
          intakeone.setPower(0);
        }
        drive_bois();
        odoY = odo.getPosY(DistanceUnit.CM);
        odoX = odo.getPosX(DistanceUnit.CM);
        odo.update();
        telemetry.addData("dist", distancesensor.getDistance(DistanceUnit.METER));
        telemetry.addData("left sti", Double.parseDouble(JavaUtil.formatNumber(gamepad1.left_stick_y, 4)));
        telemetry.addData("launchState", launchState);
        telemetry.addData("relativePosY", odoY);
        telemetry.addData("relativePosX", odoX);
        telemetry.addData("arm position", arm.getPosition());
        telemetry.addData("Launcher Motor Velocity", ((DcMotorEx) launcher).getVelocity());
        telemetry.addData("Launcher Target Velocity", 123);
        telemetry.addData("ramp", ramp.getPosition());
        telemetry.addData("launch_power", launcher.getPower());
        telemetry.addData("y", Double.parseDouble(JavaUtil.formatNumber(gamepad1.right_stick_x, 4)));
        telemetry.update();
      }
      
    }
    writer.close();
  }

  /**
   * Describe this function...
   */
  private void initMotors() {
    launcher.setDirection(DcMotor.Direction.FORWARD);
  }

  /**
   * Describe this function...
   */
  private void drive_bois() {
    double x;
    float y;
    float rx;
    double Denominator;

    backleft.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    backright.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    frontleft.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    frontright.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    x = gamepad1.left_stick_x * 1.1;
    y = -gamepad1.left_stick_y;
    rx = gamepad1.right_stick_x;
    Denominator = JavaUtil.maxOfList(JavaUtil.createListWith(JavaUtil.sumOfList(JavaUtil.createListWith(Math.abs(y), Math.abs(x), Math.abs(rx))), 1));
    if (gamepad1.rightBumperWasPressed()) {
      power_scaling += -0.3;
    }
    if (gamepad1.rightBumperWasReleased()) {
      power_scaling += 0.3;
    }
    // Denominator is the largest motor power (absoulte value) or 1.
    // This ensures all powers maintain the same ratio, but only if one is outside of the range (-1,1).
    frontleft.setPower(((y + x + rx) / Denominator) * power_scaling);
    backleft.setPower((((y - x) + rx) / Denominator) * power_scaling);
    frontright.setPower((((y - x) - rx) / Denominator) * power_scaling);
    backright.setPower((((y + x) - rx) / Denominator) * power_scaling);
  }
  
  private void drive_y(double dy){
    backleft.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    backright.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    frontleft.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    frontright.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    odo.update();
    double odoY0 = odo.getPosY(DistanceUnit.CM);
    double odoX0 = odo.getPosX(DistanceUnit.CM);
    double Driving_Power=0.4;
    double odoX, odoY;
    int isign=1;
    if(dy<0)
    {
      isign=-1;
    }
    odoY=odo.getPosY(DistanceUnit.CM);
    while((odoY-odoY0)*isign<Math.abs(dy))
    {
    if(((odoY-odoY0)*isign/Math.abs(dy))>0.7 && Driving_Power>0.1)
    {
    Driving_Power=Driving_Power*0.5;
    }
    frontleft.setPower(isign*Driving_Power);
    backleft.setPower(isign * Driving_Power);
    frontright.setPower(isign * Driving_Power);
    backright.setPower(isign * Driving_Power);
    odoY=odo.getPosY(DistanceUnit.CM);
    odo.update();
    }
    backleft.setPower(0);
    backright.setPower(0);
    frontleft.setPower(0);
    frontright.setPower(0);
  }
  
  private void drive_x(double dx){
    backleft.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    backright.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    frontleft.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    frontright.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
    odo.update();
    double odoY0 = odo.getPosY(DistanceUnit.CM);
    double odoX0 = odo.getPosX(DistanceUnit.CM);
    double Driving_Power=0.6;
    double odoX, odoY;
    int isign=1;
    if(dx<0)
    {
      isign=-1;
    }
    odoX=odo.getPosX(DistanceUnit.CM);
    while((odoX-odoX0)*isign<Math.abs(dx))
    {
    if(((odoX-odoX0)*isign/Math.abs(dx))>0.7 && Driving_Power>0.1)
    {
    Driving_Power=Driving_Power*0.5;
    }
    frontleft.setPower(-isign*Driving_Power);
    backleft.setPower(isign * Driving_Power);
    frontright.setPower(isign * Driving_Power);
    backright.setPower(-isign * Driving_Power);
    odo.update();
    odoX=odo.getPosX(DistanceUnit.CM);
    }
    backleft.setPower(0);
    backright.setPower(0);
    frontleft.setPower(0);
    frontright.setPower(0);
  }
  
  private void turn_around(double angle_deg){
    int isign=1;
    double Driving_Power=0.5;
    odo.update();
    double rotz0=odo.getHeading(AngleUnit.DEGREES);
    double rotz=odo.getHeading(AngleUnit.DEGREES);
    if(angle_deg<0)
    {
      isign=-1;
    }
    while(Math.abs(rotz-rotz0)<Math.abs(angle_deg))
    {
      if(Math.abs((rotz-rotz0)/angle_deg)>0.8)
      {
        Driving_Power=Driving_Power*0.8;
      }
      frontleft.setPower(isign * Driving_Power);
      backleft.setPower(isign * Driving_Power);
      frontright.setPower(-isign * Driving_Power);
      backright.setPower(-isign * Driving_Power);
      odo.update();
      rotz=odo.getHeading(AngleUnit.DEGREES);
      //telemetry.addData("rotation rotz", rotz);
      //telemetry.update();
    }
    //sleep(duration);
    backleft.setPower(0);
    backright.setPower(0);
    frontleft.setPower(0);
    frontright.setPower(0);
  }
  private void drive_around_2(double x, double y, double rx, double Driving_Power, int duration) {
    double Denominator = JavaUtil.maxOfList(JavaUtil.createListWith(JavaUtil.sumOfList(JavaUtil.createListWith(Math.abs(y), Math.abs(x), Math.abs(rx))), 1));
    frontleft.setPower(((y + x + rx) / Denominator) * Driving_Power);
    backleft.setPower((((y - x) + rx) / Denominator) * Driving_Power);
    frontright.setPower((((y - x) - rx) / Denominator) * Driving_Power);
    backright.setPower((((y + x) - rx) / Denominator) * Driving_Power);
    sleep(duration);
    backleft.setPower(0);
    backright.setPower(0);
    frontleft.setPower(0);
    frontright.setPower(0);
  }
  
  private void launch_boom(double launcher_power) {
    launcher.setPower(launcher_power);
    sleep(4000);
    for (int count = 0; count < 3; count++) {
      launcher.setPower(launcher_power);
      leftguider.setPower(-0.5);
      rightguider.setPower(-0.5);
      _2ndintake.setPower(1);
      intakeone.setPower(-1);
      sleep(300);
      launcher.setPower(launcher_power);
      leftguider.setPower(0);
      rightguider.setPower(0);
      _2ndintake.setPower(0);
      intakeone.setPower(0);
      sleep(1000);
    }
  }
}
