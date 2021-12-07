//date: 2021-12-07T17:16:36Z
//url: https://api.github.com/gists/289c070fef21a16fddd141d39ef17ab5
//owner: https://api.github.com/users/LinerSRT

private void updateBatteryStats(Intent intent) {

switch (intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)) {

case BatteryManager.BATTERY_STATUS_FULL:

batteryStatus.isFullyCharged = true;

break;

case BatteryManager.BATTERY_STATUS_CHARGING:

batteryStatus.isCharging = true;

break;

case BatteryManager.BATTERY_STATUS_DISCHARGING:

batteryStatus.isCharging = false;

break;

case BatteryManager.BATTERY_STATUS_NOT_CHARGING:

batteryStatus.isFullyCharged = false;

break;

case BatteryManager.BATTERY_STATUS_UNKNOWN:

default:

batteryStatus.isCharging = false;

batteryStatus.isFullyCharged = false;

break;

}

batteryStatus.level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);

batteryStatus.scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);

batteryStatus.percentage = (float) batteryStatus.level * 100 / batteryStatus.scale;

if (batteryStatus.lastPercentage != batteryStatus.percentage) {

if (batteryStatus.lastPercentage != -1) {

if (batteryStatus.isCharging) {

batteryStatus.remainChargeTime = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ?

batteryManager.computeChargeTimeRemaining() : computeChargeTimeRemaining();

} else {

batteryStatus.remainDischargeTime = computeDischargeTimeRemaining();

}

}

batteryStatus.lastPercentage = batteryStatus.percentage;

batteryStatus.lastPercentageTime = System.currentTimeMillis();

}

batteryStatus.temperature = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) / 10;

for(OnBatteryChangeCallback changeCallback:callbackList)

changeCallback.onChange(this, batteryStatus);

}