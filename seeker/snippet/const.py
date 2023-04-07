#date: 2023-04-07T16:43:01Z
#url: https://api.github.com/gists/5cba4b2276b9f4cd2fd427b43a8a87d5
#owner: https://api.github.com/users/zakir300408


PRE_STATIONARY, POST_STATIONARY, WAIT_PREDICTION, PREDICTION_DONE = range(300, 304)

IMU_LIST = {
    0: 'L_FOOT',
    1: 'R_FOOT',
    2: 'R_SHANK',
    3: 'R_THIGH',
    4: 'WAIST',
    5: 'L_SHANK',
    6: 'L_THIGH'
}
L_FOOT, R_FOOT, R_SHANK, R_THIGH, WAIST, L_SHANK, L_THIGH, = range(7)
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
ACC_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[:3]]
GYR_ALL = [field + '_' + sensor for sensor in IMU_LIST for field in IMU_FIELDS[3:]]

MAX_BUFFER_LEN = 152
GRAVITY = 9.81

WEIGHT_LOC, HEIGHT_LOC = range(2)