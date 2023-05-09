#date: 2023-05-09T17:06:02Z
#url: https://api.github.com/gists/0c1184e0c4058c2f33f847361189b059
#owner: https://api.github.com/users/AbioticFactor

    delta_rot_1 = np.degrees( np.arctan2( ( cur_pose[1] - prev_pose[1] ), ( cur_pose[0] - prev_pose[0] ) ) ) - prev_pose[2]
    delta_trans = np.sqrt( ( cur_pose[0]-prev_pose[0] )**2 + ( cur_pose[1]-prev_pose[1] )**2 )
    delta_rot_2 = cur_pose[2] - prev_pose[2] - delta_rot_1