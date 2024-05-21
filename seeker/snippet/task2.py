#date: 2024-05-21T16:54:53Z
#url: https://api.github.com/gists/c40da96dfc44b5e1c283a736f1d6ec1d
#owner: https://api.github.com/users/Egorrrad

def calculate_bengal_fire_hours(start_fires: int, reused: int) -> int:
    """

    :param start_fires: количество огней всего
    :param reused: из какого количества сгоревших огней можно сделать 2 новых
    :return:
    """
    total_fires = start_fires
    total_hours = total_fires * 2
    used_fires = total_fires
    while used_fires >= reused:
        can_reused = used_fires // reused
        used_fires = used_fires % reused

        new_fires = can_reused * 2
        total_hours += new_fires * 2

        used_fires += new_fires
    return total_hours

