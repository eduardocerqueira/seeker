#date: 2025-08-18T17:03:26Z
#url: https://api.github.com/gists/8023b949f7b6c6df54a6ad894946f9bf
#owner: https://api.github.com/users/myociss

def get_flash_params(flashes: array_type, start_time: datetime.datetime) -> List[Dict]:
    flash_params: List[Dict] = []

    north_pole = np.array([90.0,0.0,0.0])

    for flash in flashes:
        grid_points = np.unique(flash[:,5:7], axis=0).astype('int64')
        n_sources = len(flash)
        duration = flash[-1,0] - flash[0,0]
        mean_power = np.mean(flash[:,4])

        # 2d hull area: see https://github.com/deeplycloudy/lmatools/blob/8d55e11dfbbe040f58f9a393f83e33e2a4b84b4c/lmatools/flashsort/flash_stats.py#L112
        init_coords = flash[:,7:]
        mean_point = np.mean(init_coords, axis=0)
        init_coords -= (mean_point + north_pole)
        coords_ecef = to_ecef(init_coords)
        hull = ConvexHull(coords_ecef)
        
        dist_from_center = np.linalg.norm(flash[0,1:3]) / 1000.0 # km

        init_time = flash[0,0]
        seconds_from_start = init_time - (start_time.hour * 3600 + start_time.minute * 60 + start_time.second)
        init_datetime = start_time + datetime.timedelta(seconds=seconds_from_start)

        flash_params.append({'n_sources': n_sources, 'duration': duration, 'mean_power': mean_power, 'grid_points': grid_points.tolist(),
            'init_alt': flash[0,9], 'init_time': init_datetime.strftime("%m/%d/%y %H:%M:%S"), 'hull_area': hull.volume*1e-6, 
            'dist_from_center': dist_from_center})

    return flash_params