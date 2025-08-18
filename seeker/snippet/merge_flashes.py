#date: 2025-08-18T16:55:43Z
#url: https://api.github.com/gists/748284fb13855f151fae93fc3ccea51e
#owner: https://api.github.com/users/myociss

def merge_flashes(flashes: List[array_type], t_threshold: float=0.15, xyz_threshold: float=3000.0) -> List[array_type]:
    flashes_sorted = sorted(flashes, key=lambda x: x[0,0])

    flash_merges = np.zeros((len(flashes), 2)) - 1.0

    for j in range(1, len(flashes_sorted)):
        branch_flash = flashes_sorted[j]

        for i in range(j):
            base_flash = flashes_sorted[i]

            if branch_flash[0,0] - base_flash[-1,0] > t_threshold:
                continue

            dists: np.ndarray[float, np.dtype[np.float64]] = np.linalg.norm(base_flash[:,1:4] - branch_flash[0, 1:4], axis=1)

            if np.any(dists <= xyz_threshold):
                min_dist = np.min(dists)

                if flash_merges[j,1] == -1.0 or flash_merges[j,1] > min_dist:
                    merge_idx = int(flash_merges[i,0]) if flash_merges[i,0] > -1.0 else i
                    flash_merges[merge_idx,:] = np.array([merge_idx, min_dist])

    flashes_merged: List[array_type | None] = [elem for elem in flashes_sorted]

    for idx, flash in enumerate(flashes_sorted):
        merge_idx = int(flash_merges[idx, 0])

        if merge_idx > -1:
            merged_sources = np.concatenate((flash, flashes_sorted[merge_idx]))
            flashes_merged[merge_idx] = np.sort(merged_sources, axis=0)
            flashes_merged[idx] = None

    flashes_merged_remove_none: List[array_type] = [elem for elem in flashes_merged if elem is not None]

    return flashes_merged_remove_none