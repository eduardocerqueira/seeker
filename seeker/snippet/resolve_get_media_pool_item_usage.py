#date: 2024-02-28T16:51:45Z
#url: https://api.github.com/gists/3d3e254cfa07cad1c6869c4807f8d80c
#owner: https://api.github.com/users/BigRoy


def find_clip_usage(media_pool_item, project):
    """Return all Timeline Items in the project using the Media Pool Item.

    Each entry in the list is a tuple of Timeline and TimelineItem so that
    it's easy to know which Timeline the TimelineItem belongs to.
    
    Arguments:
        media_pool_item (MediaPoolItem): The Media Pool Item to search for.
        project (Project): The resolve project the media pool item resides in.

    Returns:
        List[Tuple[Timeline, TimelineItem]]: A 2-tuple of a timeline with
            the timeline item.

    """
    usage = int(media_pool_item.GetClipProperty("Usage"))
    if not usage:
        return []

    matching_items = []
    unique_id = media_pool_item.GetUniqueId()
    for timeline_idx in range(project.GetTimelineCount()):
        timeline = project.GetTimelineByIndex(timeline_idx+1)

        # Consider audio and video tracks
        for track_type in ["video", "audio"]:
            for track_idx in range(timeline.GetTrackCount(track_type)):
                timeline_items = timeline.GetItemListInTrack(track_type,
                                                             track_idx+1)
                for timeline_item in timeline_items:
                    timeline_item_mpi = timeline_item.GetMediaPoolItem()
                    if not timeline_item_mpi:
                        continue

                    if timeline_item_mpi.GetUniqueId() == unique_id:
                        matching_items.append((timeline, timeline_item))
                        usage -= 1
                        if usage <= 0:
                            # If there should be no usage left after this found
                            # entry we return early
                            return matching_items

    return matching_items