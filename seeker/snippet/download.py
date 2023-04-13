#date: 2023-04-13T17:03:52Z
#url: https://api.github.com/gists/12e7e5c4417049af22fbe7fcf8f01134
#owner: https://api.github.com/users/naderidev

import m3u8
import time
from urllib.parse import urljoin
from write import Writer


class StreamDownloader:

    __last_segment: m3u8.Segment = None

    def __init__(self, stream_url: str, file_path:str = 'video.ts'):
        self.stream_url = stream_url
        self.file_path = file_path

    # by debualt gets the first quality
    def download(self, quality_index: int = 0):
        qalities = m3u8.load(self.stream_url).playlists
        self.selected_qality_url = urljoin(
            self.stream_url, '.') + qalities[quality_index].uri

        writer = Writer(
            'local_writer',
            file_path=self.file_path,
            m3u8_base_url=urljoin(self.selected_qality_url, '.'),
            mode='+ab'
        )

        while True:

            # getting new m3u8
            new_segments = self.__get_segments()
          
            if new_segments:
                for segment in new_segments:
                    writer.write(segment)

            time.sleep(0.3)  # check time

    def __get_segments(self):
        current_m3u8 = m3u8.load(self.selected_qality_url)
        segments = []
        if self.__last_segment:
            for segment in current_m3u8.segments:
              
                # checks that the segment is not duplicated
                if segment.current_program_date_time.timestamp() > self.__last_segment.current_program_date_time.timestamp():
                    self.__last_segment = segment
                    segments.append(segment)
        else:
            segments = current_m3u8.segments
            self.__last_segment = segments[-1]

        return segments


StreamDownloader('<STREAM_URL>').download(1)

