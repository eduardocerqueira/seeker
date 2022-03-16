#date: 2022-03-16T17:02:20Z
#url: https://api.github.com/gists/a5dcbb244c91bcfdedababf3ee652609
#owner: https://api.github.com/users/McBaws

"""
I do not provide support for this unless its an actual error in the code and not related to your setup.

You'll need:
- Vapoursynth (this was written & tested on R53 with Python 3.9.6)
- pip3 install pathlib anitopy pyperclip requests requests_toolbelt
- https://github.com/HolyWu/L-SMASH-Works/releases/latest/ (Install to your usual Vapoursynth plugins64 folder)
- (if using ffmpeg) ffmpeg installed & in path

How to use:
- Drop comp.py into a folder with the video files you want
- (Recommended) Rename your files to have the typical [Group] Show - EP.mkv naming, since the script will try to parse the group and show name.
e.g. [JPBD] Youjo Senki - 01.m2ts ("fakegroup" for JPBDs so the imagefiles will have JPBD in the name)
- Change vars below (trim if needed)
- Run comp.py
"""

### Change these if you need to

# Ram Limit (in MB)
ram_limit = 6000

# Framecounts
frame_count_dark = 10
frame_count_bright = 5

# Automatically upload to slow.pics
slowpics = True

# Output slow.pics link to discord webhook (disabled if empty)
webhook_url = r""

# Upscale videos to make the clips match the highest found res
upscale = True
# Scales each screenshot to be at a higher, common resolution divisible by each video's aspect ratio
gcd_upscale = False

# Use ffmpeg as the image renderer (ffmpeg needs to be in path)
ffmpeg = True

# Choose your own frames to export
user_frames = []


"""
Used to trim clips.

Example:
trim_dict = {0: 1000, 1: 1046}

Means:
First clip should start at frame 1000
Second clip should start at frame 1046
(Clips are taken in alphabetical order of the filenames)
"""
trim_dict = {}

### Not recommended to change stuff below

import os
import time
import math
import ctypes
import random
import pathlib
import requests
import anitopy as ani
import pyperclip as pc
import vapoursynth as vs
from requests import Session
from functools import partial
from threading import Condition
from concurrent.futures import Future
from requests_toolbelt import MultipartEncoder
from typing import Any, Dict, List, Optional, BinaryIO, TextIO, Union, Callable, Type, TypeVar, Sequence, cast
RenderCallback = Callable[[int, vs.VideoFrame], None]
VideoProp = Union[int, Sequence[int],float, Sequence[float],str, Sequence[str],vs.VideoNode, Sequence[vs.VideoNode],vs.VideoFrame, Sequence[vs.VideoFrame],Callable[..., Any], Sequence[Callable[..., Any]]]
T = TypeVar("T", bound=VideoProp)
vs.core.max_cache_size = ram_limit

def lazylist(clip: vs.VideoNode, dark_frames: int = 8, light_frames: int = 4, seed: int = 20202020, diff_thr: int = 15):
    """
    Blame Sea for what this shits out

    A function for generating a list of frames for comparison purposes.
    Works by running `core.std.PlaneStats()` on the input clip,
    iterating over all frames, and sorting all frames into 2 lists
    based on the PlaneStatsAverage value of the frame.
    Randomly picks frames from both lists, 8 from `dark` and 4
    from `light` by default.
    :param clip:          Input clip
    :param dark_frame:    Number of dark frames
    :param light_frame:   Number of light frames
    :param seed:          seed for `random.sample()`
    :param diff_thr:      Minimum distance between each frames (In seconds)
    :return:              List of dark and light frames
    """

    dark = []
    light = []

    def checkclip(n, f, clip):

        avg = f.props["PlaneStatsAverage"]

        if 0.062746 <= avg <= 0.380000:
            dark.append(n)

        elif 0.450000 <= avg <= 0.800000:
            light.append(n)

        return clip

    s_clip = clip.std.PlaneStats()

    eval_frames = vs.core.std.FrameEval(
        clip, partial(checkclip, clip=s_clip), prop_src=s_clip
    )
    print('Rendering clip to get frames...')
    clip_async_render(eval_frames)

    dark.sort()
    light.sort()

    dark_dedupe = [dark[0]]
    light_dedupe = [light[0]]

    thr = round(clip.fps_num / clip.fps_den * diff_thr)
    lastvald = dark[0]
    lastvall = light[0]

    for i in range(1, len(dark)):

        checklist = dark[0:i]
        x = dark[i]

        for y in checklist:
            if x >= y + thr and x >= lastvald + thr:
                dark_dedupe.append(x)
                lastvald = x
                break

    for i in range(1, len(light)):

        checklist = light[0:i]
        x = light[i]

        for y in checklist:
            if x >= y + thr and x >= lastvall + thr:
                light_dedupe.append(x)
                lastvall = x
                break

    if len(dark_dedupe) > dark_frames:
        random.seed(seed)
        dark_dedupe = random.sample(dark_dedupe, dark_frames)

    if len(light_dedupe) > light_frames:
        random.seed(seed)
        light_dedupe = random.sample(light_dedupe, light_frames)

    return dark_dedupe + light_dedupe

def _get_slowpics_header(content_length: str, content_type: str, sess: Session) -> Dict[str, str]:
    """
    Stolen from vardefunc
    """
    return {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.5",
        "Content-Length": content_length,
        "Content-Type": content_type,
        "Origin": "https://slow.pics/",
        "Referer": "https://slow.pics/comparison",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "X-XSRF-TOKEN": sess.cookies.get_dict()["XSRF-TOKEN"]
    }

class RenderContext:
    """
    Stolen from lvsfunc

    Contains info on the current render operation.
    """
    clip: vs.VideoNode
    queued: int
    frames: Dict[int, vs.VideoFrame]
    frames_rendered: int
    timecodes: List[float]
    condition: Condition

    def __init__(self, clip: vs.VideoNode, queued: int) -> None:
        self.clip = clip
        self.queued = queued
        self.frames = {}
        self.frames_rendered = 0
        self.timecodes = [0.0]
        self.condition = Condition()

def get_prop(frame: vs.VideoFrame, key: str, t: Type[T]) -> T:
    """
    Stolen from lvsfunc

    Gets FrameProp ``prop`` from frame ``frame`` with expected type ``t``
    to satisfy the type checker.

    :param frame:   Frame containing props
    :param key:     Prop to get
    :param t:       Type of prop

    :return:        frame.prop[key]
    """
    try:
        prop = frame.props[key]
    except KeyError:
        raise KeyError(f"get_prop: 'Key {key} not present in props'")

    if not isinstance(prop, t):
        raise ValueError(f"get_prop: 'Key {key} did not contain expected type: Expected {t} got {type(prop)}'")

    return prop

def clip_async_render(clip: vs.VideoNode,
                      outfile: Optional[BinaryIO] = None,
                      timecodes: Optional[TextIO] = None,
                      progress: Optional[str] = "Rendering clip...",
                      callback: Union[RenderCallback, List[RenderCallback], None] = None) -> List[float]:
    """
    Stolen from lvsfunc

    Render a clip by requesting frames asynchronously using clip.get_frame_async,
    providing for callback with frame number and frame object.

    This is mostly a re-implementation of VideoNode.output, but a little bit slower since it's pure python.
    You only really need this when you want to render a clip while operating on each frame in order
    or you want timecodes without using vspipe.

    :param clip:      Clip to render.
    :param outfile:   Y4MPEG render output BinaryIO handle. If None, no Y4M output is performed.
                      Use ``sys.stdout.buffer`` for stdout. (Default: None)
    :param timecodes: Timecode v2 file TextIO handle. If None, timecodes will not be written.
    :param progress:  String to use for render progress display.
                      If empty or ``None``, no progress display.
    :param callback:  Single or list of callbacks to be preformed. The callbacks are called
                      when each sequential frame is output, not when each frame is done.
                      Must have signature ``Callable[[int, vs.VideoNode], None]``
                      See :py:func:`lvsfunc.comparison.diff` for a use case (Default: None).

    :return:          List of timecodes from rendered clip.
    """
    cbl = [] if callback is None else callback if isinstance(callback, list) else [callback]

    ctx = RenderContext(clip, vs.core.num_threads)

    bad_timecodes: bool = False

    def cb(f: Future[vs.VideoFrame], n: int) -> None:
        ctx.frames[n] = f.result()
        nn = ctx.queued

        while ctx.frames_rendered in ctx.frames:
            nonlocal timecodes
            nonlocal bad_timecodes

            frame = ctx.frames[ctx.frames_rendered]
            # if a frame is missing timing info, clear timecodes because they're worthless
            if ("_DurationNum" not in frame.props or "_DurationDen" not in frame.props) and not bad_timecodes:
                bad_timecodes = True
                if timecodes:
                    timecodes.seek(0)
                    timecodes.truncate()
                    timecodes = None
                ctx.timecodes = []
                print("clip_async_render: frame missing duration information, discarding timecodes")
            elif not bad_timecodes:
                ctx.timecodes.append(ctx.timecodes[-1]
                                     + get_prop(frame, "_DurationNum", int)
                                     / get_prop(frame, "_DurationDen", int))
            finish_frame(outfile, timecodes, ctx)
            [cb(ctx.frames_rendered, ctx.frames[ctx.frames_rendered]) for cb in cbl]
            del ctx.frames[ctx.frames_rendered]  # tfw no infinite memory
            ctx.frames_rendered += 1

        # enqueue a new frame
        if nn < clip.num_frames:
            ctx.queued += 1
            cbp = partial(cb, n=nn)
            clip.get_frame_async(nn).add_done_callback(cbp)  # type: ignore

        ctx.condition.acquire()
        ctx.condition.notify()
        ctx.condition.release()

    if outfile:
        if clip.format is None:
            raise ValueError("clip_async_render: 'Cannot render a variable format clip to y4m!'")
        if clip.format.color_family not in (vs.YUV, vs.GRAY):
            raise ValueError("clip_async_render: 'Can only render YUV and GRAY clips to y4m!'")
        if clip.format.color_family == vs.GRAY:
            y4mformat = "mono"
        else:
            ss = (clip.format.subsampling_w, clip.format.subsampling_h)
            if ss == (1, 1):
                y4mformat = "420"
            elif ss == (1, 0):
                y4mformat = "422"
            elif ss == (0, 0):
                y4mformat = "444"
            elif ss == (2, 2):
                y4mformat = "410"
            elif ss == (2, 0):
                y4mformat = "411"
            elif ss == (0, 1):
                y4mformat = "440"
            else:
                raise ValueError("clip_async_render: 'What have you done'")

        y4mformat = f"{y4mformat}p{clip.format.bits_per_sample}" if clip.format.bits_per_sample > 8 else y4mformat

        header = f"YUV4MPEG2 C{y4mformat} W{clip.width} H{clip.height} " \
            f"F{clip.fps.numerator}:{clip.fps.denominator} Ip A0:0\n"
        outfile.write(header.encode("utf-8"))

    if timecodes:
        timecodes.write("# timestamp format v2\n")

    ctx.condition.acquire()

    # seed threads
    try:
        for n in range(min(clip.num_frames, vs.core.num_threads)):
            cbp = partial(cb, n=n)  # lambda won't bind the int immediately
            clip.get_frame_async(n).add_done_callback(cbp)  # type: ignore

        while ctx.frames_rendered != clip.num_frames:
            ctx.condition.wait()
    finally:
        return ctx.timecodes  # might as well

def finish_frame(outfile: Optional[BinaryIO], timecodes: Optional[TextIO], ctx: RenderContext) -> None:
    """
    Stolen from lvsfunc

    Output a frame.

    :param outfile:   Output IO handle for Y4MPEG
    :param timecodes: Output IO handle for timecodesv2
    :param ctx:       Rendering context
    """
    if timecodes:
        timecodes.write(f"{round(ctx.timecodes[ctx.frames_rendered]*1000):d}\n")
    if outfile is None:
        return

    f: vs.VideoFrame = ctx.frames[ctx.frames_rendered]

    outfile.write("FRAME\n".encode("utf-8"))

    for i, p in enumerate(f.planes()):
        if f.get_stride(i) != p.width * f.format.bytes_per_sample:
            outfile.write(bytes(p))  # type: ignore
        else:
            outfile.write(p)  # type: ignore

def screengen(
    clip: vs.VideoNode,
    folder: str,
    suffix: str,
    frame_numbers: List = None,
    start: int = 1):
    """
    Stoled from Sea

    Mod of Narkyy's screenshot generator, stolen from awsmfunc.
    Generates screenshots from a list of frames.
    Not specifying `frame_numbers` will use `ssfunc.util.lazylist()` to generate a list of frames.
    :param folder:            Name of folder where screenshots are saved.
    :param suffix:            Name prepended to screenshots (usually group name).
    :param frame_numbers:     List of frames. Either a list or an external file.
    :param start:             Frame to start from.
    :param delim:             Delimiter for the external file.
    > Usage: ScreenGen(src, "Screenshots", "a")
             ScreenGen(enc, "Screenshots", "b")
    """

    folder_path = "./{name}".format(name=folder)


    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    for i, num in enumerate(frame_numbers, start=start):
        filename = "{path}/{:03d} - {suffix}.png".format(
            i, path=folder_path, suffix=suffix
        )

        matrix = clip.get_frame(0).props._Matrix

        if matrix == 2:
            matrix = 1

        print(f"Saving Frame {i}/{len(frame_numbers)} from {suffix}", end="\r")
        vs.core.imwri.Write(
            clip.resize.Spline36(
                format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion"
            ),
            "PNG",
            filename,
            overwrite=True,
        ).get_frame(num)

def get_highest_res(files: List[str]) -> int:
    height = 0
    width = 0
    filenum = -1
    for f in files:
        filenum+=1
        video = vs.core.lsmas.LWLibavSource(f)
        if height < video.height:
            height = video.height
            width = video.width
            max_res_file = filenum

    return width, height, max_res_file

def get_ar(files: List[str]) -> int:
    ar_w = []
    ar_h = []
    unitary_ar = []
    for f in files:
        video = vs.core.lsmas.LWLibavSource(f)
        gcd = math.gcd(video.width, video.height)
        ar_w.append(video.width / gcd)
        ar_h.append(video.height / gcd)
        unitary_ar.append(video.width / video.height)

    return ar_w, ar_h, unitary_ar

def get_frames(clip: vs.VideoNode, frames: List[int]) -> vs.VideoNode:
    out = clip[frames[0]]
    for i in frames[1:]:
        out += clip[i]
    return out

def actual_script():
    files = sorted([f for f in os.listdir('.') if f.endswith('.mkv') or f.endswith('.m2ts') or f.endswith('.mp4')])

    if len(files) < 2:
        print("Not enough video files found.")
        input()
        exit

    print('Files found: ')
    for f in files:
        if trim_dict.get(files.index(f)) is not None:
            print(f + f" (Will be trimmed to start at frame {trim_dict.get(files.index(f))})")
        else:
            print(f)

    print('\n')

    dict = ani.parse(files[0])
    collection_name = dict.get('anime_title') if dict.get('anime_title') is not None else dict.get('episode_title')

    first = vs.core.lsmas.LWLibavSource(files[0])
    if trim_dict.get(0) is not None:
        first = first[trim_dict.get(0):]


    if upscale and not gcd_upscale:
        max_width, max_height, max_res_file = get_highest_res(files)
        #MAX_WIDTH IS NOT THE LARGEST WIDTH, IT IS THE WIDTH OF THE VIDEO WITH THE LARGEST HEIGHT
                
    gcd_upscale_needed = False
    if upscale and gcd_upscale:
        max_width, max_height, max_res_file = get_highest_res(files)
        ar_w, ar_h, unitary_ar = get_ar(files)
        for file in files:
            clip = vs.core.lsmas.LWLibavSource(file)
            if clip.height != max_height and int(clip.width * (max_height / clip.height)) != clip.width * (max_height / clip.height):
                gcd_upscale_needed = True

        if gcd_upscale_needed:
            cur_width = max_width
            cur_height = max_height
            width_add = ar_w[max_res_file]
            height_add = ar_h[max_res_file]
            max_res_ar = unitary_ar[max_res_file]
            upscale_res_found = False

            while not upscale_res_found:
                cur_width+=width_add
                cur_height+=height_add
                #print("Testing:", cur_width, "x", cur_height)
                filenum = -1
                fail = False
                for file in files:
                    filenum+=1
                    if unitary_ar[filenum] > max_res_ar:
                        if cur_width / ar_w[filenum] != int(cur_width / ar_w[filenum]):
                            fail = True
                            break
                    if unitary_ar[filenum] < max_res_ar:
                        if cur_height / ar_h[filenum] != int(cur_height / ar_h[filenum]):
                            fail = True
                            break
                    if unitary_ar[filenum] == max_res_ar:
                        if cur_height / ar_h[filenum] != int(cur_height / ar_h[filenum]) or cur_width / ar_w[filenum] != int(cur_width / ar_w[filenum]):
                            fail = True
                            break
                if not fail:
                    upscale_w, upscale_h = cur_width, cur_height
                    upscale_res_found = True
                    print(f"Scaled resolution: {int(upscale_w)}x{int(upscale_h)}\n")

    if user_frames is None:
        frames = lazylist(first, frame_count_dark, frame_count_bright)
    else:
        frames = user_frames
    print(frames)
    print("\n")

    filenum = -1

    for file in files:
        filenum+=1
        dict = ani.parse(file)
        suffix = ""
        if dict.get('release_group') is not None:
            suffix = str(dict.get('release_group')).replace("[\\/:\"*?<>|]+", "")
        if not suffix:
            suffix = file.replace("[\\/:\"*?<>|]+", "")
    
        print(f"Generating screens for {file}")
        clip = vs.core.lsmas.LWLibavSource(file)
        index = files.index(file)
        if trim_dict.get(index) is not None:
            clip = clip[trim_dict.get(index):]

                    
        if ffmpeg:
            import subprocess
            matrix = clip.get_frame(0).props._Matrix
            if matrix == 2:
                matrix = 1

            if upscale and not gcd_upscale_needed and clip.height != max_height:
                if int(clip.width * (max_height / clip.height)) == clip.width * (max_height / clip.height):
                    clip = vs.core.resize.Spline36(clip, clip.width * (max_height / clip.height), max_height, format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion")
                else:
                    if clip.width/clip.height > max_width/max_height:
                        clip = vs.core.resize.Spline36(clip, max_width, int(round(clip.height * (max_width/clip.width), 0)), format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion")
                    else:
                        clip = vs.core.resize.Spline36(clip, int(round(clip.width * (max_height/clip.height), 0)), max_height, format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion")

            elif upscale and gcd_upscale and gcd_upscale_needed:
                if unitary_ar[filenum] > max_res_ar:
                    clip = vs.core.resize.Spline36(clip, upscale_w, upscale_w / unitary_ar[filenum], format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion")
                if unitary_ar[filenum] < max_res_ar:
                    clip = vs.core.resize.Spline36(clip, upscale_h * unitary_ar[filenum], upscale_h, format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion")
                if unitary_ar[filenum] == max_res_ar:
                    clip = vs.core.resize.Spline36(clip, upscale_w, upscale_h, format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion")

            else:
                clip = clip.resize.Spline36(format=vs.RGB24, matrix_in=matrix, dither_type="error_diffusion")

            clip = get_frames(clip, frames)
            clip = clip.std.ShufflePlanes([1, 2, 0], vs.RGB).std.AssumeFPS(fpsnum=1, fpsden=1)

            if not os.path.isdir("./screens"):
                os.mkdir("./screens")
            path_images = [
                "{path}/{:03d} - {suffix}.png".format(frames.index(f) + 1, path="./screens", suffix=suffix)
                for f in frames
            ]

            print(path_images)

            for i, path_image in enumerate(path_images):
                ffmpeg_line = f"ffmpeg -y -hide_banner -loglevel error -f rawvideo -video_size {clip.width}x{clip.height} -pixel_format gbrp -framerate {str(clip.fps)} -i pipe: -pred mixed -ss {i} -t 1 \"{path_image}\""
                try:
                    with subprocess.Popen(ffmpeg_line, stdin=subprocess.PIPE) as process:
                        clip.output(cast(BinaryIO, process.stdin), y4m=False)
                except:
                    None
        else:

            if upscale and not gcd_upscale_needed and clip.height != max_height:
                if int(clip.width * (max_height / clip.height)) == clip.width * (max_height / clip.height):
                    clip = vs.core.resize.Spline36(clip, clip.width * (max_height / clip.height), max_height)
                else:
                    if clip.width/clip.height > max_width/max_height:
                        clip = vs.core.resize.Spline36(clip, max_width, int(round(clip.height * (max_width/clip.width), 0)))
                    else:
                        clip = vs.core.resize.Spline36(clip, int(round(clip.width * (max_height/clip.height), 0)), max_height)

            if upscale and gcd_upscale and gcd_upscale_needed:
                if unitary_ar[filenum] > max_res_ar:
                    clip = vs.core.resize.Spline36(clip, upscale_w, upscale_w / unitary_ar[filenum])
                if unitary_ar[filenum] < max_res_ar:
                    clip = vs.core.resize.Spline36(clip, upscale_h * unitary_ar[filenum], upscale_h)
                if unitary_ar[filenum] < max_res_ar:
                    clip = vs.core.resize.Spline36(clip, upscale_w, upscale_h)

            screengen(clip, "screens", suffix, frames)
        
        print("\n")

    if slowpics:
        time.sleep(3)
        fields: Dict[str, Any] = {
            'collectionName': collection_name,
            'public': 'false',
            'optimize-images': 'true'
        }
        all_image_files = sorted([f for f in os.listdir('./screens/') if f.endswith('.png')])

        # Screen Number not dynamic yet so dont care about this being hardcoded
        for x in range(frame_count_dark + frame_count_bright):
            formatted_num = "{:03d}".format(x + 1)
            current_comp = [f for f in all_image_files if f.startswith(formatted_num + " - ")]
            print(f"Comp {formatted_num}: {current_comp}")
            fields[f'comparisons[{x}].name'] = formatted_num
            for imageName in current_comp:
                i = current_comp.index(imageName)
                image = pathlib.Path(f'./screens/{imageName}')
                fields[f'comparisons[{x}].images[{i}].name'] = image.name.split(' - ', 1)[1]
                fields[f'comparisons[{x}].images[{i}].file'] = (image.name, image.read_bytes(), 'image/png')
            
        sess = Session()
        sess.get('https://slow.pics/api/comparison')
        # TODO: yeet this
        files = MultipartEncoder(fields)

        print('\nUploading images...')
        url = sess.post(
            'https://slow.pics/api/comparison', data=files.to_string(),
            headers=_get_slowpics_header(str(files.len), files.content_type, sess)
        )
        sess.close()

        slowpics_url = f'https://slow.pics/c/{url.text}'
        print(f'Slowpics url: {slowpics_url}')
        pc.copy(slowpics_url)

        if webhook_url:
            data = {"content": slowpics_url}
            if requests.post(webhook_url, data).status_code < 300:
                print('Posted to webhook.')
            else:
                print('Failed to post on webhook!')

        time.sleep(3)

actual_script()