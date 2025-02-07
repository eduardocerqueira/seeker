#date: 2025-02-07T16:50:37Z
#url: https://api.github.com/gists/26e17636dd9e30373581af3f724895f9
#owner: https://api.github.com/users/SunsetMkt

#!/bin/bash

# ffav1.sh Copyright 2021-2023 William Barath w.barath@gmail.com
# see the embedded license (-license) and acknowledgements (-help)

#### initial state: ##########################################################

APP="ffav1.sh" AUTHOR="\"William Barath\" <w.barath@gmail.com>";
GITHUB_GISTS="https://gist.github.com/w-barath";
VERSION="2023.09.29" COPYRIGHT="Copyright 2021-2023 $AUTHOR";
CONFIG_DIR=~/.config/ffav1 SETTINGS="settings";
GIST="$GITHUB_GISTS/da1e87c811f2207c3662ba745d4fdd94";
DEPENDS="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz";
FFMPEG="ffmpeg" FFMPEG_MIN_VERSION="4.3"; NICE=19;
TAG="FFAV1" EXT="mkv" HERE=0 MAPS=() PRE=() OPTS1=() OPTS2=();
ACODEC="copy" ABRATE="64K" ASRATE="" CHANNELS=2 FILTER_A=();
VCODEC="libaom-av1" DEPTH="MAIN" PRESET="medium" CRF=26 VBRATE="800K";
FILTER_V=() PASS1=0 PASS2=0 REALTIME=0 MAIN=8 MAIN10=10;
ultrafast=9 superfast=8 veryfast=7 faster=6 fast=5 medium=4 slow=3 slower=2;
veryslow=1 placebo=0 DEBUG=0 OVERWRITE="ASK" CMD=() ARGV=("$@");
LICENSE_AGREED=no;

CONFIG_VARS=( # defines state variables to load/save with "presets"
    FFMPEG NICE TAG EXT HERE OVERWRITE MAPS OPTS1 OPTS2
    VCODEC DEPTH PRESET CRF VBRATE FILTER_V PASS1 PASS2 REALTIME
    ACODEC ABRATE ASRATE CHANNELS FILTER_A LICENSE_AGREED
);

#### support functions: ######################################################

changelog(){ # don't forget to update VERSION !
    less << EOF
$APP changelog: (by $AUTHOR except where noted)

2023.09.29 -nice, -wait, and -vmax options
2023.09.06 add channel layout options
2023.08.30 add libsvtav1 support, -gop and -fps, some documetnation fixes
2022.07.26 fixes for -10bit, -toon, -anime
2022.01.06 updates ffmpeg 5.x and libaom-av1 3.x
2021.04.06 various small usability tweaks and doc updates
2021.02.22 option/argument parser now accepts -opt arg and --opt=arg
2021.02.21 fastload option for muxing mp4 stream containers
2021.02.20 here option to put output in CWD.  Improve docs.
2021.02.19 cover and fit scaling, crop, ff, and fatal/quit/note 
2021.02.17 batch processing and say/die with text labels
2021.02.16 tone mapping and audio sample rate conversion
2021.02.04 license agreement
2021.02.03 named settings presets and changelog
2021.01.31 first release

$APP TODO / Wishlist:
    threading control options
    piping video through external encoders

    more libaom-av1 encoder thoughts: https://forum.doom9.org/showthread.php?t=183907

    automated splitting/joining to work around poor performance / thread scaling:
        ffmpeg -i input.mp4 -c copy -map 0 -segment_time 00:20:00 -f segment \
            -reset_timestamps 1 output%03d.mp4
        ffmpeg -i "concat:output001.mp4|output002.mp4|output003.mp4" -c copy output.mp4

    use hwaccel for input and output, support 10-bit input:
        https://trac.ffmpeg.org/wiki/Hardware/VAAPI
        ffmpeg -y -hwaccel vaapi -hwaccel_output_format vaapi -i in_1080p7.1.mkv \
            -c:v h264_vaapi -b:v 1800k -af "channelmap=channel_layout=5.1" \
            -c:a aac -b:a 360k -movflags faststart out_vaapi5.1.mp4

other static ffmpeg builds:
    # this version tracks all stable ffmpeg and dependency stable releases, but lacks vaapi
    https://hub.docker.com/r/mwader/static-ffmpeg/
    sudo docker pull mwader/static-ffmpeg
    docker run --rm -v "\$PWD:/out" \$(echo -e \
        'FROM alpine\nCOPY --from=mwader/static-ffmpeg:latest /ff* /\nENTRYPOINT cp /ff* /out' \
        | docker build -q -)
    docker run --rm mwader/static-ffmpeg -v quiet -f data -i versions.json \
        -map 0:0 -c text -f data -
EOF
    quit;
}

help(){
    check_license || {
        PROLOGUE="* Agree to the license before using $APP."\
            $'\n\n'"See $ $APP -license"$'\n\n';
    }
    less <<EOF
${PROLOGUE}Usage: $ $APP <input.ext> [option [arg]]... [-o output.ext] [-ff options]
       $ $APP [-batch glob] [option [arg]]...
-------------------------------------------------------------------------------
   Option     |    Arg    |       Description
-------------------------------------------------------------------------------
-h --help     |           | show help for $APP
-v -version   |           | output $APP version number ($VERSION)
-changelog    |           | show $APP changelog
-license      |           | show the license and prompt for agreement
-agree        |           | alternative method for providing license agreement
-debug        |           | show encoder commandline without executing
-install      |           | download and install this gist into ~/bin
-depends      |           | download and install ffmpeg into ~/bin
-------------------------------------------------------------------------------
 file options |    Arg    |       Description
-------------------------------------------------------------------------------
-title        | title     | set the file/stream title (default is output file name)
-tag          | name      | default is FFAV1
-e -ext       | mkv       | default is mkv
-fs -faststart|           | mp4 streaming container
-i -input     | in file   | input file if input was not passed as first argument
-o -output    | out file  | default is "input.TAG.EXT"
-b -batch     | glob      | encode all files matching glob
-ow -overwrite| [arg]     | =CLOBBER|ASK|SKIP behaviour for existing files
-no-ow        |           | same as -ow=SKIP
-here         |           | write output files to the current working directory
-save         | [preset]  | create a settings preset and exit - default "settings"
-load         | preset    | load a settings preset
-remove       | [preset]  | remove a settings preset and exit - default "settings"
-show         |           | show loaded settings preset and exit
-list         |           | show a list of all settings presets and exit
-------------------------------------------------------------------------------
stream options|    Arg    | options selecting video / audio / subtitle streams
-------------------------------------------------------------------------------
-map          | map       | add manual ffmpeg stream map
-v1           |           | select only first video stream
-a1           |           | select only first audio stream
-s1           |           | select only first subtitle stream
-va1          |           | select only first video and audio streams
-vas1         |           | select only first video, audio, and subtitle streams
-ss -start    | time      | start timestamp/seconds [[HH:]MM:]SS[.ms]
-to -end      | time      | end           ""               ""
-t  -duration | time      | duration time/seconds          ""
-------------------------------------------------------------------------------
video options |    Arg    | options affecting video compression / filtering
-------------------------------------------------------------------------------
-1  -pass1    |           | fast pass1 log generation
-2  -pass2    |           | slow pass2 video encoding/output
-12 -pass12   |           | run pass 1 and then pass 2
    -tonemap  | method    | convert to SDR with (none clip gamma =hable mobius) tonemap
-8  -8bit     |           | convert to SDR with mobius tonemap
-10 -10bit    |           | forces yuv420p10le and MAIN10 encoding (slow+quality)
-vc -vcodec   | codec     | =libaom-av1 video compression codec 
-vb -vbrate   | bitrate   | =800K video bitrate
-cq -crf      | CRF       | =26, =32 for -anime, =35 for -toon
-rt -realtime |           | real-time encoding for AV1 with libaom-av1 - not recommended
-p  -preset   | quality   | =4 aka medium 9=ultrafast, 1=veryslow
    -anime    |           | optimise for CGI/anime (Lion King/Nausicaä) (slow)
    -toon     |           | optimise for cel-shaded cartoons/games (Futurama) (slow)
-di -deint    | type      | =mcdeint,w3fdif,kerndeint,yadif,nnedi (nnedi=slow!!!)
-d  -decimate |           | automatically remove similar frames (make VFR)
-s  -scale    | x:y       | "1280:-1" for 720p or "iw/2:ih/2" for 1/4 scale
    -fit      | x:y       | scales to fit inside the given rectangle
    -cover    | x:y       | scales and crops to cover the given rectangle - dumb pan&scan
    -vmax     | pixels    | scale down if an axis is larger than pixels
    -crop     | w:h[:x:y] | crops the video to w:h, optionally from x:y top left
-dn -denoise  |           | hqdn3d makes noisy sources easier to compress
-db -deband   |           | deband gradients - may remove subtle texture
-dB -deblock  |           | deblock - fix heavy source compression artefacts
-um -unsharp  | x:y:amt   | =3:5:1 unsharp mask enhances edges / increases local contrast
-vf -vfilter  | filter(s) | add manual ffmpeg input video filters
-g -gop       | frames    | set GOP size - 150 to 300 for 30fps files, 60 for 30fps streaming
-fps          | fps       | framerate for the output - doesn't change duration / audio
-------------------------------------------------------------------------------
audio options |    Arg    | options affecting audio compression / filtering
-------------------------------------------------------------------------------
-ac  -acodec  | codec     | =copy audio compression codec
-ab  -abrate  | bitrate   | =64k audio bitrate (libopus bit better than fm radio quality)
-ar  -asrate  | hz        | audio sample rate in hz, ie 32k, 44100, 48k
-af  -afilter | filter(s) | add manual ffmpeg input audio filters
-8ch -7.1     |           | use 7.1 channel layout
-6ch -5.1     |           | use 5.1 channel layout
-5ch -4.1     |           | use 4.1 channel layout
-qch -quad    |           | use quadraphonic FL+FR+BL+BR layout
-4ch -4.0     |           | use 4.0 FL+FR+FC+BC layout
-3ch -2.1     |           | use 2.1 layout
-2ch -stereo  |           | use stereo layout - discards LFE
-1ch -mono    |           | downmix to mono, fixes phase, halves given audio bitrate
-0ch -na      |           | no audio in output
-------------------------------------------------------------------------------
 misc options |    Arg    | other options
-------------------------------------------------------------------------------
-n   -nice    | niceness  | =19 set niceness in 0..19
-w   -wait    | pid       | wait for pid to exit before running
-ff=          | "options" | ffmpeg options * for all passes
-ff1=         | "options" | ffmpeg options * for pass 1
-ff2=         | "options" | ffmpeg options * for pass 2
                            * NOTE: MUST use -ff="..." arg form; -ff "..." will not work.
-------------------------------------------------------------------------------


$APP acceps -option, -option arg, -option=arg as well as --option variants
    if arg begins with '-' then you MUST use -option=-arg
    if arg contains spaces, try -option "a b c" or -option="a b c"

examples:

$APP -license                   # read and choose whether to agree to the license
$APP <input>                    # converts to VBR AV1 and copies audio to input.TAG.EXT
$APP <input> -ff="args ..."     # pass arbitrary arguments to ffmpeg
$APP <input> -vc libsvtav1 -ff="-g 300"     # use SVT_AV1 and change GOP size from 60 to 300 frames 
$APP <input> -crf 26 -vr 1200k  # encode quality between CRF 26 and 1200kbps video 
$APP <input> -preset medium     # use the well-known libx264 quality preset names
$APP <input> -2ch -ac libopus   # converts to VBR AV1 and opus stereo audio
$APP <input> -scale 960:-1      # scales down to 960px wide, maintaining aspect ratio
$APP <input> -toon              # clean up and convert a cartoon to AV1 video
$APP <input> -10                # force MAIN10 coding of 8-bit source (NOT HDR!)
$APP <input> -8                 # force SDR via "none" tonemap - clamps brightness
$APP <input> -tonemap mobius:40 # force SDR via "mobius" - 30=moody ... 70=specular
$APP <input> -tonemap hable     # force SDR via "hable" - good detail, wacky colour  
$APP <input> -tonemap gamma     # force SDR via "gamma" - poor contrast and saturation
$APP <input> -va1               # select the first audio and video streams only
$APP <input> -pass12            # perform (both) 2-pass encoding
$APP <input> -tag MINE          # output file will will end in MINE.EXT
$APP <input> -ext webm          # converts into webm container
$APP <input> -faststart         # sets .mp4 extension and adds streaming support 
$APP -pass12 -save 2pass        # create / replace "2pass" settings preset
$APP -load 2pass -show          # load and show "2pass" settings preset
$APP <input> -load 2pass        # load saved settings "2pass" and encode the file
$APP -tag MINE -save            # save default settings preset
$APP -list                      # show all saved settings presets
$APP -remove                    # remove default settings preset
$APP -remove 2pass              # remove saved settings "2pass"

# Batch operation creating max 960px size copies in '\$QUEUE\done' folder:
$APP -no-ow -here -nice 19 -vc libsvtav1 -crf 35 -vb 600k -vmax 960 -ac libopus -ab 64k -save batch;
cd \$QUEUE; mkdir done; cd done;
$APP -b "../*.mkv" -load load batch;
# having added more files to the queue directory, a new batch waiting for the previous to exit:
$APP -w PID -b "../*.mkv" -load load batch

-------------------------------------------------------------------------------


Notes:

This script is not aimed at professional users.  It's aimed at people who want
to create lightweight copies of their videos to save disk space for archival
purposes, or to create copies for streaming from their own servers.  


Video codec hints:

AV1/opus works well with 2016+ iPad models up to 1080p with VLC, and most
Android TV models. It also plays great in Firefox and Chrome in .webm containers,
including HDR support. Safari support is not there yet in 2023. Use tiles if you
experience stuttering.  Generally that only happens with HDR10 content at >768p

libsvtav1 defaults to a 30-frame GOP.  That allows 1s seeking in streaming media
but it's pretty inefficient for local files where a 5-10s seek is fine.  So for
much better compression use -gop=[150..300] for 30fps content etc.

When encoding small files using 2 passes, libsvtav1 seems not to write a pass 1
log for small input files.  libaom-av1 works fine for these small files.


Video filter hints:

Order of filtering steps is important.  Deblocking relies on block structure,
so scaling or deinterlacing before deblocking will prevent deblocking from
working properly.  Deinterlacing works best with the original field data, so
it should come before scaling.  So deblocking, then deinterlacing, scaling,
decimation, then any other filters.

An exception is interlaced content in a progressive bitstream. This is called
combing - NOT interlacing. In this case use a de-combing filter before all other
filters.  Sometimes the combing will be scaled and de-combing will cause jello 
artefacts. Try to find an un-scaled source as the scaling damage most likely
can't be undone.

Scaling should happen before decimation because the decimation filter makes
decisions based on features of potential output frames, and because there's no
point to scaling frames that won't be output.

Tonemapping (conversion to SDR) optimises work done in later steps so it's often
advantageous to do before other filters for performance reasons.  Debanding,
unsharp mask, and denoising will all behave differently when applied before
tonemapping and scaling.  Quality may be higher (and slower) but their relative
strength and the selectiveness of their effects will vary. For example debanding
loses much less detail when used ahead of tonemapping.


Audio hints:

If you want to preserve 5.1 / 7.1 audio quality, don't supply an audio codec.
Changing the codec will always lower quality.  If you must change the codec and
you want better than stereo (the default) then you must supply the desired
channel layout, ie "-ac libopus -5.1 -ab 360k" otherwise stereo is the default.


Batch encoding hints:

When processing batches the glob must be quoted to avoid shell expansion.  For
example, encode all mp4 and mkv files by using -batch '*.{mp4,mkv}'.  The 
output files will be written to the same folder as the source files.  If you
want to write the processed output to a different folder ie 'ipad/' then try
$ cd ipad; $APP -batch "../*.{mp4,mkv}" -here [options...]


Parallel encoding hints:

With the -no-ow flag it's possible to run a batch in parallel to use more
cores for codecs that don't scale well to the available cores.  Run the batch
in multiple separate shells.  They will each skip over each other's work.

libsvtav1 scales well to 8 cores, so run the batch 4x in parallel for 32 cores

libaom-av1 is most efficient with threading disabled, using 2 cores, unless
you use 2/4 tiles in which case it scales well to 4/6 cores, but that hurts
bd-rate for resolutions lower than 4k.

-------------------------------------------------------------------------------


Acknowledgements:

This script contains the combined wisdom I've gleaned from many searches on
FFMPEG's Trac server, Doom9, StackOverflow, Netflix, Meta, streaming blogs.

If you have more wisdom to add, see the CONTRIBUTING section.


Thanks are specially extended to:

Fabrice Bellard, FFMPEG maintainers:
    http://ffmpeg.org - video Swiss Army Knife which this script automates.

John Van Sickle:
    https://johnvansickle.com/ffmpeg - daily static ffmpeg builds

-------------------------------------------------------------------------------


INSTALLING:

$ $APP -install
$ $APP -depends
$ $APP -license

-------------------------------------------------------------------------------


CONTRIBUTING:

You can contribute in various ways.  You can provide bug reports, make
suggestions for improvements, contribute code, donate to FFMPEG, you can share
this script with other users, and you can help others use it better, or you can
just [like] the gist so I get a measure of the value of supporting it.


I accept bug reports in story form.  Word your report like so:

    Hi.  I wanted to <INSERT YOUR OBJECTIVES>, so I tried: 
    $APP [options]
    I expected <INSERT YOUR EXPECTED RESULT>, but instead I got 
    <INSERT YOUR ACTUAL RESULT>.
    Here's what I see: <INSERT PASTE / SCREENSHOT>.

Post your bug report as a comment here:

$APP: $GIST

Don't forget to [like] if you find this helpful.


If you find there's features you often access via -ff="...", consider leaving a
comment with the options you're passing.  If they're handy and/or popular
I will add them to the script so others may enjoy easy access to them.


I accept suggestions in a similar form to bug reports.  Tell me a story about
what you're trying to achieve, in what way you feel you're unable to get there,
and what you feel is reasonable for me to do to help you get what you need.


I accept patches if they follow the design rules, don't break existing usage,
and either fix a bug or add a feature that's in demand.


Project design rules and the thought processes behind them.

All state variable and their defaults are in the "initial state" section. The
state variables are logically grouped.  Locality of reference saves a lot of
work when troubleshooting and re-familiarising oneself, so you'll see it
throughout this code.

The license, changelog, and help functions are toward the top of the script so
that new users reading the source will find them quickly.

The console output and state construction functions are close to the option
parser which uses them heavily, saving the developer some scrolling.

For the most part, support functions are arranged in the same order they appear
in the option parser.  The options are in turn ordered in the parser according
to where they've been organized in the help command.

The option parser sets state variables which provide direction to the code
which constructs the ffmpeg commandline.  This allows folding options so that
ie $APP -10 args... -8 doesn't create needless work for ffmpeg.  It also
makes the generated commands consistent, which makes them easier to read and
understand.


Coding style:

    Write elegant code.

    Elegant is defined as espressive, minimal, reusable, conventional.

    Expressive means it does a lot and makes sense doing it.
    Minimal means redundancies have been avoided.
    Reusable means it's crafted with the future in mind.
    Conventional means it leverages well-understood patterns.

    Use 4 space indents, 96-char lines, \$(foo), [[ ]], (()).
    Use local for local variables, and ';' after statements.
    Use semantic values and identifiers where possible.
    Don't use eval where declare or printf -v will do.
    Use a single line for conditionals where it's expressive:

    [[ \$foo = "bar" ]] && { success; say "yay!"; } || fatal "Oh no!";

    ...is more semantic than an if then else block, and less scrolling.


Send patches to my email with subject "$APP patch re: <context>" and I will
respond within 7 days.


Thanks in advance!

- $AUTHOR


My gists: $GITHUB_GISTS

-------------------------------------------------------------------------------

EOF
    quit;
}

license(){
    less <<EOF
$APP is a third-party shell script.  Consider all that implies.

If you don't accept full responsibility, don't use it.

Comments welcome on my GitHub Gist page: $GITHUB_GISTS
Thanks kindly appreciated.  Patches thoughtfully considered.  Complaints ignored.

-------------------------------------------------------------------------------


$APP User Licence Agreement:

$APP $COPYRIGHT

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials provided
with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

-------------------------------------------------------------------------------
EOF

    check_license || {
        read -p "If you agree to the license agreement, write \"I agree\" > " LICENSE_AGREED;
        if check_license; then 
            exec $0 -agree;
        else 
            fatal "license not agreed";
        fi
    }
    quit;
}

check_license(){
    [[ $LICENSE_AGREED = "I agree" ]];
}

agree(){
    LICENSE_AGREED="I agree";
    say "Thankyou for agreeing to the license.";
    save_config; # always exits
}


batch(){ # batch process files
    BATCH=($1);
    shift;
    ((SIZE=${#BATCH[@]}, SIZE>0)) && {
        note batch "glob produced $SIZE batch entries:";
        join $'\n' BATCH;
        for file in "${BATCH[@]}"; do
            [[ -f $file ]] || { note batch "skipping non-existing '$file'"; continue; }
            note batch "processing '$file'";
            $0 "$file" "$@";
        done;
        die batch "completed";
    }
    die batch "missing file glob.  see $APP -h";
}

save_config(){
    [[ -d $CONFIG_DIR ]] || { 
        mkdir "$CONFIG_DIR" && exec $0 "${ARGV[@]}";
        fatal "Unable to create settings folder";
    }
    set -- "${1:-$SETTINGS}"; # default argument
    show_config > $CONFIG_DIR/$1 \
        && quit "Settings preset \"$1\" saved successfully" \
        || fatal "Unable to save settings preset \"$1\"";
}

load_config(){
    [[ -f $CONFIG_DIR/$1 ]] || fatal "No such settings preset \"$1\"";
    source $CONFIG_DIR/$1 || fatal "Unable to load settings preset \"$1\"";
}

remove_config(){
    set -- "${1:-$SETTINGS}"; # default argument
    [[ -f $CONFIG_DIR/$1 ]] || fatal "Settings preset \"$1\" does not exist";
    rm -f $CONFIG_DIR/$1 || fatal "Unable to delete \"$1\"";
    quit "Settings preset \"$1\" removed";
}

show_config(){
    declare -p "${CONFIG_VARS[@]}" | cut -d ' ' -f 3-;
}

list_configs(){
    [[ -d $CONFIG_DIR ]] || fatal "no settings presets saved yet";
    cd $CONFIG_DIR;
    local list=(*);
    [[ $list = '*' && ${#list} = 1 ]] && list='<none saved>';
    say 'Settings presets:';
    join ', ' list;
    quit;
}

countdown(){ # seconds=5
    local i BREAK=0
    trap "BREAK=1" SIGINT;
    for (( i=${1:-5}; i; i--)); do 
        ((BREAK)) && break;
        printf "$i "; sleep 1; 
    done;
    trap - SIGINT;
    return $BREAK;
}

check_output(){ # returns false if file exists, exits if CTRL-C is pressed.
    [[ -f $1 ]] || return 0;
    [[ $OVERWRITE = CLOBBER ]] && return 1;
    [[ $OVERWRITE = SKIP ]] && fatal "Skipping existing \"$1\"";
    warn "File \"$1\" already exists.  Continuing in 5 seconds";
    countdown 5 || fatal "CTRL-C pressed.  Job skipped.";
    return 1;
}

format_version(){ # formats version strings in-place for comparison
    while (( $# )); do 
        declare -a parts="(${!1//[.-]/ })";
        printf -v $1 '%06d' "${parts[@]}" 2>/dev/null;
        shift;
    done;
}

compare_version(){ # left >= right
    local left=$1 right=$2;
    format_version left right;
    [[ $right < $left ]];
}

check_depends(){    
    if [[ -x $FFMPEG ]]; then
        local version=$($FFMPEG -version |head -1 |cut -d ' ' -f 3;)
        compare_version $version $FFMPEG_MIN_VERSION && return;
        warn "ffmpeg $version is too old.";
        warn "Upgrade ffmpeg to at least $FFMPEG_MIN_VERSION";
    else
        note fatal "ffpmeg not installed.";
    fi
    quit "to install ffmpeg run $0 -depends ";
}

install(){
    cd ~/bin || fatal "No ~/bin folder to install to";
    URL="https://gist.github.com";
    
    URL+=$(curl $GIST 2>/dev/null \
        | sed '/href.*\/raw\//!d' | sed -e 's/[^"]*"//' -e 's/".*//';
    ) && curl -L $URL >~/bin/"$APP";
}

install_depends(){
    cd ~/bin || fatal "No ~/bin folder to install in";
    if curl $DEPENDS | tar Jxv &>/dev/null; then
        mv -f ffmpeg-*-static/{ffmpeg,ffprobe} .
        rm -rf ffmpeg-*-static;
        FFMPEG="$(which ffmpeg)";
        say "Dependendencies installed successfully";
        save_config;
    else
        fatal "failed downloading dependencies";    
    fi
}

# quote variables by reference - for printing shell script fragments
shell_quote(){ while (( $# )); do declare -n dest=$1; dest=${dest@Q}; shift; done; }

# local a=(1 2 3) b=''; join ', ' a b '(' ')';  # declare -- b="(1, 2, 3)"
# join ', ' a - '( ' ' )';                      # "( 1, 2, 3 )"
join(){ # glue in [out|-] [before] [after]
    local -n in="$2"; local buf out;
    [[ $3 && $3 != - ]] && local -n out="$3";
    ((${#in[@]}>1)) && printf -v buf -- "$1%s" "${in[@]:1}";
    [[ $3 && $3 != - ]] && out="$4$in$buf$5" || echo "$4$in$buf$5";
}

_note(){
    local good=0 batch=1 warn=2 fatal=3 \
        MESSAGES=("* $APP: " "* $APP batch: " "* $APP warning: " "!!! $APP: ");
    (($#>1)) && echo "${MESSAGES[$1]}${*:2}";
    code=$(($1)) # assign numeric in caller scope
}

#### console output support: ##################################################

note(){ local code; _note "$@"; }               # intrumented message
say(){ note good "$@"; }                        # informative message
warn(){ note warn "$@"; }                       # cautionary message
die(){ local code; _note "$@"; exit $code; }    # instrumented exit
quit(){ die good "$@"; }                        # healthy exit
fatal(){ die fatal "$@"; }                      # unhealthy exit

#### state construction support: ##############################################

cmd(){ CMD+=("$@"); }                           # use this *after* option parser
map(){ MAPS+=(-map "$1"); }                     # add stream maps - before filters
filter_a(){ FILTER_A+=("$@"); }                 # add audio filters
filter_v(){ FILTER_V+=("$@"); }                 # add video filters
pre(){ PRE+=("$@"); }                           # options before input
opt(){ OPTS1+=("$@"); OPTS2+=("$@"); }          # add post-filter options
opt1(){ OPTS1+=("$@"); }                        # add pass1 options
opt2(){ OPTS2+=("$@"); }                        # add pass2 / 1-pass VBR options

#### load saved defaults: ####################################################

[[ -f $CONFIG_DIR/$SETTINGS ]] && source $CONFIG_DIR/$SETTINGS;

#### option parser: ##########################################################

[[ -f $1 ]] && { INPUT=$1; OUTPUT=$1; shift; }

# accept -opt -opt-skewer -opt_snake -opt+tree -camelCase etc
# and -opt arg -opt=arg --opt arg --opt=arg variants
# --opt=-1 form is required where arg starts with a hyphen
while (( $# )); do

    # echo "processing args: $@";

    [[ $1 =~ ^-?(-[0-9A-Za-z][-_+0-9A-Za-z]*)(=?)(.*)$ ]];
    OPT=${BASH_REMATCH[1]}; ARG=${BASH_REMATCH[3]}; shift;
    [[ ${BASH_REMATCH[2]} || ${1:0:1} = - ]] || { ARG=$1; shift; }

    case "$OPT" in
    -h | -help | --help)
        help;
    ;;
    -v | -version)
        say "version $VERSION"$'\n';
        $FFMPEG -version
        quit;
    ;;
    -changelog)
        changelog;
    ;; 
    -license)
        license;
    ;;
    -agree)
        agree;
    ;;
    -debug)                 # show the commandline instead of running it
        cmd echo; DEBUG=1;
    ;;
    -title)                 # set the file title
        TITLE=$ARG;
    ;;
    -tag )                  # sets the file tag
        TAG=$ARG;
    ;;
    -e | -ext)              # set the file output mux format and extension
        EXT=$ARG;
    ;;
    -fs | -faststart)       # mp4 container with streaming support
        EXT=mp4
        opt2 -movflags +faststart
    ;;
    -i | -input)            # set the input file name
        [[ -f $ARG ]] && INPUT=$ARG || fatal "input file $ARG does not exist";
    ;;
    -o | -output)           # set the output file name
        OUTPUT=$ARG;
    ;;
    -ow | -overwrite)       # CLOBBER, SKIP, ASK
        OVERWRITE=${ARG^^};
        OVERWRITE=${OVERWRITE:-CLOBBER};
    ;;
    -no-ow) 
        OVERWRITE=SKIP;
    ;;
    -b | -batch)            # batch process
        batch "$ARG" "$@";
    ;;
    -here)                  # write output file to the working directory
        HERE=1;
    ;;
    -save)                  # saves the current settings and exits
        save_config "$ARG";
    ;;
    -load)                  # loads a named settings preset
        load_config "$ARG";
    ;;
    -remove)                # deletes a named settings preset and exits
        remove_config "$ARG";
    ;;
    -show)                  # shows the current settings and exits
        show_config && quit || fatal "Error in CONFIG_VARS";
    ;;
    -list)                  # lists all saved settings presets and exits
        list_configs;
    ;;
    -install)
        install || fatal "Failed installing $APP";
        quit "$APP Installed successfully";
    ;;
    -depends)
        install_depends;
    ;;
    -m | -map)              # add ffmpeg map option
        map "$ARG";
    ;;
    -v1)                    # map first video stream only
        map 0:v:0;
    ;;
    -a1)                    # map first audio steam only
        map 0:a:0;
    ;;
    -s1)                    # map first subtitle stream only
        map 0:s:0;
    ;;
    -va1)                   # map first video and audio streams only
        map 0:v:0; map 0:a:0;
    ;;
    -vas1)                  #map first video, audio, and subtitle streams only
        map 0:v:0; map 0:a:0; map 0:s:0;
    ;;
    -ss | -start)           # encoding start timetamp
        pre -ss "$ARG";
    ;;
    -t | -duration)         # encoding duration
        pre -t "$ARG";
    ;;
    -to | -end)             # encoding end timestamp
        pre -to "$ARG";
    ;;
    -1 | -pass1)            # creates rate control optimisation log
        PASS1=1;
    ;;
    -2 | -pass2)            # encodes optimised using pass1 log
        PASS2=1;
    ;;
    -12 | -pass12)          # run pass 1 then pass 2
        PASS1=1 PASS2=1;
    ;;
    -tonemap)               # tonemap from HDR to SDR
        DEPTH=MAIN;
        filter_v "zscale=t=linear" "tonemap=${ARG:-hable}" "zscale=t=bt709";
    ;;
    -TV)                    # some HDR10 TV sources lack colourspace:
        filter_v "zscale=tin=smpte2084:min=bt2020nc:pin=bt2020:rin=tv:t=smpte2084:m=bt2020nc:p=bt2020:r=tv";
    ;;
    -8 | -8bit)             # main / 8-bit encoding profile (default)
        DEPTH=MAIN;
    ;;
    -10 | -10bit)           # main10 / 10-bit encoding profile
        DEPTH=MAIN10;       # +3~ bd-rate for same CRF, 2-3x slower
    ;;
    -vc | -vcodec | -c:v)   # set the video codec (libaom-av1)
        VCODEC=$ARG;        # libx264, libx265, libvpx-vp9, libaom-av1, libsvtav1
    ;;
    -vb | -vbrate | -b:v)   # set target / constrained bit rate
        VBRATE=$ARG;        # 384k 512k 1M 5M etc
    ;;
    -crf | -cq)             # constant quality encoding
        CRF=$ARG;           # 1080p 35 for libaom-av1, 32 for x265/x264 with veryslow
                            # 720p 30 for libaom-av1, 28 for x265/x264 with veryslow
    ;;
    -rt | -realtime)        # enable real-time encoding for libaom-AV1.  Not recommended
        REALTIME=1;
    ;;
    -p | -preset)           # 9-0 lower is slower; libsvtav1 has 13-0
        PRESET=${ARG,,}     # accepts x264 preset names as well, ie 'medium'
    ;;
    -anime)                 # for detailed animations ie Lion King and Nausicaa
        filter_v "unsharp"; ((CRF+=6, DEPTH=MAIN10));
    ;;
    -toon)                  # for cell-shaded / line drawing cartoons ie Futurama
        filter_v "deband,unsharp=3:3:-1,unsharp=5:5:3"; ((CRF+=9, DEPTH=MAIN10));
    ;;
    -di | deint)            # approximate progressive frames from interlaced fields
        filter_v "${ARG:mcdeint}";
        ;;
    -d | -decimate)         # drops similar frames ( makes VFR)
        filter_v "mpdecimate";
    ;;
    -s | -scale)            # scale to x:y size, ie 960:540, 1920:-1, -8,720
        filter_v "scale=$ARG";
    ;;
    -fit)                   # scale to fit given rectangle
        filter_v "scale=$ARG:force_original_aspect_ratio=decrease";
    ;;
    -cover)                 # scale / crop to cover given rectangle
        filter_v "scale=$ARG:force_original_aspect_ratio=increase,crop=$ARG";
    ;;
    -crop)                  # ffpmeg crop filter
        filter_v "crop=$ARG";
    ;;
    -vmax)                  # scale down if an axis is larger than ARG
        filter_v "scale=iw*min(1\,if(gt(iw\,ih)\,$ARG/iw\,($ARG*sar)/ih)):(floor((ow/dar)/2))*2";
    ;;
    -dn | -denoise)         # makes noisy sources more compressible
        filter_v "hqdn3d";
    ;;
    -db | -deblock)         # blurs blocky source artefacts
        filter_v "deblock";
    ;;
    -dp | -deband)          # smoothly blend posterised gradients
        filter_v "deband";
    ;;
    -um | -# | -unsharp)    # sharpens edges without enhancing noise
        filter_v "unsharp=${ARG:-3:5:1}";
    ;;
    -vf | -vfilter)         # add one or more FFMPEG video filters
        filter_v "$ARG";
    ;;
    -fps)                   # re-samples the frame rate
        filter_v "fps=$ARG";
    ;;
    -g | -gop)              # sets the preferred GOP size
         opt "-g $ARG";
    ;;
    -ac | -acodec | -c:a)   # set the audio codec (copy)
        ACODEC=$ARG;        # aac, ac3, libmp3lame, libvorbis, libopus
    ;;
    -ab | -abrate | -b:a)   # set target audio bitrate
        ABRATE=$ARG;        # 32k 48k 64k 96k 112k 144k 176k
    ;;
    -ar | -asrate)          # set audio sample rate
        ASRATE=$ARG;
    ;;
    -8ch | -7.1)            # downmix to 7.1 (from 22.2?)
        CHANNELS=8;
        filter_a "channelmap=channel_layout=7.1"
    ;;
    -6ch | -5.1)            # downmix to 5.1
        CHANNELS=6;
        filter_a "channelmap=channel_layout=5.1"
    ;;
    -5ch | -4.1)            # downmix to 4.1
        CHANNELS=5;
        filter_a "channelmap=channel_layout=4.1"
    ;;
    -4ch | -4.0)            # downmix to 4.0 = FL+FR+FC+BC
        CHANNELS=4;
        filter_a "channelmap=channel_layout=4.0"
    ;;
    -qch | -quad)            # downmix to quadraphonic = FL+FR+BL+BR
        CHANNELS=4;
        filter_a "channelmap=channel_layout=quad"
    ;;
    -3ch | -2.1)            # downmix to 2.1
        CHANNELS=3;
        filter_a "channelmap=channel_layout=2.1"
    ;;
    -2ch | -stereo)         # downmix to stereo
        CHANNELS=2;
        filter_a "channelmap=channel_layout=stereo"
    ;;
    -1ch | -mono)           # downmix to mono and prevent channel subtraction
        CHANNELS=1;
        filter_a "asplit[a],aphasemeter=video=0,ametadata=select:key=lavfi.aphasemeter.phase:value=-0.005:function=less,pan=1c|c0=c0,aresample=async=1:first_pts=0,[a]amix"
        filter_a "channelmap=channel_layout=mono"
    ;;
    -0ch | -na)             # disable audio
        CHANNELS=0;
    ;;
    -af | -afilter)         # add one or more FFMPEG audio filters
        filter_a "$ARG";
    ;;
    -n | --nice)              # set niceness - default is 19
        NICE=$ARG;
    ;;
    -w | --wait)              # wait for a process to exit
        printf "waiting for pid $ARG to exit";
        while [[ -d /proc/$ARG ]]; do sleep 10; printf '.'; done; echo;
    ;;
    -ff)
        opt "$ARG";
    ;;
    -ff1)
        opt1 "$ARG";
    ;;
    -ff2)
        opt2 "$ARG";
    ;;
    *)
        warn "unknown option \"$OPT\" ignored.  see $0 -h for help";
    esac;
done;

check_license || exec $0 -license;
check_depends;

#### commandline construction: ################################################

[[ $INPUT ]] || exec $0 -h; # no input file - show help
[[ $TAG ]] && EXT="$TAG.$EXT";
OUTPUT="${OUTPUT%.*}.$EXT";
((HERE)) && OUTPUT=$(basename "$OUTPUT");
[[ $TITLE ]] || TITLE=$(basename "$OUTPUT");

if ((DEBUG)); then
    shell_quote INPUT OUTPUT TITLE;
else 
    $0 "${ARGV[@]}" -debug; # show generated command(s)
fi

# generate filter chains:
[[ $FILTER_V ]] && join ',' FILTER_V FILTER_V '-vf ';
[[ $FILTER_A ]] && join ',' FILTER_A FILTER_A '-af ';

# video rate control:
[[ $CRF = 0 ]] && { CRF=''; } || { CRF="-crf $CRF"; }
[[ $VBRATE = 0 ]] && { VBRATE=''; } || { VBRATE="-b:v $VBRATE"; }

# build the command:
(( NICE )) && FFMPEG="nice -n $NICE $FFMPEG";
cmd $FFMPEG -hide_banner -y "${PRE[@]}" -i "$INPUT" "${MAPS[@]}" \
    -c:v $VCODEC $CRF $VBRATE $FILTER_V \
    -metadata title="$TITLE" -metadata:s:v:0 title="$TITLE";

# perform PASS1 log generation if requested:
((PASS1)) && {
    say "pass 1:";
    check_output "$INPUT-0.log";
    # workaround ffmpeg progress reporting bug for 1st pass:
    AUDIO="-c:a copy";
    # NOTE: no turbo setting needed for VP8/9 as they switch to realtime internally for pass 1
    [[ $VCODEC = libx264 || $VCODEC = libx265 ]] && TURBO="-preset superfast"
    [[ $VCODEC = libsvtav1 ]] && TURBO="-preset 12 -tiles 1x1";
    [[ $VCODEC = libaom-av1 ]] && \
    	TURBO="-row-mt 1 -usage 1 -aom-params cpu-used=10:tile-columns=1:tile-rows=1";
    "${CMD[@]}" -pix_fmt yuv420p $AUDIO -passlogfile "$INPUT" \
        "${OPTS1[@]}" $TURBO -pass 1 -f null /dev/null;
}

# per-codec performance flags, mostly needed due to backward AV1 presets
if [[ $VCODEC = libaom-av1 ]]; then
    # work around the FFPMEG bug limiting "cpu-used" to 0..8:
    ((REALTIME || PRESET > 7)) && cmd -usage 1;         # LibAOM bad flag name contest
    cmd -row-mt 1 -aom-params "cpu-used=$((PRESET))";   # the winner.is...
elif [[ $VCODEC = libsvtav1 ]]; then
    cmd -preset $((PRESET));
else
    cmd -preset $((9 - PRESET));        # normalise quality / effort preset number
fi

# video encoder bit depth for accuracy tradeoff / encoding efficiency, NOT HDR
if ((DEPTH == MAIN)); then          # low-accuracy = -33%ish wallclock
    cmd -pix_fmt yuv420p;
elif ((DEPTH == MAIN10)); then      # high-accuracy = +50%ish wallclock
    if [[ $VCODEC = libaom-av1 || $VCODEC = libsvtav1 ]]; then
        cmd -pix_fmt yuv420p10le;   # AV1 supports 10-bit in main profile 
    else
        cmd -pix_fmt yuv420p10le -profile main10;
    fi
fi

if ((CHANNELS)); then
    cmd -c:a $ACODEC -metadata:s:a:0 title="$TITLE";
    if [[ $ACODEC != "copy" ]]; then
        # half the bitrate for mono:
        ((CHANNELS == 1)) && { ABRATE=$((${ABRATE/[kK]/000}/2000))K; }
        cmd -ac $CHANNELS -b:a $ABRATE $FILTER_A;
        [[ $ASRATE ]] && cmd -ar $ASRATE;
    fi
else
    cmd -na;
fi

# perform PASS2 encoding if requested:
((PASS2)) && {
    say "pass 2:";
    check_output "$OUTPUT";
    "${CMD[@]}" -passlogfile "$INPUT" "${OPTS2[@]}" -pass 2 "$OUTPUT"; 
}

# plain CRF / VBR single pass:
((PASS1+PASS2)) || {
    say "single-pass VBR:"
    check_output "$OUTPUT";
    "${CMD[@]}" "${OPTS2[@]}" "$OUTPUT";
}
