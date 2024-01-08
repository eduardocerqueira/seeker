#date: 2024-01-08T16:51:43Z
#url: https://api.github.com/gists/8c232091d6c9a154199238a1f309c19e
#owner: https://api.github.com/users/Psynosaur

vvencFFapp: VVenC, the Fraunhofer H.266/VVC Encoder, version 1.10.0 [Windows][GCC 13.2.0][64 bit][SIMD=AVX2]

#======== General Options ================
  -h,   --help [0]                   show default help
        --fullhelp [0]               show full help
  -v,   --Verbosity [verbose]        verbosity level (0: silent, 1: error, 2: warning, 3: info, 4: notice, 5: verbose,
                                     6: debug)
        --stats [1]                  enable or disable printing of statistics (fps, bitrate, estimation of encoding
                                     time)
        --version [0]                show version
        --additional []              additional options as string (e.g: "bitrate=1000000 passes=1")
  -c   []                            configuration file name
        --WriteConfig []             write the encoder config into configuration file
  -w,   --WarnUnknowParameter [0]    warn for unknown configuration parameters instead of failing
        --SIMD []                    SIMD extension to use (SCALAR, SSE41, SSE42, AVX, AVX2, AVX512), default: the
                                     highest supported extension

#======== Input Options ================
  -i,   --InputFile []               Original YUV input file name or '-' for reading from stdin
  -s,   --Size [0x0]                 Input resolution (WidthxHeight)
        --InputBitDepth [8]          Bit-depth of input file
  -f,   --FramesToBeEncoded [0]      Number of frames to be encoded (default=all)
  -fr,  --FrameRate [0]              Temporal rate (framerate numerator) e.g. 25,30, 30000, 50,60, 60000
        --FrameScale [1]             Temporal scale (framerate denominator) e.g. 1, 1001
        --fps [0/1]                  Framerate as int or fraction (num/denom)
        --TicksPerSecond [27000000]  Ticks Per Second for dts generation, (1..27000000, -1: ticks per frame=1)
        --LeadFrames [0]             Number of leading frames to be read before starting the encoding, use when
                                     splitting the video into overlapping segments
        --TrailFrames [0]            Number of trailing frames to be read after frames to be encoded, use when splitting                                     the video into overlapping segments
  -fs,  --FrameSkip [0]              number of frames to skip at start of input YUV [off]
        --segment [off]              when encoding multiple separate segments, specify segment position to enable segment concatenation (first, mid, last) [off]
                                      first: first segment
                                      mid  : all segments between first and last segment
                                      last : last segment
        --y4m [0]                    force y4m input (only needed for input pipe, else enabled by .y4m file extension)
        --logofile []                set logo overlay filename (json)
        --SourceWidth [0]            Source picture width
        --SourceHeight [0]           Source picture height
        --ConformanceWindowMode [1]  Window conformance mode (0:off, 1:automatic padding, 2:padding, 3:conformance
        --ConfWinLeft [0]            Left offset for window conformance mode 3
        --ConfWinRight [0]           Right offset for window conformance mode 3
        --ConfWinTop [0]             Top offset for window conformance mode 3
        --ConfWinBottom [0]          Bottom offset for window conformance mode 3
        --HorizontalPadding [0]      Horizontal source padding for conformance window mode 2
        --VerticalPadding [0]        Vertical source padding for conformance window mode 2
        --InputChromaFormat [420]    input file chroma format (400, 420, 422, 444)
        --PackedInput [0]            Enable 10-bit packed YUV input data ( pack 4 samples( 8-byte) into 5-bytes
                                     consecutively.
        --MaxPicSize [0x0]           Maximum resolution (maxWidth x maxHeight)
        --MaxPicWidth [0]            Maximum picture width
        --MaxPicHeight [0]           Maximum picture height
        --HorCollocatedChroma [1]    Specifies location of a chroma sample relatively to the luma sample in horizontal
                                     direction in the reference picture resampling(0: horizontally shifted by 0.5 units
                                     of luma samples, 1: collocated)
        --VerCollocatedChroma [0]    Specifies location of a chroma sample relatively to the luma sample in vertical
                                     direction in the cross-component linear model intra prediction and the reference
                                     picture resampling(0: horizontally co-sited, vertically shifted by 0.5 units of
                                     luma samples, 1: collocated)
        --ClipInputVideoToRec709Range [0]
                                     Enable clipping input video to the Rec. 709 Range on loading when InternalBitDepth
                                     is less than MSBExtendedBitDepth

#======== Output options ================
  -b,   --BitstreamFile []           Bitstream output file name
  -o,   --ReconFile []               Reconstructed YUV output file name
        --OutputBitDepth [0]         Bit-depth of output file

#======== Threading, performance ================
  -t,   --Threads [0]                Number of threads
        --preset [medium]            select preset for specific encoding setting (faster, fast, medium, slow, slower,
                                     medium_lowDecEnergy)
        --Tiles [1x1]                Set number of tile columns and rows
        --MaxParallelFrames [-1]     Maximum number of frames to be processed in parallel(0:off, >=2: enable parallel
                                     frames)
        --WppBitEqual [-1]           Ensure bit equality with WPP case (0:off (sequencial mode), 1:copy from wpp line
                                     above, 2:line wise reset)
        --EnablePicPartitioning [0]  Enable picture partitioning (0: single tile, single slice, 1: multiple
                                     tiles/slices)
        --TileColumnWidthArray [[]]  Tile column widths in units of CTUs. Last column width in list will be repeated
                                     uniformly to cover any remaining picture width
        --TileRowHeightArray [[]]    Tile row heights in units of CTUs. Last row height in list will be repeated
                                     uniformly to cover any remaining picture height
        --TileParallelCtuEnc [1]     Allow parallel CTU block search in different tiles
        --FppLinesSynchro [0]        (experimental) Number of CTU-lines synchronization due to MV restriction for FPP
                                     mode

#======== Slice decision options ================
  -ip,  --IntraPeriod [0]            Intra period in frames (0: use intra period in seconds (refreshsec), else:
                                     n*gopsize)
  -rs,  --RefreshSec [1]             Intra period/refresh in seconds
  -dr,  --DecodingRefreshType [cra]  intra refresh type (idr, cra, cra_cre: CRA, constrained RASL picture encoding,
                                     none, rpsei: Recovery Point SEI)
  -g,   --GOPSize [32]               GOP size of temporal structure (16,32)
        --PicReordering [1]          Allow reordering of pictures (0:off, 1:on), should be disabled for low delay
                                     requirements
        --POC0IDR [0]                start encoding with POC 0 IDR

#======== Rate control, Perceptual Quantization ================
        --NumPasses [-1]             number of rate control passes (1,2)
        --Passes [-1]                number of rate control passes (1,2)
        --Pass [-1]                  rate control pass for two-pass rate control (-1,1,2)
        --LookAhead [-1]             Enable pre-analysis pass with picture look-ahead (-1,0,1)
        --RCStatsFile []             rate control statistics file
        --TargetBitrate [0]          Rate control: target bitrate [bits/second], use e.g. 1.5M, 1.5Mbps, 1500k,
                                     1500kbps, 1500000bps, 1500000
        --MaxBitrate [0]             Rate control: approximate maximum instantaneous bitrate [bits/second] (0: no rate
                                     cap; least constraint)
  -qpa, --PerceptQPA [0]             Enable perceptually motivated QP adaptation, XPSNR based (0:off, 1:on)
        --STA [-1]                   Enable slice type adaptation at GOPSize>8 (-1: auto, 0: off, 1: adapt slice type,
                                     2: adapt NAL unit type)
        --RCInitialQP [0]            Rate control: initial QP. With two-pass encoding, this specifies the first-pass
                                     base QP (instead of using a default QP). Activated if value is greater than zero
        --PerceptQPATempFiltIPic [-1]
                                     Temporal high-pass filter in QPA activity calculation for key pictures (0:off,
                                     1:on, 2:on incl. temporal pumping reduction, -1:auto)

#======== Quantization parameters ================
  -q,   --QP [32]                    Qp value (0-63)
        --SameCQPTablesForAllChroma [1]
                                     0: Different tables for Cb, Cr and joint Cb-Cr components, 1 (default): Same tables                                     for all three chroma components
        --IntraQPOffset [-3]         Qp offset value for intra slice, typically determined based on GOP size
        --LambdaFromQpEnable [1]     Enable flag for derivation of lambda from QP
        --LambdaModifier [1x1x1x1x1x1x1]
                                     Lambda modifier list for temporal layers. If LambdaModifierI is used, this will not                                     affect intra pictures
        --LambdaModifierI [[]]       Lambda modifiers for Intra pictures, comma separated, up to one the number of
                                     temporal layer. If entry for temporalLayer exists, then use it, else if some are
                                     specified, use the last, else use the standard LambdaModifiers.
        --IQPFactor [-1]             Intra QP Factor for Lambda Computation. If negative, the default will scale lambda
                                     based on GOP size (unless LambdaFromQpEnable then IntraQPOffset is used instead)
        --QpInValCb [17,22,34,42]    Input coordinates for the QP table for Cb component
        --QpInValCr [[]]             Input coordinates for the QP table for Cr component
        --QpInValCbCr [[]]           Input coordinates for the QP table for joint Cb-Cr component
        --QpOutValCb [17,23,35,39]   Output coordinates for the QP table for Cb component
        --QpOutValCr [[]]            Output coordinates for the QP table for Cr component
        --QpOutValCbCr [[]]          Output coordinates for the QP table for joint Cb-Cr component
        --MaxCuDQPSubdiv [-1]        Maximum subdiv for CU luma Qp adjustment
        --MaxCuChromaQpOffsetSubdiv [-1]
                                     Maximum subdiv for CU chroma Qp adjustment - set less than 0 to disable
        --CbQpOffset [0]             Chroma Cb QP Offset
        --CrQpOffset [0]             Chroma Cr QP Offset
        --CbQpOffsetDualTree [0]     Chroma Cb QP Offset for dual tree
        --CrQpOffsetDualTree [0]     Chroma Cr QP Offset for dual tree
        --CbCrQpOffset [-1]          QP Offset for joint Cb-Cr mode
        --CbCrQpOffsetDualTree [0]   QP Offset for joint Cb-Cr mode in dual tree
        --SliceChromaQPOffsetPeriodicity [-1]
                                     Used in conjunction with Slice Cb/Cr QpOffsetIntraOrPeriodic. Use 0 (default) to
                                     disable periodic nature.
        --SliceCbQpOffsetIntraOrPeriodic [0]
                                     Chroma Cb QP Offset at slice level for I slice or for periodic inter slices as
                                     defined by SliceChromaQPOffsetPeriodicity. Replaces offset in the GOP table.
        --SliceCrQpOffsetIntraOrPeriodic [0]
                                     Chroma Cr QP Offset at slice level for I slice or for periodic inter slices as
                                     defined by SliceChromaQPOffsetPeriodicity. Replaces offset in the GOP table.
        --LumaLevelToDeltaQPMode [0] Luma based Delta QP 0(default): not used. 1: Based on CTU average

#======== Profile, Level, Tier ================
        --Profile [auto]             profile (main_10, main_10_still_picture)
        --Level [auto]               level limit (1.0, 2.0,2.1, 3.0,3.1, 4.0,4.1, 5.0,5.1,5.2, 6.0,6.1,6.2,6.3, 15.5)
        --Tier [main]                tier for interpretation of level (main, high)
        --SubProfile [0]             Sub-profile idc
        --MaxBitDepthConstraint [10] Bit depth to use for profile-constraint for RExt profiles. (0: automatically choose                                     based upon other parameters)
        --IntraConstraintFlag [0]    Value of general_intra_constraint_flag to use for RExt profiles (not used if an
                                     explicit RExt sub-profile is specified)

#======== VUI and SEI options ================
        --Sdr [off]                  set SDR mode + BT.709, BT.2020, BT.470 color space. use: off, sdr|sdr_709,
                                     sdr_2020, sdr_470bg
        --Hdr [off]                  set HDR mode + BT.709 or BT.2020 color space (+SEI messages for hlg) If maxcll or
                                     masteringdisplay is set, HDR10/PQ is enabled. use: off, pq|hdr10,
                                     pq_2020|hdr10_2020, hlg, hlg_2020
  -dph, --SEIDecodedPictureHash [off]
                                     Control generation of decode picture hash SEI messages, (0:off, 1:md5, 2:crc,
                                     3:checksum)
        --SEIBufferingPeriod [0]     Control generation of buffering period SEI messages
        --SEIPictureTiming [0]       Control generation of picture timing SEI messages
        --SEIDecodingUnitInfo [0]    Control generation of decoding unit information SEI message.
        --EnableDecodingParameterSet [0]
                                     Enable writing of Decoding Parameter Set
  -aud, --AccessUnitDelimiter [auto] Enable Access Unit Delimiter NALUs, (default: auto - enable only if needed by
                                     dependent options)
  -vui, --VuiParametersPresent [auto]
                                     Enable generation of vui_parameters(), (default: auto - enable only if needed by
                                     dependent options)
  -hrd, --HrdParametersPresent [1]   Enable generation of hrd_parameters(), (0: off, 1: on; default: 1)
        --AspectRatioInfoPresent [0] Signals whether aspect_ratio_idc is present
        --AspectRatioIdc [0]         aspect_ratio_idc
        --SarWidth [0]               horizontal size of the sample aspect ratio
        --SarHeight [0]              vertical size of the sample aspect ratio
        --ColourDescriptionPresent [0]
                                     Signals whether colour_primaries, transfer_characteristics and matrix_coefficients
                                     are present
        --ColourPrimaries [unknown]  Specify color primaries (0-13): reserved, bt709, unknown, empty, bt470m, bt470bg,
                                     smpte170m, smpte240m, film, bt2020, smpte428, smpte431, smpte432
        --TransferCharacteristics [unknown]
                                     Specify opto-electroni transfer characteristics (0-18): reserved, bt709, unknown,
                                     empty, bt470m, bt470bg, smpte170m, smpte240m, linear, log100, log316, iec61966-2-4,                                     bt1361e, iec61966-2-1, bt2020-10, bt2020-12, smpte2084, smpte428, arib-std-b67
        --MatrixCoefficients [unknown]
                                     Specify color matrix setting to derive luma/chroma from RGB primaries (0-14): gbr,
                                     bt709, unknown, empty, fcc, bt470bg, smpte170m, smpte240m, ycgco, bt2020nc,
                                     bt2020c, smpte2085, chroma-derived-nc, chroma-derived-c, ictcp
        --ChromaLocInfoPresent [0]   Signals whether chroma_sample_loc_type_top_field and
                                     chroma_sample_loc_type_bottom_field are present
        --ChromaSampleLocTypeTopField [0]
                                     Specifies the location of chroma samples for top field
        --ChromaSampleLocTypeBottomField [0]
                                     Specifies the location of chroma samples for bottom field
        --ChromaSampleLocType [0]    Specifies the location of chroma samples for progressive content
        --OverscanInfoPresent [0]    Indicates whether conformant decoded pictures are suitable for display using
                                     overscan
        --OverscanAppropriate [0]    Indicates whether conformant decoded pictures are suitable for display using
                                     overscan
        --VideoFullRange [0]         Indicates the black level and range of luma and chroma signals
        --MasteringDisplayColourVolume [[]]
                                     SMPTE ST 2086 mastering display colour volume info SEI (HDR), vec(uint) size 10,
                                     x,y,x,y,x,y,x,y,max,min where: "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)"range: 0 <=
                                     GBR,WP <= 50000, 0 <= L <= uint; GBR xy coordinates in increment of 1/50000,
                                     min/max luminance in units of 1/10000 cd/m2
        --MaxContentLightLevel [[]]  Specify content light level info SEI as "cll,fall" (HDR) max. content light level,
                                     max. frame average light level, range: 1 <= cll,fall <= 65535'
        --PreferredTransferCharacteristics [auto]
                                     Specify preferred transfer characteristics SEI and overwrite transfer entry in VUI
                                     (0-18): reserved, bt709, unknown, empty, bt470m, bt470bg, smpte170m, smpte240m,
                                     linear, log100, log316, iec61966-2-4, bt1361e, iec61966-2-1, bt2020-10, bt2020-12,
                                     smpte2084, smpte428, arib-std-b67

#======== Quality reporting metrics ================
        --MSEBasedSequencePSNR [0]   Emit sequence PSNR (0: only as a linear average of the frame PSNRs, 1: also based
                                     on an average of the frame MSEs
        --PrintHexPSNR [0]           Emit hexadecimal PSNR for each frame (0: off , 1:on
        --PrintFrameMSE [0]          Emit MSE values for each frame (0: off , 1:on
        --PrintSequenceMSE [0]       Emit MSE values for the whole sequence (0: off , 1:on)

#======== Bitstream options ================
        --CabacZeroWordPaddingEnabled [1]
                                     Add conforming cabac-zero-words to bit streams (0: do not add, 1: add as required)

#======== Coding structure parameters ================
        --ReWriteParamSets [1]       Enable rewriting of Parameter sets before every (intra) random access point
        --IDRRefParamList [0]        Enable indication of reference picture list syntax elements in slice headers of IDR                                     pictures

#======== Misc. options ================
  -cf,  --ChromaFormatIDC [0]        intern chroma format (400, 420, 422, 444) or set to 0 (default), same as
                                     InputChromaFormat
        --UseIdentityTableForNon420Chroma [1]
                                     True: Indicates that 422/444 chroma uses identity chroma QP mapping tables; False:
                                     explicit Qp table may be specified in config
        --InputBitDepthC [0]         As per InputBitDepth but for chroma component. (default:InputBitDepth)
        --InternalBitDepth [10]      Bit-depth the codec operates at. (default: MSBExtendedBitDepth). If different to
                                     MSBExtendedBitDepth, source data will be converted
        --OutputBitDepthC [0]        As per OutputBitDepth but for chroma component. (default: use luma output
                                     bit-depth)
        --MSBExtendedBitDepth [0]    bit depth of luma component after addition of MSBs of value 0 (used for
                                     synthesising High Dynamic Range source material). (default:InputBitDepth)
        --MSBExtendedBitDepthC [0]   As per MSBExtendedBitDepth but for chroma component. (default:MSBExtendedBitDepth)
        --WaveFrontSynchro [0]       Enable entropy coding sync
        --EntryPointsPresent [1]     Enable entry points in slice header
        --TreatAsSubPic [0]          Allow generation of subpicture streams. Disable LMCS, AlfTempPred and JCCR
        --ExplicitAPSid [0]          Set ALF APS id
        --AddGOP32refPics [0]        Use different QP offsets and reference pictures in GOP structure
        --NumRefPics [222111]        Number of reference pictures in RPL (0: default for RPL, <10: apply for all
                                     temporal layers, >=10: each decimal digit specifies the number for a temporal
                                     layer, last digit applying to the highest TL)
        --NumRefPicsSCC [0]          Number of reference pictures in RPL for SCC pictures (semantic analogue to
                                     NumRefPics, -1: equal to NumRefPics)

#======== Low-level QT-BTT partitioning options ================
        --CTUSize [128]              CTU size
        --MinQTISlice [8]            Min QT size for (luma in) I slices
        --MinQTLumaISlice [8]        Min QT size for (luma in) I slices
        --MinQTNonISlice [8]         Min QT size for P/B slices
        --MinQTChromaISliceInChromaSamples [4]
                                     Min QT size for chroma in I slices (in chroma samples, i.e. inclusive subsampling)
        --MaxMTTDepth [221111]       Max MTT depth for P/B slices (<10: apply for all temporal layers, >=10: each
                                     decimal digit specifies the depth for a temporal layer, last digit applying to the
                                     highest TL)
        --MaxMTTDepthI [2]           Max MTT depth for (luma in) I slices
        --MaxMTTDepthISliceL [2]     Max MTT depth for (luma in) I slices
        --MaxMTTDepthISliceC [-1]    Max MTT depth for chroma in I slices
        --MaxBTLumaISlice [32]       Max BT size for (luma in) I slices
        --MaxBTChromaISlice [64]     Max BT size for chroma in I slices
        --MaxBTNonISlice [128]       Max BT size for P/B slices
        --MaxTTLumaISlice [32]       Max TT size for (luma in) I slices
        --MaxTTChromaISlice [32]     Max TT size for chroma in I slices
        --MaxTTNonISlice [64]        Max TT size for P/B slices
        --DualITree [1]              Use separate luma and chroma QTBTT trees for intra slice (if off, luma constraint
                                     apply to all channels)
        --Log2MaxTbSize [6]          Maximum transform block size in logarithm base 2
        --Log2MinCodingBlockSize [2] Minimum coding block size in logarithm base 2

#======== Coding tools ================
        --CostMode [lossy]           Use alternative cost functions: choose between 'lossy', 'sequence_level_lossless',
                                     'lossless' (which forces QP to LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP) and
                                     'mixed_lossless_lossy' (which used
                                     QP'=LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP_PRIME for pre-estimates of
                                     transquant-bypass blocks).
        --ASR [1]                    Adaptive motion search range
        --HadamardME [1]             Hadamard ME for fractional-pel
        --FastHAD [0]                Use fast sub-sampled hadamard for square blocks >=32x32
        --RDOQ [1]                   Rate-Distortion Optimized Quantization mode
        --RDOQTS [1]                 Rate-Distortion Optimized Quantization mode for TransformSkip
        --SelectiveRDOQ [0]          Enable selective RDOQ (0: never, 1: always, 2: for natural content)
        --JointCbCr [1]              Enable joint coding of chroma residuals (0:off, 1:on)
        --CabacInitPresent [-1]      Enable cabac table index selection based on previous frame
        --LCTUFast [1]               Fast methods for large CTU
        --PBIntraFast [1]            Intra mode pre-check dependent on best Inter mode, skip intra if it is not probable                                     (0:off ... 2:fastest)
        --FastMrg [3]                Fast methods for inter merge
        --AMaxBT [-1]                Adaptive maximal BT-size
        --FastQtBtEnc [1]            Fast encoding setting for QTBT
        --ContentBasedFastQtbt [1]   Signal based QTBT speed-up
        --FEN [3]                    fast encoder setting
        --ECU [0]                    Early CU setting (1: ECU limited to specific block size and TL, 2: unconstrained
                                     ECU)
        --FDM [1]                    Fast decision for Merge RD Cost
        --DisableIntraInInter [0]    Flag to disable intra CUs in inter slices
        --FastUDIUseMPMEnabled [1]   If enabled, adapt intra direction search, accounting for MPM
        --FastMEForGenBLowDelayEnabled [1]
                                     If enabled use a fast ME for generalised B Low Delay slices
        --MTSImplicit [1]            Enable implicit MTS when explicit MTS is off

        --TMVPMode [1]               TMVP mode enable(0: off 1: for all slices 2: for certain slices only)
        --DepQuant [1]               Enable dependent quantization
        --QuantThrVal [-1]           Quantization threshold value for DQ last coefficient search
        --SignHideFlag [0]           Enable sign data hiding
        --MIP [1]                    Enable MIP (matrix-based intra prediction)
        --FastMIP [3]                Fast encoder search for MIP (matrix-based intra prediction)
        --MaxNumMergeCand [6]        Maximum number of merge candidates
        --MaxNumAffineMergeCand [5]  Maximum number of affine merge candidates
        --Geo [3]                    Enable geometric partitioning mode (0:off, 1:on)
        --MaxNumGeoCand [5]          Maximum number of geometric partitioning mode candidates
        --FastIntraTools [1]         SpeedUPIntraTools:LFNST,ISP,MTS. (0:off, 1:speed1, 2:speed2)
        --IntraEstDecBit [2]         Intra estimation decimation binary exponent for first pass directional modes
                                     screening (only test each (2^N)-th mode in the first estimation pass)
        --SMVD [3]                   Enable Symmetric MVD (0:off 1:vtm 2:fast 3:faster

        --AMVR [5]                   Enable Adaptive MV precision Mode (IMV)
        --IMV [5]                    Enable Adaptive MV precision Mode (IMV)
        --LMChroma [1]               Enable LMChroma prediction
        --MRL [1]                    MultiRefernceLines
        --BDOF [1]                   Enable bi-directional optical flow
        --BIO [1]                    Enable bi-directional optical flow
        --DMVR [1]                   Decoder-side Motion Vector Refinement
        --EncDbOpt [2]               Encoder optimization with deblocking filter 0:off 1:vtm 2:fast
        --EDO [2]                    Encoder optimization with deblocking filter 0:off 1:vtm 2:fast
        --LMCSEnable [2]             Enable LMCS luma mapping with chroma scaling (0:off 1:on 2:use SCC detection to
                                     disable for screen coded content)
        --LMCS [2]                   Enable LMCS luma mapping with chroma scaling (0:off 1:on 2:use SCC detection to
                                     disable for screen coded content)
        --LMCSSignalType [0]         Input signal type (0:SDR, 1:HDR-PQ, 2:HDR-HLG)
        --LMCSUpdateCtrl [0]         LMCS model update control (0:RA, 1:AI, 2:LDB/LDP)
        --LMCSAdpOption [0]          LMCS adaptation options: 0:automatic, 1: rsp both (CW66 for QP<=22), 2: rsp TID0
                                     (for all QP), 3: rsp inter(CW66 for QP<=22), 4: rsp inter(for all QP).
        --LMCSInitialCW [0]          LMCS initial total codeword (0~1023) when LMCSAdpOption > 0
        --LMCSOffset [6]             LMCS chroma residual scaling offset
        --ALF [1]                    Adpative Loop Filter
        --ALFSpeed [0]               ALF speed (skip filtering of non-referenced frames) [0-1]
        --CCALF [1]                  Cross-component Adaptive Loop Filter
        --UseNonLinearAlfLuma [0]    Non-linear adaptive loop filters for Luma Channel
        --UseNonLinearAlfChroma [0]  Non-linear adaptive loop filters for Chroma Channels
        --MaxNumAlfAlternativesChroma [8]
                                     Maximum number of alternative Chroma filters (1-8, inclusive)
        --ALFTempPred [-1]           Enable usage of ALF temporal prediction for filter data

        --ALFUnitSize [-1]           Size of ALF Search Unit [-1:default size(CTU)]

        --PROF [1]                   Enable prediction refinement with optical flow for affine mode
        --Affine [4]                 Enable affine prediction
        --AffineType [1]             Enable affine type prediction
        --MMVD [3]                   Enable Merge mode with Motion Vector Difference
        --MmvdDisNum [6]             Number of MMVD Distance Entries
        --AllowDisFracMMVD [1]       Disable fractional MVD in MMVD mode adaptively
        --MCTF [2]                   Enable GOP based temporal filter. (0:off, 1:filter all frames, 2:use SCC detection
                                     to disable for screen coded content)
        --MCTFSpeed [2]              MCTF Fast Mode (0:best quality ... 4:fastest operation)
        --MCTFUnitSize [-1]          Size of MCTF operation area (block size for motion compensation).
        --MCTFFutureReference [1]    Enable referencing of future frames in the GOP based temporal filter. This is
                                     typically disabled for Low Delay configurations.
        --MCTFFrame [[]]             Frame to filter Strength for frame in GOP based temporal filter
        --MCTFStrength [[]]          Strength for  frame in GOP based temporal filter.
        --BIM [1]                    Block importance mapping (basic temporal RDO based on MCTF).
        --FastLocalDualTreeMode [1]  Fast intra pass coding for local dual-tree in intra coding region (0:off, 1:use
                                     threshold, 2:one intra mode only)
        --QtbttExtraFast [3]         Non-VTM compatible QTBTT speed-ups
        --FastTTSplit [5]            Fast method for TT split
        --SbTMVP [1]                 Enable Subblock Temporal Motion Vector Prediction (0: off, 1: on)
        --CIIP [0]                   Enable CIIP mode, 0: off, 1: vtm, 2: fast, 3: faster
        --SBT [0]                    Enable Sub-Block Transform for inter blocks (0: off 1: vtm, 2: fast, 3: faster)
        --LFNST [1]                  Enable LFNST (0: off, 1: on)
        --MTS [0]                    Multiple Transform Set (MTS)
        --MTSIntraMaxCand [3]        Number of additional candidates to test for MTS in intra slices
        --ISP [3]                    Intra Sub-Partitions Mode (0: off, 1: vtm, 2: fast, 3: faster)
        --TransformSkip [2]          Transform skipping, 0: off, 1: TS, 2: TS with SCC detection
        --TransformSkipLog2MaxSize [4]
                                     Specify transform-skip maximum log2-size. Minimum 2, Maximum 5
        --ChromaTS [0]               Transform skipping for chroma, 0:off, 1:on (requires transform skipping)
        --BDPCM [2]                  BDPCM (0:off, 1:luma and chroma, 2: BDPCM with SCC detection)
        --RPR [-1]                   Reference Sample Resolution (0: disable, 1: eneabled, 2: RPR ready
        --IBC [2]                    IBC (0:off, 1:IBC, 2: IBC with SCC detection)
        --IBCFastMethod [3]          Fast methods for IBC. 1:default, [2..6] speedups
        --BCW [0]                    Enable Generalized Bi-prediction(Bcw) 0: disabled, 1: enabled, 2: fast
        --FastInferMerge [0]         Fast method to skip Inter/Intra modes. 0: off, [1..4] speedups
        --NumIntraModesFullRD [-1]   Number modes for full RD intra search [-1, 1..3] (default: -1 auto)
        --ReduceIntraChromaModesFullRD [1]
                                     Reduce modes for chroma full RD intra search
        --FirstPassMode [0]          Mode for first encoding pass when using rate control (0: default, 1: faster, 2:
                                     faster with temporal downsampling, 3: faster with resolution downsampling, 4:
                                     faster with temporal and resolution downsampling)

#======== Motion search options ================
        --FastSearch [4]             Search mode (0:Full search 1:Diamond 2:Deprecated 3:Enhanced Diamond 4:
                                     FastDiamond)
        --FastSearchSCC [2]          Search mode for SCC (0:use non SCC-search 1:Deprecated 2:DiamondSCC
                                     3:FastDiamondSCC)
  -sr,  --SearchRange [384]          Motion search range
        --BipredSearchRange [4]      Motion search range for bipred refinement
        --MinSearchWindow [96]       Minimum motion search window size for the adaptive window ME
        --ClipForBiPredMEEnabled [0] Enable clipping in the Bi-Pred ME.
        --FastMEAssumingSmootherMVEnabled [1]
                                     Enable fast ME assuming a smoother MV.
        --IntegerET [0]              Enable early termination for integer motion search
        --FastSubPel [1]             Enable fast sub-pel ME (1: enable fast sub-pel ME, 2: completely disable sub-pel
                                     ME)
        --ReduceFilterME [2]         Use reduced filter taps during subpel refinement (0 - use 8-tap; 1 - 6-tap; 2 -
                                     4-tap)

#======== Loop filters (deblock and SAO) ================
        --LoopFilterDisable [0]
        --LoopFilterOffsetInPPS [1]
        --LoopFilterBetaOffset_div2 [0]
        --LoopFilterTcOffset_div2 [0]
        --LoopFilterCbBetaOffset_div2 [0]
        --LoopFilterCbTcOffset_div2 [0]
        --LoopFilterCrBetaOffset_div2 [0]
        --LoopFilterCrTcOffset_div2 [0]
        --DeblockLastTLayers [0]     Deblock only the highest n temporal layers, 0: all temporal layers are deblocked
        --DisableLoopFilterAcrossTiles [0]
                                     Loop filtering applied across tile boundaries or not (0: filter across tile
                                     boundaries  1: do not filter across tile boundaries)
        --DisableLoopFilterAcrossSlices [0]
                                     Loop filtering applied across tile boundaries or not (0: filter across slice
                                     boundaries  1: do not filter across slice boundaries)
        --SAO [0]                    Enable Sample Adaptive Offset (1: always, 2: only for screen content frames)
        --SaoEncodingRate [-1]       When >0 SAO early picture termination is enabled for luma and chroma
        --SaoEncodingRateChroma [-1] The SAO early picture termination rate to use for chroma (when m_SaoEncodingRate is                                     >0). If <=0, use results for luma
        --SaoLumaOffsetBitShift [0]  Specify the luma SAO bit-shift. If negative, automatically calculate a suitable
                                     value based upon bit depth and initial QP
        --SaoChromaOffsetBitShift [0]
                                     Specify the chroma SAO bit-shift. If negative, automatically calculate a suitable
                                     value based upon bit depth and initial QP

#======== Summary options (debugging) ================
        --SummaryOutFilename ['']    Filename to use for producing summary output file. If empty, do not produce a
                                     file.
        --SummaryPicFilenameBase ['']
                                     Base filename to use for producing summary picture output files. The actual
                                     filenames used will have I.txt, P.txt and B.txt appended. If empty, do not produce
                                     a file.
        --SummaryVerboseness [0]     Specifies the level of the verboseness of the text output

#======== Reconstructed video options ================
        --ClipOutputVideoToRec709Range [0]
                                     Enable clipping output video to the Rec. 709 Range on saving when OutputBitDepth is                                     less than InternalBitDepth
        --PYUV [0]                   Enable output 10-bit and 12-bit YUV data as 5-byte and 3-byte (respectively) packed                                     YUV data. Ignored for interlaced output.

#======== Tracing ================
        --tracechannellist [0]       list all available tracing channels
        --tracerule ['']             tracing rule (ex: "D_CABAC:poc==8" or "D_REC_CB_LUMA:poc==8")
        --tracefile ['']             tracing file