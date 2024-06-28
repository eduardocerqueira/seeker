#date: 2024-06-28T16:40:44Z
#url: https://api.github.com/gists/4cb485da9358a6b58d6ecfc91c1b78cc
#owner: https://api.github.com/users/mauigna06

# copypaste this code in the console
#
# set variables
export SLICER_CODE_PATH=/home/$USER/Slicers/Slicer
export SLICER_SUPERBUILD_PATH=/home/$USER/Slicers/SlicerR
export SLICER_BUILD_PATH=$SLICER_SUPERBUILD_PATH/Slicer-build
export BRP_AND_OTHER_EXTENSIONS_CODE=/media/$USER/Nuevo_vol/Lin/Documents/Github
export BRP_AND_OTHER_EXTENSIONS_BUILD=/home/$USER/SExtentions
export BRP_BUILD=$BRP_AND_OTHER_EXTENSIONS_BUILD/BRPR
export NUMBER_OF_SLICER_COMPILATION_JOBS=2
# build Slicer
cd $SLICER_SUPERBUILD_PATH
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DQt5_DIR:PATH=/opt/qt/5.15.2/gcc_64/lib/cmake/Qt5 $SLICER_CODE_PATH
make -j$NUMBER_OF_SLICER_COMPILATION_JOBS
# Slicer build successful
#
# clone BRP depends on extensions
cd $BRP_AND_OTHER_EXTENSIONS_CODE
git clone git@github.com:sebastianandress/Slicer-SurfaceWrapSolidify.git
git clone git@github.com:SlicerIGT/SlicerMarkupsToModel.git
git clone git@github.com:PerkLab/SlicerSandbox.git
#
# configure and build first dependency SurfaceWrapSolidify
cd $BRP_AND_OTHER_EXTENSIONS_BUILD
mkdir SWSR
cd SWSR
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DSlicer_DIR:PATH=$SLICER_BUILD_PATH $BRP_AND_OTHER_EXTENSIONS_CODE/Slicer-SurfaceWrapSolidify
make
# configure and build second dependency MarkupsToModel
cd $BRP_AND_OTHER_EXTENSIONS_BUILD
mkdir M2MR
cd M2MR
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DSlicer_DIR:PATH=$SLICER_BUILD_PATH $BRP_AND_OTHER_EXTENSIONS_CODE/SlicerMarkupsToModel
make
# configure and build third dependency Sandbox
cd $BRP_AND_OTHER_EXTENSIONS_BUILD
mkdir SandboxR
cd SandboxR
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DSlicer_DIR:PATH=$SLICER_BUILD_PATH $BRP_AND_OTHER_EXTENSIONS_CODE/SlicerSandbox/
make
#
# all dependencies where built, now configure and build BoneReconstructionPlanner
mkdir -p $BRP_BUILD
cd $BRP_BUILD
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DSlicer_DIR:PATH=$SLICER_BUILD_PATH -DSurfaceWrapSolidify_DIR:PATH=$BRP_AND_OTHER_EXTENSIONS_BUILD/SWSR -DMarkupsToModel_DIR:PATH=$BRP_AND_OTHER_EXTENSIONS_BUILD/M2MR -DSandbox_DIR:PATH=$BRP_AND_OTHER_EXTENSIONS_BUILD/SandboxR $BRP_AND_OTHER_EXTENSIONS_CODE/SlicerBoneReconstructionPlanner
make
