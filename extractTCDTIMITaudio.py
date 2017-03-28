import os
# from utils.helpFunctions import *
# from utils.copyFilesOfType import *
from fixTCDTIMITwavStructure import *
from getPhnFiles import *

srcDir = "/media/matthijs/TOSHIBA_EXT/TCDTIMIT/original/"
dstDir = os.path.expanduser("~/TCDTIMIT/TCDTIMITaudio")

## PHN: generate phoneme files
print("EXTRACTING PHNS...")
print("lipspeakers:")
generatePHN("MLFfiles/lipspeaker_labelfiles.mlf", dstDir)
print("volunteers:")
generatePHN("MLFfiles/volunteer_labelfiles.mlf", dstDir)

## WAV
# get the wav files
print("EXTRACTING WAVS...")
# copyFilesOfType(srcDir, os.path.expanduser("~/TCDTIMIT/TCDTIMITaudio"), "wav", interactive=False)
# fix the wav structure
print("fixing directory structure...")
fixTCDTIMITwavStructure(os.path.expanduser("~/TCDTIMIT/TCDTIMITaudio"))