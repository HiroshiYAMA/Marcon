#!/bin/bash

SLEEP_SEC=${1:-0}

sleep ${SLEEP_SEC}

export DISPLAY=:0.0

LCD_NAME='WaveShare WS170120'
# LCD_NAME='WaveShare WaveShare'

###########################################
# Rotation RIGHT.
###########################################
XRANDR_ROT='right'
XINPUT_ROT='0 1 0 -1 0 1 0 0 1'

# ###########################################
# # Rotation LEFT.
# ###########################################
# XRANDR_ROT='left'
# XINPUT_ROT='0 -1 1 1 0 0 0 0 1'

# display rotation.
xrandr --output HDMI-0 --rotate ${XRANDR_ROT}

# pointing device rotation.
LCD_ID=$(xinput | grep "${LCD_NAME}" | perl -pe 's/^.*\Wid=([0-9]+)\W.*$/${1}/')
xinput set-prop ${LCD_ID} 'Coordinate Transformation Matrix' ${XINPUT_ROT}
