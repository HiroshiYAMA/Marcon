#!/bin/bash

SLEEP_SEC=${1:-0}

sleep ${SLEEP_SEC}

export DISPLAY=:0.0

# display rotation.
xrandr --output HDMI-0 --rotate right

# pointing device rotation.
LCD_ID=$(xinput | grep 'WaveShare WS170120' | perl -pe 's/^.*\Wid=([0-9]+)\W.*$/${1}/')
xinput set-prop ${LCD_ID} 'Coordinate Transformation Matrix' 0 1 0 -1 0 1 0 0 1
