#!/usr/bin/env bash

# TODO: this needs to be fixed.

OUTPUT=$1
BUILD_PATH=$2

if [ -d ${BUILD_PATH} ]; then
	if [ -d ${BUILD_PATH}/Release/ ]; then
		RELEASE_OUTPUT=${BUILD_PATH}/Release/${OUTPUT}
		if [ -e ${RELEASE_OUTPUT} ]; then
			echo "${RELEASE_OUTPUT}";
		fi
	elif [ -d ${BUILD_PATH}/RelWithDebInfo/ ]; then
		RELWITHDEBINFO_OUTPUT=${BUILD_PATH}/RelWithDebInfo/${OUTPUT}
		if [ -e ${RELWITHDEBINFO_OUTPUT} ]; then
			echo "${RELWITHDEBINFO_OUTPUT}";
		fi
	elif [ -d ${BUILD_PATH}/Debug/ ]; then
		DEBUG_OUTPUT=${BUILD_PATH}/Debug/${OUTPUT}
		if [ -e ${DEBUG_OUTPUT} ]; then
			echo "${DEBUG_OUTPUT}";
		fi
	else
		exit 1
	fi
else
	exit 1
fi
