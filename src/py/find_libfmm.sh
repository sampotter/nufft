#!/usr/bin/env bash

if [ -d ../cpp/build ]; then
	if [ -d ../cpp/build/Release/ ]; then
		if [ -e ../cpp/build/Release/libfmm.dylib ]; then
			echo "../cpp/build/Release/libfmm.dylib";
		fi
	elif [ -d ../cpp/build/RelWithDebInfo/ ]; then
		if [ -e ../cpp/build/RelWithDebInfo/libfmm.dylib ]; then
			echo "../cpp/build/Release/libfmm.dylib";
		fi
	elif [ -d ../cpp/build/Debug/ ]; then
		if [ -e ../cpp/build/Debug/libfmm.dylib ]; then
			echo "../cpp/build/Release/libfmm.dylib";
		fi
	else
		echo "ERROR: no suitable build found";
	fi
else
	echo "ERROR: build directory doesn't exist";
fi
