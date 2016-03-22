LIBFMM = libfmm.dylib
LIBFMM_BUILD_PATH = ../cpp/Build/
LIBFMM_PATH = $(shell ./find_cmake_output.sh $(LIBFMM) $(LIBFMM_BUILD_PATH))
PYTHON = $(shell which python3)

all: libfmm.dylib _fmm.py

clean:
	rm -rf libfmm.dylib
	rm -rf _fmm.py
	rm -rf __pycache__

_fmm.py: fmm.py fmm_build.py
	$(PYTHON) fmm_build.py

libfmm.dylib: $(LIBFMM_PATH)
	# TODO: come up with a way of checking if $(LIBFMM_PATH) has been
	# set correctly at this point.
	cp $(LIBFMM_PATH) .
