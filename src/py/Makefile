LIBFMM = libfmm.dylib
LIBFMM_BUILD_PATH = ../cpp/build/
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
	echo $(LIBFMM_PATH)
	ifeq ($(LIBFMM_PATH), "")
		$(error could not find $(LIBFMM) in $(LIBFMM_BUILD_PATH))
	endif
	cp $(LIBFMM_PATH) .
