LIBFMM_PATH = $(shell ./find_libfmm.sh)
PYTHON = $(shell which python3)

all: libfmm.dylib _fmm.py

clean:
	rm -rf libfmm.dylib
	rm -rf _fmm.py
	rm -rf __pycache__

_fmm.py: fmm.py fmm_build.py
	$(PYTHON) fmm_build.py

libfmm.dylib: $(LIBFMM_PATH)
	cp $(LIBFMM_PATH) .
