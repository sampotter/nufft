LIBFMM = libfmm.dylib
LIBFMM_BUILD_PATH = ../cpp/Build/
LIBFMM_PATH = $(shell ./find_cmake_output.sh $(LIBFMM) $(LIBFMM_BUILD_PATH))
PYTHON = $(shell which python3)
PLOT_PDFS = \
	rad_vs_per_for_mse_wrt_n.pdf \
	p_vs_n_wrt_mse.pdf \
	L_vs_p_wrt_time.pdf \
	rad_vs_persum_with_n_wrt_mse.pdf \
	gt_vs_rad_vs_per_with_bandlimit_wrt_time.pdf \
	radial_error_vs_target_location.pdf \
	per_error_vs_target_location.pdf \
	rad_vs_per_wrt_n_with_fixed_Y.pdf

all: libfmm.dylib _fmm.py $(PLOT_PDFS)

clean:
	rm -rf libfmm.dylib
	rm -rf _fmm.py
	rm -rf __pycache__
	rm -rf $(PLOT_PDFS)

$(PLOT_PDFS): make_plots.py plot.py
	$(PYTHON) make_plots.py

_fmm.py: fmm.py fmm_build.py
	$(PYTHON) fmm_build.py

libfmm.dylib: $(LIBFMM_PATH)
	# TODO: come up with a way of checking if $(LIBFMM_PATH) has been
	# set correctly at this point.
	cp $(LIBFMM_PATH) .
