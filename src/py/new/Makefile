all: _nfft.py

libnfft_timer.dylib: nfft_timer/build/debug/libnfft_timer.dylib
	cp nfft_timer/build/debug/libnfft_timer.dylib .

_nfft.py: libnfft_timer.dylib
	python3.4 build_nfft.py
