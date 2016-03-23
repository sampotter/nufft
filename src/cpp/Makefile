CMAKE_BUILD_PATH = ./build
CMAKE_DEBUG_BUILD_PATH = $(CMAKE_BUILD_PATH)/Debug
CMAKE_RELEASE_BUILD_PATH = $(CMAKE_BUILD_PATH)/Release

GTAGS_FILES = GPATH GRTAGS GTAGS
CPP_FILES = $(wildcard *.cpp)
HPP_FILES = $(wildcard *.hpp)

all: cmake $(GTAGS_FILES)

cmake: CMakeLists.txt $(CMAKE_DEBUG_BUILD_PATH) $(CMAKE_RELEASE_BUILD_PATH)

test: cmake-debug-test cmake-release-test

gtags:
	gtags

$(CMAKE_BUILD_PATH):
	mkdir -p $(CMAKE_BUILD_PATH)

$(CMAKE_DEBUG_BUILD_PATH): | $(CMAKE_BUILD_PATH)
	mkdir -p $(CMAKE_DEBUG_BUILD_PATH)
	cd $(CMAKE_DEBUG_BUILD_PATH); cmake -DCMAKE_BUILD_TYPE=Debug ../..; make

$(CMAKE_RELEASE_BUILD_PATH): | $(CMAKE_BUILD_PATH)
	mkdir -p $(CMAKE_RELEASE_BUILD_PATH)
	cd $(CMAKE_RELEASE_BUILD_PATH); cmake -DCMAKE_BUILD_TYPE=Release ../..; make

cmake-debug-test: | $(CMAKE_DEBUG_BUILD_PATH)
	cd $(CMAKE_DEBUG_BUILD_PATH); ctest

cmake-release-test: | $(CMAKE_DEBUG_RELEASE_PATH)
	cd $(CMAKE_RELEASE_BUILD_PATH); ctest

$(GTAGS_FILES): $(CPP_FILES) $(HPP_FILES)
	gtags

clean:
	rm -rf $(CMAKE_BUILD_PATH)
	rm -rf GPATH
	rm -rf GRTAGS
	rm -rf GTAGS
