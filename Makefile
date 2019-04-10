CC=gcc

all: karma

.PHONY: karma
karma:
	mkdir cmake_build && \
	cmake -Bcmake_build -H. && \
	cd cmake_build && \
	make && \
	cp libkarma.so ../karma/libkarma.so && \
	cp karma_svd ../karma/karma_svd

.PHONY: clean
clean:
	rm -rf cmake_build
	rm -rf karma/libkarma.so karma/karma_svd