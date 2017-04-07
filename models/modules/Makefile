all: setup.py
	python setup.py build_ext --inplace
	rm -rf build
	$(MAKE) -C roi_align

clean:
	rm -rf utils/*.c utils/*.so
	$(MAKE) -C roi_align clean
