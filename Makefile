inplace:
	python setup.py build_ext --inplace

build: inplace
	python setup.py build

install: build
	python setup.py install

clean:
	python setup.py clean --all

cleanbuild: clean build

check: build
	./test.sh c

checkfast: build
	./test_fast.sh
