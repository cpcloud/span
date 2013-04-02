inplace:
	python setup.py build_ext --inplace

build: inplace
	python setup.py build

install:
	python setup.py install

clean:
	python setup.py clean --all

check: inplace
	./test.py	

checkfast: inplace
	./test.py -f
