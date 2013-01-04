build:
	python setup.py build

inplace build_ext:
	python setup.py build_ext --inplace

install all:
	python setup.py install

clean:
	python setup.py clean --all

check test:
	./test.sh
