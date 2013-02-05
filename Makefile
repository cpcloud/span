build:
	python setup.py build
	python setup.py build_ext --inplace

install:
	python setup.py install

clean:
	python setup.py clean --all

check:
	python setup.py build_ext --inplace
	./test.sh c

checkfast:
	python setup.py build_ext --inplace
	./test_fast.sh
