inplace:
	python setup.py build_ext --inplace

build: inplace
	python setup.py build

install:
	python setup.py install

clean:
	python setup.py clean --all

check: inplace
	nosetests --nologcapture -s -w ./span --with-coverage --cover-package=span \
		--cover-branches --cover-html

checkfast: inplace
	./test_fast.sh
