import os

from distutils.core import setup

if __name__ == '__main__':
    with open('README.txt') as f:
        readme = f.read()
    setup(name='span',
          version='0.1',
          author='Phillip Cloud',
          author_email='cpcloud@gmail.com',
          packages=['span'],
          scripts=[os.path.join('bin', 'serv2mat.py')],
          license='LICENSE.txt',
          description='Spike train analysis',
          long_description=readme,
          install_requires=['', ''])
