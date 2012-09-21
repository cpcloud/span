"""
"""

from future_builtins import map

import abc
import ftplib
import getpass
import os


from span.utils.progressbar import AnimatedProgressBar


class AbstractServer(object):
    """Abstract base class for a server object
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def download_file(self, filename, verbose=True):
        """Download a file from the server
        """
        pass


class ArodServer(AbstractServer):
    """Encapsulate the ARC lab's server
    """
    def __init__(self, username='Adrian', hostname='192.168.70.4',
                 password=None):
        """Constructor.

        Parameters
        ----------
        username : str, optional
            User to log on as.
        hostname : str, optional
            The host name of the server or an IP address.
        """
        super(ArodServer, self).__init__()
        self.username, self.hostname = username, hostname
        if password is None:
            password = getpass.getpass()
        self.ftp = ftplib.FTP(self.hostname, self.username, password)
        self.progress_bar = None

    def __del__(self):
        self.ftp.close()

    def download_file(self, filename, verbose=True):
        """Download a file from the server using FTP.

        Parameters
        ----------
        filename : str
            The name of the file to download from the server.

        Returns
        -------
        local_path : str
            The local path to the file downloaded
        """
        filename = os.path.join(os.sep, 'home', filename.lstrip('~' + os.sep))
        local_path = os.path.join(os.getcwd(), os.path.basename(filename))
        print filename
        remote_filesize = self.ftp.size(filename)

        if os.path.exists(local_path):
            if os.path.getsize(local_path) == remote_filesize:
                return local_path

        self.progress_bar = AnimatedProgressBar(end=remote_filesize, width=50)
        if verbose:
            print os.path.basename(local_path)

        with open(local_path, 'wb') as local_file:
            def callback(chunk):
                """Callback function for RETR FTP operation.

                Parameters
                ----------
                chunk : str
                """
                local_file.write(chunk)
                self.progress_bar += len(chunk)
                self.progress_bar.show_progress()

            self.ftp.retrbinary('RETR {0}'.format(filename), callback)
        print
        return local_path

    def download_files(self, filenames):
        """Download files from the arod server using multithreading.

        Parameters
        ----------
        server : ArodServer
        filenames : sequence
            Files to download from the server

        Returns
        -------
        r : list
            Names of the downloaded files
        """

        return list(map(self.download_file, filenames))
