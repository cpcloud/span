"""
"""

import abc
import ftplib

from progressbar import AnimatedProgressBar


class AbstractServer(object, metaclass=abc.ABCMeta):
    """
    """
    @abc.abstractmethod
    def download_file(self, filename):
        """
        """
        pass


class ArodServer(AbstractServer):
    """
    """
    def __init__(self, username='Adrian', hostname='192.168.70.4'):
        """Constructor.

        Parameters
        ----------
        username : str, optional
            User to log on as.
        hostname : str, optional
            The host name of the server or an IP address.
        """
        super(ArodServer, self).__init__()
        self.username, self.password = username, getpass.getpass()
        self.hostname = hostname
        self.ftp = ftplib.FTP(self.hostname, self.username, self.password)

    def download_file(self, filename):
        """Download a file from the server using FTP.

        Parameters
        ----------
        filename : str
            The name of the file to download from the server

        Returns
        -------
        local_path : str
            The local path to the file downloaded
        """
        filename = '/home/' + filename.lstrip('~')
        local_path = os.path.join(os.getcwd(), os.path.basename(filename))
        self.progress_bar = AnimatedProgressBar(end=self.ftp.size(filename),
                                                width=50)
        print('%s' % os.path.basename(local_path))
        
        with open(local_path, 'wb') as local_file:
            def callback(chunk):
                local_file.write(chunk)
                self.progress_bar += len(chunk)
                self.progress_bar.show_progress()
            self.ftp.retrbinary('RETR {0}'.format(filename), callback)
        return local_path
