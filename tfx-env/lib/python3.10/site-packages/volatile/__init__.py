from contextlib import contextmanager
from errno import ENOENT
import os
import shutil
import socket
import tempfile


@contextmanager
def dir(suffix='', prefix='tmp', dir=None, force=True):
    """Create a temporary directory.

    A contextmanager that creates and returns a temporary directory, cleaning
    it up on exit.

    The force option specifies whether or not the directory is removed
    recursively. If set to `False` (the default is `True`), only an empty
    temporary directory will be removed. This is helpful to create temporary
    directories for things like mountpoints; otherwise a failing unmount would
    result in all files on the mounted volume to be deleted.

    :param suffix: Passed on to :func:`tempfile.mkdtemp`.
    :param prefix: Passed on to :func:`tempfile.mkdtemp`.
    :param dir: Passed on to :func:`tempfile.mkdtemp`.
    :param force: If true, recursively removes directory, otherwise just
                  removes if empty. If directory isn't empty and `force` is
                  `False`, :class:`OSError` is raised.
    :return: Path to the newly created temporary directory.
    """
    name = tempfile.mkdtemp(suffix, prefix, dir)

    try:
        yield name
    finally:
        try:
            if force:
                shutil.rmtree(name)
            else:
                os.rmdir(name)
        except OSError as e:
            if e.errno != 2:  # not found
                raise


@contextmanager
def file(mode='w+b', suffix='', prefix='tmp', dir=None, ignore_missing=False):
    """Create a temporary file.

    A contextmanager that creates and returns a named temporary file and
    removes it on exit. Differs from temporary file functions in
    :mod:`tempfile` by not deleting the file once it is closed, making it safe
    to write and close the file and then processing it with an external
    program.

    If the temporary file is moved elsewhere later on, `ignore_missing` should
    be set to `True`.

    :param mode: Passed on to :func:`tempfile.NamedTemporaryFile`.
    :param suffix: Passed on to :func:`tempfile.NamedTemporaryFile`.
    :param prefix: Passed on to :func:`tempfile.NamedTemporaryFile`.
    :param dir: Passed on to :func:`tempfile.NamedTemporaryFile`.
    :param ignore_missing: If set to `True`, no exception will be raised if the
                           temporary file has been deleted when trying to clean
                           it up.
    :return: A file object with a `.name`.
    """
    # note: bufsize is not supported in Python3, try to prevent problems
    #       stemming from incorrect api usage
    if isinstance(suffix, int):
        raise ValueError('Passed an integer as suffix. Did you want to '
                         'specify the deprecated parameter `bufsize`?')

    fp = tempfile.NamedTemporaryFile(mode=mode,
                                     suffix=suffix,
                                     prefix=prefix,
                                     dir=dir,
                                     delete=False)
    try:
        yield fp
    finally:
        try:
            os.unlink(fp.name)
        except OSError as e:
            # if the file does not exist anymore, ignore
            if e.errno != ENOENT or ignore_missing is False:
                raise


@contextmanager
def unix_socket(sock=None, socket_name='tmp.socket', close=True):
    """Create temporary unix socket.

    Creates and binds a temporary unix socket that will be closed and removed
    on exit.

    :param sock: If not `None`, will not created a new unix socket, but bind
                 the passed in one instead.
    :param socket_name: The name for the socket file (will be placed in a
                        temporary directory). Do not pass in an absolute path!
    :param close: If `False`, does not close the socket before removing the
                  temporary directory.
    :return: A tuple of `(socket, addr)`, where `addr` is the location of the
             bound socket on the filesystem.
    """
    sock = sock or socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    with dir() as dtmp:
        addr = os.path.join(dtmp, socket_name)
        sock.bind(addr)

        try:
            yield sock, addr
        finally:
            if close:
                sock.close()


@contextmanager
def umask(new_umask):
    prev_umask = os.umask(new_umask)
    yield prev_umask
    os.umask(prev_umask)
