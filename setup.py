import os
import re
from setuptools import setup

# Parameter defs
CWD = os.getcwd()


def get_version():
    with open('version.txt', 'r') as f:
        m = re.match("""version=['"](.*)['"]""", f.read())

    assert m, "Malformed 'version.txt' file!"
    return m.group(1)


setup(
    name='REFOC',
    version=get_version(),
    description='This is the REFOC package',
    package_dir={
        'REFOC.SRC': 'SRC',
    },
    zip_safe=False,
)
