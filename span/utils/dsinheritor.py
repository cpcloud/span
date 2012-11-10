# dsinheritor.py ---

# Copyright (C) 2012 Copyright (C) 2012 Phillip Cloud <cpcloud@gmail.com>

# Author: Phillip Cloud <cpcloud@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

class DocStringInheritor(type):
    def __new__(meta, name, bases, clsdict):
        if not ('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (c for b in bases for c in b.mro()):

                doc = mro_cls.__doc__

                if doc:
                    clsdict['__doc__'] = doc
                    break

        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (c for b in bases for c in b.mro()
                                if hasattr(c, attr)):

                    doc = getattr(getattr(mro_cls, attr), '__doc__')

                    if doc:
                        attribute.__doc__ = doc
                        break

        return type.__new__(meta, name, bases, clsdict)
