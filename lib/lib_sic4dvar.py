"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from netCDF4 import Dataset

class Sic4dvarResults(object):

    def __init__(self, sic4dvar_path):
        self.sic4dvar = Dataset(sic4dvar_path)

    def close(self):
        self.sic4dvar.close()

class Sic4dvarResultsMM(Sic4dvarResults):

    def __init__(self, sic4dvar_path):
        super().__init__(sic4dvar_path)

    def get_time(self):
        pass

class Sic4dvarResultsConfluence(Sic4dvarResults):

    def __init__(self, sic4dvar_path):
        super().__init__(sic4dvar_path)

    def get_time(self):
        pass