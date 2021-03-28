# Copyright (C) 2021 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Particle class for the swarm."""
import numpy

class Particle:
    """
    Particle class, handles the current position, velocity and keeps
    track of the best point

    Parameters
    ---------
    TO DO: ...



    """
    def __init__(self, bounds, parameters):
        self._current_position = None
        self._current_velocity = None
        self._best_position = None
        self._parameters = None
        self._best_score = -numpy.infty
        self._bounds = None

        self.parameters = parameters
        self.bnds = bounds


        self.current_position = self._get_x0()
        self.best_position = self.current_position

        self.current_velocity = self._get_v0()

    def _get_x0(self):
        """Returns newly sampled starting points."""
        return numpy.array([numpy.random.uniform(self.bnds[p][0], self.bnds[p][1])
                            for p in self.parameters])

    def _get_v0(self):
        """Returns newly sampled starting velocities."""
        return numpy.array([numpy.random.normal(
            scale=(self.bnds[p][1] - self.bnds[p][0])/10.)
            for p in self.parameters])

    @property
    def parameters(self):
        """Particle's parameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        """Sets the parameters. Ensures they are stored as a list."""
        if isinstance(parameters, str):
            parameters = [parameters]
        if not isinstance(parameters, list):
            raise ValueError("'Parameters' must be a list.")
        for par in parameters:
            if not isinstance(par, str):
                raise ValueError("'{}' must be a string.".format(par))
        # Check for duplicates
        if len(set(parameters)) != len(parameters):
            raise ValueError("Some parameters are duplicate.")
        self._parameters = parameters

    @property
    def bounds(self):
        """Particle's parameters prior boundaries."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """Sets the boundaries."""
        for par in self.parameters:
            if par not in bounds.keys():
                raise ValueError("'{}' is missing boundaries.".format(par))
            if  not isinstance(bounds[par], (list, str)):
                raise ValueError("'{}' boundary must be a list.".format(par))
            if len(bounds) != 2:
                raise ValueError("'{}' boundary must be len 2.".format(par))
            bounds = list(bounds)
            if not bounds[1] > bounds[0]:
                bounds = bounds[::-1]
            self._bounds = bounds

    @property
    def current_position(self):
        """Current position of the particle."""
        if self._current_position is None:
            return raiseValueError("Particle's position not set.")
        return self._current_position

    @current_position.setter
    def current_position(self, position):
        """Sets the current position of the particle."""
        self._current_position = position

    @property
    def current_velocity(self):
        """Current velocity of the particle."""
        if self._current_velocity is None:
            raise ValueError("Particle's velocity not set.")
        return self._current_velocity

    @current_velocity.setter
    def current_velocity(self, velocity):
        """Sets the current velocity of the partile."""
        self._current_velocity = velocity

    @property
    def best_position(self):
        """Particle's best position."""
        if self._best_position is None:
            return ValueError("Particle's best position not set.")
        return self._best_position

    @best_position.setter
    def best_position(self, position):
        """Sets the particle's best position."""
        self._best_position = position.copy()

    @property
    def best_score(self):
        """Particle's best score."""
        if self._best_score is None:
            return ValueError("Particle's best score not set.")
        return self._best_score

    @best_score.setter
    def best_score(self, best_score):
        """Sets the particle's best score."""
        if best_score <= self._best_score:
            raise ValueError("Attempting to assign a new, lower best score.")
        self._best_score = best_score
