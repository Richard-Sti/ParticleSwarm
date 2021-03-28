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
"""Swarm class."""
import numpy
from scipy.stats import truncnorm

from particle import Particle


class Swarm:
    """
    TO DO: Add documentation.
    """
    _ensemble_prob = 0.1
    _reset_prob = 0.01
    _c0 = 0.6
    _c1 = 2.8
    _c2 = 1.3

    def __init__(self, logpost, Nparticles, bounds, parameters):
        # TO DO:
        #   - Clean up

        self._particles = numpy.array([Particle(bounds, parameters)
                           for i in range(Nparticles)])

        for particle in self._particles:
            particle.best_score = logpost(particle.current_position)

        self._Niters = 0
        self.logpost = logpost

        self._chain = None
        self._Nclosest = int(self.Nparticles / 4.)
        self._Nmajority = int(self.Nparticles * 2 / 3)


    @property
    def particles(self):
        """Swarm's particles."""
        return self._particles

    @property
    def Nparticles(self):
        """Number of particles in the swarm."""
        return len(self._particles)

    @property
    def Ndims(self):
        """Dimensionality of the parameter space."""
        return self.particles[0].current_position.size

    @property
    def Niters(self):
        """Number of iterations."""
        return self._Niters

    @property
    def Nfevals(self):
        """Number of function evaluation."""
        return self.Niters * self.Nparticles

    def _local_best(self, positions, scores, x0):
        """Returns the best position from a fraction of closest neigbours."""
        ds = numpy.linalg.norm(positions - x0, axis=1)
        # Indices of the closest neighbours
        closest = numpy.argsort(ds)[1:self._Nclosest + 1]
        index = closest[numpy.argmax(scores[closest])]
        # Return the position of nearby best score neighbour
        return self.particles[index].best_position

    def _ensemble_move(self, i, choices):
        """
        Returns a position somewhere between a given particle ``i`` and
        another randomly drawn particle from a set of best scoring particles.
        """
        j = numpy.random.choice(choices)
        x0 = self.particles[i].current_position
        xf = self.particles[j].best_position
        # Add a small perturbation to avoid zero width
        ds = numpy.linalg.norm(xf - x0 * (1 + 1e-12))
        # Normal-distributed point in [0, 1], mu = 0.5, std=0.25
        rand = truncnorm.rvs(a=-2, b=2, loc=0.5, scale=0.25)
        return x0 + rand * (xf - x0)


    def _candidate_velocities(self, positions, scores):
        """Calculates the proposed velocities for the particles."""
        vnew = numpy.zeros((self.Nparticles, self.Ndims))

        for i, particle in enumerate(self.particles):
            v0 = particle.current_velocity
            x0 = particle.current_position
            # Distance to the particle's best scoring point
            dx_local = particle.best_position - x0
            # Distance to the best scoring point of neighbouring particles
            dx_global = self._local_best(positions, scores, x0) - x0

            vnew[i, :] = (self._c0 * v0
                          + self._c1 * numpy.random.rand() * dx_local
                          + self._c2 * numpy.random.rand() * dx_global)
        return vnew


    def run(self, Nsteps, atol=1e-3):
        """
        Iteratively moves the swarm of particles until a stopping
        condition is met.
        """
        scratch_chain = numpy.full((self.Nparticles, Nsteps, self.Ndims + 1),
                                   fill_value=numpy.nan)
        early = False


        for i in range(Nsteps):
            # Get current scores and positions
            if i == 0:
                scores = numpy.array([p.best_score for p in self.particles])
                positions = self._gather_positions('current')
            else:
                positions = scratch_chain[:, i - 1, :-1]
                scores = scratch_chain[:, i - 1, -1]
            # Get the proposed particles' velocities
            vnew = self._candidate_velocities(positions, scores)

            for j, particle in enumerate(self.particles):
                # With small probability do an ensemble move to update
                # position while keeping the current velocity
                if numpy.random.rand() < self._ensemble_prob:
                    sort = numpy.argsort(-scores)
                    top_sorted = sort[:self._Nclosest]
                    remainder = sort[self._Nclosest:]

                    # Particles in bad positions ensemble with good positions
                    if j in remainder:
                        choice = top_sorted
                    # Particles good positions ensemble with good positions
                    else:
                        choice = numpy.delete(top_sorted, top_sorted==j)

                    particle.current_position = self._ensemble_move(j, choice)
                else:
                    particle.current_position += vnew[j, :]
                    particle.current_velocity = vnew[j, :]
                # Calculate the new position's score
                # TO DO: Multiprocess this step
                score = self.logpost(particle.current_position)

                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.current_position
                    positions[j] = particle.current_position

                # With some small probability reset particle's velocity
                if numpy.random.rand() < self._reset_prob:
                    particle.current_velocity = particle._get_v0()


                # Update the scratch chain
                scratch_chain[j, i, :-1] = particle.current_position
                scratch_chain[j, i, -1] = score

            if self.Niters % 2 == 0 and self._attempt_termination(atol):
                early = True
                break

            self._Niters += 1
        self._chain = self._save_chain(self._chain, scratch_chain, early)

        x, f = self.final_position
        msg = {'Niters': self.Niters,
               'Nfevals': self.Nfevals,
               'Early termination': early,
               'x': x,
               'f': f}
        return msg

    def _gather_positions(self, kind='current'):
        """Gathers either current or best particles' positions."""
        # Check for valid input
        if kind not in ['current', 'best']:
            raise ValueError("Unknown position type '{}'".format(kind))

        positions = numpy.zeros((self.Nparticles, self.Ndims))
        for k, particle in enumerate(self.particles):
            if kind == 'current':
                positions[k, :] = particle.current_position
            else:
                positions[k, :] = particle.best_position
        return positions

    def _attempt_termination(self, atol):
        """
        Decides whether to terminate the swarm if majority of particles
        are located close to each other within some absolute tolerance.
        """
        positions = self._gather_positions('best')
        median = numpy.median(positions, axis=0)
        isclose = numpy.linalg.norm(positions - median, axis=1) < atol
        if isclose.sum() > self._Nmajority:
            return True
        return False

    def _save_chain(self, chain, scratch_chain, early):
        """Saves the chain. If early termination removes the scratch space."""
        if early:
            scratch_chain = scratch_chain[:, :self.Niters, :]
        if chain is None:
            chain = scratch_chain
        else:
            chain = numpy.concatenate((self._chain, scratch_chain), axis=1)
        return chain

    @property
    def final_position(self):
        """
        Polls the majority of best scoring particles and returns the median
        position.
        """
        positions = self._gather_positions(kind='best')
        scores = numpy.array([p.best_score for p in self.particles])
        mask = numpy.argsort(scores)[::-1][:self._Nmajority]
        return numpy.median(positions[mask, :], axis=0), numpy.median(scores)
