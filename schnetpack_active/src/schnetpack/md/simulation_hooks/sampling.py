"""
This module contains all routines for influencing the sampling during a molecular dynamics run.
In general, these routines are derived from :obj:`schnetpack.md.simulation_hooks.SimulationHook` and act
on the middle part of each simulator step.
Currently, accelerated molecular dynamics and metadynamics are implemented.
"""
import numpy as np
import copy
import logging
import os

import torch
from schnetpack import AtomsData
from schnetpack.md import MaxwellBoltzmannInit

from schnetpack.md.simulation_hooks import SimulationHook
from schnetpack.md.utils import MDUnits

__all__ = [
    "BiasPotential",
    "AcceleratedMD",
    "MetaDyn",
    "CollectiveVariable",
    "BondColvar",
    "AdaptiveSampling",
]


class BiasPotential(SimulationHook):
    """
    Placeholder class for bias potentials used for adaptive/accelerated sampling.
    """

    def __init__(self):
        raise NotImplementedError

    def on_step_end(self, simulator):
        """
        Needs to be implemented. This applies a bias potential after computing the forces
        and properties.

        Args:
            simulator (schnetpack.md.Simulator): Main simulator used for driving the dynamics
        """
        raise NotImplementedError


class AcceleratedMD(SimulationHook):
    """
    Hook for performing accelerated molecular dynamics [#accmd1]_ . This method distorts the potential
    energy surface in order to make deep valleys smoother. This smoothing is applied to everything
    below an energy threshold and its strength is regulated via a acceleration factor.
    The energy conversion can be used to specify the energy threshold in arbitrary units of energy, which are then
    converted to the internal units.

    Args:
        energy_threshold (float): Energy threshold in units of energy used by the calculator.
        acceleration_factor (float): Acceleration factor steering the smoothness of the bias potential
        energy_handle (str, optional): Identifier for the potential energies.
        energy_conversion (str, optional): Conversion factor for the calculator energies.

    References
    ----------
    .. [#accmd1] Hamelberg, Mongan, McCammon:
                 Accelerated molecular dynamics: a promising and efficient simulation method for biomolecules.
                 J. Chem. Phys., 120 (24), 11919--11929. 2004.
    """

    def __init__(
        self,
        energy_threshold,
        acceleration_factor,
        energy_handle="energy",
        energy_conversion=1.0,
    ):
        self.energy_threshold = energy_threshold
        self.acceleration_factor = acceleration_factor
        self.energy_handle = energy_handle
        # Convert from calculator -> internal
        self.energy_conversion = MDUnits.unit2internal(energy_conversion)

    def on_step_middle(self, simulator):
        """
        Compute the bias potential and derivatives and use them to update the
        current state of :obj:`schnetpack.md.System` in the simulator.
        While the forces are updated, the bias potential itself is stored in the
        properties dictionary of the system.

        Args:
            simulator (schnetpack.md.Simulator): Main simulator used for driving the dynamics
        """
        energies, forces = self._get_energies_forces(simulator)

        # Compute bias potential and derivatives
        bias_potential = self._function(energies)
        bias_forces = self._derivative(energies) * forces

        # Store bias potential and update forces
        simulator.system.properties["bias_potential"] = bias_potential
        simulator.system.forces = forces + bias_forces.detach()

    def _get_energies_forces(self, simulator):
        """
        Extract energies and forces from the present system and
        convert them to the units used by the bias potential.

        Args:
            simulator (schnetpack.md.Simulator): Main simulator used for driving the dynamics

        Returns:
            (torch.Tensor,torch.Tensor):
                energies:
                    torch.Tensor containing the system energies (in calculator units)

                forces:
                    torch.Tensor containing the system forces (in atomic units)

        """
        energies = (
            simulator.system.properties[self.energy_handle].float()
            * self.energy_conversion
        )
        forces = simulator.system.forces
        return energies, forces

    def _function(self, energies):
        """
        Primary bias potential function used in accelerated molecular dynamics.

        Args:
            energies (torch.Tensor): Potential energies to which the bias potential
                                     should be applied to.

        Returns:
            torch.Tensor: Computed bias function.
        """
        function = torch.pow(self.energy_threshold - energies, 2.0) / (
            self.acceleration_factor + self.energy_threshold - energies
        )
        function[energies >= self.energy_threshold] = 0
        return function

    def _derivative(self, energies):
        """
        Compute the derivative of the bias function for updating the forces.
        Effectively, this becomes a scaling factor of the original forces.

        Args:
            energies (torch.Tensor): Potential energies to which the bias potential
                                     should be applied to.

        Returns:
            torch.Tensor: Derivative of the bias function.
        """
        derivative = (
            self.acceleration_factor ** 2
            / torch.pow(
                self.acceleration_factor + self.energy_threshold - energies, 2.0
            )
            - 1.0
        )
        derivative[energies >= self.energy_threshold] = 0
        return derivative


class CollectiveVariable:
    """
    Basic collective variable to be used in combination with the :obj:`MetaDyn` hook.
    The ``_colvar_function`` needs to be implemented based on the desired collective
    variable.

    Args:
        width (float): Parameter regulating the standard deviation of the Gaussians.
    """

    def __init__(self, width):
        # Initialize the width of the Gaussian
        self.width = 0.5 * width ** 2

    def get_colvar(self, structure):
        """
        Compute the collecyive variable.

        Args:
            structure (torch.Tensor): Atoms positions taken from the system in the :obj:`schnetpack.md.Simulator`.

        Returns:
            torch.Tensor: Collective variable computed for the structure.
        """
        return self._colvar_function(structure)

    def _colvar_function(self, structure):
        """
        Placeholder for defining the particular collective variable function to be used.

        Args:
            structure (torch.Tensor): Atoms positions taken from the system in the :obj:`schnetpack.md.Simulator`.
        """
        raise NotImplementedError


class BondColvar(CollectiveVariable):
    """
    Collective variable acting on bonds between atoms.
    ``idx_a`` indicates the index of the first atom in the structure tensor, ``idx_b`` the second.
    Counting starts at zero.

    Args:
        idx_a (int): Index of the first atom in the positions tensor provided by the simulator system.
        idx_b (int): Index of the second atom.
        width (float): Width of the Gaussians applied to this collective variable. For bonds, units of Bohr are used.
    """

    def __init__(self, idx_a, idx_b, width):
        super(BondColvar, self).__init__(width)
        self.idx_a = idx_a
        self.idx_b = idx_b

    def _colvar_function(self, structure):
        """
        Compute the distance between both atoms.

        Args:
            structure (torch.Tensor): Atoms positions taken from the system in the :obj:`schnetpack.md.Simulator`.

        Returns:
            torch.Tensor: Bind collective variable.
        """
        vector_ab = structure[:, :, self.idx_b, :] - structure[:, :, self.idx_a, :]
        return torch.norm(vector_ab, 2, dim=2)


class MetaDyn(SimulationHook):
    """
    Perform a metadynamics simulation, where Gaussian potentials are deposited along collective
    variables in order to steer the sampling of a molecular dynamics trajectory. [#metadyn1]_

    Args:
        collective_variables (list): List of collective variables to be sampled
                                     (:obj:`schnetpack.md.simulation_hooks.CollectiveVariable`).
        frequency (int, optional): Frequency with which Gaussians are deposited (every n simulation steps).
        weight (float, optional): Weight of each Gaussian in units of energy (Hartee).
        store_potential (bool, optional): Whether centers and widths of the placed Gaussians should be stored.

    References
    ----------
    .. [#metadyn1] Laio, Parrinello:
                   Escaping free-energy minima.
                   Proc. Natl. Acad. Sci., 99 (20), 12562--12566, 2002.

    """

    def __init__(
        self,
        collective_variables,
        frequency=200,
        weight=1.0 / 627.509,
        store_potential=True,
    ):
        self.collective_variables = collective_variables
        self.store_potential = store_potential

        self.gaussian_centers = None
        self.gaussian_mask = None
        self.collective_variable_widths = None

        self.frequency = frequency
        self.weigth = weight
        self.n_gaussians = 0

    def on_simulation_start(self, simulator):
        """
        Initialize the tensor holding the Gaussian centers and widths. These
        will be populated during the simulation.

        Args:
            simulator (schnetpack.md.Simulator): Main simulator used for driving the dynamics
        """
        n_gaussian_centers = int(simulator.n_steps / self.frequency) + 1
        self.gaussian_centers = torch.zeros(
            n_gaussian_centers,
            len(self.collective_variables),
            device=simulator.system.device,
        )
        self.collective_variable_widths = torch.FloatTensor(
            [cv.width for cv in self.collective_variables],
            device=simulator.system.device,
        )
        self.gaussian_mask = torch.zeros(
            n_gaussian_centers, device=simulator.system.device
        )
        self.gaussian_mask[0] = 1

    def on_step_middle(self, simulator):
        """
        Based on the current structure, compute the collective variables and the
        associated Gaussian potentials. If multiple collective variables are given,
        a product potential is formed. torch.autograd is used to compute the forces
        resulting from the potential, which are then in turn used to update the system
        forces. A new Gaussian is added after a certain number of steps.

        Args:
            simulator (schnetpack.md.Simulator): Main simulator used for driving the dynamics
        """
        # Get and detach the structure from the simulator
        structure = simulator.system.positions.detach()
        # Enable gradients for bias forces
        structure.requires_grad = True

        # Compute the collective variables
        colvars = torch.stack(
            [colvar.get_colvar(structure) for colvar in self.collective_variables],
            dim=2,
        )

        # Compute the Gaussians for the potential
        gaussians = torch.exp(
            -((colvars[:, :, None, :] - self.gaussian_centers[None, None, :, :]) ** 2)
            / self.collective_variable_widths[None, None, None, :]
        )
        # Compute the bias potential and apply mask for centers not yet stored
        bias_potential = torch.prod(gaussians, dim=3) * self.gaussian_mask

        # Finalize potential and compute forces
        bias_potential = torch.sum(self.weigth * bias_potential, 2)
        bias_forces = -torch.autograd.grad(
            bias_potential, structure, torch.ones_like(bias_potential)
        )[0]

        # Store bias potential, collective variables and update system forces
        simulator.system.properties["bias_potential"] = bias_potential.detach()
        simulator.system.properties["collective_variables"] = colvars.detach()
        simulator.system.forces = simulator.system.forces + bias_forces.detach()

        if self.store_potential:
            # TODO: Much better to move this to a state dict?
            # Store information on the general shape of the bias potential
            simulator.system.properties[
                "gaussian_centers"
            ] = self.gaussian_centers.detach()
            simulator.system.properties[
                "gaussian_widths"
            ] = self.collective_variable_widths.detach()
            simulator.system.properties["gaussian_mask"] = self.gaussian_mask.detach()

        # Add a new Gaussian to the potential every n_steps
        if simulator.step % self.frequency == 0:
            self.gaussian_centers[self.n_gaussians] = colvars.detach()
            # Update the mask
            self.gaussian_mask[self.n_gaussians + 1] = 1
            self.n_gaussians += 1


class AdaptiveSampling(SimulationHook):
    def __init__(
        self,
        thresholds,
        n_samples,
        dataset,
        reset=True,
        temperature=300,
        initializer=MaxwellBoltzmannInit,
    ):

        self.thresholds = thresholds
        self.n_samples = n_samples
        self.samples = []
        self.samples_thresholds = []

        # Reinitialization
        self.reset = reset
        self.initializer = initializer(temperature)

        # Dataset for storage
        self.dataset = dataset
        if os.path.exists(self.dataset):
            logging.info(
                "Database {:s} already exists. Data will be appended.".format(
                    self.dataset
                )
            )

        # Initial system variables for reset
        self.init_positions = None
        self.init_forces = None
        self.init_cells = None

    def on_simulation_start(self, simulator):
        # Store initial configs for system reset
        self.init_positions = copy.deepcopy(simulator.system.positions)
        self.init_forces = copy.deepcopy(simulator.system.forces)
        self.init_cells = copy.deepcopy(simulator.system.cells)

    def on_step_end(self, simulator):
        # Check if a sample is required
        sample_system, sample_molecule, threshold_exceeded = self._check_uncertainty(
            simulator.system
        )

        if sample_system:
            # Collect samples based on uncertainty thresholds
            self._collect_samples(simulator.system, sample_molecule, threshold_exceeded)

            # Reinitialize velocities if requested
            if self.reset:
                logging.info("Resetting system...")
                self._reset_system(simulator.system, mask=sample_molecule)

        # If a sufficient number of samples is collected, stop sampling
        if len(self.samples) >= self.n_samples:
            self._write_database()
            exit()

    def on_simulation_end(self, simulator):
        # Wrap everything up if simulation finished without collecting all/any samples.
        self._write_database()

    def _check_uncertainty(self, system):
        threshold_exceeded = {}

        sample_system = False
        sample_molecule = torch.zeros(
            system.n_replicas, system.n_molecules, device=system.device
        ).bool()

        # Check if a sample is needed
        for prop in self.thresholds:
            # Get variance from simulator
            prop_var = system.properties["{:s}_var".format(prop)]

            # Reshaping depending if property is atomic or not
            shapes = prop_var.shape
            if shapes[2] == system.max_n_atoms:
                prop_var.view(*shapes[:2], -1)
                uncertainty = torch.sqrt(torch.sum(prop_var, dim=-1))
            else:
                prop_var.view(*shapes[:1], -1)
                uncertainty = torch.sqrt(torch.sum(prop_var, dim=-1, keepdim=True))

            # Check if uncertainty threshold is exceeded
            threshold_exceeded[prop] = self.thresholds[prop] < uncertainty

            # Checks if a) a sample is needed, b) for which replica/molecule a sample is needed, c) for which atom
            if torch.any(threshold_exceeded[prop]):
                # Check for which molecule/replica samples are required
                sample_molecule = sample_molecule | torch.any(
                    threshold_exceeded[prop], dim=-1
                )
                # Overall sample required
                sample_system = True

        return sample_system, sample_molecule, threshold_exceeded

    def _collect_samples(self, system, sample_molecule, threshold_exceeded):
        # Get structures in the form of ASE atoms (R x M is flattened)
        atoms = system.get_ase_atoms(internal_units=False)

        # Collect all replicas and molecules which need sampling
        idx_c = 0
        for idx_r in range(system.n_replicas):
            for idx_m in range(system.n_molecules):

                # Get the atoms and store the thresholds
                if sample_molecule[idx_r, idx_m] == 1:
                    self.samples.append(atoms[idx_c])
                    sample_thresholds = {}
                    for prop in threshold_exceeded:
                        sample_thresholds[prop] = (
                            threshold_exceeded[prop][idx_r, idx_m]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    self.samples_thresholds.append(sample_thresholds)
                idx_c += 1

    def _reset_system(self, system, mask=None):
        if mask is None:
            system.positions = copy.deepcopy(self.init_positions)
            system.forces = copy.deepcopy(self.init_forces)
            system.cells = copy.deepcopy(self.init_cells)
            self.initializer.initialize_system(system)
        else:
            system.positions[mask == 1, ...] = copy.deepcopy(
                self.init_positions[mask == 1, ...]
            )
            system.forces[mask == 1, ...] = copy.deepcopy(
                self.init_forces[mask == 1, ...]
            )
            if self.init_cells is not None:
                system.cells[mask == 1, ...] = copy.deepcopy(
                    self.init_cells[mask == 1, ...]
                )
            self.initializer.initialize_system(system, mask=mask)

    def _write_database(self):
        if len(self.samples) > 0:
            dataset = AtomsData(
                self.dataset, available_properties=self.samples_thresholds[0]
            )
            dataset.add_systems(self.samples, self.samples_thresholds)
            logging.info(
                "{:d} samples written to {:s}.".format(len(self.samples), self.dataset)
            )
        else:
            logging.info("No samples collected.")
