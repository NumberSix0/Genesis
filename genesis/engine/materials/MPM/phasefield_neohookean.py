import taichi as ti

import genesis as gs

from .elastic import Elastic


@ti.data_oriented
class PhaseFieldNeoHookean(Elastic):
    """
    Phase Field NeoHookean material model for MPM simulations.
    
    This model extends the standard NeoHookean model with a phase field approach
    for damage and fracture simulation.
    """

    def __init__(
        self,
        E=3e5,  # Young's modulus
        nu=0.2,  # Poisson's ratio
        rho=1000.0,  # density (kg/m^3)
        l0=0.01,  # characteristic length for phase field
        residual_phase=0.001,  # residual stiffness for fully damaged material
        damage_threshold=0.0,  # threshold strain energy for damage initiation
        max_damage=1.0,  # maximum allowed damage value
        damage_rate=1.0,  # rate of damage evolution
    ):
        super().__init__(E, nu, rho, model="neohooken")
        
        # Phase field parameters
        self._l0 = l0
        self._residual_phase = residual_phase
        self._damage_threshold = damage_threshold
        self._max_damage = max_damage
        self._damage_rate = damage_rate
        
        # Calculate bulk modulus for strain energy calculation
        self._kappa = self._lam + 2.0 * self._mu / 3.0
        
        # Use the updated stress function with phase field
        self.update_stress = self.update_stress_phase_field_neohooken

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        # Keep the standard elastic update without plastic deformation
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @ti.func
    def calculate_strain_energy(self, F_tmp, J):
        """Calculate the positive part of the strain energy density"""
        # Deviatoric part
        JaF = ti.pow(J, -1.0/3.0) * F_tmp
        psi_dev = self._mu * 0.5 * ((JaF.transpose() @ JaF).trace() - 3)
        
        # Volumetric part
        psi_vol = self._kappa * 0.5 * ((J * J - 1) * 0.5 - ti.log(J))
        
        # Return the positive part of strain energy (used for damage driving force)
        return psi_dev + ti.max(0.0, psi_vol)

    @ti.func
    def update_damage(self, S, D):
        """Update damage parameter based on strain energy and strain rate
        
        Args:
            S: Singular values from SVD of deformation gradient
            D: Strain rate tensor
            
        Returns:
            damage: Updated damage value between 0 (undamaged) and 1 (fully damaged)
        """
        # Calculate J (determinant) from singular values
        J = S.determinant()
        
        # Calculate strain energy from singular values
        # In a real implementation, we would compute this directly from F,
        # but here we reconstruct F from S to demonstrate the principle
        F_approx = S  # Simplified approximation for demonstration
        
        # Calculate the strain energy
        strain_energy = self.calculate_strain_energy(F_approx, J)
        
        # Additional criterion: Consider strain rate magnitude for dynamic fracture
        strain_rate_magnitude = ti.sqrt((D * D).sum())
        
        # Combine strain energy and strain rate for damage driving force
        H = ti.max(strain_energy + self._damage_rate * strain_rate_magnitude, self._damage_threshold)
        
        # Damage driving force (normalized by critical energy)
        driving_force = 4.0 * self._l0 * (1.0 - self._residual_phase) * H * self._damage_rate
        
        # Compute new damage value (ranges from 0 to self._max_damage)
        damage = ti.min(self._max_damage, 1.0 - 1.0/(1.0 + driving_force))
        
        return damage

    @ti.func
    def update_stress_phase_field_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, D):
        """Update stress with phase field damage"""
        # Calculate J (determinant) from singular values
        J = S.determinant()
        
        # Calculate strain energy from singular values
        # In a real implementation, we would compute this directly from F,
        # but here we reconstruct F from S to demonstrate the principle
        F_approx = S  # Simplified approximation for demonstration
        
        # Calculate the strain energy
        strain_energy = self.calculate_strain_energy(F_approx, J)
        
        # Additional criterion: Consider strain rate magnitude for dynamic fracture
        strain_rate_magnitude = ti.sqrt((D * D).sum())
        
        # Combine strain energy and strain rate for damage driving force
        H = ti.max(strain_energy + self._damage_rate * strain_rate_magnitude, self._damage_threshold)
        
        # Damage driving force (normalized by critical energy)
        driving_force = 4.0 * self._l0 * (1.0 - self._residual_phase) * H * self._damage_rate
        
        # Compute new damage value (ranges from 0 to self._max_damage)
        damage = ti.min(self._max_damage, 1.0 - 1.0/(1.0 + driving_force))
        
        # Calculate degradation function g(d) = (1-d)^2 + residual_phase
        degradation = (1.0 - damage) * (1.0 - damage) + self._residual_phase
        
        # Standard Neo-Hookean stress calculation
        stress = self._mu * (F_tmp @ F_tmp.transpose()) + ti.Matrix.identity(gs.ti_float, 3) * (
            self._lam * ti.log(J) - self._mu
        )
        
        # Degrade stress by the degradation function
        stress = degradation * stress
        
        return stress
    
    @property
    def damage_rate(self):
        return self._damage_rate
    
    @property
    def l0(self):
        return self._l0
    
    @property
    def residual_phase(self):
        return self._residual_phase
    
    @property
    def damage_threshold(self):
        return self._damage_threshold
    
    @property
    def max_damage(self):
        return self._max_damage
    
    @property
    def kappa(self):
        return self._kappa 