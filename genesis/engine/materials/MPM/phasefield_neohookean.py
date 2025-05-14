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
        E=0.8e6,  # Young's modulus
        nu=0.48,  # Poisson's ratio
        rho=1060.0,  # density (kg/m^3)
        l0=0.002,  # characteristic length for phase field
        residual_phase=0.01,  # residual stiffness for fully damaged material
        damage_threshold=2.0,  # threshold strain energy for damage initiation
        max_damage=1.0,  # maximum allowed damage value
        damage_rate=5.0,  # rate of damage evolution
        delete_threshold=0.05,  # threshold for particle deletion
        one_over_sigma_c=0.2,  # inverse of critical energy release rate
    ):
        super().__init__(E, nu, rho, model="neohooken")
        
        # Phase field parameters
        self._l0 = l0
        self._residual_phase = residual_phase
        self._damage_threshold = damage_threshold
        self._max_damage = max_damage
        self._damage_rate = damage_rate
        self._delete_threshold = delete_threshold
        self._one_over_sigma_c = one_over_sigma_c
        self._allow_damage = True
        
        # Calculate bulk modulus for strain energy calculation
        self._kappa = self._lam + 2.0 * self._mu / 3.0
        
        # Use the updated stress function with phase field
        self.update_stress = self.update_stress_phase_field_neohooken
        
        # Store maximum historical strain energy
        self._H_max = 1e10  # Maximum allowable strain energy
        self._pf_Fp = 1.0   # Phase field driving force term

    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        # Keep the standard elastic update without plastic deformation
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @ti.func
    def calculate_strain_energy(self, F, J):
        """Calculate the strain energy density according to Borden's implementation"""
        # Calculate Ja^(-1/dim) * F
        JaF = ti.pow(J, -1.0/3.0) * F
        
        # Deviatoric part of the strain energy
        psi_dev = self._mu * 0.5 * ((JaF.transpose() @ JaF).trace() - 3)
        
        # Volumetric part of the strain energy
        psi_vol = self._kappa * 0.5 * ((J * J - 1) * 0.5 - ti.log(J))
        
        # Split energy into positive and negative parts for damage evolution
        # Only positive part drives damage (Borden's implementation)
        psi_pos = psi_dev
        if J >= 1.0:
            psi_pos += psi_vol
            
        return psi_pos

    @ti.func
    def update_phase_field_Fp(self, psi_pos, H):
        """Update the phase field driving force parameter"""
        new_H = H
        new_pf_Fp = self._pf_Fp
        
        if psi_pos > H:
            new_H = ti.min(psi_pos, self._H_max)
            new_pf_Fp = 4.0 * self._l0 * (1.0 - self._residual_phase) * new_H * self._one_over_sigma_c + 1.0
            
        return new_H, new_pf_Fp

    @ti.func
    def calculate_damage(self, pf_Fp):
        """Calculate damage parameter c based on phase field driving force"""
        # c = 1 is undamaged, c = 0 is fully damaged
        # Following Borden's implementation
        c = 1.0 / pf_Fp
        
        # Limit damage to max_damage
        c = ti.max(1.0 - self._max_damage, c)
        
        return c

    @ti.func
    def update_damage(self, F_tmp, J, D):
        """
        Calculate damage parameter based on deformation history
        
        Args:
            F_tmp: Deformation gradient tensor
            J: Determinant of deformation gradient
            D: Strain rate tensor (for dynamic fracture)
            
        Returns:
            c: Damage parameter (1 = undamaged, 0 = fully damaged)
        """
        # Calculate positive part of strain energy
        psi_pos = self.calculate_strain_energy(F_tmp, J)
        
        # Consider strain rate magnitude for dynamic fracture (optional)
        # if D is not None:
        #strain_rate_magnitude = ti.sqrt((D * D).sum())
        #psi_pos += self._damage_rate * strain_rate_magnitude
        
        # Start with current values (these would be stored per particle in real implementation)
        H = self._damage_threshold
        pf_Fp = self._pf_Fp
        
        # Update history-dependent terms
        H, pf_Fp = self.update_phase_field_Fp(psi_pos, H)
        
        # Calculate damage parameter c (1 = undamaged, 0 = fully damaged)
        c = self.calculate_damage(pf_Fp)
        
        return c

    @ti.func
    def update_stress_phase_field_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir, D, damage):
        """
        Update stress with phase field damage following Borden's implementation
        
        Args:
            U, S, V: SVD decomposition of deformation gradient
            F_tmp, F_new: Deformation gradient (before and after plastic update)
            J: Determinant of deformation gradient
            Jp: Plastic component of J
            actu: Actuator signal (not used)
            m_dir: Material direction (not used)
            D: Strain rate tensor
            damage: Damage parameter c (1 = undamaged, 0 = fully damaged)
            
        Returns:
            stress: Cauchy stress tensor with phase field damage
        """
        # Calculate damage parameter
        c = damage
        
        # Calculate degradation function g(c) = c^2 + residual_phase
        g = c * c + self._residual_phase
        
        # Standard Neo-Hookean stress calculation (matching elastic.py)
        stress = self._mu * (F_tmp @ F_tmp.transpose()) + ti.Matrix.identity(gs.ti_float, 3) * (
            self._lam * ti.log(J) - self._mu
        )
        
        # For more accurate fracture simulation, decompose stress into volumetric and deviatoric parts
        # while ensuring the sum equals the standard stress from elastic.py
        
        # First calculate F^T * F 
        # FTF = F_tmp.transpose() @ F_tmp
        # trFTF = FTF.trace()  # 未使用
        
        # Calculate B = F * F^T
        B = F_tmp @ F_tmp.transpose()
        
        # Calculate deviatoric part of B
        trB = B.trace()
        devB = B - ti.Matrix.identity(gs.ti_float, 3) * (trB / 3.0)
        
        # Calculate deviatoric stress using the mu parameter
        dev_stress = self._mu * devB
        
        # Calculate volumetric stress as the difference between standard_stress and dev_stress
        # This ensures that dev_stress + vol_stress = standard_stress
        vol_stress = stress - dev_stress
        
        #dev_stress = self._mu * (F_tmp @ F_tmp.transpose())
        #vol_stress = ti.Matrix.identity(gs.ti_float, 3) * (self._lam * ti.log(J) - self._mu)
        # Apply damage degradation based on Borden's approach:
        # g(c) * dev_stress + (J >= 1 ? g(c) * vol_stress : vol_stress)
        if J >= 1.0:
            # For expansion (J >= 1), degrade both deviatoric and volumetric parts
            stress = g * dev_stress + g * vol_stress
        else:
            # For compression (J < 1), only degrade deviatoric part
            stress = g * dev_stress + vol_stress
            
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
        
    @property
    def one_over_sigma_c(self):
        return self._one_over_sigma_c
        
    @property
    def delete_threshold(self):
        return self._delete_threshold 