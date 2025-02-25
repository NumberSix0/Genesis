import taichi as ti

import genesis as gs

from .base import Base

@ti.data_oriented
class CorticalBone:
    def __init__(self, E=20e9, nu=0.3, rho=1900,
                 yield_comp=200e6, yield_tens=150e6,
                 hardening=0.1, damage_rate=2.0):
        self.mu = E / (2 * (1 + nu))  # 剪切模量
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé参数
        self.yield_comp = yield_comp  # 压缩屈服应力
        self.yield_tens = yield_tens  # 拉伸屈服应力
        self.hardening = hardening    # 硬化系数
        self.damage_rate = damage_rate
        
        # 历史变量
        self.add_field('equiv_plastic_strain', ti.f32)
        self.add_field('damage', ti.f32)

    @ti.func
    def update_stress(self, F, U, S, V, dt):
        # 弹性预测
        F_e = F @ V.transpose()  # 塑性变形后的弹性变形梯度
        J = F_e.determinant()
        C = F_e.transpose() @ F_e
        S_e = self.mu * (C - ti.Matrix.identity(float,3)) + self.lam * ti.log(J) * ti.Matrix.identity(float,3)
        
        # 各向异性屈服准则
        sigma = (F_e @ S_e @ F_e.transpose()) / J  # 柯西应力
        principal_stress = ti.sym_eig(sigma)  # 主应力计算
        tens_stress = ti.max(principal_stress[0], 0.0)
        comp_stress = ti.min(principal_stress[2], 0.0)
        
        # 损伤演化
        f_tens = tens_stress - self.yield_tens
        f_comp = comp_stress - self.yield_comp
        delta_damage = ti.max(f_tens/self.yield_tens, f_comp/self.yield_comp) * dt * self.damage_rate
        self.damage = ti.min(self.damage + delta_damage, 1.0)
        
        # 有效应力
        effective_stress = (1 - self.damage) * S_e
        return effective_stress @ V  # 返回第一P-K应力