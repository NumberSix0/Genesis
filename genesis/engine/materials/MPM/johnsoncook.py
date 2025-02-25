from .base import Base
import genesis as gs
import taichi as ti

##############################
### 韧性损伤Johnson-Cook模型 ###
##############################
@ti.data_oriented
class JohnsonCook(Base):
    """
    Johnson-Cook塑性模型, 适用于高速冲击模拟
    Johnson GR, Cook WH. A constitutive model and data for metals subjected to large strains, 
    high strain rates and high temperatures. 1983.
    """
    def __init__(
        self,
        E=210e9, 
        nu=0.3,
        rho=7850,
        lam=None,
        mu=None,
        sampler='pbs',
        yield_stress=500e6,
        # JC模型参数（典型钢材）
        A = 500e6,    # 初始屈服应力
        B = 600e6,   # 硬化模量
        n = 0.32,     # 硬化指数
        C = 0.015,    # 应变率系数
        m = 1.56,     # 热软化指数
        # 状态变量
        eps_p = ti.field(gs.ti_float),  # 等效塑性应变
        temperature = ti.field(gs.ti_float),
    ):
        super().__init__(E, nu, rho, lam, mu, sampler)
        
    @ti.func
    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        # 塑性变形更新逻辑
        eps_p = ti.sqrt(2/3)*ti.log(Jp)
        eps_dot = ... # 计算应变率
        T = ... # 获取温度
        
        # Johnson-Cook屈服条件
        sigma_y = (self.A + self.B*eps_p**self.n) * (1 + self.C*ti.log(eps_dot)) * (1 - (T/self.Tmelt)**self.m)
        
        # 径向返回映射
        deviatoric = S - S.trace()/3*ti.Matrix.identity(3)
        sigma_eq = ti.sqrt(3/2)*deviatoric.norm()
        if sigma_eq > sigma_y:
            ratio = sigma_y / sigma_eq
            S_new = ratio * deviatoric + S.trace()/3*ti.Matrix.identity(3)
            F_new = U @ S_new @ V.transpose()
            return F_new, S_new, Jp * ti.det(F_new)/J
        return F_tmp, S, Jp

    @ti.func
    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # 计算柯西应力
        return 2 * self._mu * (F_new - U @ V.transpose()) + self._lam * ti.log(J) * ti.Matrix.identity(3)