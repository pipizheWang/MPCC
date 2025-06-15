#!/usr/bin/env python3
import numpy as np
import time
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca


class MPCCOptimizer:
    def __init__(self, traj, dt=0.04, N=20):
        self.traj = traj
        self.dt = dt
        self.N = N

        # 权重参数
        self.q_contour = 50.0
        self.q_lag = 6000.0
        self.mu = 0.02
        self.r_dv = 12.0
        self.r_a = 1.0
        self.r_jerk = 100.0
        self.r_snap = 2.0

        # 约束参数
        self.max_accel = 2.2
        self.max_v_theta = 1.0
        self.max_dv_theta = 0.3
        self.max_jerk = 8.0
        self.max_snap = 8.0

        # 创建优化问题
        self.model_name = 'improved_mpcc_quad'
        self.solver = self._create_solver()

        # 状态管理
        self.last_solution = None
        self.is_initialized = False
        self.target_theta = 0.0

    def _create_model(self):
        """创建MPCC模型"""
        model = AcadosModel()
        model.name = self.model_name

        # 状态: [px, py, pz, vx, vy, vz, ax, ay, az, theta, v_theta]
        px = ca.SX.sym('px')
        py = ca.SX.sym('py')
        pz = ca.SX.sym('pz')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        vz = ca.SX.sym('vz')
        ax = ca.SX.sym('ax')
        ay = ca.SX.sym('ay')
        az = ca.SX.sym('az')
        theta = ca.SX.sym('theta')
        v_theta = ca.SX.sym('v_theta')

        x = ca.vertcat(px, py, pz, vx, vy, vz, ax, ay, az, theta, v_theta)

        # 控制输入：[jx, jy, jz, dv_theta]
        jx = ca.SX.sym('jx')
        jy = ca.SX.sym('jy')
        jz = ca.SX.sym('jz')
        dv_theta = ca.SX.sym('dv_theta')

        u = ca.vertcat(jx, jy, jz, dv_theta)

        # 系统动力学
        x_dot = ca.vertcat(
            vx, vy, vz,
            ax, ay, az,
            jx, jy, jz,
            v_theta, dv_theta
        )

        # 参数向量 [p_ref_x, p_ref_y, p_ref_z, t_ref_x, t_ref_y, t_ref_z]
        p_ref_x = ca.SX.sym('p_ref_x')
        p_ref_y = ca.SX.sym('p_ref_y')
        p_ref_z = ca.SX.sym('p_ref_z')
        t_ref_x = ca.SX.sym('t_ref_x')
        t_ref_y = ca.SX.sym('t_ref_y')
        t_ref_z = ca.SX.sym('t_ref_z')

        p = ca.vertcat(p_ref_x, p_ref_y, p_ref_z, t_ref_x, t_ref_y, t_ref_z)

        model.x = x
        model.u = u
        model.xdot = x_dot
        model.p = p
        model.f_expl_expr = x_dot
        model.f_impl_expr = x_dot - model.xdot

        return model

    def _setup_cost_function(self, ocp):
        """设置代价函数"""
        model = ocp.model
        x = model.x
        u = model.u
        p = model.p

        # 状态变量解包
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        ax, ay, az = x[6], x[7], x[8]
        theta, v_theta = x[9], x[10]

        # 控制输入（jerk）
        jx, jy, jz = u[0], u[1], u[2]
        dv_theta = u[3]

        # 参考位置和切线
        p_ref = ca.vertcat(p[0], p[1], p[2])
        t_ref = ca.vertcat(p[3], p[4], p[5])

        # 当前位置
        pos = ca.vertcat(px, py, pz)

        # 计算轨迹跟踪误差
        e = pos - p_ref
        e_l = ca.dot(e, t_ref)  # 滞后误差
        e_c = e - e_l * t_ref  # 轮廓误差

        # 阶段代价函数
        stage_cost = (
                self.q_contour * ca.dot(e_c, e_c) +
                self.q_lag * e_l ** 2 +
                self.r_jerk * (jx ** 2 + jy ** 2 + jz ** 2) +
                self.r_dv * dv_theta ** 2 -
                self.mu * v_theta
        )

        # 终端代价函数
        terminal_cost = (
                self.q_contour * 3.0 * ca.dot(e_c, e_c) +
                self.q_lag * 3.0 * e_l ** 2
        )

        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

    def _setup_constraints(self, ocp):
        """设置约束条件"""
        nx = int(ocp.model.x.size()[0])
        nu = int(ocp.model.u.size()[0])

        # 初始状态约束
        ocp.constraints.x0 = np.zeros(nx)

        # 控制输入约束
        ocp.constraints.lbu = np.array([
            -self.max_jerk, -self.max_jerk, -self.max_jerk, -self.max_dv_theta
        ])
        ocp.constraints.ubu = np.array([
            self.max_jerk, self.max_jerk, self.max_jerk, self.max_dv_theta
        ])
        ocp.constraints.idxbu = np.arange(nu)

        # 状态约束：加速度和v_theta限制 [ax, ay, az, v_theta, theta]
        ocp.constraints.lbx = np.array([
            -self.max_accel, -self.max_accel, -self.max_accel,  # 加速度下限
            0.02, 0.0  # v_theta最小值, theta最小值
        ])
        ocp.constraints.ubx = np.array([
            self.max_accel, self.max_accel, self.max_accel,  # 加速度上限
            self.max_v_theta, self.traj.total_length  # v_theta上限, theta上限
        ])
        ocp.constraints.idxbx = np.array([6, 7, 8, 10, 9])  # ax, ay, az, v_theta, theta

        # 终端状态约束
        ocp.constraints.lbx_e = np.array([
            -self.max_accel, -self.max_accel, -self.max_accel,
            0.02, 0.0
        ])
        ocp.constraints.ubx_e = np.array([
            self.max_accel, self.max_accel, self.max_accel,
            self.max_v_theta, self.traj.total_length
        ])
        ocp.constraints.idxbx_e = np.array([6, 7, 8, 10, 9])

    def _create_solver(self):
        """创建acados求解器"""
        model = self._create_model()
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N * self.dt

        self._setup_cost_function(ocp)
        self._setup_constraints(ocp)

        # 求解器选项
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1

        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.nlp_solver_max_iter = 25

        ocp.solver_options.qp_solver_cond_N = min(self.N, 8)
        ocp.solver_options.qp_solver_warm_start = 2
        ocp.solver_options.qp_solver_iter_max = 120

        # 求解精度
        ocp.solver_options.nlp_solver_tol_stat = 5e-4
        ocp.solver_options.nlp_solver_tol_eq = 5e-4
        ocp.solver_options.nlp_solver_tol_ineq = 5e-4
        ocp.solver_options.nlp_solver_tol_comp = 5e-4

        ocp.solver_options.print_level = 0
        ocp.parameter_values = np.zeros(6)

        try:
            solver = AcadosOcpSolver(ocp, json_file=f'{self.model_name}_ocp.json')
            return solver
        except Exception as e:
            print(f"创建acados求解器失败: {e}")
            raise

    def _initialize_theta(self, pos):
        """初始化theta值"""
        closest_theta, min_dist = self.find_closest_theta(pos)
        print(f"初始化：找到最近轨迹点 theta={closest_theta:.3f}, 距离={min_dist:.3f}m")

        if min_dist > 3.0:
            theta_init = 0.0
            print(f"距离轨迹过远({min_dist:.2f}m)，从起点开始")
        else:
            theta_init = closest_theta

        self.target_theta = theta_init
        self.is_initialized = True

        return theta_init, 0.1, np.zeros(3)  # theta, v_theta, acc

    def _update_theta(self, pos):
        """更新theta值（非初始化情况）"""
        if self.last_solution is None:
            return self.target_theta, 0.1, np.zeros(3)

        last_theta = self.last_solution['theta']
        last_v_theta = self.last_solution['v_theta']
        last_acc = self.last_solution.get('acc_state', np.zeros(3))

        # 处理轨迹完成重置
        if last_theta >= self.traj.total_length * 0.95:
            closest_theta, min_dist = self.find_closest_theta(pos)
            if min_dist < 2.0:
                self.target_theta = closest_theta
                print(f"平滑重置到最近点: theta={closest_theta:.3f}")
            else:
                self.target_theta = 0.0
                print("重置到轨迹起点")
            return self.target_theta, 0.1, np.zeros(3)
        else:
            # theta预测：基于上一次的v_theta
            max_step = 0.05  # 适当增大允许的最大步长
            predicted_theta = last_theta + min(last_v_theta, 0.3) * self.dt
            predicted_theta = min(predicted_theta, last_theta + max_step)
            theta_init = np.clip(predicted_theta, 0.0, self.traj.total_length)
            v_theta_init = np.clip(last_v_theta, 0.02, self.max_v_theta)

            return theta_init, v_theta_init, last_acc

    def _set_reference_trajectory(self, theta_init, v_theta_init):
        """设置参考轨迹参数
        根据初始theta计算得到预测裕度范围的位置和切线"""
        for i in range(self.N + 1):
            theta_pred = theta_init

            if i > 0:
                # 动态预测：基于v_theta的动态步长
                step = min(v_theta_init * self.dt, 0.015)  # 限制最大步长
                theta_pred = min(theta_pred + step * i, self.traj.total_length * 0.95)

            try:
                p_ref = self.traj.get_position(theta_pred).flatten()
                t_ref = self.traj.get_tangent(theta_pred).flatten()
            except:
                p_ref = self.traj.get_position(0.0).flatten()
                t_ref = self.traj.get_tangent(0.0).flatten()

            p_ref[2] -= 1.5  # 高度偏移 Todo

            params = np.concatenate([p_ref, t_ref])
            self.solver.set(i, 'p', params)

    def _set_initial_guess(self, theta_init, v_theta_init, warm_start=True):
        """设置初始猜测"""
        # 状态初始猜测
        for i in range(self.N + 1):
            # 使用动态theta步长：v_theta * dt
            theta_step = v_theta_init * self.dt
            theta_pred = theta_init + i * theta_step
            theta_pred = min(theta_pred, self.traj.total_length * 0.95)

            try:
                p_ref = self.traj.get_position(theta_pred).flatten()
            except:
                p_ref = self.traj.get_position(0.0).flatten()

            if warm_start and self.last_solution is not None and i + 1 < len(self.last_solution['x_traj']):
                x_guess = self.last_solution['x_traj'][i + 1].copy()
                # 平滑更新theta，而不是完全覆盖
                old_theta = x_guess[9]
                x_guess[9] = 0.7 * old_theta + 0.3 * theta_pred  # 加权平均
                x_guess[10] = np.clip(x_guess[10], 0.02, self.max_v_theta)
            else:
                x_guess = np.array([
                    p_ref[0], p_ref[1], p_ref[2],  # 位置
                    0.0, 0.0, 0.0,  # 速度
                    0.0, 0.0, 0.0,  # 加速度
                    theta_pred, v_theta_init  # theta状态
                ])

            self.solver.set(i, 'x', x_guess)

        # 控制输入初始猜测
        for i in range(self.N):
            if warm_start and self.last_solution is not None and i + 1 < len(self.last_solution['u_traj']):
                u_guess = self.last_solution['u_traj'][i + 1].copy()
                # 限制控制输入在合理范围内
                u_guess[:3] = np.clip(u_guess[:3], -self.max_jerk * 0.6, self.max_jerk * 0.6)
                u_guess[3] = np.clip(u_guess[3], -self.max_dv_theta, self.max_dv_theta)
            else:
                u_guess = np.array([0.0, 0.0, 0.0, 0.0])  # 初始jerk为零

            self.solver.set(i, 'u', u_guess)

    def solve(self, pos, vel, warm_start=True):
        """改进的求解函数"""
        start_time = time.time()

        pos = np.array(pos).flatten()
        vel = np.array(vel).flatten()

        # theta管理策略
        if not self.is_initialized:
            theta_init, v_theta_init, acc_init = self._initialize_theta(pos)
        else:
            theta_init, v_theta_init, acc_init = self._update_theta(pos)

        # 构建初始状态 [px, py, pz, vx, vy, vz, ax, ay, az, theta, v_theta]
        x0 = np.array([
            pos[0], pos[1], pos[2],  # 位置
            vel[0], vel[1], vel[2],  # 速度
            acc_init[0], acc_init[1], acc_init[2],  # 加速度
            theta_init, v_theta_init  # theta状态
        ])

        # 设置参考轨迹
        self._set_reference_trajectory(theta_init, v_theta_init)

        # 设置初始状态约束
        self.solver.set(0, 'x', x0)
        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)

        # 设置初始猜测
        self._set_initial_guess(theta_init, v_theta_init, warm_start)

        # 求解优化问题
        status = self.solver.solve()

        # 直接提取优化器结果（无平滑处理）
        try:
            u0 = self.solver.get(0, 'u')  # 当前控制输入
            x1 = self.solver.get(1, 'x')  # 下一状态

            # 直接使用优化器结果
            jerk_cmd = u0[:3]
            acc_cmd = x1[6:9]  # 下一时刻的加速度状态
            current_theta = x1[9]
            current_v_theta = x1[10]

            # 安全限制
            acc_norm = np.linalg.norm(acc_cmd)
            if acc_norm > self.max_accel:
                acc_cmd = acc_cmd * self.max_accel / acc_norm

            jerk_norm = np.linalg.norm(jerk_cmd)
            if jerk_norm > self.max_jerk:
                jerk_cmd = jerk_cmd * self.max_jerk / jerk_norm

            # 状态范围检查
            current_theta = np.clip(current_theta, 0.0, self.traj.total_length)
            current_v_theta = np.clip(current_v_theta, 0.02, self.max_v_theta)

            # 保存完整轨迹信息
            x_traj = [self.solver.get(i, 'x') for i in range(self.N + 1)]
            u_traj = [self.solver.get(i, 'u') for i in range(self.N)]

        except Exception as e:
            print(f"获取求解结果失败: {e}")
            acc_cmd = acc_init if np.linalg.norm(acc_init) > 0 else np.zeros(3)
            jerk_cmd = np.zeros(3)
            current_theta = theta_init
            current_v_theta = v_theta_init
            x_traj = []
            u_traj = []

        # 保存求解结果
        self.last_solution = {
            'acc_cmd': acc_cmd,
            'jerk_cmd': jerk_cmd,
            'acc_state': acc_cmd,
            'theta': current_theta,
            'v_theta': current_v_theta,
            'x_traj': x_traj,
            'u_traj': u_traj,
            'status': status,
            'solve_time': time.time() - start_time
        }

        return acc_cmd, current_theta, current_v_theta

    def find_closest_theta(self, pos):
        """找到距离当前位置最近的theta值"""
        pos = np.array(pos).flatten()
        min_dist = float('inf')
        best_theta = 0.0

        # 粗搜索
        num_samples = min(150, int(self.traj.total_length * 1.5))
        theta_samples = np.linspace(0, self.traj.total_length, num_samples)

        for theta in theta_samples:
            try:
                ref_pos = self.traj.get_position(theta).flatten()
                dist = np.linalg.norm(pos - ref_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_theta = theta
            except:
                continue

        # 精细搜索
        search_range = self.traj.total_length / num_samples * 0.5
        fine_samples = np.linspace(
            max(0, best_theta - search_range),
            min(self.traj.total_length, best_theta + search_range),
            20
        )

        for theta in fine_samples:
            try:
                ref_pos = self.traj.get_position(theta).flatten()
                dist = np.linalg.norm(pos - ref_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_theta = theta
            except:
                continue

        return best_theta, min_dist

    def get_solution_info(self):
        """获取求解器信息"""
        if self.last_solution is None:
            return {"status": "未执行求解", "solve_time": 0.0}

        return {
            'status': self.last_solution['status'],
            'solve_time': self.last_solution['solve_time'] * 1000,  # 转换为毫秒
            'theta': self.last_solution['theta'],
            'v_theta': self.last_solution['v_theta']
        }