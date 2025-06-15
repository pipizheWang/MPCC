#!/usr/bin/env python3
import numpy as np


class MPCCTrajectory:
    def __init__(self, spline_file):
        """初始化MPCC轨迹处理器

        Args:
            spline_file: 包含样条参数的NPZ文件路径
        """
        self.load_spline_parameters(spline_file)
        self.current_theta = 0.0

    def load_spline_parameters(self, filename):
        """从NPZ文件加载样条参数"""
        try:
            data = np.load(filename)
            self.total_length = float(data['total_length'])
            self.num_segments = int(data['num_segments'])
            self.s_params = data['s_params']
            self.x_coeffs = data['x_coeffs']
            self.y_coeffs = data['y_coeffs']
            self.z_coeffs = data['z_coeffs']
            print(f"成功加载轨迹，总长度: {self.total_length:.2f}, 段数: {self.num_segments}")
        except Exception as e:
            print(f"加载样条参数失败: {e}")
            raise

    def _find_segment(self, theta):
        """找到对应弧长参数的样条段索引

        Args:
            theta: 弧长参数

        Returns:
            seg_idx: 段索引
            local_s: 段内局部参数
        """
        # 确保theta在有效范围内
        theta = np.clip(theta, 0.0, self.total_length)

        # 处理边界情况
        if theta >= self.total_length:
            return self.num_segments - 1, self.s_params[-1] - self.s_params[-2]

        # 找到对应的样条段
        seg_idx = np.searchsorted(self.s_params, theta) - 1
        seg_idx = max(0, min(seg_idx, self.num_segments - 1))

        # 计算局部参数
        local_s = theta - self.s_params[seg_idx]

        return seg_idx, local_s

    def get_position(self, theta):
        """根据弧长参数theta计算位置

        Args:
            theta: 弧长参数

        Returns:
            位置向量 [x, y, z]^T
        """
        seg_idx, local_s = self._find_segment(theta)

        # 计算位置（三次样条: a0 + a1*s + a2*s^2 + a3*s^3）
        x = np.polyval([self.x_coeffs[seg_idx, 3], self.x_coeffs[seg_idx, 2],
                        self.x_coeffs[seg_idx, 1], self.x_coeffs[seg_idx, 0]], local_s)
        y = np.polyval([self.y_coeffs[seg_idx, 3], self.y_coeffs[seg_idx, 2],
                        self.y_coeffs[seg_idx, 1], self.y_coeffs[seg_idx, 0]], local_s)
        z = np.polyval([self.z_coeffs[seg_idx, 3], self.z_coeffs[seg_idx, 2],
                        self.z_coeffs[seg_idx, 1], self.z_coeffs[seg_idx, 0]], local_s)

        return np.array([[x], [y], [z]])

    def get_tangent(self, theta):
        """根据弧长参数theta计算切线向量

        Args:
            theta: 弧长参数

        Returns:
            归一化的切线向量 [tx, ty, tz]^T
        """
        seg_idx, local_s = self._find_segment(theta)

        # 计算导数（二次多项式: a1 + 2*a2*s + 3*a3*s^2）
        dx = self.x_coeffs[seg_idx, 1] + 2 * self.x_coeffs[seg_idx, 2] * local_s + \
             3 * self.x_coeffs[seg_idx, 3] * local_s ** 2
        dy = self.y_coeffs[seg_idx, 1] + 2 * self.y_coeffs[seg_idx, 2] * local_s + \
             3 * self.y_coeffs[seg_idx, 3] * local_s ** 2
        dz = self.z_coeffs[seg_idx, 1] + 2 * self.z_coeffs[seg_idx, 2] * local_s + \
             3 * self.z_coeffs[seg_idx, 3] * local_s ** 2

        tangent = np.array([[dx], [dy], [dz]])

        # 归一化切线向量
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            tangent = tangent / norm
        else:
            # 防止除零，返回默认方向
            tangent = np.array([[1.0], [0.0], [0.0]])

        return tangent

    def get_curvature(self, theta):
        """计算曲率向量（二阶导数）

        Args:
            theta: 弧长参数

        Returns:
            曲率向量 [ddx, ddy, ddz]^T
        """
        seg_idx, local_s = self._find_segment(theta)

        # 计算二阶导数（线性: 2*a2 + 6*a3*s）
        ddx = 2 * self.x_coeffs[seg_idx, 2] + 6 * self.x_coeffs[seg_idx, 3] * local_s
        ddy = 2 * self.y_coeffs[seg_idx, 2] + 6 * self.y_coeffs[seg_idx, 3] * local_s
        ddz = 2 * self.z_coeffs[seg_idx, 2] + 6 * self.z_coeffs[seg_idx, 3] * local_s

        return np.array([[ddx], [ddy], [ddz]])

    def project_point(self, pos, theta_guess=None, max_iter=10, tol=1e-4):
        """将位置投影到轨迹上找到最近点对应的弧长参数theta

        使用牛顿法求解最小化问题：min_theta ||pos - p_d(theta)||^2

        Args:
            pos: 当前位置 [x, y, z]
            theta_guess: 初始theta猜测值
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            theta: 最近点对应的弧长参数
        """
        if theta_guess is None:
            theta_guess = self.current_theta

        theta = theta_guess
        pos = np.array(pos).reshape(3, 1)

        # 牛顿迭代法
        for i in range(max_iter):
            # 获取当前点的位置、切线和曲率
            p_d = self.get_position(theta)
            t_d = self.get_tangent(theta)
            c_d = self.get_curvature(theta)

            # 计算误差
            e = pos - p_d

            # 计算目标函数的梯度：g = -2 * t_d^T * e
            grad = -2.0 * np.dot(t_d.T, e)[0, 0]

            # 计算Hessian：H = 2 * (t_d^T * t_d - e^T * c_d)
            hess = 2.0 * (np.dot(t_d.T, t_d)[0, 0] - np.dot(e.T, c_d)[0, 0])

            # 防止除零
            if abs(hess) < 1e-6:
                hess = 1e-6 if hess >= 0 else -1e-6

            # 牛顿步长
            delta_theta = -grad / hess

            # 线搜索（可选，提高稳定性）
            alpha = 1.0
            theta_new = theta + alpha * delta_theta

            # 限制步长，防止越界
            if abs(delta_theta) > 0.5:
                delta_theta = 0.5 * np.sign(delta_theta)
                theta_new = theta + delta_theta

            # 确保在有效范围内
            theta_new = np.clip(theta_new, 0.0, self.total_length)

            # 检查收敛
            if abs(theta_new - theta) < tol:
                break

            theta = theta_new

        # 更新当前theta
        self.current_theta = theta

        return theta

    def compute_errors(self, pos, theta=None):
        """计算轮廓误差和滞后误差

        Args:
            pos: 当前位置 [x, y, z]
            theta: 弧长参数（如果为None则计算投影点）

        Returns:
            e_c: 轮廓误差向量（垂直于轨迹的误差）
            e_l: 滞后误差标量（沿轨迹方向的误差）
            t_d: 切线向量
        """
        pos = np.array(pos).reshape(3, 1)

        # 如果没有指定theta，找到投影点
        if theta is None:
            theta = self.project_point(pos)

        # 计算参考位置和切线
        p_d = self.get_position(theta)
        t_d = self.get_tangent(theta)

        # 计算位置误差
        e = pos - p_d

        # 计算滞后误差（投影到切线方向）
        e_l = np.dot(t_d.T, e)[0, 0]

        # 计算轮廓误差（与切线垂直的误差分量）
        e_c = e - e_l * t_d

        return e_c, e_l, t_d

    def get_trajectory_info(self, theta):
        """获取轨迹在指定弧长处的完整信息

        Args:
            theta: 弧长参数

        Returns:
            dict: 包含位置、切线、曲率等信息
        """
        pos = self.get_position(theta)
        tangent = self.get_tangent(theta)
        curvature = self.get_curvature(theta)

        # 计算曲率半径
        curvature_norm = np.linalg.norm(curvature)
        radius_of_curvature = 1.0 / curvature_norm if curvature_norm > 1e-6 else float('inf')

        return {
            'position': pos,
            'tangent': tangent,
            'curvature': curvature,
            'radius_of_curvature': radius_of_curvature,
            'theta': theta,
            'progress_ratio': theta / self.total_length
        }