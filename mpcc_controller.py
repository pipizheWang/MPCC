#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, TwistStamped
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.clock import Clock
import csv
import os
import time
from datetime import datetime

from .mpcc_trajectory import MPCCTrajectory
from .improved_mpcc_optimizer import ImprovedMPCCOptimizer
from .ETHinspired_MPCC_optimizer import ETHInspiredMPCCOptimizer
from .mpcc_optimizer import MPCCOptimizer



class MPCCController(Node):
    def __init__(self, name):
        """初始化MPCC控制器节点"""
        super().__init__(name)
        self.get_logger().info("MPCC控制器初始化: %s" % name)

        # 声明参数
        self.declare_parameter('spline_file', 'spline_parameters.npz')
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('mpcc_rate', 25.0)
        self.declare_parameter('max_accel', 2.0)
        self.declare_parameter('traj_mode', False)
        self.declare_parameter('hover_height', 1.5)

        # 轨迹接入策略参数
        self.declare_parameter('approach_distance', 1.5)
        self.declare_parameter('approach_speed', 0.6)
        
        # P控制器参数
        self.declare_parameter('kp_pos_x', 1.2)
        self.declare_parameter('kp_pos_y', 1.2) 
        self.declare_parameter('kp_pos_z', 2.5)
        self.declare_parameter('kp_vel_x', 0.8)
        self.declare_parameter('kp_vel_y', 0.8)
        self.declare_parameter('kp_vel_z', 1.2)

        # 读取参数
        spline_filename = self.get_parameter('spline_file').get_parameter_value().string_value
        self.control_rate = self.get_parameter('control_rate').get_parameter_value().double_value
        self.mpcc_rate = self.get_parameter('mpcc_rate').get_parameter_value().double_value
        self.max_accel = self.get_parameter('max_accel').get_parameter_value().double_value
        self.hover_height = self.get_parameter('hover_height').get_parameter_value().double_value
        self.approach_distance = self.get_parameter('approach_distance').get_parameter_value().double_value
        self.approach_speed = self.get_parameter('approach_speed').get_parameter_value().double_value
        
        # P控制器增益
        self.kp_pos = np.array([
            self.get_parameter('kp_pos_x').get_parameter_value().double_value,
            self.get_parameter('kp_pos_y').get_parameter_value().double_value,
            self.get_parameter('kp_pos_z').get_parameter_value().double_value
        ])
        
        self.kp_vel = np.array([
            self.get_parameter('kp_vel_x').get_parameter_value().double_value,
            self.get_parameter('kp_vel_y').get_parameter_value().double_value,
            self.get_parameter('kp_vel_z').get_parameter_value().double_value
        ])

        # 构建样条文件的完整路径
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.spline_file = os.path.join(current_dir, spline_filename)

        # 检查文件是否存在
        if not os.path.exists(self.spline_file):
            self.get_logger().error(f"样条文件不存在: {self.spline_file}")
            alternative_paths = [
                spline_filename,
                os.path.join(os.getcwd(), spline_filename),
                os.path.join(os.path.expanduser("~"), "px4wss/zhempcc/mpcc/src/mpcc_control/mpcc_control",
                             spline_filename)
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    self.spline_file = path
                    self.get_logger().info(f"找到样条文件: {path}")
                    break
            else:
                raise FileNotFoundError(f"无法找到样条文件: {spline_filename}")

        # 系统常量
        self.gravity = 9.8
        self.thrust_efficiency = 0.8

        # 初始化状态变量
        self.current_pose = None
        self.current_velocity = None
        self.last_acc_cmd = np.zeros(3)
        self.current_theta = 0.0
        self.current_v_theta = 0.0
        
        # 初始化控制分量变量
        self.last_acc_feedforward = np.zeros(3)
        self.last_acc_feedback = np.zeros(3)
        self.last_ref_pos = np.zeros(3)
        self.last_ref_vel = np.zeros(3)
        self.last_ref_acc = np.zeros(3)

        # 轨迹接入状态管理
        self.approach_mode = True
        self.approach_target = None

        # 初始化MPCC组件
        try:
            self.traj = MPCCTrajectory(self.spline_file)

            if not self._verify_trajectory():
                raise Exception("轨迹验证失败")

            self.optimizer = MPCCOptimizer(
                self.traj,
                dt=0.1,
                N=20
            )
            self.get_logger().info("MPCC系统初始化成功")

            # 打印轨迹信息
            self.get_logger().info(f"轨迹总长度: {self.traj.total_length:.2f}")
            self.get_logger().info(f"轨迹段数: {self.traj.num_segments}")

            # 测试轨迹在几个点的值
            test_thetas = [0.0, self.traj.total_length * 0.25,
                           self.traj.total_length * 0.5, self.traj.total_length * 0.75]
            for theta in test_thetas:
                pos = self.traj.get_position(theta).flatten()
                self.get_logger().info(f"θ={theta:.2f}: 位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        except Exception as e:
            self.get_logger().error(f"MPCC初始化失败: {e}")
            raise

        # 初始化数据记录
        self._init_data_logging()

        # 创建订阅者
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            qos_best_effort
        )

        self.velocity_sub = self.create_subscription(
            TwistStamped,
            '/mavros/local_position/velocity_local',
            self.velocity_callback,
            qos_best_effort
        )

        # 创建发布者
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.attitude_pub = self.create_publisher(
            AttitudeTarget,
            '/mpcc_control/attitude',
            qos_reliable
        )

        # 创建定时器
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_callback)
        self.mpcc_timer = self.create_timer(1.0 / self.mpcc_rate, self.mpcc_callback)

        # 轨迹跟踪状态
        self.traj_active = False
        self.traj_start_time = None
        self.clock = Clock()

        # 调试计数器
        self.debug_counter = 0

        self.get_logger().info("MPCC控制器初始化完成")
        self.get_logger().info(f"  控制频率: {self.control_rate} Hz")
        self.get_logger().info(f"  MPCC频率: {self.mpcc_rate} Hz")
        self.get_logger().info(f"  接近距离阈值: {self.approach_distance} m")
        self.get_logger().info(f"  悬停高度: {self.hover_height} m")
        self.get_logger().info(f"  P控制器位置增益: [{self.kp_pos[0]:.1f}, {self.kp_pos[1]:.1f}, {self.kp_pos[2]:.1f}]")
        self.get_logger().info(f"  P控制器速度增益: [{self.kp_vel[0]:.1f}, {self.kp_vel[1]:.1f}, {self.kp_vel[2]:.1f}]")

    def _verify_trajectory(self):
        """验证轨迹的合理性"""
        try:
            if self.traj.total_length <= 0 or self.traj.total_length > 500:
                self.get_logger().error(f"轨迹长度异常: {self.traj.total_length}")
                return False

            test_thetas = [0.0, self.traj.total_length * 0.5, self.traj.total_length * 0.99]
            for theta in test_thetas:
                pos = self.traj.get_position(theta).flatten()
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    self.get_logger().error(f"轨迹点异常: theta={theta}, pos={pos}")
                    return False

            self.get_logger().info(f"轨迹验证通过 - 长度: {self.traj.total_length:.2f}")
            return True

        except Exception as e:
            self.get_logger().error(f"轨迹验证出错: {e}")
            return False

    def _init_data_logging(self):
        """初始化数据记录"""
        log_dir = os.path.expanduser("~/mpcc_logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"mpcc_log_{timestamp}.csv"
        self.log_file_path = os.path.join(log_dir, log_filename)

        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        headers = [
            "timestamp",
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "ref_pos_x", "ref_pos_y", "ref_pos_z",
            "ref_vel_x", "ref_vel_y", "ref_vel_z", 
            "ref_acc_x", "ref_acc_y", "ref_acc_z",
            "acc_cmd_x", "acc_cmd_y", "acc_cmd_z",
            "acc_ff_x", "acc_ff_y", "acc_ff_z",
            "acc_fb_x", "acc_fb_y", "acc_fb_z",
            "theta", "v_theta",
            "contour_error", "lag_error",
            "solve_time_ms",
            "solver_status",
            "approach_mode"
        ]
        self.csv_writer.writerow(headers)

        self.get_logger().info(f"数据记录到: {self.log_file_path}")

    def pose_callback(self, msg):
        """位置消息回调"""
        self.current_pose = msg

    def velocity_callback(self, msg):
        """速度消息回调"""
        self.current_velocity = msg

    def mpcc_callback(self):
        """MPCC优化求解回调"""
        if self.current_pose is None or self.current_velocity is None:
            return

        traj_mode = self.get_parameter('traj_mode').value

        if self.debug_counter % 150 == 0:
            self.get_logger().info(
                f"traj_mode: {traj_mode}, approach_mode: {self.approach_mode}, traj_active: {self.traj_active}")

        if not traj_mode:
            self.traj_active = False
            self.approach_mode = True
            self.current_theta = 0.0
            self.current_v_theta = 0.0
            if self.debug_counter % 300 == 0:
                self.get_logger().warn("轨迹模式未激活，使用悬停模式")
            return

        # 提取当前状态
        pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])

        vel = np.array([
            self.current_velocity.twist.linear.x,
            self.current_velocity.twist.linear.y,
            self.current_velocity.twist.linear.z
        ])

        # 轨迹接入策略
        if self.approach_mode:
            success = self._handle_trajectory_approach(pos, vel)
            if success:
                self.approach_mode = False
                self.traj_active = True
                self.traj_start_time = self.clock.now()
                self.get_logger().info("成功接入轨迹，开始MPCC跟踪")
            return

        try:
            # 执行MPCC优化求解，获取前馈控制和参考状态
            acc_feedforward, theta, v_theta, ref_pos, ref_vel, ref_acc = self.optimizer.solve(pos, vel, warm_start=True)

            # 计算反馈控制（P控制器）
            pos_error = ref_pos - pos
            vel_error = ref_vel - vel
            
            acc_feedback = self.kp_pos * pos_error + self.kp_vel * vel_error
            
            # 组合前馈和反馈控制
            acc_cmd = acc_feedforward + acc_feedback
            
            # 结果验证和限制
            acc_norm = np.linalg.norm(acc_cmd)
            if acc_norm > self.max_accel:
                acc_cmd = acc_cmd * self.max_accel / acc_norm

            # 检查theta和v_theta的合理性
            if theta < 0 or theta > self.traj.total_length:
                theta = max(0.0, min(theta, self.traj.total_length))

            if v_theta < 0.02 or v_theta > 1.0:
                v_theta = np.clip(v_theta, 0.02, 1.0)

            # 更新状态
            self.last_acc_cmd = acc_cmd
            self.current_theta = theta
            self.current_v_theta = v_theta
            
            # 保存控制分量用于数据记录
            self.last_acc_feedforward = acc_feedforward
            self.last_acc_feedback = acc_feedback
            self.last_ref_pos = ref_pos
            self.last_ref_vel = ref_vel
            self.last_ref_acc = ref_acc

            # 调试信息
            if self.debug_counter % 150 == 0:
                traj_ref_pos = self.traj.get_position(theta).flatten()
                traj_pos_error = np.linalg.norm(pos - traj_ref_pos)
                ff_norm = np.linalg.norm(acc_feedforward)
                fb_norm = np.linalg.norm(acc_feedback)
                self.get_logger().info(f"MPCC状态: θ={theta:.3f}, v_θ={v_theta:.3f}, 轨迹误差={traj_pos_error:.3f}m")
                self.get_logger().info(f"控制分解: 前馈={ff_norm:.2f}, 反馈={fb_norm:.2f}, 总计={np.linalg.norm(acc_cmd):.2f}")

            # 获取求解信息并记录数据
            info = self.optimizer.get_solution_info()
            e_c, e_l, _ = self.traj.compute_errors(pos, theta)
            contour_error = np.linalg.norm(e_c)
            lag_error = abs(e_l)

            self._log_data(pos, vel, acc_cmd, theta, v_theta,
                           contour_error, lag_error, info, self.approach_mode)

        except Exception as e:
            self.get_logger().error(f"MPCC求解失败: {e}")
            self.last_acc_cmd *= 0.9
            # 在异常情况下保持上一次的控制分量
            self.last_acc_feedforward = self.last_acc_cmd * 0.5
            self.last_acc_feedback = self.last_acc_cmd * 0.5

        self.debug_counter += 1

    def _handle_trajectory_approach(self, pos, vel):
        """处理轨迹接入阶段"""
        # 找到最近的轨迹点
        try:
            closest_theta, min_dist = self.optimizer.find_closest_theta(pos)
        except:
            self.get_logger().error("无法找到最近轨迹点")
            return False

        if self.debug_counter % 150 == 0:
            self.get_logger().info(f"接近模式: 距轨迹 {min_dist:.2f}m, 目标θ={closest_theta:.2f}")

        if min_dist > self.approach_distance:
            # 距离轨迹太远，移动到轨迹附近
            target_pos = self.traj.get_position(closest_theta).flatten()
            self.approach_target = target_pos

            # 计算接近控制命令
            self.last_acc_cmd = self._compute_approach_control(pos, vel, target_pos)
            return False
        else:
            # 足够接近轨迹，可以开始MPCC跟踪
            self.current_theta = closest_theta
            self.current_v_theta = 0.08  # Todo
            self.get_logger().info(f"接近完成！距轨迹 {min_dist:.2f}m，开始MPCC跟踪")
            return True

    def _compute_approach_control(self, pos, vel, target_pos):
        """计算接近轨迹的控制命令"""
        pos_error = target_pos - pos
        vel_error = -vel

        # 合理接近控制器参数 Todo 修改为悬停相同参数
        kp = np.array([1.2, 1.2, 2.0])
        kd = np.array([1.5, 1.5, 2.5])

        acc_cmd = kp * pos_error + kd * vel_error

        # 限制接近速度
        vel_norm = np.linalg.norm(vel)
        if vel_norm > self.approach_speed:
            vel_limit_acc = -vel * (vel_norm - self.approach_speed) / vel_norm * 2.0
            acc_cmd += vel_limit_acc

        # 限制加速度
        approach_max_accel = min(1.5, self.max_accel * 0.8)
        acc_norm = np.linalg.norm(acc_cmd)
        if acc_norm > approach_max_accel:
            acc_cmd = acc_cmd * approach_max_accel / acc_norm

        return acc_cmd

    def control_callback(self):
        """控制回调 - 发布姿态命令"""
        if self.current_pose is None or self.current_velocity is None:
            return

        traj_mode = self.get_parameter('traj_mode').value

        if traj_mode and (self.traj_active or self.approach_mode):
            # 使用MPCC或接近模式的加速度命令
            acc_cmd = self.last_acc_cmd
        else:
            # 悬停模式 - 使用参照成功PID代码的控制器
            acc_cmd = self._compute_hover_control()

        # 计算期望推力向量（加上重力补偿）
        F_desired = np.array([acc_cmd[0], acc_cmd[1], acc_cmd[2] + self.gravity])

        # 获取当前姿态
        quaternion = [
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w
        ]

        # 计算当前body系z轴在世界坐标系中的方向
        rotation = R.from_quat(quaternion)
        body_z = rotation.as_matrix()[:, 2]

        # 计算期望姿态和推力
        attitude_target = self._compute_attitude_target(F_desired, body_z)

        # 发布控制命令
        self.attitude_pub.publish(attitude_target)

    def _compute_hover_control(self):
        """计算悬停控制命令"""
        # 目标轨迹（悬停）
        traj_p = np.array([[0.0], [0.0], [self.hover_height]])
        traj_v = np.array([[0.0], [0.0], [0.0]])
        traj_a = np.array([[0.0], [0.0], [0.0]])

        # 当前状态
        pose = np.array([
            [self.current_pose.pose.position.x],
            [self.current_pose.pose.position.y],
            [self.current_pose.pose.position.z]
        ])

        velo = np.array([
            [self.current_velocity.twist.linear.x],
            [self.current_velocity.twist.linear.y],
            [self.current_velocity.twist.linear.z]
        ])

        sliding_gain = np.array([[0.3], [0.3], [0.5]])
        tracking_gain = np.array([[3.2], [3.2], [5.0]])

        # 计算复合误差s
        s = (velo - traj_v + sliding_gain * (pose - traj_p))

        # 前馈参考加速度
        a_r = traj_a - sliding_gain * (velo - traj_v) + np.array([[0], [0], [self.gravity]])

        # 计算期望力
        F_sp = a_r - tracking_gain * s

        # 提取加速度命令（去掉重力）
        acc_cmd = F_sp.flatten() - np.array([0, 0, self.gravity])

        # 调试输出
        if self.debug_counter % 150 == 0:
            pos_error = (pose - traj_p).flatten()
            vel_error = velo.flatten()
            self.get_logger().info(f"悬停控制 - 位置误差: ({pos_error[0]:.3f}, {pos_error[1]:.3f}, {pos_error[2]:.3f})")
            self.get_logger().info(f"悬停控制 - 速度误差: ({vel_error[0]:.3f}, {vel_error[1]:.3f}, {vel_error[2]:.3f})")
            self.get_logger().info(f"悬停控制 - 加速度命令: ({acc_cmd[0]:.3f}, {acc_cmd[1]:.3f}, {acc_cmd[2]:.3f})")

        return acc_cmd

    def _compute_attitude_target(self, F_desired, body_z, yaw_desired=0.0):
        """计算期望姿态和推力"""
        msg = AttitudeTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # 推力计算方式
        thrust_magnitude = np.dot(F_desired, body_z)

        if thrust_magnitude > 0.1:
            normalized_thrust = thrust_magnitude / self.gravity * self.thrust_efficiency
        else:
            normalized_thrust = 0.1

        # 推力限制
        msg.thrust = float(np.clip(normalized_thrust, 0.0, 1.0))

        # 计算期望姿态
        F_norm = np.linalg.norm(F_desired)
        if F_norm > 0.01:
            body_z_desired = F_desired / F_norm
        else:
            body_z_desired = np.array([0, 0, 1])

        # 构建期望旋转矩阵
        x_C = np.array([np.cos(yaw_desired), np.sin(yaw_desired), 0])

        body_y_desired = np.cross(body_z_desired, x_C)
        y_norm = np.linalg.norm(body_y_desired)
        if y_norm > 0.001:
            body_y_desired = body_y_desired / y_norm
        else:
            body_y_desired = np.array([0, 1, 0])

        body_x_desired = np.cross(body_y_desired, body_z_desired)

        # 构建旋转矩阵
        R_desired = np.column_stack([body_x_desired, body_y_desired, body_z_desired])

        # 转换为四元数
        rotation_desired = R.from_matrix(R_desired)
        quat_desired = rotation_desired.as_quat()

        msg.orientation.x = quat_desired[0]
        msg.orientation.y = quat_desired[1]
        msg.orientation.z = quat_desired[2]
        msg.orientation.w = quat_desired[3]

        # 设置控制模式
        msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                        AttitudeTarget.IGNORE_PITCH_RATE | \
                        AttitudeTarget.IGNORE_YAW_RATE

        return msg

    def _log_data(self, pos, vel, acc_cmd, theta, v_theta,
                  contour_error, lag_error, solver_info, approach_mode):
        """记录数据到CSV文件"""
        timestamp = self.clock.now().nanoseconds * 1e-9

        row = [
            timestamp,
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            self.last_ref_pos[0], self.last_ref_pos[1], self.last_ref_pos[2],
            self.last_ref_vel[0], self.last_ref_vel[1], self.last_ref_vel[2],
            self.last_ref_acc[0], self.last_ref_acc[1], self.last_ref_acc[2],
            acc_cmd[0], acc_cmd[1], acc_cmd[2],
            self.last_acc_feedforward[0], self.last_acc_feedforward[1], self.last_acc_feedforward[2],
            self.last_acc_feedback[0], self.last_acc_feedback[1], self.last_acc_feedback[2],
            theta, v_theta,
            contour_error, lag_error,
            solver_info['solve_time'],
            solver_info['status'],
            int(approach_mode)
        ]

        self.csv_writer.writerow(row)
        self.log_file.flush()

    def __del__(self):
        """析构函数 - 关闭日志文件"""
        if hasattr(self, 'log_file'):
            self.log_file.close()
            self.get_logger().info(f"日志文件已保存: {self.log_file_path}")


def main(args=None):
    """主函数"""
    try:
        rclpy.init(args=args)
        node = MPCCController("mpcc_controller")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"错误: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()