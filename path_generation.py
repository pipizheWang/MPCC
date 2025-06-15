import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def trajectory_function(t):
    """定义原始轨迹函数 - 增加轨迹长度版本"""
    # 🔧 方法1: 增加垂直速度，让轨迹在Z方向更长
    vz = 0.15  # 从0.1增加到0.15
    amplitude = 2.5  # 从2.0增加到2.5，扩大XY平面范围

    x = amplitude * np.sin(t)
    y = amplitude * np.sin(t)
    z = vz * t + 0.8

    return np.array([x, y, z])


def longer_sine_trajectory(t):
    """更长的正弦轨迹 - 多周期版本"""
    # 🔧 方法2: 增加频率和周期数
    vz = 0.12
    amplitude = 3.0
    frequency = 0.8  # 增加频率，让轨迹更复杂

    x = amplitude * np.sin(frequency * t)
    y = amplitude * np.sin(frequency * t)  # 可以改为cos创建李萨如图形
    z = vz * t + 1.0

    return np.array([x, y, z])


def extended_heart_trajectory(t):
    """扩展心形轨迹 - 更大更长版本"""
    # 🔧 方法3: 扩大心形轨迹
    R = 8.0  # 从6.0增加到8.0，扩大轨迹
    w = (2 * np.pi) / 40  # 从30增加到40，增加周期长度
    vz = 0.12  # 从0.1增加到0.12

    x = R * np.sin(w * t)
    y = R * (1 - np.cos(w * t))
    z = vz * t + 0.8

    return np.array([x, y, z])


def spiral_trajectory(t):
    """螺旋轨迹 - 新的长轨迹选项"""
    # 🔧 方法4: 螺旋轨迹，自然地创建长轨迹
    R = 2.0  # 基础半径
    growth = 0.05  # 半径增长率
    w = 0.3  # 角频率
    vz = 0.08  # Z方向速度

    radius = R + growth * t  # 螺旋半径随时间增长

    x = radius * np.cos(w * t)
    y = radius * np.sin(w * t)
    z = vz * t + 1.5

    return np.array([x, y, z])


def lemniscate_trajectory(t):
    """∞字轨迹（李萨如曲线）- 复杂长轨迹"""
    # 🔧 方法5: 8字轨迹的扩展版
    scale = 4.0  # 从3.0增加到4.0
    w1 = 0.2  # X方向频率
    w2 = 0.4  # Y方向频率（2倍关系创建8字）
    vz = 0.1

    x = scale * np.sin(w1 * t)
    y = scale * np.sin(w2 * t)
    z = vz * t + 1.0

    return np.array([x, y, z])


def arc_length_parameterize(traj_func, t_start=0, t_end=120, num_samples=2000, num_segments=30):
    """
    弧长参数化 - 增加轨迹长度版本

    🔧 关键修改：
    - t_end: 60 → 120 （时间翻倍，轨迹长度翻倍）
    - num_samples: 1000 → 2000 （更多采样点，保证精度）
    - num_segments: 20 → 30 （更多样条段，保证平滑性）

    参数:
    traj_func: 轨迹函数
    t_start: 起始时间
    t_end: 结束时间 🔧 关键参数：增加这个值来延长轨迹
    num_samples: 采样点数 🔧 建议与t_end成比例增加
    num_segments: 样条段数 🔧 建议适度增加保持平滑性
    """
    print(f"🔧 生成更长轨迹...")
    print(f"轨迹函数: {traj_func.__name__}")
    print(f"时间范围: {t_start} 到 {t_end} （时长: {t_end - t_start}s）")
    print(f"采样点数: {num_samples}")
    print(f"样条段数: {num_segments}")

    # 1. 采样原始轨迹
    t_samples = np.linspace(t_start, t_end, num_samples)
    points = np.array([traj_func(t) for t in t_samples])

    print(f"采样完成，轨迹点形状: {points.shape}")
    print(f"轨迹范围:")
    print(f"  X: {points[:, 0].min():.3f} 到 {points[:, 0].max():.3f}")
    print(f"  Y: {points[:, 1].min():.3f} 到 {points[:, 1].max():.3f}")
    print(f"  Z: {points[:, 2].min():.3f} 到 {points[:, 2].max():.3f}")

    # 2. 计算每个点的弧长
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    arc_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = arc_lengths[-1]

    print(f"🎯 总弧长: {total_length:.3f} （这是MPCC中的轨迹长度）")

    # 3. 计算等弧长间隔的参数值
    target_arc_lengths = np.linspace(0, total_length, num_segments + 1)

    # 4. 对每个目标弧长，找到对应的t值
    equal_arc_params = []
    equal_arc_points = []

    for i, target_length in enumerate(target_arc_lengths):
        if target_length <= 0:
            t_value = t_start
        elif target_length >= total_length:
            t_value = t_end
        else:
            t_value = np.interp(target_length, arc_lengths, t_samples)

        equal_arc_params.append(t_value)
        equal_arc_points.append(traj_func(t_value))

        if i % 10 == 0:
            print(f"  进度: {i}/{num_segments}, 弧长: {target_length:.3f}, t: {t_value:.3f}")

    equal_arc_params = np.array(equal_arc_params)
    equal_arc_points = np.array(equal_arc_points)

    print(f"等弧长采样完成，点数: {len(equal_arc_points)}")

    # 5. 拟合三次样条曲线
    s_params = np.linspace(0, total_length, num_segments + 1)
    splines = {}
    spline_coefficients = {}

    print("开始拟合三次样条...")
    for i, coord in enumerate(['x', 'y', 'z']):
        try:
            spline = CubicSpline(s_params, equal_arc_points[:, i])
            splines[coord] = spline

            coeffs = []
            for j in range(num_segments):
                segment_coeffs = spline.c[:, j]
                a0 = segment_coeffs[3]
                a1 = segment_coeffs[2]
                a2 = segment_coeffs[1]
                a3 = segment_coeffs[0]
                coeffs.append([a0, a1, a2, a3])

            spline_coefficients[coord] = coeffs
            print(f"  {coord}轴样条拟合完成")

        except Exception as e:
            print(f"  {coord}轴样条拟合失败: {e}")
            raise

    print("🎯 更长轨迹样条拟合完成！")

    return {
        'original_points': points,
        'original_times': t_samples,
        'arc_lengths': arc_lengths,
        'total_length': total_length,
        'equal_arc_points': equal_arc_points,
        'equal_arc_params': equal_arc_params,
        'splines': splines,
        's_params': s_params,
        'spline_coefficients': spline_coefficients,
        'num_segments': num_segments
    }


def evaluate_spline(s, result, clip=True):
    """根据弧长参数s评估三次样条曲线上的点"""
    splines = result['splines']
    total_length = result['total_length']

    if clip:
        s = np.clip(s, 0, total_length)

    if np.isscalar(s):
        x = splines['x'](s)
        y = splines['y'](s)
        z = splines['z'](s)
        return np.array([x, y, z])
    else:
        points = np.zeros((len(s), 3))
        for i, s_val in enumerate(s):
            points[i, 0] = splines['x'](s_val)
            points[i, 1] = splines['y'](s_val)
            points[i, 2] = splines['z'](s_val)
        return points


def evaluate_spline_derivative(s, result, clip=True):
    """计算样条曲线在弧长s处的一阶导数"""
    splines = result['splines']
    total_length = result['total_length']

    if clip:
        s = np.clip(s, 0, total_length)

    if np.isscalar(s):
        dx_ds = splines['x'].derivative()(s)
        dy_ds = splines['y'].derivative()(s)
        dz_ds = splines['z'].derivative()(s)
        return np.array([dx_ds, dy_ds, dz_ds])
    else:
        derivs = np.zeros((len(s), 3))
        for i, s_val in enumerate(s):
            derivs[i, 0] = splines['x'].derivative()(s_val)
            derivs[i, 1] = splines['y'].derivative()(s_val)
            derivs[i, 2] = splines['z'].derivative()(s_val)
        return derivs


def compute_parameterization_error(result, num_points=100):
    """计算弧长参数化误差|dP/ds|-1|"""
    s_values = np.linspace(0, result['total_length'], num_points)
    errors = []

    for s in s_values:
        deriv = evaluate_spline_derivative(s, result)
        norm = np.linalg.norm(deriv)
        error = abs(norm - 1.0)
        errors.append(error)

    return s_values, np.array(errors)


def save_spline_parameters(result, filename="spline_parameters.npz"):
    """保存样条参数到二进制文件"""
    data_to_save = {
        'total_length': result['total_length'],
        'num_segments': result['num_segments'],
        's_params': result['s_params'],
        'x_coeffs': np.array(result['spline_coefficients']['x']),
        'y_coeffs': np.array(result['spline_coefficients']['y']),
        'z_coeffs': np.array(result['spline_coefficients']['z'])
    }

    np.savez(filename, **data_to_save)
    print(f"🎯 更长轨迹参数已保存: {filename}")

    # 验证保存的数据
    print("验证保存的数据:")
    loaded = np.load(filename)
    print(f"  总长度: {loaded['total_length']:.3f}")
    print(f"  段数: {loaded['num_segments']}")
    print(f"  系数矩阵形状: {loaded['x_coeffs'].shape}")


def visualize_result(result, num_vis_points=300):
    """可视化更长的轨迹"""
    original_points = result['original_points']
    equal_arc_points = result['equal_arc_points']
    total_length = result['total_length']

    fig = plt.figure(figsize=(18, 14))

    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(original_points[:, 0], original_points[:, 1], original_points[:, 2], 'b-',
             label='Original Trajectory', linewidth=2)
    ax1.scatter(equal_arc_points[:, 0], equal_arc_points[:, 1], equal_arc_points[:, 2],
                c='g', marker='o', s=30, label='Equal Arc Length Points')

    s_vis = np.linspace(0, total_length, num_vis_points)
    spline_points = evaluate_spline(s_vis, result, clip=False)
    ax1.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2],
             'r-', linewidth=2, label='Arc Length Parameterized Spline')

    ax1.legend()
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')
    ax1.set_title(f'3D Extended Trajectory (Length: {total_length:.1f})')

    # XY平面投影
    ax2 = fig.add_subplot(222)
    ax2.plot(original_points[:, 0], original_points[:, 1], 'b-', label='Original', linewidth=2)
    ax2.plot(spline_points[:, 0], spline_points[:, 1], 'r-', label='Spline', linewidth=2)
    ax2.scatter(equal_arc_points[:, 0], equal_arc_points[:, 1], c='g', marker='o', s=20)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # 弧长参数化误差
    s_values, errors = compute_parameterization_error(result, num_vis_points)
    ax3 = fig.add_subplot(223)
    ax3.plot(s_values, errors)
    ax3.set_xlabel('Arc Length Parameter s')
    ax3.set_ylabel('Parameterization Error: |dP/ds|-1|')
    ax3.set_title('Arc Length Parameterization Error')
    ax3.grid(True)

    # Z轴随弧长变化
    ax4 = fig.add_subplot(224)
    ax4.plot(s_values, spline_points[:, 2], 'r-', label='Spline Z', linewidth=2)
    ax4.set_xlabel('Arc Length Parameter s')
    ax4.set_ylabel('Z Position')
    ax4.set_title('Z vs Arc Length')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()


# 主程序 - 支持更多轨迹选项
if __name__ == "__main__":
    print("🔧 选择轨迹类型（更长版本）:")
    print("1. 正弦轨迹: x=sin(t), y=sin(t), z=v*t （基础版）")
    print("2. 更长正弦轨迹: 扩大范围和频率")
    print("3. 扩展心形轨迹: 更大更长的心形")
    print("4. 螺旋轨迹: 自然长轨迹")
    print("5. ∞字轨迹: 复杂长轨迹")

    choice = input("请输入选择 (1-5, 默认2): ").strip()
    if not choice:
        choice = "2"

    trajectory_functions = {
        "1": trajectory_function,
        "2": longer_sine_trajectory,
        "3": extended_heart_trajectory,
        "4": spiral_trajectory,
        "5": lemniscate_trajectory
    }

    selected_func = trajectory_functions.get(choice, longer_sine_trajectory)

    # 参数设置 - 针对更长轨迹优化
    print("\n🔧 选择轨迹长度:")
    print("1. 标准长度 (120s)")
    print("2. 长轨迹 (180s)")
    print("3. 超长轨迹 (240s)")

    length_choice = input("请选择长度 (1-3, 默认1): ").strip()

    if length_choice == "2":
        t_end = 180
        num_samples = 6000
        num_segments = 80
    elif length_choice == "3":
        t_end = 240
        num_samples = 8000
        num_segments = 80
    else:
        t_end = 120
        num_samples = 4000
        num_segments = 60

    print(f"\n🎯 使用轨迹函数: {selected_func.__name__}")
    print(f"时间范围: 0 到 {t_end}")
    print(f"采样点数: {num_samples}")
    print(f"样条段数: {num_segments}")

    # 进行弧长参数化
    try:
        result = arc_length_parameterize(
            selected_func,
            t_start=0,
            t_end=t_end,
            num_samples=num_samples,
            num_segments=num_segments
        )

        print(f"\n🎯 轨迹总弧长: {result['total_length']:.4f}")

        # 保存样条参数到文件
        save_spline_parameters(result, "spline_parameters.npz")

        # 可视化结果
        print("\n正在生成可视化图表...")
        visualize_result(result, num_vis_points=400)

        # 输出参数化误差统计
        _, errors = compute_parameterization_error(result, 200)
        print(f"\n弧长参数化质量:")
        print(f"最大误差: {np.max(errors):.6f}")
        print(f"平均误差: {np.mean(errors):.6f}")
        print(f"误差标准差: {np.std(errors):.6f}")

        if np.max(errors) > 0.1:
            print("⚠️  警告: 弧长参数化误差较大，建议增加采样点数或样条段数")
        else:
            print("✅ 弧长参数化质量良好")

        print(f"\n🎯 更长轨迹生成完成！总长度: {result['total_length']:.2f}")

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback

        traceback.print_exc()

# ============================================================================
# 🔧 轨迹长度调整总结：
#
# 1. 时间范围调整: t_end = 60 → 120/180/240
# 2. 采样密度调整: num_samples 相应增加
# 3. 样条段数调整: num_segments 适度增加
# 4. 轨迹参数调整: amplitude, vz, frequency 等
# 5. 新轨迹类型: 螺旋轨迹、∞字轨迹等
#
# 关键是保持 samples/time 和 segments/length 的合理比例！
# ============================================================================