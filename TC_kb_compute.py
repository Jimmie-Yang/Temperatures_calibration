整理一下代码，用中文写好注释：import os
import glob
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class TC:
    def __init__(self,
                 input_dir: str,
                 fig_dir: str = None,
                 ref_rows: List[int] = None,
                 ref_temp: int = 20,
                 temps_order: List[int] = None,
                 channel_names: Tuple[str, str] = ("CH1_sig", "CH2_sig")):
        """初始化"""
        self.input_dir = input_dir
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True) if self.fig_dir else None

        self.ref_rows = [0, 1, 2] if ref_rows is None else ref_rows
        self.ref_temp = ref_temp
        self.temps_order = [0, 10, 20, 30, 40, 50, -10] if temps_order is None else temps_order
        self.channel_names = channel_names

        # 读取后填充的属性
        self.files_sorted: List[str] = []  # 排序后的文件路径
        self.temps: List[int] = []  # 每一列对应的温度
        self.ch1_data: np.ndarray = None  # shape = (n_points, n_series)
        self.ch2_data: np.ndarray = None  # shape = (n_points, n_series)
        self.x_values: np.ndarray = None  # x 轴位置 (mm)

        # 结果存储（按模块和通道）
        self.result_k: List[Dict[str, np.ndarray]] = []
        self.result_b: List[Dict[str, np.ndarray]] = []
        self.poly_k: List[Dict[str, np.ndarray]] = []  # 拟合的二次多项式系数 [a, d, c]
        self.poly_b: List[Dict[str, np.ndarray]] = []

        # 校准后的数据
        self.ch1_out_data: np.ndarray = None
        self.ch2_out_data: np.ndarray = None

    # ===================  数据加载与基础处理  =================== #
    def _sort_key(self, fn: str) -> Tuple[int, int]:
        """根据文件名排序"""
        base = os.path.basename(fn)
        m = re.match(r'([a-zA-Z]+)_T(\d+)_', base)
        if not m:
            return 99, 0
        color, t = m.group(1), int(m.group(2))
        # 颜色优先级：可根据实际需要调整
        color_order = {"black": 0, "blue": 1, "white": 2}
        if t == 110:
            t = -10  # 处理特殊温度映射
        return color_order.get(color, 99), t

    def load_data(self):
        """加载 Excel 数据"""
        pattern = os.path.join(self.input_dir, '*.xlsx')
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"未在 {self.input_dir} 下找到 xlsx 文件！")

        self.files_sorted = sorted(files, key=self._sort_key)

        # 解析温度
        self.temps = []
        for fn in self.files_sorted:
            base = os.path.basename(fn)
            m = re.match(r'.*_T(\d+)_', base)
            if m:
                t = int(m.group(1))
                if t == 110:
                    t = -10
                self.temps.append(t)
            else:
                self.temps.append(None)

        # 读取 Excel 数据
        ch1_list, ch2_list = [], []
        for fn in self.files_sorted:
            df = pd.read_excel(fn, usecols=list(self.channel_names))
            ch1_list.append(df[self.channel_names[0]].to_numpy())
            ch2_list.append(df[self.channel_names[1]].to_numpy())

        ch1_sig = np.stack(ch1_list, axis=0)  # shape = (n_series, n_points)
        ch2_sig = np.stack(ch2_list, axis=0)

        # 转置成 (n_points, n_series)
        self.ch1_data = ch1_sig.T
        self.ch2_data = ch2_sig.T

        n_points = self.ch1_data.shape[0]
        self.x_values = np.arange(1, n_points + 1) * 1.0  # 如果有步长可替换为实际值

        print(f"ch1_sig shape = {ch1_sig.shape}")
        print(f"ch2_sig shape = {ch2_sig.shape}")

    # ===================  数据校验  =================== #
    def _check_data_range(self, data: np.ndarray, cols: List[int], mod_idx: int, channel: str) -> Tuple[
        bool, List[int]]:
        """
        检查模组数据中每个位置的7列数据最大最小值差是否≤100
        新增channel参数用于明确标识通道
        """
        try:
            # 提取当前模组的所有位置数据 (n_points, 7)
            mod_data = data[:, cols]  # shape=(n_points, 7)
            if mod_data.shape[1] != len(self.temps_order):
                print(f"❌ 模组 {mod_idx}（{channel}）列数异常：实际{mod_data.shape[1]}列，预期{len(self.temps_order)}列")
                return False, []

            # 计算每个位置的最大最小值差
            range_per_pos = mod_data.max(axis=1) - mod_data.min(axis=1)  # shape=(n_points,)
            invalid_positions = np.where(range_per_pos > 100)[0].tolist()

            # 打印校验结果
            if invalid_positions:
                print(f"⚠️ 模组 {mod_idx}（{channel}）存在 {len(invalid_positions)} 个位置数据波动超过100：")
                for pos in invalid_positions[:5]:  # 只打印前5个，避免刷屏
                    min_val = mod_data[pos].min()
                    max_val = mod_data[pos].max()
                    print(f"    位置 {pos}：min={min_val:.2f}, max={max_val:.2f}, 差={max_val - min_val:.2f}")
                if len(invalid_positions) > 5:
                    print(f"    ...（省略 {len(invalid_positions) - 5} 个位置）")
            else:
                print(f"✅ 模组 {mod_idx}（{channel}）所有位置数据波动均≤100")

            return len(invalid_positions) == 0, invalid_positions
        except Exception as e:
            print(f"❌ 模组 {mod_idx}（{channel}）校验失败：{str(e)}")
            return False, []

    def check_all_modules(self):
        """单独执行所有模组的数据校验（CH1和CH2分别校验）"""
        if self.ch1_data is None:
            raise RuntimeError("请先调用 load_data() 加载数据")

        n_series = self.ch1_data.shape[1]
        cols_per_module = len(self.temps_order)
        n_modules = n_series // cols_per_module

        print("\n==== 开始校验所有模组数据稳定性（最大最小值差≤100）====")
        for mod_idx in range(n_modules):
            start = mod_idx * cols_per_module
            end = start + cols_per_module
            cols = list(range(start, end))
            # 校验CH1
            self._check_data_range(self.ch1_data, cols, mod_idx, channel="CH1")
            # 校验CH2
            self._check_data_range(self.ch2_data, cols, mod_idx, channel="CH2")
        print("==== 数据稳定性校验结束 ====\n")

    # ===================  绘图相关  =================== #
    def plot_0(self, C: bool = False):
        import matplotlib.pyplot as plt

        # 数据
        temperature = [-10, 0, 10, 20, 30, 40, 50]
        ch1 = self.ch1_data[0,  0:7]
        ch2 = self.ch2_data[14, 0:7]

        if C:
            ch1 = self.ch1_out_joint[0,  0:7]
            ch2 = self.ch2_out_joint[14, 0:7]
        # 绘图
        plt.figure(figsize=(8, 5))
        plt.plot(temperature, ch1, 's--', label='ch1', color='blue')  # 's--'表示方形点+虚线
        plt.plot(temperature, ch2, 's--', label='ch2', color='green')

        # 标签和图例
        plt.ylim(1280, 1440)  # 参数为 (最小值, 最大值)
        plt.xlabel('t(°C)')
        plt.ylabel('signal (uV/100g)')
        plt.legend()
        plt.grid(True)

        # 显示图表
        plt.tight_layout()
        if C==False:
            plt.savefig(r'F:\Temputure_Calibration\data_0723\0724_Temp\fig\极值.png',dpi=300)
        if C:
            plt.savefig(r'F:\Temputure_Calibration\data_0723\0724_Temp\fig\极值C.png', dpi=300)
        plt.show()

    def plot_signals(self, module_idx: int = None, save_name: str = None):
        """绘制原始数据（两个通道）"""
        if self.ch1_data is None:
            raise RuntimeError("请先调用 load_data()")

        # 需要绘制的列索引
        if module_idx is None:
            cols = list(range(self.ch1_data.shape[1]))
        else:
            start = module_idx * len(self.temps_order)
            end = start + len(self.temps_order)
            cols = list(range(start, end))

        fig, ax = plt.subplots(figsize=(10, 6))
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*']
        linestyles = ['-', '--', ':', '-.']

        for idx_count, col in enumerate(cols):
            m = markers[idx_count % len(markers)]
            ls = linestyles[(idx_count // len(markers)) % len(linestyles)]
            temp = self.temps[col]
            label_ch1 = f'ch1_{temp}°C'
            label_ch2 = f'ch2_{temp}°C'

            ax.plot(self.x_values, self.ch1_data[:, col], marker=m, linestyle=ls,
                    linewidth=1, label=label_ch1)
            ax.plot(self.x_values, self.ch2_data[:, col], marker=m, linestyle=ls,
                    linewidth=1, label=label_ch2)

        ax.set_xlabel('x-pos (mm)')
        ax.set_ylabel('μV/100g')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(ncol=2, fontsize='small')
        plt.tight_layout()

        if self.fig_dir and save_name:
            plt.savefig(os.path.join(self.fig_dir, save_name), dpi=600)
        plt.show()

    def plot_calibrated(self, module_idx: int, save_name: str = None):
        """绘制校准后的结果（两个通道）"""
        if self.ch1_out_data is None or self.ch2_out_data is None:
            raise RuntimeError("请先执行 calibrate_all() 进行校准")

        start = module_idx * len(self.temps_order)
        end = start + len(self.temps_order)
        cols = list(range(start, end))

        fig, ax = plt.subplots(figsize=(10, 6))
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*']
        linestyles = ['-', '--', ':', '-.']

        for idx_count, col in enumerate(cols):
            m = markers[idx_count % len(markers)]
            ls = linestyles[(idx_count // len(markers)) % len(linestyles)]
            temp = self.temps[col]
            label_ch1 = f'ch1_out_{temp}°C'
            label_ch2 = f'ch2_out_{temp}°C'

            ax.plot(self.x_values, self.ch1_out_data[:, col], marker=m, linestyle=ls,
                    linewidth=1, label=label_ch1)
            ax.plot(self.x_values, self.ch2_out_data[:, col], marker=m, linestyle=ls,
                    linewidth=1, label=label_ch2)

        ax.set_xlabel('x-pos (mm)')
        ax.set_ylabel('μV/100g (calibrated)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(ncol=2, fontsize='small')
        plt.tight_layout()

        if self.fig_dir and save_name:
            plt.savefig(os.path.join(self.fig_dir, save_name), dpi=600)
        plt.show()

    # ===================  核心计算：k,b 及其拟合  =================== #
    def _compute_kb_one_channel(self, data: np.ndarray, cols: List[int], mod_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """对一个通道和一个模组内部的 7 列数据计算 k,b"""
        # 1. 数据范围校验
        valid, invalid_positions = self._check_data_range(data, cols, mod_idx, "CH1" if "ch1" in str(data) else "CH2")
        # 2. 调整参考行（移除异常位置的参考行）
        adjusted_ref_rows = self._adjust_ref_rows(self.ref_rows, invalid_positions)

        # 3. 从调整后的参考行取数据
        S = data[adjusted_ref_rows, :][:, cols]  # shape = (len(adjusted_ref_rows), 7)

        # 4. 后续k、b计算逻辑
        temps_mod = [self.temps[c] for c in cols]
        try:
            ref_idx = temps_mod.index(self.ref_temp)
        except ValueError:
            raise ValueError(f"在模组列中未找到基准温度 {self.ref_temp}°C")
        S_prime = np.tile(S[:, [ref_idx]], (1, S.shape[1]))

        m, n = S.shape  # m = len(adjusted_ref_rows), n = 7
        k = np.zeros(n)
        b = np.zeros(n)
        for j in range(n):
            X = np.column_stack([S[:, j], np.ones(m)])
            y = S_prime[:, j]
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            k[j], b[j] = sol

        # 验证残差
        residual = S @ np.diag(k) + np.tile(b, (m, 1)) - S_prime
        print(f"残差 Frobenius 范数 (channel data cols={cols}):", np.linalg.norm(residual, 'fro'))
        return k, b

    def _adjust_ref_rows(self, ref_rows: List[int], invalid_positions: List[int]) -> List[int]:
        """剔除参考行中处于异常位置的索引"""
        adjusted = [r for r in ref_rows if r not in invalid_positions]
        if len(adjusted) < len(ref_rows):
            removed = set(ref_rows) - set(adjusted)
            print(f"🔍 从参考行中移除异常位置：{removed}，调整后参考行：{adjusted}")
        if not adjusted:
            raise ValueError("调整后参考行为空，无法计算k、b！请检查数据或修改ref_rows")
        return adjusted

    def compute_kb_all(self):
        """对所有模组、两个通道计算 k,b 并存储"""
        if self.ch1_data is None:
            raise RuntimeError("请先调用 load_data()")

        n_series = self.ch1_data.shape[1]
        cols_per_module = len(self.temps_order)
        n_modules = n_series // cols_per_module

        self.result_k = []
        self.result_b = []

        for mod_idx in range(n_modules):
            start = mod_idx * cols_per_module
            end = start + cols_per_module
            cols = list(range(start, end))

            k_ch1, b_ch1 = self._compute_kb_one_channel(self.ch1_data, cols, mod_idx)
            k_ch2, b_ch2 = self._compute_kb_one_channel(self.ch2_data, cols, mod_idx)

            self.result_k.append({"ch1": k_ch1, "ch2": k_ch2})
            self.result_b.append({"ch1": b_ch1, "ch2": b_ch2})

            # 数据
            temperature = [-10, 0, 10, 20, 30, 40, 50]
            ch1 = self.result_k
            ch2 = self.result_b

            # 绘图
            plt.figure(figsize=(8, 5))
            plt.plot(temperature, ch1, 's--', label='k', color='blue')  # 's--'表示方形点+虚线
            plt.plot(temperature, ch2, 's--', label='b', color='green')

            # 标签和图例
            plt.xlabel('t(°C)')
            plt.ylabel('k/b ')
            plt.legend()
            plt.grid(True)

            # 显示图表
            plt.tight_layout()
            plt.savefig(r'F:\Temputure_Calibration\data_0723\0724_Temp\fig\k,b.png', dpi=300)
            plt.show()


    def fit_poly_all(self):
        """将每个模组、每个通道的 k、b 与温度做二次多项式拟合"""
        if not self.result_k:
            raise RuntimeError("请先调用 compute_kb_all()")

        cols_per_module = len(self.temps_order)
        n_modules = len(self.result_k)

        self.poly_k = []
        self.poly_b = []

        for mod_idx in range(n_modules):
            # 此模组对应列的温度（顺序应与 temps_order 一致）
            temps_mod = np.array(self.temps_order, dtype=float)
            poly_k_ch = {}
            poly_b_ch = {}
            for ch in ["ch1", "ch2"]:
                k = self.result_k[mod_idx][ch]
                b = self.result_b[mod_idx][ch]
                coeff_k = np.polyfit(temps_mod, k, deg=2)  # [a, d, c]
                coeff_b = np.polyfit(temps_mod, b, deg=2)
                poly_k_ch[ch] = coeff_k
                poly_b_ch[ch] = coeff_b
                print(f"模组 {mod_idx} {ch}：k(t) = {coeff_k[0]:.4e}·t² + {coeff_k[1]:.4e}·t + {coeff_k[2]:.4e}")
                print(f"模组 {mod_idx} {ch}：b(t) = {coeff_b[0]:.4e}·t² + {coeff_b[1]:.4e}·t + {coeff_b[2]:.4e}")

            self.poly_k.append(poly_k_ch)
            self.poly_b.append(poly_b_ch)

    # ===================  校准  =================== #
    def calibrate_all(self):
        """用拟合得到的多项式对所有数据进行校准"""
        if not self.poly_k:
            raise RuntimeError("请先调用 fit_poly_all()")

        n_points, n_series = self.ch1_data.shape
        cols_per_module = len(self.temps_order)
        n_modules = n_series // cols_per_module

        self.ch1_out_data = np.zeros_like(self.ch1_data)
        self.ch2_out_data = np.zeros_like(self.ch2_data)

        for mod_idx in range(n_modules):
            start = mod_idx * cols_per_module
            end = start + cols_per_module
            cols = list(range(start, end))

            # 对应的温度列表
            temps_mod = np.array([self.temps[c] for c in cols], dtype=float)

            # 计算 k(t), b(t)
            k_ch1 = np.polyval(self.poly_k[mod_idx]["ch1"], temps_mod)
            b_ch1 = np.polyval(self.poly_b[mod_idx]["ch1"], temps_mod)
            k_ch2 = np.polyval(self.poly_k[mod_idx]["ch2"], temps_mod)
            b_ch2 = np.polyval(self.poly_b[mod_idx]["ch2"], temps_mod)

            # 广播到 (n_points, 7)
            self.ch1_out_data[:, cols] = self.ch1_data[:, cols] * k_ch1[np.newaxis, :] + b_ch1[np.newaxis, :]
            self.ch2_out_data[:, cols] = self.ch2_data[:, cols] * k_ch2[np.newaxis, :] + b_ch2[np.newaxis, :]

    # ===================  通道联合校准  =================== #
    def compute_kb_joint(self, modules_to_use=(0, 2)):
        """
        仅使用指定模组 (modules_to_use) 的数据来求联合 k、b，
        然后仍对全部模组做后续校准。
        """
        if self.ch1_data is None:
            raise RuntimeError("请先调用 load_data()")

        cols_per_module = len(self.temps_order)
        n_series = self.ch1_data.shape[1]
        n_modules = n_series // cols_per_module

        # ---------- 1. 构造“全部模组”的索引矩阵 ----------
        idx_matrix_full = np.arange(n_series).reshape(n_modules, cols_per_module)

        # ---------- 2. 只保留需要参与求解的行（模组） ----------
        modules_to_use = [m for m in modules_to_use if m < n_modules]
        if not modules_to_use:
            raise ValueError("modules_to_use 为空，无法计算 k、b")
        idx_matrix = idx_matrix_full[modules_to_use, :]
        n_sel = idx_matrix.shape[0]

        # ---------- 3. 找到基准温度列 ----------
        try:
            ref_pos = self.temps_order.index(self.ref_temp)
        except ValueError:
            raise ValueError(f"temps_order 中找不到基准温度 {self.ref_temp}")

        def _solve_channel(data):
            m = len(self.ref_rows)
            k_all = np.zeros(cols_per_module)
            b_all = np.zeros(cols_per_module)

            S_ref = data[self.ref_rows, :][:, idx_matrix[:, ref_pos].ravel()]  # (m, n_sel*1)
            S_ref = S_ref.reshape(m * n_sel, 1)

            for j in range(cols_per_module):
                cols_j = idx_matrix[:, j].ravel()
                S_j = data[self.ref_rows, :][:, cols_j].reshape(m * n_sel, 1)

                X = np.column_stack([S_j, np.ones_like(S_j)])
                y = S_ref.squeeze(-1)
                sol, *_ = np.linalg.lstsq(X, y, rcond=None)
                k_all[j], b_all[j] = sol

            return k_all, b_all

        self.joint_k = {"ch1": _solve_channel(self.ch1_data)[0],
                        "ch2": _solve_channel(self.ch2_data)[0]}
        self.joint_b = {"ch1": _solve_channel(self.ch1_data)[1],
                        "ch2": _solve_channel(self.ch2_data)[1]}


        print(f"⚙️  k,b 已用模组 {modules_to_use} 求得 (共 {n_sel} 个模组)")



    def fit_poly_joint(self):
        """将联合校准得到的 k,b 与温度做二次多项式拟合"""
        if not hasattr(self, 'joint_k'):
            raise RuntimeError("请先调用 compute_kb_joint()")

        temps = np.array(self.temps_order, dtype=float)
        self.joint_poly_k = {}
        self.joint_poly_b = {}
        for ch in ["ch1", "ch2"]:
            k = self.joint_k[ch]
            b = self.joint_b[ch]
            coeff_k = np.polyfit(temps, k, deg=2)
            coeff_b = np.polyfit(temps, b, deg=2)
            self.joint_poly_k[ch] = coeff_k
            self.joint_poly_b[ch] = coeff_b
            print(f"{ch}：k(t) = {coeff_k[0]:.4e}·t² + {coeff_k[1]:.4e}·t + {coeff_k[2]:.4e}")
            print(f"{ch}：b(t) = {coeff_b[0]:.4e}·t² + {coeff_b[1]:.4e}·t + {coeff_b[2]:.4e}")

            # print(f"{ch}：k(t) = {coeff_k[0]:.4e}·t^4 + {coeff_k[1]:.4e}·t^3 + {coeff_k[2]:.4e}·t² + {coeff_k[3]:.4e}·t + {coeff_k[4]:.4e}")
            # print(f"{ch}：b(t) = {coeff_b[0]:.4e}·t^4 + {coeff_b[1]:.4e}·t^3 + {coeff_b[2]:.4e}·t² + {coeff_b[3]:.4e}·t + {coeff_b[4]:.4e}")

    def calibrate_joint(self):
        """使用联合拟合的多项式对所有数据进行校准"""
        if not hasattr(self, 'joint_poly_k'):
            raise RuntimeError("请先调用 fit_poly_joint()")

        n_points, n_series = self.ch1_data.shape
        cols_per_module = len(self.temps_order)
        n_modules = n_series // cols_per_module

        self.ch1_out_joint = np.zeros_like(self.ch1_data)
        self.ch2_out_joint = np.zeros_like(self.ch2_data)

        for mod_idx in range(n_modules):
            start = mod_idx * cols_per_module
            end = start + cols_per_module
            cols = list(range(start, end))
            temps_mod = np.array([self.temps[c] for c in cols], dtype=float)

            # 二次函数拟合写法
            k1 = np.polyval(self.joint_poly_k['ch1'], temps_mod)
            b1 = np.polyval(self.joint_poly_b['ch1'], temps_mod)
            k2 = np.polyval(self.joint_poly_k['ch2'], temps_mod)
            b2 = np.polyval(self.joint_poly_b['ch2'], temps_mod)

            self.ch1_out_joint[:, cols] = self.ch1_data[:, cols] * k1[np.newaxis, :] + b1[np.newaxis, :]
            self.ch2_out_joint[:, cols] = self.ch2_data[:, cols] * k2[np.newaxis, :] + b2[np.newaxis, :]

            #分段函数拟合写法self.joint_k
            k1 = self.joint_k['ch1']
            b1 = self.joint_b['ch1']
            k2 = self.joint_k['ch2']
            b2 = self.joint_b['ch2']

            self.ch1_out_joint[:, cols] = self.ch1_data[:, cols] * k1[np.newaxis, :] + b1[np.newaxis, :]
            self.ch2_out_joint[:, cols] = self.ch2_data[:, cols] * k2[np.newaxis, :] + b2[np.newaxis, :]

    def plot_calibrated_joint(self, module_idx: int = None, save_name: str = None):
        """绘制联合校准后的结果"""
        if not hasattr(self, 'ch1_out_joint'):
            raise RuntimeError("请先调用 calibrate_joint()")

        if module_idx is None:
            cols = list(range(self.ch1_data.shape[1]))
        else:
            start = module_idx * len(self.temps_order)
            end = start + len(self.temps_order)
            cols = list(range(start, end))

        fig, ax = plt.subplots(figsize=(10, 6))
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*']
        linestyles = ['-', '--', ':', '-.']

        for idx_count, col in enumerate(cols):
            m = markers[idx_count % len(markers)]
            ls = linestyles[(idx_count // len(markers)) % len(linestyles)]
            temp = self.temps[col]
            label_ch1 = f'ch1_joint_{temp}°C'
            label_ch2 = f'ch2_joint_{temp}°C'

            ax.plot(self.x_values, self.ch1_out_joint[:, col], marker=m, linestyle=ls,
                    linewidth=1, label=label_ch1)
            ax.plot(self.x_values, self.ch2_out_joint[:, col], marker=m, linestyle=ls,
                    linewidth=1, label=label_ch2)

        ax.set_xlabel('x-pos (mm)')
        ax.set_ylabel('μV/100g (joint calibrated)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(ncol=2, fontsize='small')
        plt.tight_layout()

        if self.fig_dir and save_name:
            plt.savefig(os.path.join(self.fig_dir, save_name), dpi=600)
        plt.show()

    def plot_residual_heatmaps(self,
                                   use_calibrated: bool = True,
                                   cmap: str = "coolwarm",
                                   fmt: str = ".1f"):
            """
            画三个模组的残差热力图（每个模组一张图）。
            参数
            -------
            use_calibrated : True  → 使用 self.ch*_out_joint
                            False → 使用原始 self.ch*_data
            cmap           : 颜色映射，默认 'coolwarm'
            fmt            : 误差值文本格式，默认保留 1 位小数
            """
            # ---------- 数据源检查 ----------
            if use_calibrated:
                if not (hasattr(self, "ch1_out_joint") and hasattr(self, "ch2_out_joint")):
                    raise RuntimeError("请先执行 calibrate_joint() 获得校准数据")
                ch1_src = self.ch1_out_joint
                ch2_src = self.ch2_out_joint
            else:
                ch1_src = self.ch1_data
                ch2_src = self.ch2_data

            n_points, n_series = ch1_src.shape
            cols_per_module = len(self.temps_order)
            n_modules = n_series // cols_per_module

            # 找到基准温度列在 temps_order 中的位置
            try:
                ref_pos = self.temps_order.index(self.ref_temp)
            except ValueError:
                raise ValueError(f"temps_order 中不包含基准温度 {self.ref_temp}")

            for mod_idx in range(n_modules):
                # ---------- 当前模组对应的列 ----------
                start = mod_idx * cols_per_module
                end = start + cols_per_module
                cols = list(range(start, end))

                # 基准列索引
                ref_col = cols[ref_pos]

                '''
                临时-力校准
                '''

                y1_data = ch1_src[:, [ref_col]]  # 通道1先验数据作为y1
                y2_data = ch2_src[:, [ref_col-6]]  # 通道2先验数据作为y2
                x_values = self.x_values if len(self.x_values) > 0 else np.linspace(1, 15,
                                                                                    len(y1_data))

                # 计算最小二乘所需的累加项
                S_xx = np.sum(x_values ** 2)
                S_x = np.sum(x_values)
                S_xy1 = np.sum(y1_data * x_values)
                S_y1 = np.sum(y1_data)
                S_xy2 = np.sum(y2_data * x_values)
                S_y2 = np.sum(y2_data)
                N = len(x_values)

                # 计算分母（避免除零）
                denom = (N * S_xx) - (S_x ** 2)
                if denom == 0:
                    raise ValueError("分母为零，可能所有x值相同，无法估计模板参数")

                # 最小二乘估计k1、b1、k2、b2
                k_1 = ((N * S_xy1) - (S_x * S_y1)) / denom
                b_1 = ((S_xx * S_y1) - (S_x * S_xy1)) / denom
                k_2 = ((N * S_xy2) - (S_x * S_y2)) / denom
                b_2 = ((S_xx * S_y2) - (S_x * S_xy2)) / denom
                self.result_k1b1k2b2 = np.array([k_1, b_1, k_2, b_2])  # 存储估计结果
                print(f"Module {mod_idx}:估计的k1={k_1:.4f}, b1={b_1:.4f}, k2={k_2:.4f}, b2={b_2:.4f}")
                [alphak_2, alphab_2] = [k_2/k_1, b_2/b_2]
                print(f"Module {mod_idx}:估计的alpha1={alphak_2:.4f}, alpha2={alphab_2:.4f}")

                # ---------- 计算残差 ----------
                ch1_res = np.round(100*(ch1_src[:, cols] - ch1_src[:, [ref_col]])/ch1_src[:, [ref_col]],0).astype(int)  # (n_points, 7)
                ch2_res = np.round(100*(ch2_src[:, cols] - ch2_src[:, [ref_col]]) / ch2_src[:, [ref_col]],0).astype(int)  # (n_points, 7)

                # 拼接成 (n_points, 14)：CH1 在前、CH2 在后
                n_rows, n_cols = ch1_res.shape
                residual_mat = np.empty((n_rows, n_cols * 2), dtype=ch1_res.dtype if ch1_res.dtype == ch2_res.dtype else object)
                residual_mat[:, ::2] = ch1_res
                residual_mat[:, 1::2] = ch2_res
                # ---------- 绘图 ----------
                vmax = np.nanmax(np.abs(residual_mat))
                fig, ax = plt.subplots(figsize=(14, 8))
                im = ax.imshow(residual_mat,
                               aspect='auto',
                               cmap=cmap,
                               vmin=-vmax,
                               vmax=vmax)

                # 轴标签
                x_labels = []
                for t in self.temps_order:
                    x_labels.extend([f"{t}°C\nCH1", f"{t}°C\nCH2"])
                ax.set_xticks(np.arange(residual_mat.shape[1]), labels=x_labels)
                ax.set_xlabel("Temperature & Channel")
                ax.set_ylabel("Position index")

                # 网格线（可选美化）
                ax.set_xticks(np.arange(residual_mat.shape[1] + 1) - .5, minor=True)
                ax.set_yticks(np.arange(residual_mat.shape[0] + 1) - .5, minor=True)
                ax.grid(which='minor', color='black', linewidth=0.2)
                ax.tick_params(which='minor', length=0)

                # 在每个格子中心写数值
                for i in range(residual_mat.shape[0]):
                    for j in range(residual_mat.shape[1]):
                        val = residual_mat[i, j]
                        ax.text(j, i, format(val, fmt),
                                ha='center', va='center',
                                fontsize=11,
                                color='white' if abs(val) > vmax * 0.4 else 'black')

                # 色条
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Proportion %")

                # 标题 & 保存
                mode = "calibrated" if use_calibrated else "raw"
                ax.set_title(f"Module {mod_idx} residuals ({mode}, ref={self.ref_temp}°C)")

                if self.fig_dir:
                    fname = f"residual_heatmap_module_{mod_idx}_{mode}.png"
                    plt.savefig(os.path.join(self.fig_dir, fname), dpi=600, bbox_inches='tight')

                plt.show()


    # ===================  运行流程  =================== #
    def run_all(self):
        """默认执行联合校准流程"""
        # 1) 读取数据
        self.load_data()
        # 2) 强制执行所有模组数据校验
        self.check_all_modules()

        # 2.5) 绘图
        self.plot_0()

        # 3) 原始绘图
        self.plot_signals(module_idx=None, save_name="raw_all_modules.png")
        for mod_idx in range(self.num_modules):
            self.plot_signals(module_idx=mod_idx, save_name=f"raw_module_{mod_idx}.png")

        self.plot_residual_heatmaps(use_calibrated=False)

        # 4) 校准流程
        self.compute_kb_joint(modules_to_use=(0, 2))  # ← 只用模组 0、2
        self.fit_poly_joint()
        self.calibrate_joint()

        self.plot_calibrated_joint(module_idx=None, save_name="joint_calibrated_all.png")
        for mod_idx in range(self.num_modules):
            # self.plot_calibrated_joint(module_idx=mod_idx, save_name=f"joint_calibrated_module_{mod_idx}.png")

        # 4.5) 绘图
        self.plot_0(C = True)

        # 5) 残差热力图
        self.plot_residual_heatmaps(use_calibrated=True)

    @property
    def num_modules(self) -> int:
        return self.ch1_data.shape[1] // len(self.temps_order) if self.ch1_data is not None else 0


# ===================  脚本入口  =================== #
if __name__ == '__main__':
    input_dir = r'F:\Temputure_Calibration\data_0723'
    fig_dir = r'F:\Temputure_Calibration\fig_0724'
    input_dir = r'F:\Temputure_Calibration\data_0723\0724_Temp'
    fig_dir = r'F:\Temputure_Calibration\data_0723\0724_Temp\fig'

    tc = TC(input_dir=input_dir,
            fig_dir=fig_dir,
            ref_rows=[0, 1,8,13,14],  # 可根据需要调整参考行
            ref_temp=30,  # 基准温度
            temps_order=[-10, 0, 10, 20, 30, 40, 50])
            # temps_order = [10, 20, 30, 40, 50])

    # 执行完整流程（含所有绘图）
    tc.run_all()

    # 若想分步调试，可单独调用：
    # tc.load_data()
    # tc.check_all_modules()  # 单独执行数据校验
    # tc.plot_signals(module_idx=None, save_name="raw_all.png")
    # tc.compute_kb_all()
    # tc.fit_poly_all()
    # tc.calibrate_all()
    # tc.plot_calibrated(module_idx=0, save_name="calib_mod0.png")