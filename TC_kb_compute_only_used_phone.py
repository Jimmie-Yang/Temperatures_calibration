"""
@Author: LXH (refactored by ChatGPT)
@Date  : 2025-07-24

功能概述：
1. 读取多个 Excel 文件中的 CH1_sig / CH2_sig 列，按文件名规则排序并解析温度。
2. 每 7 列为一个模组（module），共 3 个模组；温度顺序固定为 [0, 10, 20, 30, 40, 50, -10]。
3. 先绘制原始数据（两个通道）曲线对比。
4. 对每个模组、每个通道计算校准系数 k、b：
   - 从若干参考位置行（ref_rows）取数据，设定基准温度（ref_temp = 20°C）。
   - 通过最小二乘拟合 y = k * x + b，得到每个温度对应的 k_j、b_j。
5. 将 k、b 与温度 t（摄氏度）拟合为二次多项式。
6. 用拟合的多项式对整列数据做校准并绘图展示。

使用说明：
- 根据实际数据路径修改 input_dir、fig_dir。
- 可根据需要调整 ref_rows、ref_temp。
"""

import os
import glob
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TC:
    def __init__(self,
                 input_dir: str,
                 fig_dir: str = None,
                 ref_rows: List[int] = None,
                 ref_temp: int = 20,
                 temps_order: List[int] = None,
                 channel_names: Tuple[str, str] = ("CH1_sig", "CH2_sig")):
        """初始化
        Args:
            input_dir     : 数据所在文件夹
            fig_dir       : 图像保存文件夹，为 None 则不保存
            ref_rows      : 用于拟合 k,b 的参考行索引列表（位置下标，从 0 开始）
            ref_temp      : 基准温度（用于计算 S'）
            temps_order   : 每个模组内部温度的固定顺序
            channel_names : Excel 中对应的两列列名
        """
        self.input_dir = input_dir
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True) if self.fig_dir else None

        self.ref_rows = [0, 1, 2] if ref_rows is None else ref_rows
        self.ref_temp = ref_temp
        self.temps_order = [0, 10, 20, 30, 40, 50, -10] if temps_order is None else temps_order
        self.channel_names = channel_names

        # 读取后填充的属性
        self.files_sorted: List[str] = []   # 排序后的文件路径
        self.temps: List[int] = []          # 每一列对应的温度
        self.ch1_data: np.ndarray = None    # shape = (n_points, n_series)
        self.ch2_data: np.ndarray = None    # shape = (n_points, n_series)
        self.x_values: np.ndarray = None    # x 轴位置 (mm)

        # 结果存储（按模块和通道）
        # 结构：result_k[mod]["ch1"] -> np.ndarray(7,)
        self.result_k: List[Dict[str, np.ndarray]] = []
        self.result_b: List[Dict[str, np.ndarray]] = []
        self.poly_k:   List[Dict[str, np.ndarray]] = []  # 拟合的二次多项式系数 [a, d, c]
        self.poly_b:   List[Dict[str, np.ndarray]] = []

        # 校准后的数据
        self.ch1_out_data: np.ndarray = None
        self.ch2_out_data: np.ndarray = None

    # ===================  数据加载与基础处理  =================== #
    def _sort_key(self, fn: str) -> Tuple[int, int]:
        """根据文件名排序，格式示例：black_T0_*.xlsx
        先按颜色顺序，再按温度值排序（把 110 当作 -10）
        """
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
        """加载 Excel 数据，并将 CH1/CH2 转置为 (n_points, n_series) 格式。"""
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

    # ===================  绘图相关  =================== #
    def plot_signals(self, module_idx: int = None, save_name: str = None):
        """绘制原始数据（两个通道）
        Args:
            module_idx: 指定模组索引（0/1/2）。None 表示绘制所有列。
            save_name  : 保存文件名（不含路径），只有在 fig_dir 不为 None 时有效。
        """
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
        """绘制校准后的结果（两个通道）。
        Args:
            module_idx: 模组索引
            save_name : 保存文件名（不含路径）
        """
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
    def _compute_kb_one_channel(self, data: np.ndarray, cols: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """对一个通道和一个模组内部的 7 列数据计算 k,b。
        Args:
            data : shape = (n_points, n_series)
            cols : 本模组对应的列索引（长度=7）
        Returns:
            k: (7,)
            b: (7,)
        """
        # 从参考行取数据
        S = data[self.ref_rows, :][:, cols]  # shape = (len(ref_rows), 7)

        # 找到基准温度位置（列索引）
        # 先得到 cols 对应的温度列表
        temps_mod = [self.temps[c] for c in cols]
        try:
            ref_idx = temps_mod.index(self.ref_temp)
        except ValueError:
            raise ValueError(f"在模组列中未找到基准温度 {self.ref_temp}°C")

        S_prime = np.tile(S[:, [ref_idx]], (1, S.shape[1]))  # 每列都用 ref_idx 的列值

        m, n = S.shape  # m = len(ref_rows), n = 7
        k = np.zeros(n)
        b = np.zeros(n)
        for j in range(n):
            X = np.column_stack([S[:, j], np.ones(m)])  # (m,2)
            y = S_prime[:, j]                          # (m,)
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            k[j], b[j] = sol

        # 验证残差（可选打印）
        residual = S @ np.diag(k) + np.tile(b, (m, 1)) - S_prime
        print(f"残差 Frobenius 范数 (channel data cols={cols}):", np.linalg.norm(residual, 'fro'))

        return k, b

    def compute_kb_all(self):
        """对所有模组、两个通道计算 k,b 并存储。"""
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

            k_ch1, b_ch1 = self._compute_kb_one_channel(self.ch1_data, cols)
            k_ch2, b_ch2 = self._compute_kb_one_channel(self.ch2_data, cols)

            self.result_k.append({"ch1": k_ch1, "ch2": k_ch2})
            self.result_b.append({"ch1": b_ch1, "ch2": b_ch2})

    def fit_poly_all(self):
        """将每个模组、每个通道的 k、b 与温度做二次多项式拟合。"""
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
        """用拟合得到的多项式对所有数据进行校准。
        输出 self.ch1_out_data / self.ch2_out_data。
        """
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

    # ===================  运行流程  =================== #
    def run_all(self):
        """完整流程：加载数据 -> 绘原始图 -> 计算 k,b -> 拟合 -> 校准 -> 绘校准后的图"""
        self.load_data()
        # 原始数据整体展示
        self.plot_signals(module_idx=None, save_name="raw_all_modules.png")

        # 按模组分别展示原始数据
        for mod_idx in range(self.num_modules):
            self.plot_signals(module_idx=mod_idx, save_name=f"raw_module_{mod_idx}.png")

        # 计算 k,b
        self.compute_kb_all()
        # 拟合二次多项式
        self.fit_poly_all()
        # 校准
        self.calibrate_all()

        # 按模组绘制校准后结果
        for mod_idx in range(self.num_modules):
            self.plot_calibrated(module_idx=mod_idx, save_name=f"calibrated_module_{mod_idx}.png")

    @property
    def num_modules(self) -> int:
        return self.ch1_data.shape[1] // len(self.temps_order) if self.ch1_data is not None else 0


# ===================  脚本入口  =================== #
if __name__ == '__main__':
    input_dir = r'F:\Temputure_Calibration\data_0723'
    fig_dir = r'F:\Temputure_Calibration\fig_0724'

    tc = TC(input_dir=input_dir,
            fig_dir=fig_dir,
            ref_rows=[0,1,2,6,7,12,13,14],   # 可根据需要调整参考行
            ref_temp=20,          # 基准温度
            temps_order=[-10, 0, 10, 20, 30, 40, 50])

    # 执行完整流程（含所有绘图）
    tc.run_all()

    # 若想分步调试，可单独调用各方法：
    # tc.load_data()
    # tc.plot_signals(module_idx=None, save_name="raw_all.png")
    # tc.compute_kb_all()
    # tc.fit_poly_all()
    # tc.calibrate_all()
    # tc.plot_calibrated(module_idx=0, save_name="calib_mod0.png")
