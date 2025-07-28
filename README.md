# Temperatures_calibration
这个储存库是实习期间为手机loadcell侧键项目做的温度补偿项目

# 温度校准工具

本仓库提供了一个基于 Python 编写的温度信号校准工具，封装在 `TC` 类中。该工具用于从 Excel 文件读取传感器数据、校验数据稳定性、计算校准参数（k、b）、进行多项式拟合，并可视化原始与校准后的数据。

## 功能概览

- 📂 从 Excel 文件批量加载不同温度下的信号数据
- ✅ 自动校验各模组通道在不同温度下的信号稳定性（最大值与最小值差 ≤ 100）
- 📈 利用最小二乘法回归求解每组的校准系数 `k` 和 `b`
- 🔁 拟合 `k(t)` 与 `b(t)` 的二次多项式函数用于插值校准
- 🧪 提供标准校准和联合校准（仅使用部分模组数据）两种方式
- 🗺️ 生成每个模组在校准前/后的残差热力图，直观显示误差分布
- 🖼️ 自动保存所有绘图结果到指定路径

## 目录结构说明

- `input_dir`：输入文件夹，包含命名格式为 `xxx_T30_.xlsx` 的 Excel 文件
- `fig_dir`：输出图像保存路径

## 类：`TC`

### 初始化方法

```python
TC(input_dir, fig_dir=None, ref_rows=None, ref_temp=20, temps_order=None, channel_names=("CH1_sig", "CH2_sig"))
```

- `ref_rows`：参考行索引列表
- `ref_temp`：参考温度（如 30°C）
- `temps_order`：温度顺序（默认 `[-10, 0, 10, 20, 30, 40, 50]`）

### 主要方法功能列表

| 方法名 | 功能描述 |
|--------|----------|
| `load_data()` | 读取所有 Excel 中的信号数据 |
| `check_all_modules()` | 校验每个模组的信号稳定性 |
| `compute_kb_all()` | 对每个模组/通道计算 k、b 参数 |
| `fit_poly_all()` | 对 k(t) 和 b(t) 拟合二次函数 |
| `calibrate_all()` | 应用拟合结果进行数据校准 |
| `compute_kb_joint(modules_to_use)` | 联合选定模组进行参数估计 |
| `fit_poly_joint()` | 拟合联合校准的 k(t)、b(t) |
| `calibrate_joint()` | 应用联合校准参数进行校正 |
| `plot_signals()` | 绘制原始数据曲线图 |
| `plot_calibrated()` | 绘制标准校准后的图像 |
| `plot_calibrated_joint()` | 绘制联合校准后的图像 |
| `plot_residual_heatmaps()` | 残差热力图（支持对比） |
| `plot_0()` | 提取极值点绘图（如第1和15点） |
| `run_all()` | 执行完整联合校准流程 |

## 配置脚本入口
在脚本底部修改：
```
python
Copy
Edit
input_dir = r'/path/to/your/data'
fig_dir   = r'/path/to/output/figures'
tc = TC(
    input_dir=input_dir,
    fig_dir=fig_dir,
    ref_rows=[0,1,2,3,4],   # 参考行索引列表
    ref_temp=20,            # 基准温度（℃）
    temps_order=[-10,0,10,20,30,40,50]  # 温度顺序
)
```

### 示例使用

```python
if __name__ == '__main__':
    tc = TC(input_dir="路径/数据", fig_dir="路径/输出", ref_rows=[0,1,8,13,14], ref_temp=30)
    tc.run_all()
```

## 环境依赖

- Python ≥ 3.7
- `numpy`, `pandas`, `matplotlib`, `openpyxl`

安装依赖：

```bash
pip install numpy pandas matplotlib openpyxl
```

## 许可协议

MIT License，自由使用，欢迎引用。



















