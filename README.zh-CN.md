# MetSMILES 0.3.0

面向分子金属配合物和周期结构的配位环境注释。支持 Python 3.9–3.12；方法版本 **`manuscript-v1`**，序列化版本 **`canonical-sorted-v1`**。

[English](README.md) · [详细方法规范](METHODS.md) · [更新日志](CHANGELOG.md)

## 范围说明

MetSMILES 0.3.0 是论文所述配位环境表示的参考实现（方法配置 `manuscript-v1`，序列化配置 `canonical-sorted-v1`）。它从成对 XYZ/SDF 结构生成配位后缀与配体前缀，并从 CIF 输入生成配位与周期注释。训练数据、模型权重及训练/评估流程由作者另行维护，不属于本生成器包。

按论文约定，供体诱导图中两个及以上相连原子编码为 `%` 条目；这是对相连供体的表示约定，不是电子结构层面的判断。全碳五元环贡献三个几何位点，两原子分量取中点，其余分量取质心。几何评分使用高斯核（默认 σ=15°）与 CN 2–6 理想模板匹配。输出采用规范化 token 顺序；论文图示中的字符串条目顺序可能与规范输出不同，但组成相同。

自动配体前缀由 RDKit 从 SDF 提取：仅删除指定的目标金属原子，其余片段保留 SDF 键级、形式电荷与自由基信息。化学上不可用的输入会得到明确的类型化错误，而不是悄悄输出不完整字符串。

## 安装与运行

在解压后的发布根目录执行：

```bash
python -m pip install .
python -m pip install '.[molecular]'         # 自动分子SDF前缀需要RDKit
python -m pip install '.[crystal]'           # CIF主linker与空间群需要RDKit/spglib
python -m pip install '.[dev,molecular,crystal]'
```

基础依赖是 numpy、pandas；`dev` 提供 pytest。以上是用户自行安装的命令，导入代码不会下载依赖。也可设置 `PYTHONPATH=src`，不安装直接使用。已有运行依赖时，可用 `pip install --no-index --no-deps 本地wheel文件` 离线安装。构建需要 setuptools≥77 和 wheel（或构建前端）。

```python
from metsmiles import generate_metsmiles_record, generate_global_metsmiles

# 默认完整自动前缀；无RDKit或化学净化失败会明确报错。
r = generate_metsmiles_record('examples/data/405.xyz', 'examples/data/405.sdf')
print(r['SMILES_1'], r['SMILES_3'], r['prefix_source'])

# 显式前缀优先，包括空字符串。代码不验证用户所给前缀的化学正确性。
r = generate_metsmiles_record('examples/data/405.xyz', None,
                              base_smiles='[Cl-].[Cl-].[Cl-].[Cl-]')

# 只有XYZ时，必须明确指定仅后缀，不能静默省略前缀。
r = generate_metsmiles_record('examples/data/405.xyz', None,
                              prefix_mode='suffix-only')
```

文件 API 和目录 API 接受 `prefix_mode='auto'|'suffix-only'`，默认 auto。`base_smiles is not None` 总是覆盖自动提取。自动分子前缀只删除显式目标金属，保留所有其他片段、重复片段、电荷、同位素、编码的立体信息及自由基；按 canonical isomeric SMILES 片段字典序排序。先不净化读 SDF，删除目标金属后再净化；不猜电荷、不修键级、不为删除金属后的价态缺口补隐式氢，也不按 ID 查询参考答案。移除金属键可暴露自由基，这些不会偷偷转成历史闭壳层前缀。`PrefixError`、`PrefixDependencyError`、`PrefixChemistryError` 给出缺SDF、缺RDKit及化学失败的明确边界。金属-only 输入可合法为空前缀，并有专门状态。

`generate_metsmiles_records` 返回逐中心记录，各条使用**整个结构的前缀**；`generate_global_metsmiles` 返回全局串、`centers` 和 `clusters`。低层 `global_metsmiles_for_molecule` 没有SDF上下文，只拼接给定前缀，否则明确为 suffix-only。默认目标元素只有 Fe，其他金属通过 `target_metal_elements=['Co','Ni']` 等显式指定。

## CLI 与参数

```bash
python -m metsmiles examples/data/405.xyz --sdf examples/data/405.sdf
python -m metsmiles examples/data/405.xyz --prefix-mode suffix-only
python -m metsmiles examples/data --prefix-mode suffix-only --output-csv coordination.csv
python -m metsmiles examples/data --output-csv complete.csv
python -m metsmiles examples/data/mil-88b.cif --topology acs
python -m metsmiles examples/data/mil-88b.cif --no-ligands --no-symmetry-detection
```

默认接触阈值 γ=1.16、Gaussian σ=15°、无SDF时的配体内距离连通系数1.20。CLI 支持 `--bond-tolerance`、`--geometry-sigma`、`--connectivity-tolerance`、可重复的 `--metal`、`--method-profile manuscript-v1`。单/多中心按中心数量显式选择，不以捕获任意 ValueError 后重试掩盖前缀错误。目录使用同名 XYZ/SDF；`--base-smiles-csv` 两列为 `gdb_idx,smiles_1`。失败行记录 `status=error`、异常类型及信息，不伪造 SMILES；有失败时CLI退出码非零。成功和失败行均保留方法、参数信息。

## 方法要点及语法

1. 按 `d≤γ(rM+ri)` 判定供体接触；SDF 只给内部连接，不决定是否接触。半径表以源码为准，如 Fe1.52、C0.69、N0.71、O0.66 Å。
2. 在**接触原子诱导图**上分组，不沿非接触原子扩展整配体。单原子 `&N`，多原子 `%CO`；所有接触均进入分析，不恢复历史跳过位点逻辑。
3. 两原子取中点。实际全碳、无弦、连通五环取最近锚点原子本身及其余两对相邻碳的中点，共三位点；不是三个质心，也不按原子输入顺序任意配对。其他组取单质心，非Cp五组同样如此。
4. 有效CN是几何位点数，不是接触原子数或电子计数。CN2–6匹配 Linear、TrigonalPlanar、Tetrahedral/SquarePlanar、TrigonalBipyramid/SquarePyramid、Octahedral；其他CN为unknown，不截取最近六点。
5. 测量角与模板角排序后配对，用 Gaussian 均值评分。`Confidence` 是相似度，不是校准概率；没有低分拒识阈值，CN6仅一个候选。不能从几何串推出氧化态、自旋或完整异构体。
6. 供体/组内元素/多原子条目按字典序规范排序。Figure2 实际结构 `train_iron_ref/48` 的规范后缀为 `[{Octahedral}Fe&C&C%CC%CCCCC]`，图中 `%CCCCC%CC` 顺序不同；两者组成相同、条目顺序不同。

全局扩展保留 `:uk` 桥联、`{cluster:...}`、`xN`、`|` 与 `{periodic:...}`。簇header后的xN为簇数，中心后的xN为该类中心数；这些不能恢复完整原子—配体—位点连接图。

## CIF 与分子前缀不是同一策略

CIF 展开对称位点，按度量球搜索截断半径内**全部周期像**，处理斜晶胞；维数为各连通分量的cycle-translation rank最大值，不是跨轴数，也不限于minimum image。商晶胞簇不自动证明有限独立SBU。

保留原有有限片段提取策略：RDKit尝试候选电荷/键级，主前缀选**最大片段，且仅在该片段有限、可转换时生成**，不等于全部化学组成。最大片段无限或不可转换时，自动前缀不可用，不静默改用更小片段；其他片段和份数保存在 `ligands`。无限/截断片段不生成假分子SMILES。无RDKit、转换失败或禁用提取时，前缀来源/状态明确为缺失或禁用，不冒充完整分子auto成功。无spglib则保留文件声明空间群。`topology='acs'` 是显式提供标签，**不是自动识别**。MIL-88B要写acs需显式传入。

## 消融与测试

- `mask_donor_entries`：论文noDonor逐条目替换，`&N→&X`、`%CO→%X`，不是`%XX`。
- `remove_pi_marker`：noPi，仅中心内`%→&`。
- `remove_geometry`：noGeo，仅删除中心几何token。
- `remove_donor_list`：删除供体列表，单独保留为不同操作；`geometry_token_only`也不是noDonor。

`add_representation_columns` 的 `SMILES_NO_DONOR`、`SMILES_NO_PI_MARKER`、`SMILES_NO_GEO` 分别映射以上前三种。保留有机 `%10`、桥联、多中心、簇、重复与周期字段。

```bash
python examples/run_example.py
python examples/run_example.py --prefix-mode suffix-only
python examples/run_mof.py
python examples/run_paper_examples.py  # 三个论文相关示例，需要molecular/crystal依赖
python examples/run_paper_examples.py --suffix-only --no-ligands --no-symmetry-detection
python -m pytest -q -W error
```

三个论文相关实例见 [examples/README.md](examples/README.md)：Figure 2a 的 Cp* 配合物、与 Figure 2b 对应的四Cl四面体配位基元、MIL-88B。运行器核对预期输出并展示 noDonor/noPi/noGeo，实例定义及参考结果分别保存在 `examples/paper_cases.json` 和 `examples/paper_golden.json`。

Figure2复制结构及来源哈希在 `examples/figure2`；实际自动前缀与给定参考前缀分开测试。CI 定义 Python3.9–3.12最小/扩展依赖及wheel/sdist检查；本地单一版本执行不冒称已跑遍所有Python版本。

## 未解决范围与发布

不包含训练器、模型权重、14描述符流程、tokenizer或统一的图训练定义。

MIT许可证及真实原署名保留于 LICENSE、CITATION.cff。版本统一0.3.0。仓库地址：https://github.com/majing-group/MetSmiles
