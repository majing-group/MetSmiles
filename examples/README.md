# Examples and provenance

All paths below are relative to this source release. Coordinates were copied unchanged from the existing study example collection; no full study data table is bundled.

| Files under `data/` | Reference canonical suffix |
|---|---|
| `0.xyz`, `0.sdf` | `[{Octahedral}Fe&Cl&Cl&N&N&O&O]` |
| `405.xyz`, `405.sdf` | `[{Tetrahedral}Fe&Cl&Cl&Cl&Cl]` |
| `493.xyz`, `493.sdf` | `[{Tetrahedral}Fe&C&C%CC%CC]` |
| `498.xyz`, `498.sdf` | `[{TrigonalBipyramid}Fe&C&N%CCCCC]` |
| `mil-88b.cif` | two Fe3 quotient-cell clusters, each with three equivalent Octahedral Fe centers; periodic rank 3 |

`figure2/48.xyz` and `figure2/48.sdf` are the exact paired Cp* example from the study shared `train_iron_ref` collection, ID48, identified from the SI ligand prefix. `figure2/reference.json` records provenance, unchanged input hashes, the supplied reference base and both diagram/canonical suffix orders. The canonical suffix is `[{Octahedral}Fe&C&C%CC%CCCCC]` (effective CN6); the diagram reverses the two π entries. Automatic SDF prefix and supplied reference prefix are different paths and are tested separately; neither is looked up at runtime.

```bash
python examples/run_example.py                # molecular extra: automatic SDF prefix
python examples/run_example.py --prefix-mode suffix-only
python examples/run_mof.py                    # crystal extras for reference BDC/group
python -m metsmiles examples/figure2/48.xyz --sdf examples/figure2/48.sdf
python -m metsmiles examples/data --prefix-mode suffix-only --output-csv coordination.csv
python -m metsmiles examples/data/mil-88b.cif --no-ligands --no-symmetry-detection
```

The molecular walkthrough explicitly labels the no-SDF branch suffix-only. MIL-88B's example supplies `topology='acs'`; no topology is inferred. With RDKit/spglib the detected group is `P6_3/mmc`, point group `6/mmm`, and the main linker is BDC(2-). Without these optional dependencies, CIF graph analysis still runs, the declared group is used and linker SMILES can be unavailable. Only the largest fragment can occupy the CIF prefix, and only when it is finite and convertible; a failed or infinite largest fragment is not silently replaced with a smaller one. The prefix does not encode all composition.

## Three actual paper examples / 三个论文实例

Run all three with one command after installing the release and optional extras:

```bash
python -m pip install -e '.[molecular,crystal]'
python examples/run_paper_examples.py > paper_examples.json
```

The script resolves inputs relative to its own file, so it also runs from another working directory. JSON contains the figure/source mapping, relative input files, expected and freshly computed canonical suffixes, effective coordination numbers, method/serialization profiles, automatic `SMILES_1` and its provenance, full strings, topology-supplied status, and noDonor/noPi/noGeo variants. It does not read golden outputs to generate a result. `paper_cases.json` records mappings and SHA256 hashes; `paper_golden.json` contains stable fields from a fresh release-API execution (RDKit 2025.09.3), not supplied reference prefixes.

| Paper mapping | Reused input | Independently checked interpretation |
|---|---|---|
| Figure 2a Cp* encoding schematic, `train_iron_ref/48` | `figure2/48.xyz`, `.sdf` | Raw nine carbon contacts partition into 1+1+2+5 in the SDF-induced donor graph; two CO + one olefin midpoint + three Cp sites give effective CN6. Canonical suffix is `[{Octahedral}Fe&C&C%CC%CCCCC]`. |
| Figure 2b tetrahedral four-Cl environment, `test_exp/405` | `data/405.xyz`, `.sdf` | Exactly four Cl donors; all six independently calculated Cl–Fe–Cl angles are within 0.10 degrees of the ideal tetrahedral angle; suffix is `[{Tetrahedral}Fe&Cl&Cl&Cl&Cl]`. |
| Figure 2b MIL-88B(Fe), manuscript P0028/P0030 | `data/mil-88b.cif` | Independent periodic-distance O shells and API results give six FeO6 sites, two triply shared O atoms, and two Fe3 quotient-cell groups. Detected group `P6_3/mmc`, point group `6/mmm`, bonded-component rank 3; `acs` is supplied explicitly. |

Mapping is recorded in `paper_cases.json` using manuscript/SI figure references, dataset identifiers, the Cp* ligand prefix, input paths and SHA256 hashes. The full Cp* ligand prefix in the SI matches `train_iron_ref/48` fragment-wise. Both 48 and 405 input pairs were compared byte-for-byte with their corresponding dataset records. The four-Cl example demonstrates the Figure 2b coordination motif; it does not establish an atom-by-atom coordinate mapping from the schematic. The MIL-88B CIF is identical to the framework example used for the manuscript check, but no independent deposition identifier is asserted. The full study collection is not a runtime dependency.

### Prefix and representation boundaries

* Automatic molecular prefixes are extracted from the SDF without a supplied base. For 48 the automatic prefix is `C[C]1C(C)=C(C)C(C)=C1C.[CH2][CH]CC.[C][O].[C][O]`: the SDF records single C–O bonds, which serialize as `[C][O]`. The supplied reference base instead uses `[C]=O` (see `figure2/reference.json`); both paths are tested separately. The coordination suffix reorders the two π entries relative to the diagram.
* For 405 the four `[Cl]` fragments are retained, including duplicates and SDF-authoritative neutral radical annotations. No chloride charge or Fe oxidation state is inferred. Neither molecular input has an assigned ligand stereocenter; tests do not invent stereochemistry.
* MIL-88B's extracted main linker is BDC(2-), `O=C([O-])c1ccc(C(=O)[O-])cc1`, not the entire framework composition. CIF charge-candidate search differs from the molecular SDF-authoritative prefix policy. The `x2` cluster and `x3` local-site multiplicities mean **six**, not three, total Fe centers.
* noDonor masks each coordination entry with one `X`, retaining `&`, `%` and bridge tags; it does not delete the donor list. For Cp*, it gives `[{Octahedral}Fe&X&X%X%X]`. noPi gives `[{Octahedral}Fe&C&C&CC&CCCCC]`; noGeo gives `[Fe&C&C%CC%CCCCC]`. Helpers act on full freshly generated strings while preserving the ligand prefix and periodic/cluster grammar. These are representation ablations, not model-performance reproductions.

For a core-only, explicitly reduced run:

```bash
python examples/run_paper_examples.py --suffix-only --no-ligands --no-symmetry-detection
python -m pytest -p no:cacheprovider tests/test_paper_examples.py
```

`--suffix-only` affects molecular prefixes only; `--no-ligands` suppresses CIF linker extraction; `--no-symmetry-detection` uses the CIF declaration (`P1`, point group `1`) instead of detected `P6_3/mmc`. The expected suffix in JSON changes accordingly. The full runner fails with an actionable installation/flag message when an optional dependency is missing rather than silently reporting a reduced result as the paper branch. Only optional-dependency-specific tests skip when unavailable; input-hash, raw geometric/graph, explicit-prefix and reduced-run tests remain active.

These are small executable **representation** examples, not an electronic-structure benchmark. Geometry tokens do not predict spin or oxidation state. Geometry similarity scores are not probabilities; notably the Cp* score is about 0.4495, so its constrained-template `Octahedral` classification is not proof of strict Oh symmetry. See [METHODS.md](../METHODS.md) for the full specification.

中文：以上三个实例有论文映射证据与真实结构文件，不是人造几何。48的来源是SI完整配体前缀逐片段精确匹配，405输入与对应数据记录逐字节一致，但仅作为Figure 2b四Cl配位基元示例，不声称与示意图逐原子坐标对应。自动SDF前缀保留SDF键级（48的CO序列化为[C][O]），与显式参考前缀分开测试；图示后缀与规范后缀另有π项排序差异。MIL-88B的acs是显式参数，不是自动拓扑识别；两个Fe3簇共六个Fe，主linker不代表完整组成。测试包含独立角度、供体子图、周期距离及电荷/自由基检查，而非把给定答案再读回作端到端验证。

中文：分子例子支持自动SDF前缀及显式仅后缀。Figure2实际ID48输入、来源哈希及给定前缀单独保存，图示排序与规范排序不逐字相同。MIL-88B的acs为例子显式参数，不是自动拓扑识别；CIF主前缀仅在最大片段有限且可转换时生成，不代表完整组成。
