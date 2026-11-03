# Changelog

## 0.3.0 — 2026-11-03

- Introduce explicit `manuscript-v1` method and `canonical-sorted-v1` serialization profiles, carried by reports and records.
- Apply the SI connected-donor-component convention: all connected groups of two or more contacts are `%` entries; all contacts participate. This is not electronic π-bond proof.
- Use a midpoint for two donors; an actual bonded all-carbon five-cycle has a topology-based nearest anchor plus two pair midpoints (three sites). Other components, including non-Cp five-atom groups, use one centroid as an explicit ambiguity completion.
- Add molecular RDKit SDF prefix extraction: remove only target metals; retain all other fragments, duplicates, formal charges, stereochemistry, isotopes and radicals; no charge search, bond-order repair or implicit-H completion. Default auto failures are explicit typed errors. Explicit bases override extraction; suffix-only use must be requested.
- Correct manuscript noDonor to one X per coordination entry. Retain donor deletion as a separately named helper and add noGeo.
- Select molecular CLI mode by center count instead of retrying arbitrary ValueErrors. Directory failures are explicit typed rows and cause nonzero CLI exit.
- Preserve strict paired-input validation, Gaussian sigma=15 scoring, CN2–6 templates, robust CIF parsing, all-image metric searches and cycle-translation-rank periodicity. Zero-site geometry is consistently unknown.
- Add actual Figure2 paired structure/provenance and separate automatic versus supplied-prefix tests; document explicit MIL-88B acs metadata and largest-linker CIF scope.
- Add repository URLs and bilingual documentation; retain existing author/license attribution; synchronize package and citation version.

This release changes representation semantics and default molecular prefix behavior relative to 0.2.0.

中文：0.3.0将论文方法、序列化、前缀和消融操作显式版本化；非Cp五原子分量按约定取单质心。默认auto需可用SDF/RDKit，失败不静默返回不完整串；仅后缀调用需显式suffix-only。
