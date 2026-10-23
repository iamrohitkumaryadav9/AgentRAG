hybrid-bge vs bm25: lost 27, won 64, tied 209 of 300 queries (sum of losses -7.96, sum of gains +20.23 nDCG@10).

| group | queries | distinct gold docs |
| --- | --- | --- |
| dense-blind-spot | 8 | 6 |
| dense-weak | 3 | 3 |
| consensus-demotion | 9 | 9 |

| query id | nDCG@10 change | gold-doc rank: BM25 / dense / hybrid | group | query |
| --- | --- | --- | --- | --- |
| 1363 | -1.000 | 1 / >100 / 14 | dense-blind-spot | Venules have a thinner or absent smooth layer compared to arterioles. |
| 303 | -0.631 | 2 / >100 / 17 | dense-blind-spot | DMRT1 is a sex-determining gene that is epigenetically regulated by the MHM region. |
| 295 | -0.569 | 1 / 8 / 4 | consensus-demotion | Crosstalk between dendritic cells (DCs) and innate lymphoid cells (ILCs) is important in the regulation of intestinal homeostasis. |
| 674 | -0.569 | 1 / 10 / 4 | consensus-demotion | LDL cholesterol has no involvement in the development of cardiovascular disease. |
| 128 | -0.500 | 3 / >100 / 13 | dense-blind-spot | Arterioles have a larger lumen diameter than venules. |
| 1362 | -0.500 | 3 / >100 / 11 | dense-blind-spot | Venules have a larger lumen diameter than arterioles. |
| 3 | -0.500 | 1 / 5 / 3 | consensus-demotion | 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants. |
| 845 | -0.500 | 1 / 5 / 3 | consensus-demotion | Neutrophil extracellular traps (NETs) are released by ANCA-stimulated neutrophils. |
| 324 | -0.431 | 4 / >100 / 16 | dense-blind-spot | Deleting Raptor reduces G-CSF levels. |
| 1024 | -0.369 | 1 / 7 / 2 | consensus-demotion | Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes. |
| 507 | -0.369 | 1 / 5 / 2 | consensus-demotion | Helminths interfere with immune system control of macrophages activated by IL-4 favor Mycobacterium tuberculosis replication. |
| 690 | -0.330 | 2 / 99 / 9 | dense-blind-spot | Less than 10% of the gabonese children with Schimmelpenning-Feuerstein-Mims syndrome (SFM) had a plasma lactate of more than 5mmol/L. |
| 1179 | -0.301 | 9 / >100 / 33 | dense-blind-spot | The PRR MDA5 has a central DExD/H RNA helices domain. |
| 1100 | -0.275 | 2 / 14 / 6 | dense-weak | Statins increase blood cholesterol. |
| 1099 | -0.200 | 2 / 8 / 4 | consensus-demotion | Statins decrease blood cholesterol. |
| 619 | -0.150 | 2 / 44 / 5 | dense-weak | Increased vessel density along with a reduction in fibrosis decreases the efficacy of chemotherapy treatments. |
| 1019 | -0.131 | 2 / 3 / 3 | consensus-demotion | Rapid phosphotransfer rates govern fidelity in two component systems |
| 727 | -0.131 | 2 / 6 / 3 | consensus-demotion | Ly6C hi monocytes have a lower inflammatory capacity compared to their Ly6C lo counterparts. |
| 314 | -0.113 | 3 / 21 / 5 | dense-weak | Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophic G-to-A mutations in the viral genome. |
| 452 | -0.088 | 3 / 90 / 6 | dense-blind-spot | Gene expression does not vary appreciably across genetically identical cells. |
