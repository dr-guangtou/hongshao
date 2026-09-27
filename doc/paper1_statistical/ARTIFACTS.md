# Archived artifacts inspected for paper planning

Source root: `/Users/shuang/Dropbox/work/project/massive/hongshao`.

Private destination: `data/processed/paper1_audit_snapshot_20260925/` in
the planning worktree. Each relative path below is preserved underneath it.
Copied September 26, 2026. Source SHA-256 was unchanged across each copy;
`cmp` verified identical bytes. No source artifact or running process was
modified. Artifacts remain gitignored; this inventory is the durable record.

The six score/summary files and seven manifests were read. All nine PNGs
were visually inspected. These are existing figures, not newly generated QA.
The inspection establishes what the archives say, not that the fits have
been independently reproduced.

## Figure review

- Exp02 modes: compact cumulative shape variation is clear; add reconstructed
  profile examples and held-out reconstruction before using in the paper.
- Exp06 connection: useful assembly-to-shape association; only halo mass is
  controlled, so it cannot support the strict stellar-mass-controlled claim.
- Exp08 skill/calibration: history gain and shuffled reference are clear;
  replace the physical-parameter comparator with matched information controls.
- Exp08 painting: intuitive individual examples at similar halo mass, but
  three selected objects are an illustration, not the population test.
- Exp22 full profile: inner improvement is visible, but its population-mean
  shape reference is weaker than the required mass-dependent reference.
- Exp22 generative: the histogram illustrates direct-annulus sampling;
  it does not certify the full CoG generator.
- Exp62 integrated information: modest, epoch-dependent shape gain; the
  strict-control line and shuffled band refer to different control sets.
- Exp62 robustness comparison: radial support and sample change together;
  separate them in the paper.
- Exp62 average normalized CoGs: mean profiles agree closely; the tiny
  mean residuals must not be confused with accurate individual shapes.

## Checksums

```text
94d39556fe6f5c4593b2e44d3fa823e833d1d7523124c697139d061be3bb90b4  experiments/exp01_aperture_mah_corr/outputs/scatter_reduction.csv
ddf15f83b368af637b28fdbf14111bf9add6b40e62329b7b338b87e241108616  experiments/exp06_mah_pca/outputs/representation_comparison.csv
75190007c88cbfbed7f71aa2feeeb359ed40226921f452e0a8c859c6dd1c1a47  experiments/exp08_emulator/outputs/emulator_scores.csv
102c3ad075cc8484f3f2b5769d7a31fe299a4a85c6bb8b1c5358c5a38067e9e4  experiments/exp16_secondary_c200c/outputs/c200c_scores.csv
57a16093d35d67440f927e712ed41580ea4a94e41c0ff26e15a6d717306a6d1b  experiments/exp22_full_profile_predict/outputs/profile_scores.csv
aa0b81fa323c443baa518e301a42559b875c8d1cef2a382ed2de12b949d73967  experiments/exp62_cog_fit_atlas/outputs/stage7/full/summary.json
12d17ee35d17cd0a993163c878a13513ada093ae3bcc344db134b68cc250e9d4  experiments/exp62_cog_fit_atlas/outputs/stage7/full/manifest.json
c51d8328c4b9c6da7589718d9596720007763168da1d73d64999339d81f84d5b  experiments/exp02_profile_pca/figures/exp02_modes.png
19d894872900d5193ed3a89e00b675618a18fb481b8def5139a5950cff70cf0e  experiments/exp06_mah_pca/figures/exp06_connection.png
c8cc2969d83360c6ee3cd45cde7db52a352ff9e01f46bf4e957eeb41eb10a224  experiments/exp08_emulator/figures/exp08_skill_calibration.png
289cfe512ea83efd63a132d9561962e7df2db13a0240f53744a1a5e3b069f521  experiments/exp08_emulator/figures/exp08_painting.png
dd269e61d46195515b4d51dd33e161f0c7bd51308d0798eaa693dce9c77cd331  experiments/exp22_full_profile_predict/figures/exp22_full_profile.png
7ee60eb1a61d8bfd553461127f4f95f98adb7bd40ec234ab01432e3451f84bee  experiments/exp22_full_profile_predict/figures/exp22_generative.png
a5dd677ad8d5d52543a6011aa2f826595b85903b697302040523e43147e9fa9e  experiments/exp62_cog_fit_atlas/figures/stage7/full/exp62_stage7_integrated_information.png
26d05194126de32f84ffec9dafbf9c09bdaa2048dc7dd10d5757e77e5ec96a04  experiments/exp62_cog_fit_atlas/figures/stage7/full/exp62_stage7_robustness_comparison.png
057f60770fada44a467e452cf3202e9e2da294647a239600b5a1352fae9a3eca  experiments/exp62_cog_fit_atlas/figures/stage7/full/exp62_stage7_average_cog_halo_mass_bins.png
4255f3fcdd1be9ff81b66e2fd124eb41a569ee239a31b8ad31f5812363c49561  experiments/exp01_aperture_mah_corr/outputs/manifest.json
52927082faada798e25e3f869713adc292deef2e00dfd104c116e791b48bc91f  experiments/exp02_profile_pca/outputs/manifest.json
5059c2886ae6c28c32dbe6bde1f4b13157e2180c29db96b49aee27d645817a92  experiments/exp06_mah_pca/outputs/manifest.json
43772b2c92bf828ff571fda41d77fdd2b7b1f235b3a1b66f3b4bb84dc5e05a67  experiments/exp08_emulator/outputs/manifest.json
d2601c3f42e981fde9ec2304eb4220aebd6515f2c2a7b26c53b4394ea98da219  experiments/exp16_secondary_c200c/outputs/manifest.json
9cf896d3864ec64cb6100b8962f018552717ed2086ac23698956130e0e30aa8d  experiments/exp22_full_profile_predict/outputs/manifest.json
```
