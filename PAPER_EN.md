# Parallel Gated Fusion of LSTM and Transformer for Multi-Step Electricity Demand Forecasting

**Authors:** Zhaoyi Sun¹,², Ping Liu²

**Affiliations:**
- ¹ EFREI Paris, Panthéon-Assas University, 94800 Villejuif, France
- ² School of Information Engineering, Wuhan University of Technology, 430070 Wuhan, China

## Abstract

Multi-step electricity demand forecasting exhibits regime- and metric-dependent behavior: models that perform well on aggregate error do not necessarily preserve peak loads, and more complex neural architectures may offer limited value on smoother periodic series. This study investigates whether feature-level parallel gated fusion provides a useful peak-sensitive learning-based design for volatile load forecasting and how its relative value changes across different load regimes. We propose PGF-Net, which combines parallel LSTM and Transformer encoders through an element-wise learnable gate, and evaluate it under a leakage-free direct forecasting protocol with 96-hour inputs, 24-hour outputs, chronological splitting, and five-seed reporting. On the UCI household benchmark, PGF-Net achieves the lowest Peak Load Error among the learning-based baselines (PLE 34.56±0.82%), whereas DLinear provides lower aggregate MSE and MAE and S-Naive remains the strongest overall peak reference. On the weekly benchmark, S-Naive remains the strongest overall reference on the smoother forecast-trajectory series. After a fairness-stabilized rerun that prevents pathological early stopping, DLinear becomes the strongest model on the same-protocol `nat_demand` benchmark across MSE, MAE, sMAPE, PLE, and Ramp_MAE, while PGF-Net remains the strongest nonlinear model and stays close on aggregate accuracy. These findings support a conditional, metric-dependent interpretation of parallel gated fusion rather than a claim of universal superiority.

## 1. Introduction

Electricity demand forecasting is a central component of modern power-system management. In day-ahead scheduling, however, the practical task is not merely to fit an average trend: the forecasting model must simultaneously track short-term local fluctuations, long-range temporal dependency, and peak-sensitive behavior under chronologically evolving load conditions. This challenge becomes more pronounced as renewable penetration rises and end-user demand becomes more heterogeneous, because forecasting errors directly affect reserve allocation, spinning-reserve planning, balancing emissions, and demand-side operational efficiency \cite{ferdosian2024reserve,aziz2024peak}.

This study focuses on direct 24-hour multi-step forecasting from the previous 96 hours. Within this scope, the paper addresses two explicitly bounded research questions. The first is whether feature-level parallel gated fusion helps a learning-based model preserve both local variation and long-context information while improving peak-sensitive behavior on volatile household-style load series. The second is whether the same design should still be preferred once the target series becomes smoother and more strongly periodic. The practical boundary is limited to univariate public benchmarks under leakage-free chronological splitting, and the academic gap is therefore not the absence of forecasting models in general, but the lack of clear evidence on these two scoped questions.

Existing approaches leave this question insufficiently resolved. Classical statistical methods remain interpretable but often struggle with nonlinear multi-scale dynamics. LSTM-based models provide a strong inductive bias for local sequential evolution, yet long-history optimization remains difficult in practice. Transformer-based models strengthen long-range dependency modeling, but they can smooth rapid local transitions and may increase computational burden. Existing hybrid designs are also commonly serial or fixed-weight, which means later modules receive already transformed features and cannot adaptively decide, at the feature level, how much local or global information should be retained \cite{hochreiter1997long,vaswani2017attention,nie2023time,zeng2023transformers}.

To address these questions, we propose PGF-Net, a parallel gated fusion architecture in which LSTM and Transformer encoders process the same input sequence simultaneously and are combined by an element-wise learnable gate. This design defines the technical path of the paper: direct 24-step prediction, training-only normalization, chronological splitting, unified multi-seed evaluation, and joint reporting of aggregate metrics and operationally relevant peak-sensitive metrics. The paper therefore tests, in a controlled way, whether gated fusion improves peak-aware learning-based forecasting on the volatile benchmark and whether that advantage survives comparison against strong seasonal and linear baselines on smoother regimes.

The study is organized around two bounded research questions about volatile versus smoother load regimes. We introduce PGF-Net as a parallel feature-level fusion method rather than a serial or fixed-weight hybrid, and we compare it against LSTM, PatchTST, DLinear, and S-Naive under a unified chronological protocol to examine both the promise and the boundary of gated fusion.

The main contributions are:
- We design PGF-Net, which fuses parallel LSTM and Transformer representations with an element-wise learnable gate instead of fixed or serial mixing.
- We provide a reproducible CPU-based evaluation protocol with chronological splitting, training-only normalization, direct 24-step forecasting, and multi-seed reporting.
- We report both aggregate-error and operationally relevant metrics (PLE and Ramp_MAE), and summarize both positive and negative findings across heterogeneous load regimes.

## 2. Related Work

Recurrent methods remain foundational in time-series forecasting. LSTM \cite{hochreiter1997long} introduces gated memory dynamics that improve temporal modeling under nonlinear variation, and subsequent hybrid recurrent designs such as LSTNet \cite{lai2018modeling} combine convolutional and recurrent components to capture multi-scale behavior. Broader evaluations of recurrent families and load-focused reviews also report that local temporal continuity is captured effectively, while long-context representation remains a persistent challenge \cite{chung2014empirical,hippert2001neural}.

Attention-based models substantially expand long-horizon capacity. Transformer \cite{vaswani2017attention} establishes self-attention as a general sequence modeling mechanism, while Informer \cite{zhou2021informer} and Autoformer \cite{wu2021autoformer} improve computational efficiency and decomposition ability for long-range forecasting. PatchTST \cite{nie2023time} and iTransformer \cite{liu2023itransformer} provide strong recent evidence that tokenization and channel-wise inversion can improve long-horizon forecasting robustness. Interpretable multi-horizon designs such as TFT further strengthen operational analysis \cite{lim2021temporal}.

Linear decomposition models have recently shown surprising competitiveness. DLinear \cite{zeng2023transformers} demonstrates that trend-seasonal decomposition with lightweight linear mapping can provide robust performance in practical settings. Complementary evidence from N-BEATS and large-scale forecasting surveys also shows that architectural simplicity, basis expansion, and careful benchmarking can remain highly competitive under distributional variation \cite{oreshkin2019nbeats,benidis2021deeplearning}.

Existing hybrid integration strategies are often serial or fixed-weight. In contrast, our work emphasizes adaptive feature-level fusion between recurrent and attention-based representations. The proposed gate does not treat branches as static complements; it provides a feature-wise mixing mechanism whose empirical value is tested here against two explicit benchmarks: whether it improves peak-aware learning-based forecasting on a volatile household series, and whether it remains preferable once the series becomes smoother and more periodic. Recent domain studies in electricity generation economics and integrated green energy scheduling further motivate adaptive fusion under regime shifts \cite{an2022economic,bian2025lstm}.

## 3. Methodology

### 3.1 Data Preprocessing and Leakage Prevention

We follow a strict chronological split protocol to avoid temporal leakage. Each dataset is partitioned into training (70%), validation (15%), and test (15%) subsets in time order. MinMaxScaler is fitted only on the training subset, and validation and test subsets are transformed using training statistics. This protocol ensures that no future information is introduced during normalization and follows established guidance on stable neural-network preprocessing \cite{sola1997importance}. In the current univariate setup, hourly `Global_active_power` is used as both input variable and prediction target, without exogenous covariates.

### 3.2 Parallel Gated Fusion Architecture

PGF-Net consists of parallel encoding, adaptive gated fusion, and a direct multi-step prediction head.

Given input $X \in \mathbb{R}^{B \times L \times D}$, we first inject temporal priors by encoding hour-of-day and day-of-week with `nn.Embedding` ($d_{model}=32$). These embeddings are added to the projected load sequence. The recurrent branch captures local temporal evolution:

$$
H_{lstm} = \text{LSTM}(X), \quad h_{lstm} = H_{lstm}[:, -1, :].
$$

In parallel, the attention branch models long-range interactions through a Transformer encoder with sinusoidal positional encoding:

$$
\begin{aligned}
H_{trans} &= \text{TransformerEncoder}(\text{Linear}(X) + \text{PosEnc} + \text{TimeEmbed}), \\
h_{trans} &= H_{trans}[:, -1, :].
\end{aligned}
$$

In implementation, the positional encoding uses the standard fixed Transformer frequencies with maximum length 5000 and no separate positional-dropout layer. Because all experiments use a fixed input length of 96, the model never extrapolates beyond the predefined positional range. The hour-of-day and day-of-week embeddings are two separate learnable tables of dimension 32, and their outputs are summed with the projected load feature rather than concatenated. All learnable modules use the default PyTorch parameter initialization; we do not introduce manual Xavier/He overrides, so the reported behavior corresponds to the current implementation included in the revision package.

To combine both representations, we compute an element-wise gate:

$$
z = \sigma(W_z \cdot [h_{lstm}; h_{trans}] + b_z),
$$

where $z \in (0,1)^d$ and $\sigma$ is the sigmoid function. The fused representation is

$$
h_{fused} = z \odot h_{lstm} + (1 - z) \odot h_{trans},
$$

with $\odot$ denoting element-wise multiplication. This formulation allows each feature channel to adaptively favor local or global information. Unlike scalar mixing, element-wise gating provides finer control over branch mixing when temporal patterns change across samples, although the present study does not claim that this mechanism is uniformly superior across all datasets \cite{dauphin2016language}.

The prediction head maps $h_{fused}$ to the 24-step horizon:

$$
\hat{Y} = \text{Linear}(h_{fused}),
$$

with output shape $\mathbb{R}^{B \times 24}$.

### 3.3 Training Strategy

We optimize the models using mean squared error (MSE) with Adam-family optimizers \cite{kingma2014adam}. The default learning-based setting uses learning rate 0.001 with validation-loss early stopping (patience 8) to reduce overfitting risk \cite{prechelt1998early}. After the `nat_demand` DLinear audit revealed pathological fast stopping, that specific benchmark/model pair was rerun with a fairness-stabilized configuration (AdamW, learning rate 3e-4, minimum 15 epochs, patience 12, and ReduceLROnPlateau), while the remaining models keep the shared default setting. We therefore report the final `nat_demand` DLinear numbers from the stabilized rerun rather than from the earlier prematurely stopped runs.

## 4. Experiments

### 4.1 Experimental Setup

All experiments run in a CPU-only environment on Intel(R) Core(TM) Ultra 9 275HX with 31.4 GiB RAM, using Python 3.9+ and PyTorch 1.13+. We evaluate on three datasets.

The first is UCI Individual Household Electric Power Consumption (2006-2010, 2,075,259 records) \cite{hebrail2012individual}. Data are resampled hourly with forward-fill, and `Global_active_power` is used as target with training-only MinMax normalization. The chronological split is:
- Train: 2006-12-16 17:00:00 to 2009-09-20 12:00:00
- Validation: 2009-09-20 13:00:00 to 2010-04-24 16:00:00
- Test: 2010-04-24 17:00:00 to 2010-11-26 21:00:00

The second dataset is Weekly Pre-dispatch Load Forecast (2016-2020, 40,152 hourly records). The released file contains only `load_forecast`; no paired realized-demand column is provided. We therefore treat this benchmark as an auxiliary pre-dispatch forecast-trajectory task rather than a pure metered-demand forecasting benchmark. We use hourly mean aggregation with forward-fill, keep `load_forecast` as target, and apply training-only MinMax normalization. The split is:
- Train: 2016-01-02 00:00:00 to 2019-03-18 01:00:00
- Validation: 2019-03-18 02:00:00 to 2019-11-23 23:00:00
- Test: 2019-11-24 00:00:00 to 2020-07-31 23:00:00

This setup complements prior spatially resolved and process-industry load modeling cases and supports cross-regime comparison \cite{robinius2017topdown,hu2019shortterm}.

The third dataset is the continuous electricity load file (`continuous dataset.csv`) used in the current workspace for revision experiments, where we extract hourly `nat_demand` as target and keep chronological splitting with:
- Train: 2015-01-03 01:00:00 to 2018-11-04 09:00:00
- Validation: 2018-11-04 10:00:00 to 2019-08-31 16:00:00
- Test: 2019-08-31 17:00:00 to 2020-06-27 00:00:00

The forecasting task is direct multi-step prediction from past 96 hours to future 24 hours. Baselines include LSTM, DLinear, PatchTST, and S-Naive, where S-Naive repeats the most recent 24 hours for the next 24 hours. For PatchTST, we use a channel-independent internal reimplementation with patch length 16, stride 8, `d_model=128`, 8 attention heads, 3 encoder layers, and dropout 0.1. We report this baseline to provide a modern patch-based Transformer reference, but we interpret its results cautiously because the present implementation is a simplified in-house reproduction rather than a full official-repository replication, and the benchmark setting is univariate. All learning-based baselines share the same input/output horizon, chronological split, optimizer family, early-stopping policy, and random-seed protocol for fairness. The UCI, weekly, and `nat_demand` settings are all reported with same-protocol five-seed coverage, and UCI is additionally examined with a 30-origin expanding-window evaluation. This configuration is sufficient for the scoped cross-regime question of the paper, while broader claims remain bounded by the current univariate setup and dataset coverage.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/dataset_statistics_three_datasets.png`
> **Caption**: Figure 1: Regenerated dataset statistics for the three benchmarks (hourly train/validation/test split sizes).

Table 1: Cross-dataset characteristics computed on the hourly target series after resampling. Outlier rate is computed with the 1.5×IQR rule on each resampled target series.

| Dataset | Target type | CV | ACF(24) | ACF(168) | Missing (%) | Outlier (%) |
| :-- | :-- | --: | --: | --: | --: | --: |
| UCI Household | actual demand | 0.822 | 0.437 | 0.456 | 1.25 | 2.14 |
| Weekly Pre-dispatch | forecast series | 0.162 | 0.784 | 0.872 | 0.00 | 0.18 |
| `nat_demand` | actual demand | 0.162 | 0.809 | 0.893 | 0.00 | 0.01 |

These values quantify why model rankings differ across datasets. UCI has much higher relative variability and substantially weaker daily/weekly autocorrelation than the other two series, whereas the weekly pre-dispatch and `nat_demand` targets are smoother and more periodic. These descriptive differences are consistent with the observed dataset-dependent rankings, while DLinear and S-Naive remain stronger on the smoother operational sequences.

### 4.2 Prediction Setting

To ensure a fair comparison, all learning-based models follow the same direct forecasting protocol. For each sample, the model receives an input window of length $T=96$ and predicts the complete horizon $H=24$ in one forward pass. This setup differs from recursive forecasting, where one-step outputs are repeatedly fed back as inputs.

The direct strategy aligns training and inference objectives and avoids cumulative error propagation across rollout steps. In our notation, the input is $[x_{t-95}, \ldots, x_t]$, and the output is $[\hat{y}_{t+1}, \ldots, \hat{y}_{t+24}]$.

### 4.3 Main Results

We report mean±std on hold-out test sets over the specified seeds. Standard metrics are MSE, MAE, and sMAPE \cite{hyndman2006another,hyndman2009forecast}. To assess operational relevance, we also report Peak Load Error (PLE) and Ramp_MAE. For a 24-step target window `y_i={y_{i,1},...,y_{i,H}}` and prediction `ŷ_i={ŷ_{i,1},...,ŷ_{i,H}}`, we compute `PLE = (1/N) Σ_i |max_h ŷ_{i,h} - max_h y_{i,h}| / (max_h y_{i,h} + ε) × 100`, and `Ramp_MAE = (1 / (N(H-1))) Σ_i Σ_{h=2}^H |(ŷ_{i,h}-ŷ_{i,h-1}) - (y_{i,h}-y_{i,h-1})|`. Both metrics are computed sample-wise on each forecast window and then averaged across the evaluation set.

Table 2: Results on the Household Power Consumption dataset (CPU; mean±std over 5 seeds).

| Model | Params | MSE | MAE | sMAPE | PLE | Ramp |
| :---- | -----: | --: | --: | ----: | --: | ---: |
| DLinear | 4,656 | 0.3673±0.0032 | 0.4511±0.0049 | 49.72±0.56 | 35.52±0.71 | 0.3924±0.0019 |
| PGF-Net | 20,920 | 0.3878±0.0253 | 0.4694±0.0292 | 54.20±4.37 | 34.56±0.82 | 0.4172±0.0219 |
| PatchTST | 632,216 | 0.4429±0.0489 | 0.5110±0.0401 | 57.08±4.98 | 36.05±2.00 | 0.4726±0.0704 |
| LSTM | 13,720 | 0.4402±0.0502 | 0.5201±0.0470 | 56.81±5.19 | 49.38±8.56 | 0.3782±0.0038 |
| S-Naive | 0 | 0.5516±0.0000 | 0.4906±0.0000 | 47.58±0.00 | 27.10±0.00 | 0.4901±0.0000 |

On the household benchmark, the updated five-seed results show a more conservative picture than the earlier three-seed snapshot. DLinear now achieves the strongest aggregate MSE and MAE, while PGF-Net retains the lowest PLE among the learning-based baselines. This pattern indicates that parallel fusion can still be useful for peak-sensitive behavior on the most variable dataset in our study, but it does not justify a blanket superiority claim over all baselines or all metrics. Figure 2 now shows four representative cases only, using PGF-Net, DLinear, and S-Naive to avoid visual overload. The regenerated layout removes the previous blank axes and overlapping labels while preserving the main qualitative comparison: PGF-Net often follows both diurnal structure and rapid local changes more closely, whereas DLinear and S-Naive remain stronger in some smoother periodic cases.

At the same time, the stronger PLE of S-Naive indicates that this residential series retains pronounced daily periodicity. We therefore interpret PGF-Net not as a replacement for pure seasonal extrapolation in every peak scenario, but as a model that can better capture deviations from regular daily structure than the other learning-based baselines while remaining conditional on the evaluation metric.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/forecast_comparison_grid.png`
> **Caption**: Figure 2: Four representative 24-hour forecasting cases on the UCI household benchmark using PGF-Net, DLinear, and S-Naive.

Figure 3 regenerates the UCI peak scatter by overlaying representative seed-0 PGF-Net and DLinear checkpoints together with the deterministic S-Naive baseline. The updated panel makes the stronger periodic peak alignment of S-Naive explicit, while PGF-Net still lies closer than DLinear on several high-peak cases.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/peak_scatter.png`
> **Caption**: Figure 3: Regenerated predicted versus observed peak loads on the UCI household benchmark.

Average CPU training time per seed in the updated five-seed UCI runs is 1.61 s for DLinear, 110.14 s for PGF-Net, 146.15 s for LSTM, and 59.13 s for PatchTST. Although PGF-Net adds a second encoder branch, it remains practical for offline day-ahead retraining, but its efficiency advantage over the simplest baselines is limited.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/inference_latency.png`
> **Caption**: Figure 4: Regenerated inference latency comparison across forecasting models (UCI benchmark setup).

To improve interpretability beyond point estimates, the updated UCI table is reported over five seeds with corresponding standard deviations, and all comparative statements in this revision are treated as descriptive rather than confirmatory. The current evidence therefore supports a metric-dependent interpretation: PGF-Net remains competitive and peak-aware on UCI, DLinear is stronger on aggregate error, and S-Naive remains a particularly strong periodic reference.

We also extend UCI evaluation to a 30-origin expanding-window rolling-origin protocol with weekly-spaced forecast origins. Each origin retrains the learning-based models from scratch on the corresponding expanding window before evaluating the next 24 hours. Across these origins, DLinear yields lower mean MSE (0.5928) and MAE (0.5742) than PGF-Net (0.7183, 0.6044), whereas PGF-Net achieves a lower mean PLE (43.41%) than DLinear (52.48%); S-Naive attains the lowest mean sMAPE (52.26%). The manuscript now also includes the current horizon-wise mean and interquartile-range profiles for both MAE and MSE across h=1,...,24, rather than only describing them in the generated outputs. The current local package additionally contains per-origin distributions, raw prediction tables, and loss-differential files. For the horizon-wise DM analysis, we use squared-error loss differentials across the 30 origin-level samples, with lag 0, sample-variance standardization, and Benjamini-Hochberg correction within each baseline family. After correction, only horizon 16 remains below q<0.05 versus DLinear, while no horizon remains significant versus S-Naive. We therefore interpret the rolling-origin evidence as stronger descriptive time-domain validation of the same metric-dependent pattern rather than as proof of uniform superiority.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/walkforward_horizon_profiles_uci.png`
> **Caption**: Horizon-wise MAE and MSE profiles over the current 30 UCI rolling-origin forecasts. Solid lines show horizon means and shaded bands show interquartile ranges across origins.

### 4.4 Model Analysis

To inspect internal behavior, we visualize gate activations in Figure 5. Values near 1 indicate higher reliance on local recurrent information, and values near 0 indicate stronger reliance on global attention features. Importantly, the horizontal axis shows 48 consecutive forecast origins generated by consecutive 96-hour input windows; it does not denote the 24-step output horizon. This view therefore reflects how the feature-wise gate vector changes from one forecast origin to the next rather than how weights vary across future steps. We now complement the heatmap with quantitative statistics aggregated over all five UCI seeds: the gate mean is 0.309, the standard deviation is 0.100, and the interquartile range is [0.236, 0.370]. Moreover, 70.43% of activations lie in [0.2, 0.4], 15.00% lie in [0.4, 0.6], 1.63% lie in [0.6, 0.8], and none exceed 0.8. Correlations with simple load-level and absolute-ramp signals are weak (r=0.013 and r=0.008), while correlations with volatility-related signals are slightly larger but still modest (history volatility r=0.100). We therefore interpret the gate as a feature-wise adaptive bias toward the Transformer branch rather than a strong regime-switching controller driven by a single scalar load indicator.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/gate_analysis_heatmap.png`
> **Caption**: Figure 5: Gate activations across 48 consecutive forecast origins on the UCI benchmark. Each column corresponds to the feature-wise gate vector generated from one 96-hour input window; the horizontal axis does not represent the 24-step output horizon.

We also analyze errors across load regimes by stratifying MAE and PLE into low-, medium-, high-, and peak-load conditions. In the regenerated Figure 6, the bins are defined by quartiles of the true 24-step future peak on the UCI test windows: Low <= 1.8318, Medium (1.8318, 2.3059], High (2.3059, 2.8006], and Peak > 2.8006 kW, with 1272, 1286, 1255, and 1257 samples, respectively. Within each bin, PLE is computed sample-wise from the predicted and observed window maxima and then averaged. The updated figure therefore removes the previous empty low bin and makes the binning rule explicit. PGF-Net remains competitive in the higher-load bins, but the results should still be interpreted as conditional descriptive evidence rather than proof of universally better error distributions.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/error_by_load_condition.png`
> **Caption**: Figure 6: Grouped MAE and PLE summary across quartile-based low-, medium-, high-, and peak-load bins on the UCI benchmark.

Table 3: Results on the Weekly Pre-dispatch dataset (same protocol, mean±std over 5 seeds).

| Model | Params | MSE | MAE | sMAPE | PLE | Ramp |
| :---- | -----: | --: | --: | ----: | --: | ---: |
| DLinear | 4,656 | 70181.66±10257.76 | 217.33±19.45 | 17.69±1.83 | 20.82±0.70 | 237.90±65.64 |
| PGF-Net | 20,920 | 12745.29±1578.84 | 87.39±5.45 | 7.26±0.39 | 8.89±0.85 | 32.30±2.76 |
| PatchTST | 632,216 | 21068.42±8462.68 | 116.02±23.26 | 9.62±1.93 | 9.67±1.25 | 42.25±12.67 |
| LSTM | 13,720 | 20697.09±7304.02 | 109.58±21.28 | 9.04±1.79 | 9.92±0.95 | 35.54±6.16 |
| S-Naive (n=5) | 0 | 11316.76±0.00 | 65.22±0.00 | 5.39±0.00 | 5.46±0.00 | 15.38±0.00 |

On the weekly pre-dispatch benchmark, the completed same-protocol five-seed results show S-Naive as the strongest reference, while PGF-Net is the best among the learning-based models on MSE and MAE. LSTM and PatchTST form a middle tier, whereas DLinear shows substantially higher variability under the current implementation and training configuration and performs worst on this smoother operational forecast series. The released DLinear audit indicates that this weekly underperformance is consistent across all five seeds rather than being driven by a single failed run. These results again indicate that model rankings remain strongly dataset-dependent and that the most suitable inductive bias can differ sharply from the UCI household benchmark \cite{hong2016probabilistic}.

Table 4: Nat demand results under a common data/evaluation protocol with validation-selected model-specific optimization settings.

| Model | MSE | MAE | sMAPE | PLE | Ramp |
| :---- | --: | --: | ----: | --: | ---: |
| DLinear | 8220.13±118.37 | 65.59±1.02 | 5.41±0.09 | 5.89±0.05 | 21.18±0.91 |
| PGF-Net | 8381.85±395.70 | 68.48±2.09 | 5.77±0.17 | 6.42±0.21 | 28.76±2.63 |
| PatchTST | 10278.50±854.97 | 75.34±3.66 | 6.19±0.28 | 6.75±0.50 | 31.88±3.82 |
| LSTM | 13716.32±154.87 | 88.52±1.26 | 7.35±0.09 | 8.11±0.26 | 33.93±1.24 |
| S-Naive | 12190.45±0.00 | 73.14±0.00 | 6.01±0.00 | 6.00±0.00 | 22.17±0.00 |

### 4.5 Supplementary Benchmark on an Additional Public Dataset

Table 4 presents the performance comparison on the nat_demand benchmark. All models use the same data split, input/output configuration, seed set, and evaluation protocol, while model-specific optimization settings are selected using validation data. These results incorporate the fairness-stabilized DLinear training configuration, ensuring a reliable and robust comparison against the neural baselines.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/overall_comparison_three_datasets.png`
> **Caption**: Figure 7: Cross-dataset metric comparison (MSE, MAE, PLE) across UCI, Weekly Pre-dispatch, and Kaggle Nat Demand benchmarks under the same protocol, using dataset-specific subplot scales for readability.

Figure 7 summarizes cross-dataset MSE, MAE, and PLE using the completed same-protocol result tables for all three datasets. To avoid the previous readability problem caused by mismatched dataset scales, the regenerated figure now uses dataset-specific subplots rather than a shared linear axis. The updated `nat_demand` panel now shows DLinear as the strongest model on all three displayed metrics after the fairness-stabilized rerun, while PGF-Net remains the strongest nonlinear alternative. Figure 8 reports raw per-seed MSE points from the corrected `nat_demand` table, making the post-rerun stability of DLinear explicit rather than reinforcing the earlier misleading failure-driven spread.

> **[Insert Figure Here]**
> **File**: `paper-template/figures/robustness_boxplot_kaggle.png`
> **Caption**: Figure 8: Raw per-seed MSE comparison on the same-protocol `nat_demand` benchmark across all five models.

### 4.6 Ablation Study of the Gated Fusion Design

To address the ablation request, we update the ablation on the main UCI benchmark and expand the comparison to controlled fusion alternatives, including concatenation, fixed 0.5 averaging, scalar fusion, branch-removal variants, and temporal-encoding removals. To keep the protocol consistent across tables, the full-model PGF-Net row in Table 5 reuses the same five-seed UCI result reported in Table 2. The remaining rows are evaluated under the ablation protocol. The table shows that the full PGF-Net remains competitive and achieves the lowest MSE among the tested fusion variants, but the margins over `w/o Positional Encoding` and `w/o Transformer` are modest. We therefore interpret the ablation evidence as supportive of branch composition and protocol-sensitive fusion design, but not as proof that the gate alone is universally beneficial across all datasets and metrics.

| Variant | MSE | MAE | PLE | Ramp |
| :------ | --: | --: | --: | ---: |
| PGF-Net | 0.3878±0.0253 | 0.4694±0.0292 | 34.56±0.82 | 0.4172±0.0219 |
| Concatenation | 0.4229±0.0109 | 0.5000±0.0094 | 36.23±1.93 | 0.4297±0.0040 |
| Fixed 0.5 | 0.4289±0.0092 | 0.5086±0.0082 | 36.46±2.11 | 0.4409±0.0062 |
| Scalar fusion | 0.4264±0.0118 | 0.5042±0.0132 | 37.11±1.25 | 0.4405±0.0065 |
| w/o Transformer | 0.4158±0.0097 | 0.4951±0.0088 | 35.86±1.38 | 0.4281±0.0024 |
| w/o LSTM | 0.5027±0.0158 | 0.5845±0.0178 | 43.33±0.95 | 0.4484±0.0068 |
| w/o Time Embedding | 0.4750±0.0135 | 0.5564±0.0120 | 42.50±0.67 | 0.4160±0.0057 |
| w/o Positional Encoding | 0.4084±0.0090 | 0.4913±0.0069 | 34.90±1.60 | 0.4314±0.0053 |

Given that the updated ablation now uses five seeds on UCI but still covers only one main dataset and one horizon configuration, the observed gaps are treated as stronger descriptive evidence rather than final confirmatory proof. The current results are sufficient to show that branch composition and fusion choice matter, but not sufficient to claim that the gate itself is consistently beneficial across datasets.

## 5. Practical Implications

The empirical findings suggest that PGF-Net is most useful in demand regimes where local spikes and long-context trends coexist, such as household-level and highly behavior-driven load profiles. For smoother operational series, strong linear or seasonal baselines remain preferable because they offer lower computational cost and comparable or better aggregate error. From a deployment perspective, the main implication is regime-aware model selection based on verified series characteristics rather than architecture preference alone. A separate automatic regime classifier is outside the scope of the present study and is not implemented here.

## 6. Limitations

This study has several limitations. First, although we expanded from two to three datasets, the current setup still uses limited exogenous information and does not exploit weather or market covariates, which constrains broader real-world generalization. Second, the weekly pre-dispatch benchmark is an operational forecast series rather than metered realized demand, so its interpretation is narrower than that of the other two datasets. Third, although the `nat_demand` benchmark is now covered by same-protocol five-seed results for all five models, the paper still studies only three univariate datasets and therefore cannot claim broad cross-domain generalization. Fourth, the new UCI rolling-origin analysis spans 30 weekly-spaced forecast origins, yet it is still restricted to one dataset and one fixed input/output configuration. Fifth, the gate analysis is now quantitative, but the released correlations indicate only modest links to simple scalar signals; stronger mechanism claims would still require richer covariates and dedicated intervention tests. Broader deployment claims therefore require richer benchmark diversity, stricter protocol alignment, stronger uncertainty-aware reporting, and more complete mechanism analysis \cite{benidis2021deeplearning}.

## 7. Conclusion

This study investigated the regime- and metric-dependent value of parallel LSTM-Transformer gated fusion for direct 24-step electricity demand forecasting. PGF-Net combines recurrent and attention-based representations through an element-wise feature gate and was evaluated against learning-based and seasonal baselines using both aggregate and peak-oriented metrics.

The results reveal a clear accuracy-peak-regime trade-off. On the volatile UCI household benchmark, PGF-Net achieves the lowest Peak Load Error among the learning-based baselines, indicating value as a peak-sensitive neural forecasting option. However, DLinear provides lower aggregate MSE and MAE, while S-Naive remains the strongest overall peak reference. On the smoother weekly pre-dispatch series, S-Naive performs best overall and PGF-Net is the strongest learning-based model on aggregate error. On the fairness-corrected same-protocol `nat_demand` benchmark, DLinear becomes the strongest model across aggregate and peak-sensitive metrics, while PGF-Net remains the strongest nonlinear alternative rather than the overall winner. The 30-origin UCI rolling-origin evaluation reinforces the same broader conclusion: DLinear leads mean MSE and MAE across origins, whereas PGF-Net leads mean PLE.

The central implication is therefore not that one architecture is universally optimal. Model selection should depend on the temporal regime and operational objective: lightweight seasonal or linear methods remain preferable for strongly periodic series and some aggregate-error objectives, whereas PGF-Net can still be considered when a peak-sensitive nonlinear alternative is desired on more volatile series such as UCI. Even after completing the fairness-corrected `nat_demand` rerun and the 30-origin UCI rolling-origin evaluation, broader generalization and deployment claims still require richer benchmark diversity and quantitative gate analysis.

## Reproducibility Statement

We report dataset ranges, split timestamps, model settings, optimization hyperparameters, and hardware environment to support transparent reproduction. All preprocessing is performed with training-only statistics to prevent leakage. Multi-seed results are reported with mean±std. The current local package used for this manuscript version contains the hold-out multi-seed tables, 30-origin UCI rolling-origin outputs, horizon-wise DM test files with correction metadata, gate-analysis CSV summaries, load-bin metadata, and the fairness-corrected DLinear audit and rerun outputs derived from the canonical per-seed tables.

## Code Availability

The source code, scripts, environment specification, and revision outputs are publicly available at [repository URL](https://github.com/aszybaiye/Parallel-Gated-Fusion-of-LSTM-and-Transformer-for-Multi-Step-Electricity-Demand-Forecasting). The current package materials include per-seed CSV files, raw rolling-origin predictions, DM-test details, gate-analysis summaries, and DLinear rerun/audit tables together with a package manifest for verification.

## Funding

No external funding was received for this study.

## Availability of Data and Materials

The UCI Individual household electric power consumption dataset is publicly available from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

The weekly pre-dispatch file and the `nat_demand` file used in the current workspace are described in the accompanying project documentation and are included in the current local package together with the code, plotting scripts, per-seed outputs, rolling-origin raw files, audit tables, and package manifest used by this manuscript version.

## Authors' Contributions

Zhaoyi Sun: conceptualization, methodology, implementation, experiments, writing-original draft. Ping Liu: supervision, validation, writing-review and editing. All authors read and approved the final manuscript.

## Competing Interests

The authors declare that they have no competing interests.

## Ethics Approval and Consent to Participate

Not applicable.

## Consent for Publication

Not applicable.

## 8. References

1. Robinius, M., Stein, F., Schwane, A., Stolten, D. A Top-Down Spatially Resolved Electrical Load Model. *Energies* **10**(3), 361 (2017). doi:10.3390/en10030361.
2. Hu, Y., Li, J., Hong, M., Ren, J., Lin, R., Liu, Y., Liu, M., Man, Y. Short term electric load forecasting model and its verification for process industrial enterprises based on hybrid GA-PSO-BPNN algorithm: A case study of papermaking process. *Energy* **170**, 1215--1227 (2019). doi:10.1016/j.energy.2018.12.208.
3. An, J., Mikhaylov, A., Dincer, H., Yuksel, S. Economic modelling of electricity generation: long short-term memory and Q-rung orthopair fuzzy sets. *Heliyon* **8**(12), e12345 (2022). doi:10.1016/j.heliyon.2022.e12345.
4. Bian, R., He, R., Yan, X., Tong, X. LSTM-Driven Predictive Scheduling for Green Energy-Hydrogen-Methanol Integrated System. *Chemical Engineering Transactions* **120**, 511--516 (2025). doi:10.3303/CET25120086.
5. Benidis, K., Rangapuram, S. S., Flunkert, V., Wang, B., Maddix, D. C., Turkmen, C., Gasthaus, J., Schneider, M., Salinas, D., Stella, L., Aubet, F.-X., Callot, L., Januschowski, T. Deep Learning for Time Series Forecasting: Tutorial and Literature Survey. *ACM Computing Surveys* **55**(6), 1--36 (2022). doi:10.1145/3533382.
6. Hochreiter, S., Schmidhuber, J. Long Short-Term Memory. *Neural Computation* **9**(8), 1735--1780 (1997). doi:10.1162/neco.1997.9.8.1735.
7. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. Attention Is All You Need. In: *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 5998--6008 (2017).
8. Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. In: *International Conference on Learning Representations (ICLR)* (2023).
9. Zeng, A., Chen, M., Zhang, L., Xu, Q. Are Transformers Effective for Time Series Forecasting? In: *AAAI Conference on Artificial Intelligence*, vol. 37, pp. 11121--11130 (2023). doi:10.1609/aaai.v37i9.26317.
10. Lai, G., Chang, W.-C., Yang, Y., Liu, H. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. In: *Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 95--104 (2018). doi:10.1145/3209978.3210006.
11. Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., Zhang, W. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. In: *AAAI Conference on Artificial Intelligence*, vol. 35, pp. 11106--11115 (2021).
12. Wu, H., Xu, J., Wang, J., Long, M. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. In: *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 22419--22430 (2021).
13. Chung, J.-Y., Gulcehre, C., Cho, K., Bengio, Y. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *arXiv preprint arXiv:1412.3555* (2014).
14. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., Long, M. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. In: *International Conference on Learning Representations (ICLR)* (2024).
15. Lim, B., Arik, S. O., Loeff, N., Pfister, T. Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting* **37**(4), 1748--1764 (2021). doi:10.1016/j.ijforecast.2021.03.012.
16. Oreshkin, B. N., Carpov, D., Chapados, N., Bengio, Y. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In: *International Conference on Learning Representations (ICLR)* (2020).
17. Tashman, L. J. Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting* **16**(4), 437--450 (2000). doi:10.1016/S0169-2070(00)00065-0.
18. Sola, J., Sevilla, J. Importance of input data normalization for the application of neural networks to complex industrial problems. *IEEE Transactions on Nuclear Science* **44**(3), 1464--1468 (1997). doi:10.1109/23.589532.
19. Dauphin, Y., Fan, A., Auli, M., Grangier, D. Language Modeling with Gated Convolutional Networks. In: *International Conference on Machine Learning (ICML)*, pp. 933--942 (2017).
20. Kingma, D. P., Ba, J. Adam: A Method for Stochastic Optimization. In: *International Conference on Learning Representations (ICLR)* (2015).
21. Prechelt, L. Early Stopping - But When? In: *Neural Networks: Tricks of the Trade*, pp. 55--69. Springer, Berlin, Heidelberg (1998). doi:10.1007/3-540-49430-8_3.
22. Hebrail, G., Berard, A. Individual household electric power consumption data set. *UCI Machine Learning Repository* (2012).
23. Hyndman, R. J., Koehler, A. B. Another look at measures of forecast accuracy. *International Journal of Forecasting* **22**(4), 679--688 (2006). doi:10.1016/j.ijforecast.2006.03.001.
24. Hyndman, R. J., Athanasopoulos, G., Bergmeir, C., Caceres, G. A., Chhay, L., Kuroptev, K., O'Hara-Wild, M., Petropoulos, F., Razbash, S., Wang, E., Yasmeen, F. forecast: Forecasting Functions for Time Series and Linear Models. *R package version 8.15* (2021).
25. Wilcoxon, F. Individual Comparisons by Ranking Methods. *Biometrics Bulletin* **1**(6), 80--83 (1945). doi:10.2307/3001968.
26. Hong, T., Pinson, P., Shu, F., Zareipour, H., Troccoli, A., Hyndman, R. J. Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. *International Journal of Forecasting* **32**(3), 896--913 (2016). doi:10.1016/j.ijforecast.2016.02.001.
27. Hippert, H. S., Pedreira, C. E., Souza, R. C. Neural networks for short-term load forecasting: a review and evaluation. *IEEE Transactions on Power Systems* **16**(1), 44--55 (2001). doi:10.1109/59.910780.
28. Ferdosian, M., Abdi, H., Karimi, S., Kharrati, S. Short-term load and spinning reserve prediction based on LSTM and ANFIS with PSO algorithm. *The Journal of Engineering* (2024). doi:10.1049/tje2.12356.
29. Aziz, A., Mahmood, D., Qureshi, M. S., Qureshi, M. B., Kim, K. AI-based peak power demand forecasting model focusing on economic and climate features. *Frontiers in Energy Research* 12, 1328891 (2024). doi:10.3389/fenrg.2024.1328891.
