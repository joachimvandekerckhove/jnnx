# ddm3mv JNNX Package

Raw-I/O emulator for the ddm3mv model.

## Inputs
- v: [-2.0, 2.0]
- a: [0.5, 2.0]
- t0: [0.15, 0.45]

## Outputs
- mu_acc
- mu_rt_mean
- mu_rt_var
- chol_1
- chol_2
- chol_3
- chol_4
- chol_5
- chol_6

## Observation transform (v2.0)

Raw physical summaries are passed to JAGS; `obs_transform.json` bakes the forward map (column transforms + StandardScaler) into the compiled module.

```json
{
  "column_transforms": ["identity", "log1p", "log1p"],
  "scaler_mean": [0.82, 0.389, -2.61],
  "scaler_scale": [0.12, 0.195, 0.71]
}
```

JAGS usage: `obs[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)` with `obs` in physical units (accuracy, mean RT, RT variance).
