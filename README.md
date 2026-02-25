# alonshalem

## SymPy Z-transform solution (question system)

This repository now includes a symbolic solver for:

\[
y[n] - a^2 y[n-2] = \frac{1}{2}x[n]
\]

assuming zero initial conditions and a causal system.

### Setup

```bash
python3 -m pip install -r requirements.txt
```

### Run

```bash
python3 z_transform_solver.py
```

The script prints:
- the Z-domain equation,
- \(H(z)=Y(z)/X(z)\),
- poles,
- causal ROC,
- stability condition on \(a\),
- symbolic impulse response.

### Test

```bash
python3 -m unittest -v
```