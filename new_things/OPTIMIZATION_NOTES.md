# NEST Optimization Research Notes

## Current Performance Issue

The bottleneck is NOT NEST itself, but **Python loop overhead** when building connection lists:
- Building 14.4M connection dictionaries in Python takes 20-30 minutes
- The actual NEST `Connect()` call is very fast once data is prepared

## NEST Capabilities (from research)

### Benchmarked Performance:
- ✅ NEST 3.9 can handle **65 billion synapses** (we have 14.4M)
- ✅ Networks with 5.8M neurons tested successfully
- ✅ Our network (52K neurons, 15M synapses) is well within NEST capabilities

### Key Optimizations Available:

#### 1. **Use NEST's Built-in Connection Rules**
Instead of manually creating connection lists, use NEST's algorithmic connectors:
- `FixedIndegree`, `FixedOutdegree`
- `FixedTotalNumber`
- Probability-based connections
- **This is 10-100x faster** than manual lists

#### 2. **Connection Set Algebra (CSA)**
Pre-compute connectivity patterns efficiently:
```python
import csa
conn_spec = csa.random(p=0.1)  # Much faster than loops
nest.Connect(pre, post, conn_spec)
```

#### 3. **NumPy Vectorization**
Replace Python loops with NumPy operations:
```python
# Current (slow):
for syn in synapses:
    connections.append({...})

# Optimized (fast):
connections = {
    'source': synapses[:, 0].astype(int).tolist(),
    'target': synapses[:, 1].astype(int).tolist(),
    'weight': (synapses[:, 2] * scale).tolist(),
    'delay': synapses[:, 3].tolist()
}
```

#### 4. **GPU Acceleration (NEST 3.9+)**
- Direct GPU network construction
- Multi-GPU support
- Dramatically faster for large networks

#### 5. **Batch Initialization**
Our current approach already uses batch Connect - good!
The issue is building the input lists.

## Recommended Future Improvements

### Immediate (10x speedup):
1. **Pre-compute connection arrays** in NumPy before any loops
2. **Filter zero weights** using NumPy boolean indexing
3. Use **vectorized operations** for weight scaling

### Example Optimization:
```python
# Current slow approach:
for syn in recurrent:
    src, tgt, w, d = int(syn[0]), int(syn[1]), float(syn[2]), float(syn[3])
    ntype = neurons[tgt]
    w_scaled = w * vsc[ntype] / 1000.0
    if w_scaled != 0:
        conns.append({'source': src, 'target': tgt, 'weight': w_scaled, 'delay': d})

# Optimized (100x faster):
# Vectorize ALL operations at once
src = recurrent[:, 0].astype(int)
tgt = recurrent[:, 1].astype(int)
w = recurrent[:, 2]
d = recurrent[:, 3]
ntype_vec = neurons[tgt]  # Fancy indexing - very fast
w_scaled = w * vsc[ntype_vec] / 1000.0
mask = np.abs(w_scaled) > 1e-10  # Boolean mask

# Create connection dict directly
conns = {
    'source': src[mask].tolist(),
    'target': tgt[mask].tolist(),
    'weight': w_scaled[mask].tolist(),
    'delay': np.maximum(d[mask], 1.0).tolist()
}
```

### Advanced (100x+ speedup):
1. **Use NEST GPU** module
2. **Load connections from HDF5** directly into NEST
3. **Sparse matrix format** for connectivity

## Performance Targets

With optimizations:
- **Setup time:** < 30 seconds (currently 20-30 minutes)
- **Simulation time:** < 10 seconds for 1000ms
- **Total runtime:** < 1 minute (currently 25+ minutes)

## References
- NEST 3.9 Documentation: https://nest-simulator.readthedocs.io/
- NEST GPU: NEST Conference 2024
- Connection Patterns: https://nest-simulator.readthedocs.io/en/stable/tutorials/
- Performance Benchmarks: NEST can handle 65B synapses
