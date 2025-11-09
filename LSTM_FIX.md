# LSTM Build Error Fix

## Issue
```
ERROR - Failed to build LSTM model: list index out of range
```

## Root Cause
The `build_model()` method was hardcoded for 3 LSTM layers:
```python
model.add(LSTM(self.layers[0], ...))  # Layer 1
model.add(LSTM(self.layers[1], ...))  # Layer 2
model.add(LSTM(self.layers[2], ...))  # Layer 3 - ERROR!
```

But the new configuration only has 2 layers:
```python
LSTM_LAYERS = [32, 16]  # Only 2 layers!
```

Accessing `self.layers[2]` caused "list index out of range" error.

## Fix Applied

Changed `build_model()` in `models/lstm_model.py` to dynamically build layers:

```python
# BEFORE (hardcoded for 3 layers)
model.add(LSTM(self.layers[0], return_sequences=True, input_shape=input_shape))
model.add(Dropout(0.2))
model.add(LSTM(self.layers[1], return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(self.layers[2], return_sequences=False))  # ERROR!
model.add(Dropout(0.2))

# AFTER (dynamic for any number of layers)
for i, units in enumerate(self.layers):
    if i == 0:
        model.add(LSTM(units, return_sequences=(i < len(self.layers) - 1), input_shape=input_shape))
    else:
        model.add(LSTM(units, return_sequences=(i < len(self.layers) - 1)))
    model.add(Dropout(0.2))
```

## Benefits

✅ **Flexible**: Works with any number of layers (1, 2, 3, or more)  
✅ **Automatic**: Correctly sets `return_sequences` for each layer  
✅ **Future-proof**: Can change `LSTM_LAYERS` without code changes  

## Configuration Support

Now supports any layer configuration:

```python
# 1 layer
LSTM_LAYERS = [64]

# 2 layers (current)
LSTM_LAYERS = [32, 16]

# 3 layers (original)
LSTM_LAYERS = [64, 32, 16]

# 4 layers
LSTM_LAYERS = [128, 64, 32, 16]
```

## Additional Fix

Also added better error handling in `models/ensemble_model.py`:
- Check if `clean_df` is empty after cleaning
- Log specific error message
- Prevent prediction failures

## Restart Required

```bash
python start_app.py
```

## Expected Output

```
INFO - Training LSTM model...
INFO - Created 723 sequences for LSTM training
INFO - Sequence shape: (723, 30, 93)
INFO - LSTM model built: ~15000 parameters  ✅ (should work now!)
INFO - LSTM model trained successfully
INFO - Training MAE: X.XX
INFO - Full training completed: 4/4 models  ✅
INFO - Adjusted weights: {
    'kalman': 0.10,
    'random_forest': 0.30,
    'lightgbm': 0.35,
    'lstm': 0.25  ✅ (should be 0.25, not 0.0!)
}
```

## Summary

✅ **LSTM build error**: Fixed with dynamic layer building  
✅ **Supports any layer count**: 1, 2, 3, or more layers  
✅ **Better error handling**: Added checks for empty data  

**Ready to restart!** 🚀
