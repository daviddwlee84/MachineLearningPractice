# Feature Engineering

## Tips for Getting New Feature

Model may not extract some message like human does. We can build new feature by combining other features to gain accuracy.

* Think about business situation
* Learn from the open source kernel and the discussion session

### What to define a good feature

* This feature gain a lot of improvement
* Feature importance

## Tips for Memory Usage

Feature engineering will need lots of RAM (usually > 16GB).

### Pandas

#### astype

Most of the time, you won't need very precise data. Use not-so-precise data type will save memory and time.

* float: float32 (default float64)
* string: int32

#### [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)

#### [dataframe.groupby.agg](https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.core.groupby.DataFrameGroupBy.agg.html)

* [Summarising, Aggregating, and Grouping data in Python Pandas](https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/)

- np.ptp (peak to peak)
  - Range of values (maximum - minimum) along an axis.

#### [pandas.Series.dt.month](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.dt.month.html)

### Pickle

#### MacOS file large than 4GB

* [stackoverflow](https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb)

```py
def dump_bigger(data, output_file):
    """
    pickle.dump big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)
```

## Tips for parameter

### Gain parameter

#### Bayesian Optimization

#### Optuna

#### Hyperopt

### Test parameter

#### K-Fold Cross-Validation

## Links

* [Wiki - Feature engineering](https://en.wikipedia.org/wiki/Feature_engineering)
* [機器學習筆記 - 特徵工程](https://feisky.xyz/machine-learning/basic/feature-engineering.html)
