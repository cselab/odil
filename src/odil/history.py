import numpy as np
import pickle


class History:

    def __init__(self, csvpath=None, warmup=0):
        '''
        warmup: `int`
            Write only if `self.count > warmup`.
            Useful if unknown columns appear after the first row.
        '''
        self.data = dict()  # Data columns.
        self.count = 0  # Current number of entries.
        self.warmup = warmup
        self.csvcount = 0  # Current number of entries written to CSV.
        self.csvpath = csvpath
        self.csvkeys = None
        self.csvfile = open(csvpath, 'w') if csvpath is not None else None

    @staticmethod
    def _none_like(value):
        if value is None:
            return None
        elif type(value) in [float, np.float32, np.float64]:
            return 0.
        elif type(value) == int:
            return 0
        else:
            assert "Unknown type: " + str(type(value))

    def append(self, key, value=None):
        assert value is None or isinstance(value, (
            int,
            float,
            str,
            np.float32,
            np.float64,
            np.ndarray,
        )), ("Unexpected type: " + str(type(value)))
        if isinstance(value, np.ndarray):
            assert value.shape == (1, ) or len(value.shape) == 0
            value = value.item()
        if key not in self.data:
            assert value is not None
            self.data[key] = [self._none_like(value)] * self.count
        if value is None:
            assert len(self.data[key]) > 0, 'Expected non-empty column ' + key
            value = self._none_like(self.data[key][-1])
        self.data[key].append(value)

    def commit(self):  # Finish the current entry.
        maxlen = max(len(self.data[k]) for k in self.data)
        fail = ''
        for k, v in self.data.items():
            if len(v) < maxlen:
                fail += k + ','
        if fail:
            raise RuntimeError('Missing values for columns: ' + fail)
        self.count += 1

    def get(self, key, default=None):
        return self.data.get(key, default)

    def append_dict(self, newdict):
        for k, v in newdict.items():
            self.append(k, v)

    def write(self, nocommit=False):
        '''
        Writes pending rows to current file.
        '''
        if not nocommit:
            self.commit()
        if self.count <= self.warmup:
            return
        if self.csvfile is None:
            return
        if self.csvkeys is not None:
            if len(self.data.keys()) != len(self.csvkeys):
                newkeys = list(set(self.data.keys()) - set(self.csvkeys))
                raise RuntimeError(
                    "Unexpected keys in history: {:}".format(newkeys))
        if self.csvcount == 0:
            # Write header.
            self.csvkeys = list(self.data.keys())
            self.csvfile.write(','.join(self.csvkeys) + '\n')
        while self.csvcount < self.count:
            row = [self.data[key][self.csvcount] for key in self.data]
            line = ','.join(map(str, row))
            self.csvfile.write(line + '\n')
            self.csvcount += 1
        self.csvfile.flush()

    def save(self, path):
        '''
        Saves data to pickle.
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        '''
        Overwrites data from pickle.
        '''
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            self.csvkeys = list(self.data.keys())
            self.count = len(next(iter(self.data.values())))
            self.write(nocommit=True)

    def close(self):
        self.csvfile.close()
