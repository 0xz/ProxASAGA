
import os, tarfile, scipy, cffi
import numpy as np
from scipy import sparse
from sklearn.datasets import load_svmlight_file

from tick.dataset.download_helper import load_dataset, get_data_home

ffi = cffi.FFI()
ffi.cdef("""
    extern int cffi_prox_asaga(double* x, double* A_data, int64_t* A_indices, int64_t* A_indptr, double* b,
        double* d, int64_t n_samples, int64_t n_features, int64_t n_threads, double alpha, double beta,
        double step_size, int64_t max_iter, double* trace_x, double* trace_time, int64_t iter_freq);
    """)

dataset_path = os.path.join(get_data_home(), 'binary/url/url_svmlight.tar.gz')

dir_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(dir_path + '/bin/build/libasaga.so'):
  raise ValueError("asaga lib not found - build with mkn first")
C = ffi.dlopen(dir_path + '/bin/build/libasaga.so')

def load_url_dataset_day(cache_path, days):
    tar_file = tarfile.open(dataset_path, "r:gz")
    n_features = 3231961
    X, y = None, None
    for day in days:
        data_filename = 'url_svmlight/Day{}.svm'.format(day)
        with tar_file.extractfile(data_filename) as data_file:
            X_day, y_day = load_svmlight_file(data_file, n_features=n_features)
        if X is None:
            X, y = X_day, y_day
        else:
            X = scipy.sparse.vstack((X, X_day))
            y = np.hstack((y, y_day))
    return X, y


def fetch_url_dataset(n_days=10, data_home=None, verbose=True):
    dataset = None
    print(dataset_path)
    if os.path.exists(dataset_path):
        try:
            dataset = load_url_dataset_day(dataset_path, range(n_days))
        except Exception as e:
            print('Cache loading failed')
            print(e)
    if dataset is None:
        raise ValueError("dataset is None")
    return dataset

def _compute_D(A):
    # .. estimate diagonal elements of the reweighting matrix (D) ..
    n_samples = A.shape[0]
    tmp = A.copy()
    tmp.data[:] = 1.
    d = np.array(tmp.sum(0), dtype=np.float).ravel()
    idx = (d != 0)
    d[idx] = n_samples / d[idx]
    d[~idx] = 0.
    return d


def _logistic_loss(A, b, alpha, beta, x):
    # loss function to be optimized, it's the logistic loss
    z = A.dot(x)
    yz = b * z
    idx = yz > 0
    out = np.zeros_like(yz)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
    out = out.mean() + .5 * alpha * x.dot(x) + beta * np.sum(np.abs(x))
    return out

def minimize_SAGA(A, b, alpha, beta, step_size, max_iter=100, n_jobs=1):
    n_samples, n_features = A.shape
    A = sparse.csr_matrix(A, dtype=np.float)
    indices = A.indices.astype(np.int64)
    indptr = A.indptr.astype(np.int64)
    x = np.zeros(n_features)

    d = _compute_D(A)
    print('Delta (sparsity measure) = %s' % (1.0 / np.min(d[d != 0])))

    trace_x = np.zeros((max_iter + 1, n_features))
    trace_time = np.zeros(max_iter + 1)
    C.cffi_prox_asaga(
        ffi.cast("double *", x.ctypes.data), ffi.cast("double *", A.data.ctypes.data),
        ffi.cast("int64_t *", indices.ctypes.data), ffi.cast("int64_t *", indptr.ctypes.data),
        ffi.cast("double *", b.ctypes.data), ffi.cast("double *", d.ctypes.data), n_samples, n_features,
        n_jobs, alpha, beta, step_size, max_iter, ffi.cast("double *", trace_x.ctypes.data),
        ffi.cast("double *", trace_time.ctypes.data), ffi.cast("int64_t", n_samples))
    print('.. computing trace ..')
    func_trace = np.array([
        _logistic_loss(A, b, alpha, beta, xi) for xi in trace_x])

    return x, trace_time[:-1], func_trace[:-1]


if __name__ == '__main__':
    from scipy import sparse
    import matplotlib.pyplot as plt

    X, y = fetch_url_dataset()

    n_samples, n_features = X.shape
    
    beta = 1e-10
    alpha = 1. / n_samples

    L = 0.25 * np.max(X.multiply(X).sum(axis=1)) + alpha * n_samples
    print('data loaded')


    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    step_size_SAGA = 1.0 / (3 * L)
    markers = ['^', 'h', 'o', 's', 'x', 'p', 'h', 'o', 's', 'x', 'p']

    max_trace_time = 0
    max_epochs = 200
    for i, n_jobs in enumerate([1, 2, 4, 8, 16]):

        print('Running %s jobs' % n_jobs)

        max_iter = int(max_epochs / n_jobs)

        x, trace_time, func_trace = minimize_SAGA(
            X, y, alpha, beta, step_size_SAGA, max_iter=max_iter, n_jobs=n_jobs)
        fmin = np.min(func_trace)

        n_epochs = len(trace_time) * n_jobs

        label = '{} c, {:.2g} s/e, {:.2g} s/i'.format(n_jobs, trace_time[-1] / n_epochs, trace_time[-1] / len(trace_time))
        axes[0].plot(trace_time, func_trace - fmin, label=label, marker=markers[i], markersize=10, lw=3)
        axes[1].plot(np.arange(len(func_trace)) * n_jobs, func_trace - fmin, label=label, marker=markers[i], markersize=10, lw=3)

        max_trace_time = max(trace_time[-1], max_trace_time)

    for ax in axes:
      ax.grid()
      ax.legend()
      
      # ax.set_ylim(ymin=1e-10)
      ax.set_yscale('log')

    axes[0].set_xlim((0, max_trace_time * .7))
    axes[0].set_xlabel("time")
    axes[1].set_xlim((0, max_epochs * .7))
    axes[1].set_xlabel("number of epochs")

    plt.show()


# if __name__ == '__main__':
#     from scipy import sparse
#     import pylab as plt

#     X, y = fetch_url_dataset()

#     n_samples, n_features = X.shape
    
#     beta = 1e-10
#     alpha = 1. / n_samples

#     L = 0.25 * np.max(X.multiply(X).sum(axis=1)) + alpha * n_samples
#     print('data loaded')

#     step_size_SAGA = 1.0 / (3 * L)
#     markers = ['^', 'h', 'o', 's', 'x', 'p', 'h', 'o', 's', 'x', 'p']
#     for i, n_jobs in enumerate([1, 2, 3, 4, 6, 8, 9, 12, 14, 16]):
#         print('Running %s jobs' % n_jobs)
#         x, trace_time, func_trace = minimize_SAGA(
#             X, y, alpha, beta, step_size_SAGA, max_iter=int(200 / n_jobs), n_jobs=n_jobs)
#         fmin = np.min(func_trace)
#         plt.plot(trace_time, func_trace - fmin, label='using %s cores' % n_jobs, marker=markers[i], markersize=10, lw=3)
#     plt.grid()
#     plt.legend()
#     plt.xlim((0, trace_time[-1] * .7))
#     plt.ylim(ymin=1e-10)
#     plt.yscale('log')
#     plt.show()
