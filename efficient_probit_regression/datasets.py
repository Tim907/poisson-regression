import abc
import bz2
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.datasets import (
    fetch_covtype,
    fetch_kddcup99,
    load_iris,
    load_svmlight_file,
    make_blobs,
)
from sklearn.preprocessing import scale

from efficient_probit_regression import settings
from efficient_probit_regression.probit_model import PoissonModel

_logger = settings.get_logger()

_rng = np.random.default_rng()


def add_intercept(X):
    """ Adds intercept."""
    return np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)


class BaseDataset(abc.ABC):
    def __init__(self, p, add_intercept=True, use_caching=True, cache_dir=None):
        self.use_caching = use_caching
        self.p = p
        if cache_dir is None:
            cache_dir = settings.DATA_DIR
        self.cache_dir = cache_dir

        if use_caching and not self.cache_dir.exists():
            self.cache_dir.mkdir()

        self.X = None
        self.y = None
        self.beta_opt_dir = {}
        self.add_intercept = add_intercept
        self.inHull = None

    @abc.abstractmethod
    def load_X_y(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    def _load_X_y_cached(self):

        X_path = self.get_binary_path_X()
        y_path = self.get_binary_path_y()
        inHull_path = self.cache_dir / f"{self.get_name()}_inHull.npy"

        if self.use_caching:
            if X_path.exists() and y_path.exists() and inHull_path.exists():
                _logger.info(
                    f"Loading cached versions of X and y found at {X_path} and {y_path} and {inHull_path}..."
                )
                X = np.load(X_path)
                if self.add_intercept:
                    X = add_intercept(X)
                y = np.load(y_path)
                self.inHull = np.load(inHull_path)
                _logger.info("Done.")
                return X, y

        _logger.info("Loading X and y...")
        X, y = self.load_X_y()
        _logger.info("Done.")
        np.save(X_path, X)
        np.save(y_path, y)

        if type(self) is not SyntheticSimplexGaussian:
            self.inHull = find_convex_hull(X)
            np.save(inHull_path, self.inHull)

        _logger.info(f"Saved X, y and inHull at {X_path}, {y_path} and {inHull_path}.")

        if self.add_intercept:
            X = add_intercept(X)

        return X, y

    def _compute_beta_opt(self, p):
        """
        Computes beta_opt.
        """
        model = PoissonModel(p=p, epsilon=0, X=self.get_X(), y=self.get_y())
        model.fit()
        beta_opt = model.get_params()
        return beta_opt

    def _get_beta_opt_cached(self, p):
        if not self.use_caching:
            _logger.info(f"Computing beta_opt for p={p}...")
            beta_opt = self._compute_beta_opt(p)
            _logger.info("Done.")
            return beta_opt

        beta_opt_path = self.get_binary_path_beta_opt(p)
        # if beta_opt_path.exists():
        #     _logger.info(
        #         f"Loading cached version of beta_opt for p={p} found at {beta_opt_path}..."  # noqa
        #     )
        #     beta_opt = np.load(beta_opt_path)
        #     _logger.info("Done.")
        #     return beta_opt

        _logger.info(f"Computing beta_opt for p={p}...")
        beta_opt = self._compute_beta_opt(p)
        _logger.info("Done.")
        np.save(beta_opt_path, beta_opt)
        _logger.info(f"Saved beta_opt for p={p} at {beta_opt_path}.")

        return beta_opt

    def _assert_data_loaded(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_X_y_cached()

    def get_binary_path_X(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_X.npy"

    def get_binary_path_y(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_y.npy"

    def get_binary_path_beta_opt(self, p) -> Path:
        return self.cache_dir / f"{self.get_name()}_beta_opt_p_{p}.npy"

    def get_X(self):
        """Returns the data matrix from the data object."""
        self._assert_data_loaded()
        return self.X

    def get_y(self):
        """Returns the target data y."""
        self._assert_data_loaded()
        return self.y

    def get_n(self):
        """Returns the number of rows of the data."""
        self._assert_data_loaded()
        return self.X.shape[0]

    def get_d(self):
        """Returns the dimension of the data. d can also be regarded as the number of features."""
        self._assert_data_loaded()
        return self.X.shape[1]

    def get_beta_opt(self, p: int):
        """Returns the optimized/estimated parameters beta. 
        
        :param p: the order of the p-generalized probit-model.
        
        :return: returns the estimated parameters """
        if p not in self.beta_opt_dir.keys():
            self.beta_opt_dir[p] = self._get_beta_opt_cached(p)

        return self.beta_opt_dir[p]


class Covertype(BaseDataset):
    """
    Dataset Homepage:
    https://archive.ics.uci.edu/ml/datasets/Covertype
    """

    features_continuous = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]

    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        """ Returns name of the data set."""
        return "covertype"

    def load_X_y(self):
        _logger.info("Fetching covertype from sklearn...")
        sklearn_result = fetch_covtype(as_frame=True)
        df = sklearn_result.frame

        _logger.info("Preprocessing...")

        # Cover_Type 2 gets the label 1, everything else gets the label -1.
        # This ensures maximum balance.
        y = df["Cover_Type"].apply(lambda x: 1 if x == 2 else -1).to_numpy()
        df = df.drop("Cover_Type", axis="columns")

        # scale the continuous features to mean zearo and variance 1
        # and leave the 0/1 features as is
        X_continuous = df[self.features_continuous].to_numpy()
        X_continuous = scale(X_continuous)

        features_binary = sorted(set(df.columns) - set(self.features_continuous))
        X_binary = df[features_binary].to_numpy()

        # put binary features and scaled features back together
        X = np.append(X_continuous, X_binary, axis=1)

        return X, y


class KDDCup(BaseDataset):
    """
    Dataset Homepage:
    https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    """

    features_continuous = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
    ]

    features_discrete = [
        "protocol_type",
        "service",
        "flag",
        "land",
        "logged_in",
        "is_host_login",
        "is_guest_login",
    ]

    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        """ Returns name of data set."""
        return "kddcup"

    def load_X_y(self):
        _logger.info("Fetching kddcup from sklearn...")
        sklearn_result = fetch_kddcup99(as_frame=True, percent10=True)
        df = sklearn_result.frame

        _logger.info("Preprocessing...")

        # convert label "normal." to -1 and everything else to 1
        y = df.labels.apply(lambda x: -1 if x.decode() == "normal." else 1).to_numpy()

        # get all the continuous features
        X_continuous = df[self.features_continuous]

        # the feature num_outbound_cmds has only one value that doesn't
        # change, so drop it
        X_continuous = X_continuous.drop("num_outbound_cmds", axis="columns")

        # convert to numpy array
        X_continuous = X_continuous.to_numpy()

        # scale the features to mean 0 and variance 1
        X = scale(X_continuous)

        return X, y


class Webspam(BaseDataset):
    """
    Dataset Source:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#webspam
    """

    dataset_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.bz2"  # noqa: E501

    def __init__(self, drop_sparse_columns=True, use_caching=True):
        self.drop_sparse_columns = drop_sparse_columns
        super().__init__(use_caching=use_caching)

    def get_name(self):
        """ Returns name of data set."""
        if self.drop_sparse_columns:
            return "webspam"
        else:
            return "webspam_with_sparse"

    def get_raw_path(self):
        return self.cache_dir / f"{self.get_name()}.csv"

    def download_dataset(self):
        _logger.info(f"Downloading data from {self.dataset_url}")

        context = ssl._create_unverified_context()
        with urllib.request.urlopen(self.dataset_url, context=context) as f:
            contents = f.read()

        _logger.info("Download completed.")
        _logger.info("Extracting data...")

        file_raw = bz2.open(io.BytesIO(contents))
        X_sparse, y = load_svmlight_file(file_raw)

        # convert scipy Compressed Sparse Row array into numpy array
        X = X_sparse.toarray()

        df = pd.DataFrame(X)
        df["LABEL"] = y

        _logger.info(f"Writing .csv file to {self.get_raw_path()}")

        df.to_csv(self.get_raw_path(), index=False)

    def load_X_y(self):
        """ Loads data and returns X and y."""
        if not self.get_raw_path().exists():
            _logger.info(f"Couldn't find dataset at location {self.get_raw_path()}")
            self.download_dataset()

        df = pd.read_csv(self.get_raw_path())

        _logger.info("Preprocessing the data...")

        y = df["LABEL"].to_numpy()
        df = df.drop("LABEL", axis="columns")

        # drop all columns that only have constant values
        # drop all columns that contain only one non-zero entry
        for cur_column_name in df.columns:
            cur_column = df[cur_column_name]
            cur_column_sum = cur_column.astype(bool).sum()
            unique_values = cur_column.unique()
            if len(unique_values) <= 1:
                df = df.drop(cur_column_name, axis="columns")
            if self.drop_sparse_columns and cur_column_sum == 1:
                df = df.drop(cur_column_name, axis="columns")

        X = df.to_numpy()

        # scale the features to mean 0 and variance 1
        X = scale(X)

        return X, y


class Iris(BaseDataset):
    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        """" Returns name of data set."""
        return "iris"

    def load_X_y(self):
        """ Loads data and returns X and y."""
        X, y = load_iris(return_X_y=True)
        X = scale(X)
        y = np.where(y == 1, 1, -1)

        return X, y


class Example2D(BaseDataset):
    def __init__(self):
        super().__init__(add_intercept=True, use_caching=False)

    def get_name(self):
        """ Returns name of data set."""
        return "example-2d"

    def load_X_y(self):
        """ Loads data set and returns X and y."""
        centers = np.array([[-1, -1], [1, 1], [4.5, 5]])
        X, y = make_blobs(
            n_samples=[80, 80, 15],
            n_features=2,
            centers=centers,
            cluster_std=[1, 1, 0.5],
            random_state=1,
        )
        y = np.where(y == 2, 0, y)

        y = 2 * y - 1

        return X, y



def find_convex_hull(X):
    """
    Returns a boolean vector describing which rows of X are on the convex hull of X.
    """
    _logger.info(f"Finding the convex hull...")

    def in_hull(points, x):
        c = np.zeros(len(points))
        lp = linprog(c, A_eq=points.T, b_eq=x, options={"disp": False})
        return lp.success

    def are_hull_points(X):
        in_hull_logic = np.full(X.shape[0], False)
        for i in range(X.shape[0]):
            all_but_i = np.delete(np.arange(X.shape[0]), i)
            unskippable = all_but_i[np.invert(in_hull_logic[all_but_i])]  # remove in-hull points
            print(i, "/", X.shape[0], " | To Consider: ", len(unskippable))
            in_hull_logic[i] = in_hull(X[unskippable, :], X[i, :])  # boolean result for point i
        return in_hull_logic

    def divide_and_conquer_convex_hull(X):
        if X.shape[0] <= 5:  # Base case
            return are_hull_points(X)

        # Divide the set into two halves
        mid = X.shape[0] // 2
        left_half = X[:mid, :]
        right_half = X[mid:, :]

        # Recursively find convex hulls of the halves
        left_hull = divide_and_conquer_convex_hull(left_half)
        right_hull = divide_and_conquer_convex_hull(right_half)

        # Merge the convex hulls of two halves
        removeable_points = np.hstack((left_hull, right_hull))
        unskippable_points = np.arange(X.shape[0])[np.invert(removeable_points)]
        in_hull_logic = removeable_points
        in_hull_logic[unskippable_points] = are_hull_points(X[unskippable_points, :])
        print(X.shape, "/", len(unskippable_points) / X.shape[0], " / ", np.sum(in_hull_logic))
        return in_hull_logic

    # Check equality of branch and bound implementation
    # print(np.array_equal(are_hull_points(X[:10000, :]), divide_and_conquer_convex_hull(X[:10000, :])))
    # divide-and-conquer is about twice as slow

    inHull = are_hull_points(X)
    _logger.info(f"Convex hull calculated. In-hull are {np.sum(inHull)}/{X.shape[0]} points")
    return inHull


class Synthetic(BaseDataset):
    def __init__(self, n, d, p, variant, seed, use_caching=True):
        super().__init__(p=p, add_intercept=False, use_caching=use_caching)
        self.n = n
        self.d = d
        self.seed = seed
        if variant != 1 and variant != 2:
            raise ValueError("Variant should be one of {1, 2}.")
        self.variant = variant

    def get_name(self):
        """" Returns name of data set."""
        return f"synthetic_n{self.n}_d{self.d}_p{self.p}_variant{self.variant}_seed{self.seed}"

    def get_raw_path(self):
        return self.cache_dir / f"{self.get_name()}.csv"

    def load_X_y(self):
        """ Loads data and returns X and y."""
        n = self.n
        d = self.d
        p = self.p
        np.random.seed(self.seed)

        z_vec = np.zeros((6, d))
        for i in range(6):
            z_vec[i, :] = np.random.choice([-1, 1], size=d)
        mat = np.zeros((0, d))
        for i in range(2):
            mat = np.vstack((mat, z_vec[0]))
        for i in range(10):
            mat = np.vstack((mat, z_vec[1]))
        # Fill up z_[3-6] till n reached
        mat = np.vstack((mat, z_vec[np.tile(np.arange(start=2, stop=6), n // 4), :]))
        mat = mat[0:n, :]

        delta = 0.1
        Z = mat# + delta * np.random.standard_normal(size=(n, d))
        X = np.hstack((np.ones((n, 1)), Z)) # add intercept

        beta_snake = np.random.standard_normal((d, 1))
        b = np.max((1, 1.1 * np.abs(np.min(np.matmul(Z, beta_snake)))))
        beta = np.vstack([b, beta_snake])

        # Synthetic dataset 1
        lambdas = (np.matmul(X, beta) ** p).reshape(n)
        y = np.random.poisson(lambdas)

        # Synthetic dataset 2
        if self.variant == 2:
            logic = np.random.choice((False, True), n)
            y = np.ceil(lambdas)
            y[logic] = np.floor(lambdas)[logic]

        return X, y




class OnlineRetail(BaseDataset):
    """
    Dataset Source:
    https://archive.ics.uci.edu/dataset/352/online+retail
    """

    dataset_url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"  # noqa: E501

    def __init__(self, use_caching=True):
        super().__init__(use_caching=use_caching)

    def get_name(self):
        """ Returns name of data set."""
        return "onlineRetail"


    def get_raw_path(self):
        return self.cache_dir / f"{self.get_name()}.csv"

    def download_dataset(self):
        _logger.info(f"Downloading data from {self.dataset_url}")

        context = ssl._create_unverified_context()
        with urllib.request.urlopen(self.dataset_url, context=context) as f:
            contents = f.read()

        _logger.info("Download completed.")
        _logger.info("Extracting data...")

        with zipfile.ZipFile(io.BytesIO(contents), 'r') as z:
            # Assuming there's only one file in the zip archive and it's an Excel file
            file_name = z.namelist()[0]
            with z.open(file_name) as excel_file:
                # Read Excel file using pandas
                df = pd.read_excel(excel_file)

        df["LABEL"] = df["Quantity"]
        df = df[["InvoiceNo", "StockCode", "UnitPrice", "CustomerID", "Country", "LABEL"]]

        _logger.info(f"Writing .csv file to {self.get_raw_path()}")

        df.to_csv(self.get_raw_path(), index=False)

    def load_X_y(self):
        """ Loads data and returns X and y."""
        if not self.get_raw_path().exists():
            _logger.info(f"Couldn't find dataset at location {self.get_raw_path()}")
            self.download_dataset()

        df = pd.read_csv(self.get_raw_path())

        _logger.info("Preprocessing the data...")

        y = df["LABEL"].to_numpy()
        df = df.drop("LABEL", axis="columns")

        # drop all columns that only have constant values
        # drop all columns that contain only one non-zero entry
        for cur_column_name in df.columns:
            cur_column = df[cur_column_name]
            cur_column_sum = cur_column.astype(bool).sum()
            unique_values = cur_column.unique()
            if len(unique_values) <= 1:
                df = df.drop(cur_column_name, axis="columns")

        X = df.to_numpy()

        # scale the features to mean 0 and variance 1
        X = scale(X)

        return X, y



class Diabetes(BaseDataset):
    """
    Dataset Source:
    https://archive.ics.uci.edu/dataset/352/online+retail
    """

    dataset_url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"  # noqa: E501

    def __init__(self, p, use_caching=True):
        super().__init__(p=p, use_caching=use_caching)

    def get_name(self):
        """ Returns name of data set."""
        return "diabetes"


    def get_raw_path(self):
        return self.cache_dir / f"{self.get_name()}.csv"

    def download_dataset(self):
        _logger.info(f"Downloading data from {self.dataset_url}")

        context = ssl._create_unverified_context()
        with urllib.request.urlopen(self.dataset_url, context=context) as f:
            contents = f.read()

        _logger.info("Download completed.")
        _logger.info("Extracting data...")

        with zipfile.ZipFile(io.BytesIO(contents), 'r') as z:
            # Assuming there's only one file in the zip archive and it's an Excel file
            file_name = z.namelist()[0]
            with z.open(file_name) as csv_file:
                df = pd.read_csv(csv_file)

        df["LABEL"] = df["num_medications"] # response poisson distributed variable
        df = df[df["gender"].isin(["Male", "Female"])] # Remove the 3 observations with "Unknown/Invalid" gender
        df["IsFemale"] = df["gender"] == "Female"
        df["change"] = df["change"] == "Ch"
        df["diabetesMed"] = df["diabetesMed"] == "Yes"
        df["admission.Emergency"] = df["admission_type_id"] == 1
        df["admission.Urgent"] = df["admission_type_id"] == 2
        df["admission.Elective"] = df["admission_type_id"] == 3
        df = df[["LABEL", "IsFemale", "time_in_hospital", "num_lab_procedures", "num_procedures", "number_outpatient",
                 "number_emergency", "number_inpatient", "number_diagnoses", "change", "diabetesMed",
                 "admission.Emergency", "admission.Urgent", "admission.Elective"]]

        _logger.info(f"Writing .csv file to {self.get_raw_path()}")

        df.to_csv(self.get_raw_path(), index=False)

    def load_X_y(self):
        """ Loads data and returns X and y."""
        if not self.get_raw_path().exists():
            _logger.info(f"Couldn't find dataset at location {self.get_raw_path()}")
            self.download_dataset()

        df = pd.read_csv(self.get_raw_path())

        _logger.info("Preprocessing the data...")

        y = df["LABEL"].to_numpy()
        df = df.drop("LABEL", axis="columns")

        # drop all columns that only have constant values
        # drop all columns that contain only one non-zero entry
        for cur_column_name in df.columns:
            cur_column = df[cur_column_name]
            cur_column_sum = cur_column.astype(bool).sum()
            unique_values = cur_column.unique()
            if len(unique_values) <= 1:
                df = df.drop(cur_column_name, axis="columns")

        X = df.to_numpy().astype(float)

        # scale the features to mean 0 and variance 1
        # X = scale(X)

        return X, y


class SyntheticSimplexGaussian(BaseDataset):
    def __init__(self, n, d, p, variant, seed, use_caching=True):
        super().__init__(p=p, add_intercept=False, use_caching=use_caching)
        self.n = n
        self.d = d
        self.seed = seed
        if variant != 1 and variant != 2:
            raise ValueError("Variant should be one of {1, 2}.")
        self.variant = variant

    def get_name(self):
        """" Returns name of data set."""
        return f"syntheticSimplexGaussian_n{self.n}_d{self.d}_p{self.p}_variant{self.variant}_seed{self.seed}"

    def get_raw_path(self):
        return self.cache_dir / f"{self.get_name()}.csv"

    def load_X_y(self):
        """Loads data and returns X and y."""
        n = self.n  # number of points
        d = self.d  # dimensionality
        p = self.p  # Exponent für die Berechnung von lambdas
        np.random.seed(self.seed)

        # 1. build Simplex-points: nullvector and standard basis vectors
        simplex_points = np.vstack([np.zeros(6), np.eye(6)])

        # 2. generate n-7 Gauss-points, shift and scale
        l1_limit = 0.99  # L_1-Norm-boundary
        gaussian_points = np.random.normal(loc=0, scale=1.0, size=(n - 7, 6))  # standard gaussian
        gaussian_points = gaussian_points - gaussian_points.min()  # shift to non-negative numbers
        # make L_1-Norm < 1
        norms = np.linalg.norm(gaussian_points, ord=1, axis=1)
        gaussian_points = gaussian_points * (l1_limit / norms.max())

        # 3. create final points
        Z = np.vstack([simplex_points, gaussian_points])
        X = np.hstack((np.ones((n, 1)), Z))  # add intercept

        # 4. save convex hull
        inHull_path = self.cache_dir / f"{self.get_name()}_inHull.npy"
        self.inHull = np.hstack((np.repeat(False, 7), np.repeat(True, n - 7)))
        np.save(inHull_path, self.inHull)

        # 4. beta coefficients
        beta_snake = 10 * np.random.standard_normal((d, 1))
        b = np.max((1, 2 * np.abs(np.min(np.matmul(Z, beta_snake)))))
        beta = np.vstack([b, beta_snake])

        # 5. Synthetic variant 1
        lambdas = (np.matmul(X, beta) ** p).reshape(n)
        y = np.random.poisson(lambdas)

        # 6. Synthetic variant 2
        if self.variant == 2:
            logic = np.random.choice((False, True), n)
            y = np.ceil(lambdas)
            y[logic] = np.floor(lambdas)[logic]

        return X, y