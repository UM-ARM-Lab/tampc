from sklearn.cluster import KMeans
import copy
import numpy as np


class KMeansWithAutoK:
    """KMeans with automatic number of cluster selection based on a form of the elbow method"""

    def __init__(self, max_k=10, inertia_ratio=0.6, **kwargs):
        """inertia ratio (0,1) where higher value prefers more number of clusters as it tolerates smaller decreases"""
        self.max_k = max_k
        self.inertia_ratio = inertia_ratio
        self.kmeans_args = kwargs

    def fit(self, x):
        sse = []
        max_k = min(self.max_k, len(x))
        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k, **self.kmeans_args).fit(x)
            sse.append(kmeans.inertia_)
            if k > 1 and sse[-1] > self.inertia_ratio * sse[-2]:
                return KMeans(n_clusters=k - 1, **self.kmeans_args).fit(x)
        return KMeans(n_clusters=max_k, **self.kmeans_args).fit(x)


class OnlineSklearnContactSet:
    """Variant of contact set where we replace our clustering method with an sklearn based one

    One concern is that the clustering won't be stable between updates since the label ID is meaningless"""

    def __init__(self, cluster_method, inertia_ratio=0.8, delay_new_clusters_until=5):
        self.last_labels = None
        self.cluster_method = cluster_method
        # self.cluster_method = MiniBatchKMeans(n_clusters=1)
        self.inertia_ratio = inertia_ratio
        self.delay_new_clusters_until = delay_new_clusters_until

        self.data = None

    def update(self, x, u, dx):
        xx = x[:2].reshape(1, -1)
        if self.data is None:
            self.data = xx
            self.cluster_method.fit(self.data)
        else:
            self.data = np.concatenate([self.data, xx], axis=0)
            self._fit_online()

        # filter by moving all members of the current cluster by dx
        this_cluster = self.cluster_method.labels_[-1]

        # noise labels aren't actually clusters
        if this_cluster != -1:
            members_of_this_cluster = self.cluster_method.labels_ == this_cluster
            self.data[members_of_this_cluster, :2] += dx[:2]
        return self.cluster_method.labels_

    def _fit_online(self):
        raise NotImplementedError()

    def final_labels(self):
        if self.data is None:
            return []
        return self.cluster_method.predict(self.data)

    def moved_data(self):
        if self.data is None:
            return None
        return self.data[:, :2]


class OnlineSklearnFixedClusters(OnlineSklearnContactSet):
    def _fit_online(self):
        # decide if data should be part of existing clusters or if a new cluster should be created
        use_same_n_clusters = self.cluster_method
        use_more_n_clusters = copy.deepcopy(self.cluster_method)
        use_more_n_clusters.n_clusters += 1

        use_same_n_clusters.fit(self.data)
        use_more_n_clusters.fit(self.data)
        # using more cluster improved inertia sufficiently, so use new one
        if len(self.data) >= self.delay_new_clusters_until and \
                use_more_n_clusters.inertia_ < use_same_n_clusters.inertia_ * self.inertia_ratio:
            self.cluster_method = use_more_n_clusters
        else:
            self.cluster_method = use_same_n_clusters


class OnlineAgglomorativeClustering(OnlineSklearnContactSet):
    def _fit_online(self):
        # use as is, just rerun
        self.cluster_method.fit(self.data)

    def final_labels(self):
        if self.data is None:
            return []
        # some of these don't have predict for some reason
        return self.cluster_method.fit_predict(self.data)


def process_labels_with_noise(labels):
    if len(labels) == 0:
        return labels
    noise_label = max(labels) + 1
    for i in range(len(labels)):
        # some methods use -1 to indicate noise; in this case we have to assign a cluster so we use a single element
        if labels[i] == -1:
            labels[i] = noise_label
            noise_label += 1
    return labels
