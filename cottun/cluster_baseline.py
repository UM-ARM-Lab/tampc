from sklearn.cluster import KMeans


class KMeansWithAutoK:
    """KMeans with automatic number of cluster selection based on a form of the elbow method"""

    def __init__(self, max_k=10, inertia_ratio=0.6, **kwargs):
        """inertia ratio (0,1) where higher value prefers more number of clusters as it tolerates smaller decreases"""
        self.max_k = max_k
        self.inertia_ratio = inertia_ratio
        self.kmeans_args = kwargs

    def fit(self, x):
        sse = []
        for k in range(1, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, **self.kmeans_args).fit(x)
            sse.append(kmeans.inertia_)
            if k > 1 and sse[-1] > self.inertia_ratio * sse[-2]:
                return KMeans(n_clusters=k - 1, **self.kmeans_args).fit(x)
        return KMeans(n_clusters=self.max_k, **self.kmeans_args).fit(x)
