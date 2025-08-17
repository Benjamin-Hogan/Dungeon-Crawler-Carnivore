import os
import urllib.request
import numpy as np

MODEL_URL = "https://github.com/openfoodfacts/off-nutrition-table-extractor/raw/master/nutrition_extractor/data/frozen_inference_graph.pb"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "frozen_inference_graph.pb")


class Detector:
    """TensorFlow object detector using OpenFoodFacts nutrition model."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._sess = None
        self._graph = None
        self._image_tensor = None
        self._boxes = None
        self._scores = None
        self._classes = None
        self._ensure_model()
        try:
            self._load()
        except Exception as exc:
            # defer heavy failures until detection time
            self._load_error = exc
        else:
            self._load_error = None

    def _ensure_model(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if not os.path.exists(self.model_path):
            urllib.request.urlretrieve(MODEL_URL, self.model_path)

    def _load(self) -> None:
        import tensorflow as tf  # lazily import
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(self.model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        self._graph = graph
        self._sess = tf.compat.v1.Session(graph=graph)
        self._image_tensor = graph.get_tensor_by_name("image_tensor:0")
        self._boxes = graph.get_tensor_by_name("detection_boxes:0")
        self._scores = graph.get_tensor_by_name("detection_scores:0")
        self._classes = graph.get_tensor_by_name("detection_classes:0")

    def detect(self, image: np.ndarray):
        """Return (boxes, scores, classes) for the given BGR image."""
        if self._load_error is not None:
            raise RuntimeError("Detector failed to load") from self._load_error
        if self._sess is None:
            self._load()
        expanded = np.expand_dims(image, axis=0)
        boxes, scores, classes = self._sess.run(
            [self._boxes, self._scores, self._classes],
            feed_dict={self._image_tensor: expanded},
        )
        return boxes[0], scores[0], classes[0]
