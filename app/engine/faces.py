"""Face detection, embedding and comparison.

**Direction, stated once and loudly.** ``compare`` returns **cosine
similarity**: it runs from -1 to 1, and *higher means more alike*. Candidate
formation's merge signal is a **distance**, where *lower* means more alike.
:func:`similarity_to_distance` converts, and everything in this module that
crosses that boundary goes through it. Getting the sign backwards here would
merge strangers and split one person, silently and confidently, which is why
the two are named differently rather than sharing the word "score".

What a face match is worth, and what it is not:

* A face match is **direct evidence** about the person. A name match is not: a
  name is a label many people share. That asymmetry is why a face similarity
  above threshold anchors a record on its own, with no name and no context.
* **Absence is not disagreement.** A record with no image, an image that will
  not download, or an image with no detectable face yields *no signal* rather
  than a low one. None and 0.0 are different answers and are kept different.
* **Two faces in one photo make a record ambiguous.** Every similarity is
  recorded, the best is used, and the ambiguity is flagged so a reviewer can
  see that the match may be to the wrong person in a group shot.

These similarities are uncalibrated. The thresholds below are conventional
values for the model, not measurements against a labelled set for this task.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from app.adapters.base import AssertionDraft
from app.cache import cached_image

#: Predicate carrying an image a record showed.
IMAGE_PREDICATE = "image_url"


@dataclass(frozen=True)
class FaceEmbedding:
    """One detected face, as a unit-length vector."""

    vector: tuple[float, ...]
    det_score: float = 0.0
    bbox: tuple[float, float, float, float] | None = None

    def __repr__(self) -> str:
        return f"<FaceEmbedding dim={len(self.vector)} det={self.det_score:.2f}>"


@dataclass(frozen=True)
class FaceComparison:
    """The outcome of comparing one record's images against the subject.

    ``similarities`` holds every face-pair comparison made, so a group photo
    leaves a trail rather than a single number. ``ambiguous`` is true when more
    than one face was found in the record's image, which means the best match
    may belong to somebody standing next to the subject.
    """

    best_similarity: float
    similarities: tuple[float, ...] = ()
    ambiguous: bool = False
    faces_in_image: int = 0
    image_url: str | None = None

    @property
    def distance(self) -> float:
        """The merge signal's units: lower is more alike."""
        return similarity_to_distance(self.best_similarity)

    def __repr__(self) -> str:
        flag = " ambiguous" if self.ambiguous else ""
        return f"<FaceComparison similarity={self.best_similarity:.3f}{flag}>"


@dataclass(frozen=True)
class FaceConfig:
    """Model choice and the thresholds that decide what counts as a match."""

    model_name: str = "buffalo_l"
    #: Where model weights live. InsightFace defaults to ~/.insightface, which
    #: is the system drive and is not always where the space is. Set
    #: INSIGHTFACE_HOME to move it.
    model_root: str | None = None
    det_size: tuple[int, int] = (640, 640)
    #: Which buffalo_l models to load. The pack ships five; this system reads
    #: only a detection box, a detection score and an embedding, so the 3D
    #: landmark model (143MB on disk, ~150MB resident), the 2D landmark model
    #: and the gender/age classifier are never consulted.
    #:
    #: Loading all five costs 422MB resident against 271MB for these two, which
    #: is the difference between fitting and not fitting on a 512MB instance —
    #: a deployed run was killed mid-search for exactly this reason. The saving
    #: is incidental to a better argument: a system that scores identity has no
    #: business loading a gender classifier it never asks.
    allowed_modules: tuple[str, ...] = ("detection", "recognition")
    #: Detections below this confidence are discarded rather than compared.
    min_det_score: float = 0.50

    #: Cosine similarity at or above which two faces are treated as the same
    #: person. Higher is stricter. Conventional for this model rather than
    #: calibrated, and deliberately not shared with the anchor threshold below.
    same_person_similarity: float = 0.45

    #: Similarity at or above which a face match anchors a record by itself,
    #: with no name and no context. Set higher than same_person_similarity,
    #: because admitting a record on a face alone is a stronger claim than
    #: proposing a merge that other evidence can still overturn.
    anchor_alone_similarity: float = 0.55

    #: Images compared per record. A page may carry many; comparing all of them
    #: turns one record into an unbounded amount of work.
    max_images_per_record: int = 3


#: Given image bytes, return the faces found. Injected so this module can be
#: exercised without a model, and so a different detector can be substituted.
Embedder = Callable[[bytes], list[FaceEmbedding]]


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def compare(a: FaceEmbedding, b: FaceEmbedding) -> float:
    """Cosine **similarity** of two faces. Higher is more alike, range -1..1.

    Returns 0.0 for a zero-length vector, which is the "no information" value
    for cosine rather than a claim that the faces differ.
    """
    if len(a.vector) != len(b.vector):
        raise ValueError(
            f"embedding dimensions differ: {len(a.vector)} vs {len(b.vector)}"
        )
    dot = sum(x * y for x, y in zip(a.vector, b.vector))
    norm_a = math.sqrt(sum(x * x for x in a.vector))
    norm_b = math.sqrt(sum(y * y for y in b.vector))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def similarity_to_distance(similarity: float) -> float:
    """Convert a similarity to the distance the merge rule expects.

    The only place the sign flips. Formation asks "how far apart", this module
    measures "how alike", and one subtraction separates them.
    """
    return 1.0 - similarity


# --------------------------------------------------------------------------
# InsightFace
# --------------------------------------------------------------------------


class InsightFaceEmbedder:
    """buffalo_l detection and embedding, loaded on first use.

    The import and the model load are deferred because both are expensive and
    neither is needed to run the rest of the system. A missing InsightFace is
    reported as a clear error at call time rather than breaking every import.
    """

    def __init__(self, config: FaceConfig | None = None) -> None:
        self.config = config or FaceConfig()
        self._app = None

    def _load(self):
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "InsightFace is not installed. Install insightface and "
                "onnxruntime, or inject a different embedder."
            ) from exc

        root = self.config.model_root or os.environ.get("INSIGHTFACE_HOME")
        kwargs = {"name": self.config.model_name}
        if root:
            kwargs["root"] = root
        if self.config.allowed_modules:
            kwargs["allowed_modules"] = list(self.config.allowed_modules)
        app = FaceAnalysis(**kwargs)
        # ctx_id=-1 selects CPU. There is no GPU assumption anywhere here.
        app.prepare(ctx_id=-1, det_size=self.config.det_size)
        self._app = app
        return app

    def __call__(self, image_bytes: bytes) -> list[FaceEmbedding]:
        """Detect and embed every face in one image."""
        if not image_bytes:
            return []
        try:  # pragma: no cover - depends on environment
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "numpy and opencv are required to decode images for InsightFace."
            ) from exc

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            # Not a decodable image. No signal, rather than a bad one.
            return []

        faces = []
        for face in self._load().get(image):
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            if score < self.config.min_det_score:
                continue
            vector = getattr(face, "normed_embedding", None)
            if vector is None:
                vector = getattr(face, "embedding", None)
            if vector is None:
                continue
            bbox = getattr(face, "bbox", None)
            faces.append(
                FaceEmbedding(
                    vector=tuple(float(v) for v in vector),
                    det_score=score,
                    bbox=tuple(float(b) for b in bbox) if bbox is not None else None,
                )
            )
        return faces


_default_embedder: InsightFaceEmbedder | None = None


def embed(image_bytes: bytes, config: FaceConfig | None = None) -> list[FaceEmbedding]:
    """Detect and embed every face in an image. Empty list when there are none."""
    global _default_embedder
    if _default_embedder is None or config is not None:
        _default_embedder = InsightFaceEmbedder(config)
    return _default_embedder(image_bytes)


# --------------------------------------------------------------------------
# Matching records against a subject photo
# --------------------------------------------------------------------------


def image_urls(drafts: Iterable[AssertionDraft], config: FaceConfig) -> list[str]:
    """Distinct image URLs a record carries, in order, capped."""
    urls: list[str] = []
    for draft in drafts:
        if draft.predicate != IMAGE_PREDICATE:
            continue
        url = draft.raw_value or draft.normalized_value
        if url and url not in urls:
            urls.append(url)
        if len(urls) >= config.max_images_per_record:
            break
    return urls


class FaceMatcher:
    """Compares record images against one subject photo.

    Every stage can decline to answer. No photo, no image on the record, an
    image that will not download, an image with no face: each returns ``None``,
    meaning "no signal". Only an actual comparison produces a number.
    """

    def __init__(
        self,
        subject_photo_url: str | None = None,
        config: FaceConfig | None = None,
        embedder: Embedder | None = None,
        fetch: Callable[[str], bytes | None] = cached_image,
    ) -> None:
        self.config = config or FaceConfig()
        self._embedder = embedder or InsightFaceEmbedder(self.config)
        self._fetch = fetch
        self.subject_photo_url = subject_photo_url
        self.subject_faces: list[FaceEmbedding] = []
        self.subject_error: str | None = None

        if subject_photo_url:
            self._load_subject(subject_photo_url)

        # Embeddings are cached per URL for the life of the matcher, so a page
        # appearing under several records is embedded once.
        self._by_url: dict[str, list[FaceEmbedding]] = {}

    # --- subject ----------------------------------------------------------

    def _load_subject(self, url: str) -> None:
        body = self._fetch(url)
        if not body:
            self.subject_error = "subject photo could not be fetched"
            return
        try:
            faces = self._embedder(body)
        except Exception as exc:  # noqa: BLE001 - a broken model is not a match
            self.subject_error = f"{type(exc).__name__}: {exc}"
            return
        if not faces:
            self.subject_error = "no face detected in the subject photo"
            return
        self.subject_faces = faces

    @property
    def available(self) -> bool:
        """True when there is a subject face to compare against."""
        return bool(self.subject_faces)

    @property
    def subject_is_ambiguous(self) -> bool:
        """More than one face in the supplied photo is itself a problem."""
        return len(self.subject_faces) > 1

    # --- records ----------------------------------------------------------

    def _faces_for(self, url: str) -> list[FaceEmbedding]:
        if url in self._by_url:
            return self._by_url[url]
        body = self._fetch(url)
        faces: list[FaceEmbedding] = []
        if body:
            try:
                faces = self._embedder(body)
            except Exception:  # noqa: BLE001 - unreadable image, not a mismatch
                faces = []
        self._by_url[url] = faces
        return faces

    def match_record(
        self, drafts: Sequence[AssertionDraft]
    ) -> FaceComparison | None:
        """Best face match between the subject and this record's images.

        ``None`` means no comparison was possible. It never means "not a
        match": a record with no image is not evidence against the subject.
        """
        if not self.available:
            return None

        best: float | None = None
        best_url: str | None = None
        best_face_count = 0
        every: list[float] = []
        ambiguous = False

        for url in image_urls(drafts, self.config):
            faces = self._faces_for(url)
            if not faces:
                continue
            if len(faces) > 1:
                ambiguous = True
            for candidate_face in faces:
                for subject_face in self.subject_faces:
                    similarity = compare(subject_face, candidate_face)
                    every.append(similarity)
                    if best is None or similarity > best:
                        best = similarity
                        best_url = url
                        best_face_count = len(faces)

        if best is None:
            return None

        return FaceComparison(
            best_similarity=best,
            similarities=tuple(every),
            ambiguous=ambiguous,
            faces_in_image=best_face_count,
            image_url=best_url,
        )

    def image_matches(self, drafts: Sequence[AssertionDraft]) -> list[tuple[str, float]]:
        """Best similarity to the subject for each image a record carries.

        Per image rather than per record, because a page can show several faces
        and only some of them are the person. An image that will not download,
        or holds no detectable face, is absent from the result rather than
        present with a low score: it was never compared, and a caller must not
        be able to mistake "not checked" for "checked and failed".
        """
        if not self.available:
            return []

        results: list[tuple[str, float]] = []
        for url in image_urls(drafts, self.config):
            faces = self._faces_for(url)
            if not faces:
                continue
            best = max(
                compare(subject_face, candidate_face)
                for candidate_face in faces
                for subject_face in self.subject_faces
            )
            results.append((url, best))
        return results

    # --- candidates -------------------------------------------------------

    def candidate_distance(self, left, right) -> float | None:
        """Face **distance** between two candidates, for the merge rule.

        Lower means more alike, which is the opposite of everything else in
        this module. The conversion happens here, once.
        """
        best: float | None = None
        for left_url in image_urls(left.assertions, self.config):
            left_faces = self._faces_for(left_url)
            if not left_faces:
                continue
            for right_url in image_urls(right.assertions, self.config):
                if right_url == left_url:
                    continue
                right_faces = self._faces_for(right_url)
                for a in left_faces:
                    for b in right_faces:
                        similarity = compare(a, b)
                        if best is None or similarity > best:
                            best = similarity
        if best is None:
            return None
        return similarity_to_distance(best)

    def as_record_matcher(self) -> Callable[[Sequence[AssertionDraft]], FaceComparison | None]:
        """The callable candidate formation expects for anchoring."""
        return self.match_record

    def as_distance_function(self) -> Callable[[object, object], float | None]:
        """The callable candidate formation expects for merging."""
        return self.candidate_distance
