# Backend image. The face model is downloaded at BUILD time, never at runtime.
#
# InsightFace fetches buffalo_l (~300MB) on first use by default. On a fresh
# container that means the first request with a photo pays a long download, and
# a restart pays it again — so the model is baked into the image here and the
# runtime is told where to find it. A build that cannot reach the model fails
# loudly instead of producing an image that will fail later on a user's request.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_HOME=/opt/insightface

# opencv and onnxruntime need these at runtime, not just to build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- bake the model -------------------------------------------------------
# Prepared with the same det_size the application uses, so the weights that get
# cached are the ones that get loaded. The assertion turns a silent partial
# download into a failed build.
RUN python -c "\
import insightface;\
app = insightface.app.FaceAnalysis(name='buffalo_l', root='/opt/insightface');\
app.prepare(ctx_id=-1, det_size=(640, 640));\
assert app.models, 'buffalo_l did not load';\
print('baked buffalo_l into the image')\
"

COPY app ./app
COPY scripts ./scripts
# Served by /fixtures. These are the cases a reviewer can run without spending
# a search, which matters more on a public deploy than it does locally.
COPY fixtures ./fixtures

# Everything written at runtime lives under /app/cache: responses, fetched
# images, and uploads (app/uploads.py puts UPLOADS_DIR inside CACHE_DIR). One
# directory, so one persistent disk mount covers all of it — see render.yaml.
# There is deliberately no /app/uploads: it existed, nothing used it, and an
# empty directory with the obvious name is an invitation to mount a disk in
# the wrong place.
RUN mkdir -p /app/cache

EXPOSE 8000

# Render supplies $PORT. One worker: the run guard and the in-flight run
# registry are both in-process, so a second worker would double the daily
# budget and lose half the run lookups.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
