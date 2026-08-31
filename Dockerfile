FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Batch paths set this themselves; the default keeps a container that has a
    # key configured from spending completions on prose nobody asked for.
    LLM_NARRATE=1

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
# Everything the image needs is pinned in requirements.txt — nothing is
# installed here that is not recorded there.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The data artifact and the trained model are committed, but rebuild them so a
# container is never serving a stale .joblib, and fail the BUILD rather than
# the first request if anything is inconsistent.
RUN python data/prepare_data.py && python baseline/train.py

EXPOSE 8501

# $PORT is set by Railway / Cloud Run; 8501 is the local default.
CMD streamlit run app.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true
