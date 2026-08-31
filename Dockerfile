FROM python:3.11-slim

# Batch paths (evaluate, cohort triage) switch narration off themselves, so
# this default only affects single-case runs, where the prose is wanted.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
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

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:'+__import__('os').environ.get('PORT','8501')+'/_stcore/health',timeout=4).read()==b'ok' else 1)"

# `exec` matters, and shell form is needed for ${PORT}.
#
# Without exec, /bin/sh stays PID 1 and Streamlit runs as its child, so Docker's
# SIGTERM goes to the shell and never reaches the app — `docker stop` then waits
# the full 10s grace period and SIGKILLs it. Measured: 10s before this change,
# under 1s after. On Cloud Run or Railway that difference is a slow, unclean
# shutdown on every single deploy.
CMD ["sh", "-c", "exec streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
