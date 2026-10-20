# ragdoc in a container: the browser UI over a folder of your documents.
#
#   docker build -t ragdoc .
#   docker run --rm -p 5000:5000 -e ANTHROPIC_API_KEY -v "$PWD/docs:/docs" ragdoc
#
# Then open http://localhost:5000

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAGDOC_HOST=0.0.0.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the container starts
# without a ~30 MB download and works offline.
RUN python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"

COPY *.py ./

EXPOSE 5000

# Mount your documents at /docs. Pass a different path to index something else.
ENTRYPOINT ["python", "web.py"]
CMD ["/docs"]
