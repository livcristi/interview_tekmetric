## My response

So, this is my response (Liviu-Cristian Terebeș) to the ML task. I split it into initially analysing the data and training a few models, followed
by the implementation to actually serve it as a possible microservice.

## Data Analysis and Model Training

In the `eda` folder, I added my Jupyter notebook which contains the exploratory data analysis for the received data,
highlighting the main found patterns, chosen approaches, with trainings and a summary for the chosen models. On short,
for an initial OOD detection I used the embeddings of a simple transformer model and a nearest-neighbor approach based
on the training data, since the "unknown" entries presented a different distribution than "known" entries. For the
classification itself, I also used a transformer based model (and trained a fully connected layer after it), since it
had the highest accuracy among the tested algorithms.

## Model Serving

After the analysis, I worked on writing the code for serving the trained model as a microservice using FastAPI as a web
server. The work is under the `src` folder and uses data from the `data` folder (the model weights and the items used
for the initial OOD detection / anomaly detection).

For serving the model and keeping a relatively fine software architecture, I split the logic into multiple classes,
following the SRP, but we basically have a `service\RepairService` class that, when receiving text (either single or in
batches), searches it through the cache (`cache\RedisCache`, based on Redis), then checks if it is an anomaly (using a
`similarity\SimilarityAnomalyDetector` based on the similarity distance) and then passes it through the model (
`models\EmbeddingsRepairClassifier`). I chose Redis for cache since it's a common approach that has multiple advantages
other plain in-memory methods (like scaling, higher accessibility, possibly disk persistence, etc.) but also left in an
in-memory cache. To start Redis locally, I added a `docker\docker-compose.yaml` file. The HTTP requests are handled by a
`FastAPI` app with `uvicorn` workers. The code also is configurable, with the possibility of switching the cache type,
thresholds or main embeddings model. For the dependency management, I also used `uv`, since it is a fantastic tool that
keeps gaining popularity for production environments. There are also some tests in the `tests` folder.

The main entry point is the `src\main.py` file.