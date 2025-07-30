# docker logs -f compute_worker
# docker logs -f compute_worker_1
# docker logs -f compute_worker_2

docker stop compute_worker
docker rm compute_worker
docker pull codalab/competitions-v2-compute-worker:cpu1.1
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file .env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:cpu1.1

docker stop compute_worker_1
docker rm compute_worker_1
docker pull codalab/competitions-v2-compute-worker:cpu1.1
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file .env \
    --name compute_worker_1 \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:cpu1.1

docker stop compute_worker_2
docker rm compute_worker_2
docker pull codalab/competitions-v2-compute-worker:cpu1.1
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file .env \
    --name compute_worker_2 \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:cpu1.1
