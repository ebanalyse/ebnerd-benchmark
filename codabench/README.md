# Running the RecSys'24 Challenge on Codabench with Docker

The **RecSys'24 Challenge** was hosted on [Codabench](https://www.codabench.org/) (huge shoutout to their team!).  
Each submission during the challenge was evaluated using a **hidden test set** on compute workers. These workers were hosted on virtual machines provided by **[Ekstra Bladet](https://ekstrabladet.dk/)** and **[JP/Politikens Media Group](https://jppol.dk/en/)**.

**Competition site:** [https://www.codabench.org/competitions/2469/](https://www.codabench.org/competitions/2469/)

Now that the challenge has concluded, the original virtual machines are no longer active.  
However, **you can reproduce the setup on your own machine**—whether it’s a physical computer or a cloud-based VM.  
You can even add multiple compute workers to a queue to process submissions simultaneously.

---

## Before You Begin

We **highly recommend reading the official Codabench documentation** for detailed instructions:

- **[Compute Worker Management Setup](https://github.com/codalab/codabench/wiki/Compute-Worker-Management---Setup#setup-compute-worker)**  
- **[Submission Guide](https://www.codabench.org/competitions/2469/)**  

The instructions below outline **how we ran it during the challenge** using **Docker**.

---

# Step-by-Step Setup

## 1. Install Docker

Run the following commands to install Docker and ensure your user is added to the Docker group:

```bash
curl https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

## 2. Pull the Codabench Compute Worker Image
```bash
docker pull codalab/competitions-v2-compute-worker:latest
```

## 3. Start a CPU Worker
We ran using CPU workers.
You will need an `.env` file containing the `BROKER_URL` required to connect to the challenge.
Note, in this repository, we named it simply `env`.

## 4. Run the Docker Container
```bash
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:latest
```

## 5. Check Logs and Running Jobs
To view logs:
```bash
docker logs -f compute_worker
```

To check running jobs:
```bash
docker ps
```

## 6. Update the Docker Image
docker stop compute_worker
docker rm compute_worker
docker pull codalab/competitions-v2-compute-worker:latest 
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file .env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:latest

## 7. Running Multiple Workers
To add additional workers, simply change the container name (e.g., compute_worker_1):
```bash
docker stop compute_worker_1
docker rm compute_worker_1
docker pull codalab/competitions-v2-compute-worker:latest
docker run \
    -v /codabench:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file .env \
    --name compute_worker_1 \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:latest
```
For convenience, we have also included a script (`codabench_docker.sh`) that starts **3 workers automatically**.  
You can run it with:

```bash
source codabench_docker.sh
```