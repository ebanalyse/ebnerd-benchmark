# Running the RecSys'24 Challenge on Codabench with Docker

The **RecSys'24 Challenge** was hosted on [Codabench](https://www.codabench.org/) (huge shoutout to their team!). Each submission during the challenge was evaluated using a **hidden test set** on compute workers. These workers were hosted on virtual machines provided by [Ekstra Bladet](https://ekstrabladet.dk/) and [JP/Politikens Media Group](https://jppol.dk/en/).

**Competition site:** [www.codabench.org/competitions/2469](https://www.codabench.org/competitions/2469/)

Now that the challenge has concluded, the original virtual machines are no longer active. However, **you can reproduce the setup on your own machine**—whether it’s a physical computer or a cloud-based VM. You can even add multiple compute workers to a queue to process submissions simultaneously.

---

## Before You Begin

We **highly recommend reading the official Codabench documentation** for detailed instructions:

- **[Compute Worker Management Setup](https://github.com/codalab/codabench/wiki/Compute-Worker-Management---Setup#setup-compute-worker)**  
- Got to the tab **[Submission Guidelines](https://www.codabench.org/competitions/2469/)**  

The instructions below outline **how we ran it during the challenge** using **Docker**.

Please ensure your current working directory is:
```bash
cd codabench
```
We are running the compute workers on a **Linux operating system** on a the AWS `t3.xlarge`.

Once the Docker container is running, your submission will be evaluated. **Important**: Remember to follow the submission guidelines strictly.

The evaluation process can take several hours (e.g., 4–5 hours). Keep an eye on your worker, as technical issues can occur and you may need to reset it. You can view error logs directly on Codabench:
`Submissions => LOGS => Scoring Logs`, this is extremely useful for debugging.

---

# Step-by-Step Setup

## 1. Install Docker

Run the following commands to install Docker and ensure your user is added to the Docker group:

```bash
curl https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```
Log in to the new docker group:
```bash
newgrp docker
```

## 2. Pull the Codabench Compute Worker Image
```bash
docker pull codalab/competitions-v2-compute-worker:cpu1.1
```

## 3. Start a CPU Worker
We ran using CPU workers. You will need an `.env` file containing the `BROKER_URL` required to connect to the challenge.

## 4. Run the Docker Container
```bash
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
```bash
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
```
## 7. Running Multiple Workers
To add additional workers, simply change the container name (e.g., compute_worker_1):
```bash
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
```
For convenience, we have also included a script (`codabench_docker.sh`) that starts **3 workers automatically**.  
You can run it with:

```bash
source codabench_docker.sh
```

# Issues
As [Codabench](https://www.codabench.org/) is constantly evolving, we have experienced issues, such as remote worker connections being dropped or temporary updates on the site affecting our competition. We recommend reaching out and creating a [Git issue](https://github.com/codalab/codabench/issues) if you identify something unusual in your logs. The Codabench team has always been extremely helpful, and we have always managed to find a solution. Examples include:
[#1951](https://github.com/codalab/codabench/issues/1951),
[#1677](https://github.com/codalab/codabench/issues/1677),
[#1471](https://github.com/codalab/codabench/issues/1471),
[#1469](https://github.com/codalab/codabench/issues/1469),
[#1455](https://github.com/codalab/codabench/issues/1455), and
[#1369](https://github.com/codalab/codabench/issues/1369)

