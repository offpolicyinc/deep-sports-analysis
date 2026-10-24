"""Google Cloud: step 2 (dsa.gpu.app's models) on a Compute Engine VM with an L4, instead of Modal. The bucket
gs://<project>-dsa is laid out as the data root (dsa.paths), like the Modal volume. The laptop pushes what the
videos need and starts a VM; the VM copies that to its own disk, runs the videos RUNS at a time, copies the results
back and deletes itself. The same command again, once the VM is gone, brings the results back (and starts a new VM
for what is still missing, after a spot VM was taken back).

The project's GPU quota is one GPU at a time across all regions (raising it was refused in Oct 2026), so one VM
takes all its videos. An L4 runs phone footage at about a third of real time (Modal's L40S: about two thirds).

    python -m dsa.gpu.gcp run others/2026_e,others/2026_g     # push, start the VM; again afterwards: collects
    python -m dsa.gpu.gcp status                                      # the VMs, and the tail of their logs
    python -m dsa.gpu.gcp ls projects/others
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dsa.paths import ROOT

PROJECT = os.environ.get("DSA_GCP_PROJECT", "wc3env-pw")
BUCKET = f"gs://{PROJECT}-dsa"
REGION = "us-central1"                            # the bucket; VMs go wherever an L4 is free (US egress: ~$0.01/GB)
ZONES = ["us-central1-a", "us-central1-b", "us-central1-c", "us-east4-a", "us-east4-c", "us-east1-b", "us-east1-c",
         "us-east1-d", "us-west1-a", "us-west1-b", "us-west1-c", "us-west4-a", "us-west4-c"]
MACHINE, GPU = "g2-standard-16", "nvidia-l4"      # 16 cores, 64 GB: ffmpeg decodes and crops for two runs at once
IMAGE = ("deeplearning-platform-release", "common-cu129-ubuntu-2204-nvidia-580")   # NVIDIA driver 580: CUDA 13
RUNS = 2                                          # videos side by side on the one GPU
SPOT = False        # spot: about a third of the price, but on 23 Oct 2026 us-central1 took two back within
                    # 12 minutes, before a 70-min video could finish; `run` again starts what's left either way
GPU_EXTRA = ["onnxruntime-gpu==1.30.0", "timm==0.6.13", "tabulate==0.9.0"]   # as dsa.gpu.app's Modal image
SKIP = r"(^|.*/)(\.git|__pycache__|\.DS_Store|annotation-tool)(/.*|$)|(^|.*/)\._[^/]*$"
REPO = Path(__file__).resolve().parents[3]

STARTUP = r"""#!/bin/bash
# The worker: Python and the project's pinned packages, the inputs from the bucket, step 2, results back, then gone.
set -x
meta() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
BUCKET=$(meta attributes/dsa-bucket); NAME=$(meta name); ZONE=$(meta zone | cut -d/ -f4)
LOG=/var/log/dsa.log
exec > >(tee -a $LOG) 2>&1
( while true; do sleep 60; gcloud storage cp -q $LOG $BUCKET/logs/gcp/$NAME.log; done ) &
finish() { gcloud storage cp -q $LOG $BUCKET/logs/gcp/$NAME.log; gcloud compute instances delete $NAME --zone $ZONE -q; }
trap finish EXIT
for i in $(seq 30); do nvidia-smi && break; sleep 10; done
apt-get update -q && apt-get install -yq ffmpeg libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
mkdir -p /opt/dsa /vol && cd /opt/dsa
gcloud storage rsync -r $BUCKET/dsa /opt/dsa
gcloud storage cp $BUCKET/logs/gcp/$NAME.jobs.json /opt/dsa/jobs.json
# the installed environment, kept in the bucket per requirements: later VMs unpack it instead of installing
ENV=$BUCKET/env/$(cat requirements.txt gpu.txt | sha1sum | cut -c1-12).tar
if gcloud storage cp $ENV /tmp/env.tar; then
  tar -xf /tmp/env.tar -C / && rm /tmp/env.tar
else
  uv venv -p 3.11 .venv
  VIRTUAL_ENV=/opt/dsa/.venv uv pip install -r requirements.txt
  VIRTUAL_ENV=/opt/dsa/.venv uv pip uninstall onnxruntime onnxruntime-gpu
  VIRTUAL_ENV=/opt/dsa/.venv uv pip install --no-deps $(cat gpu.txt)
  tar -cf /tmp/env.tar /opt/dsa/.venv /root/.local/share/uv/python && gcloud storage cp /tmp/env.tar $ENV && rm /tmp/env.tar
fi
. .venv/bin/activate
export DSA_DATA=/vol RF_HOME=/vol/models/rfdetr PYTHONPATH=/opt/dsa/src
python -m dsa.gpu.gcp worker /opt/dsa/jobs.json --bucket $BUCKET
"""


def gs(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", "storage", *args], check=check, text=True, capture_output=quiet)


def listing(rel: str, bucket: str = BUCKET) -> set[str]:
    """Every object under <bucket>/<rel>, as paths relative to the bucket (empty when there is nothing)."""
    r = gs("ls", "-r", f"{bucket}/{rel}", check=False, quiet=True)
    n = len(bucket) + 1
    return {x[n:] for x in r.stdout.split() if x.startswith(bucket) and not x.endswith(("/", ":"))}


def push(rels: list[str]) -> None:
    """Upload ROOT/<rel> (a file or a folder) to the bucket, only what is missing or changed there (rsync)."""
    for rel in dict.fromkeys(rels):
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing on the laptop (docs/STORAGE.md)")
        if src.is_dir():
            gs("rsync", "-r", "-q", "-x", SKIP, str(src), f"{BUCKET}/{rel}")
        elif rel not in listing(rel):
            gs("cp", "-q", str(src), f"{BUCKET}/{rel}")


def pull(rel: str) -> Path:
    """Bucket -> ROOT/<rel>, streamed (gcloud's cp hung finishing its download: a tracker file left locked by
    an interrupted copy of the same object), named <rel> only once whole."""
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "wb") as fh:
        subprocess.run(["gcloud", "storage", "cat", f"{BUCKET}/{rel}"], stdout=fh, check=True)
    tmp.rename(out)
    return out


def push_code() -> None:
    """The dsa package and its pinned dependencies (pyproject.toml), for the VM's venv."""
    import tomllib

    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    stage = ROOT / "logs/gcp/dsa"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "requirements.txt").write_text("\n".join(deps + GPU_EXTRA) + "\n")
    (stage / "gpu.txt").write_text(GPU_EXTRA[0] + "\n")
    gs("rsync", "-r", "-q", "-x", SKIP, str(stage), f"{BUCKET}/dsa")
    gs("rsync", "-r", "-q", "--delete-unmatched-destination-objects", "-x", SKIP + r"|.*\.egg-info/.*", str(REPO / "src"), f"{BUCKET}/dsa/src")


def vm(name: str, jobs: list[tuple]) -> str:
    """Start the worker VM in the first zone with an L4 free, spot first; returns where."""
    spec = [[j[0], str(j[1]), *j[2:]] for j in jobs]                   # (spec, rel, name, ..., people, out)
    tmp = ROOT / "logs/gcp" / f"{name}.jobs.json"
    tmp.write_text(json.dumps(spec))
    gs("cp", "-q", str(tmp), f"{BUCKET}/logs/gcp/{name}.jobs.json")
    script = ROOT / "logs/gcp/startup.sh"
    script.write_text(STARTUP)
    for spot, zone in [(True, z) for z in ZONES] * SPOT + [(False, z) for z in ZONES]:  # on demand when no spot is free
        r = subprocess.run(["gcloud", "compute", "instances", "create", name, f"--project={PROJECT}", f"--zone={zone}",
                            f"--machine-type={MACHINE}", f"--accelerator=type={GPU},count=1",
                            "--maintenance-policy=TERMINATE",
                            *(["--provisioning-model=SPOT", "--instance-termination-action=DELETE"] if spot else []),
                            f"--image-project={IMAGE[0]}", f"--image-family={IMAGE[1]}",
                            "--boot-disk-size=200GB", "--boot-disk-type=pd-ssd", "--scopes=cloud-platform",
                            f"--metadata=dsa-bucket={BUCKET},install-nvidia-driver=True",
                            f"--metadata-from-file=startup-script={script}", "--labels=app=dsa"],
                           text=True, capture_output=True)
        if r.returncode == 0:
            return f"{zone} ({'spot' if spot else 'on demand'})"
        print(zone, "spot" * spot, r.stderr.strip().splitlines()[-1][-90:], flush=True)
    sys.exit("no zone had an L4 free")


def bucket() -> None:
    if subprocess.run(["gcloud", "storage", "buckets", "describe", BUCKET], capture_output=True).returncode:
        gs("buckets", "create", BUCKET, f"--project={PROJECT}", f"--location={REGION}", "--uniform-bucket-level-access")


def running() -> list[str]:
    r = subprocess.run(["gcloud", "compute", "instances", "list", f"--project={PROJECT}", "--filter=labels.app=dsa",
                        "--format=value(name)"], text=True, capture_output=True, check=True)
    return r.stdout.split()


def run(videos: list[str], seconds: float = 0) -> None:
    """Start a VM for what the bucket lacks (none while one is running), and bring back what it has."""
    from dsa.gpu.app import collect, inputs, plan_all

    bucket()
    out = "test" if seconds else "."
    jobs, todo = plan_all(videos, seconds, False, listing)
    busy = running()
    print(f"{len(jobs)} to run: {[(j[0], 'strokes' * j[6], 'people' * j[7]) for j in jobs]}; running: {busy}",
          flush=True)
    if jobs and not busy:
        push_code()
        push(inputs(jobs))
        name = f"dsa-{time.strftime('%m%d-%H%M')}"
        print(name, "started in", vm(name, [j + (out,) for j in jobs]), flush=True)
    for v, (lengths, fps) in todo.items():
        print(v, "collected", collect(v, out, lengths, fps, listing, pull), flush=True)


def worker(jobs_file: str, bucket_url: str) -> None:
    """On the VM: each job's inputs from the bucket, dsa.gpu.app.models RUNS at a time (one process each), results
    up as each finishes. A job that fails is tried once more on its own."""
    from concurrent.futures import ThreadPoolExecutor

    from dsa.gpu.app import inputs

    jobs = [tuple(j) for j in json.loads(Path(jobs_file).read_text())]
    for rel in inputs([(j[0], Path(j[1]), *j[2:]) for j in jobs]):
        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        gs("cp", "-r", "-q", f"{bucket_url}/{rel}", str(dst.parent) + "/")

    def one(j) -> bool:
        spec, rel, out = j[0], j[1], j[8]
        t = time.time()
        r = subprocess.run([sys.executable, "-m", "dsa.gpu.gcp", "one", json.dumps(j)])
        res = Path(ROOT / rel / out)
        for f in ("raw.npz", "strokes/scores.npz"):
            if (res / f).exists():
                gs("cp", "-q", str(res / f), f"{bucket_url}/{Path(rel) / out / f}")
        print(spec, "exit", r.returncode, f"{time.time() - t:.0f} s", flush=True)
        return r.returncode == 0

    with ThreadPoolExecutor(RUNS) as pool:
        ok = list(pool.map(one, jobs))
    for j, k in zip(jobs, ok):
        if not k:
            one(j)


def one(job: str) -> None:
    from dsa.gpu.app import models

    spec, _, name, lengths, fps, tv, strokes, people, out = json.loads(job)
    print(spec, models(spec, name, lengths, fps, out, tv, strokes, people), flush=True)


def status() -> None:
    for n in running():
        print("running:", n)
    for rel in sorted(listing("logs/gcp")):
        if rel.endswith(".log"):
            print(f"== {rel}")
            print("\n".join(gs("cat", f"{BUCKET}/{rel}", quiet=True).stdout.splitlines()[-15:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "ls", "worker", "one"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--bucket", default=BUCKET)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.arg.split(","), a.seconds)
    elif a.cmd == "status":
        status()
    elif a.cmd == "ls":
        print("\n".join(sorted(listing(a.arg))))
    elif a.cmd == "worker":
        worker(a.arg, a.bucket)
    else:
        one(a.arg)
