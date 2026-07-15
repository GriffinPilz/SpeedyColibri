#!/usr/bin/env bash
# Container entrypoint for SpeedyColibri on DGX Spark.
#
# Passes arguments straight through to `coli`. The cluster layout is taken from
# the environment (single-node by default); colibri-cluster::ClusterConfig::from_env
# reads the same COLI_NUM_NODES / COLI_NODE_RANK variables.
set -euo pipefail

: "${COLI_NUM_NODES:=1}"
: "${COLI_NODE_RANK:=0}"

if [[ "${COLI_NUM_NODES}" != "1" ]]; then
  echo "[cluster] node ${COLI_NODE_RANK}/${COLI_NUM_NODES} (expert-parallel)" >&2
  # NOTE: multi-node transport (RDMA/RoCE over ConnectX-7) is not wired yet —
  # single-node is the current target. See DEPLOYMENT.md.
fi

exec coli "$@"
