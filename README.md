# Unawesome AI Fabric Engineering

> Networking that moves data between GPUs in AI clusters: RDMA, GPU-to-NIC data paths, collectives, transports, and cluster fabrics.

Written for GPU performance engineers moving into the network.

It assumes you already know GPU kernels, profiling, inference engines, and the basics of distributed inference. The list is ordered from one NIC to one GPU-NIC path, collectives, inference transfer, transports, and whole fabrics. Read **Start here** first. After that, use it as a reference.

Every resource assumes real NICs, switches, and GPUs. See [Footnotes](#footnotes).

## Contents

- [Start here: the minimum mental model](#start-here-the-minimum-mental-model)
- [1. RDMA fundamentals](#1-rdma-fundamentals)
  - [Verbs and memory registration](#verbs-and-memory-registration)
  - [NIC behavior](#nic-behavior)
  - [Measurement](#measurement)
- [2. The GPU-to-NIC data path](#2-the-gpu-to-nic-data-path)
  - [GPUDirect and PCIe](#gpudirect-and-pcie)
  - [GPU-initiated networking](#gpu-initiated-networking)
- [3. Collectives](#3-collectives)
  - [Algorithms](#algorithms)
  - [Libraries](#libraries)
- [4. Point-to-point transfer for inference](#4-point-to-point-transfer-for-inference)
- [5. Transport and congestion control](#5-transport-and-congestion-control)
- [6. Fabric design for AI clusters](#6-fabric-design-for-ai-clusters)
- [7. Scale-up fabrics](#7-scale-up-fabrics)
- [8. Cloud fabrics](#8-cloud-fabrics)
- [9. Operating fabrics at scale](#9-operating-fabrics-at-scale)
- [Frontier](#frontier)

## Start here: the minimum mental model

Read these in order.

- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) - NVLink, NVSwitch, scale-out InfiniBand, and collective costs from the GPU's point of view.
- [RDMA Aware Networks Programming User Manual](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17) - Verbs objects, queue pairs, memory registration, and transport types on NVIDIA NICs.
- [Design Guidelines for High Performance RDMA Systems](https://www.usenix.org/conference/atc16/technical-sessions/presentation/kalia) - How NIC caches, PCIe transactions, and verb choice decide RDMA throughput.
- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/) - How a PCIe device reads and writes GPU memory directly, and what the driver must pin.
- [RDMA over Commodity Ethernet at Scale](https://doi.org/10.1145/2934872.2934908) - RoCEv2 with PFC in a production datacenter, including deadlocks and pause storms.
- [RDMA over Ethernet for Distributed Training at Meta Scale](https://doi.org/10.1145/3651890.3672233) - A RoCE fabric built for AI training: topology, routing, and congestion choices.
- [Demystifying NCCL](https://arxiv.org/abs/2507.04786) - NCCL's protocols, channels, and ring and tree algorithms, measured.
- [NCCL performance methodology](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) - Algorithm bandwidth versus bus bandwidth, and how to compare a collective against link speed.

## 1. RDMA fundamentals

### Verbs and memory registration

- [rdma-core](https://github.com/linux-rdma/rdma-core) - The userspace verbs library, NIC providers, and man pages.
- [ibv_reg_mr(3)](https://man7.org/linux/man-pages/man3/ibv_reg_mr.3.html) - Memory regions, access flags, on-demand paging, and dma-buf registration.
- [ibv_advise_mr(3)](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_advise_mr.3.md) - Prefetching on-demand paging translations before the hot path.
- [Page Fault Support for Network Controllers](https://doi.org/10.1145/3037697.3037710) - The design behind on-demand paging in RDMA NICs.
- [Linux InfiniBand subsystem](https://docs.kernel.org/infiniband/index.html) - Kernel-side user verbs, MAD access, and core interfaces.

### NIC behavior

- [Mellanox Adapters Programmer's Reference Manual](https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf) - The ConnectX command interface, work queue formats, and offloads that later ConnectX parts extend.
- [FaSST](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/kalia) - Two-sided datagram RPCs over RDMA, and when they beat one-sided reads.
- [eRPC: Datacenter RPCs can be General and Fast](https://www.usenix.org/conference/nsdi19/presentation/kalia) - Fast RPCs over lossy fabrics with congestion control in software.
- [Collie](https://www.usenix.org/conference/nsdi22/presentation/kong) - A systematic search for RDMA NIC performance anomalies.
- [Husky: Understanding RDMA Microarchitecture Resources for Performance Isolation](https://www.usenix.org/conference/nsdi23/presentation/kong) - Shared NIC resources that let one tenant degrade another.

### Measurement

- [perftest](https://github.com/linux-rdma/perftest) - `ib_write_bw`, `ib_read_lat`, and the standard verbs benchmarks.
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) - The standard collective benchmarks.
- [InfiniBand Architecture Specification](https://www.infinibandta.org/ibta-specification/) - How to obtain the specification that defines InfiniBand and RoCE transport.

## 2. The GPU-to-NIC data path

### GPUDirect and PCIe

- [Linux dma-buf](https://docs.kernel.org/driver-api/dma-buf.html) - The kernel buffer-sharing mechanism used to register GPU memory without a peer-memory module.
- [PCI peer-to-peer DMA](https://docs.kernel.org/driver-api/pci/p2pdma.html) - Kernel support and constraints for device-to-device DMA across PCIe.
- [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) - PCI ACS, IOMMU, and platform settings that silently break or slow GPU-NIC paths.
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) - Topology files, NIC selection, GPUDirect levels, and InfiniBand settings.

### GPU-initiated networking

- [NVSHMEM and GPUDirect Async](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/) - InfiniBand GPUDirect Async: GPU threads post RDMA work to the NIC directly.
- [NVSHMEM](https://docs.nvidia.com/nvshmem/api/index.html) - The GPU-initiated communication library behind it.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) - Expert dispatch and combine over NVLink and RDMA, including GPU-initiated low-latency kernels.
- [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html) - GPU-controlled packet I/O on ConnectX NICs.
- [Device Memory TCP](https://docs.kernel.org/networking/devmem.html) - TCP that receives payloads directly into device memory.

## 3. Collectives

### Algorithms

- [Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521) - Ring, recursive doubling, and Rabenseifner algorithms with their cost models.
- [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf) - Why ring all-reduce reaches the bandwidth lower bound.
- [TACCL](https://www.usenix.org/conference/nsdi23/presentation/shah) - Synthesizing topology-aware collective algorithms.
- [Swing](https://www.usenix.org/conference/nsdi24/presentation/de-sensi) - All-reduce on torus fabrics by short-cutting rings.
- [SHArP](https://doi.org/10.1109/COMHPC.2016.006) - In-network reduction inside InfiniBand switches.

### Libraries

- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) - The reference for NVIDIA collectives.
- [NCCL net plugins](https://github.com/NVIDIA/nccl/tree/master/plugins/net) - The interface a transport implements to carry NCCL traffic.
- [NCCL 2.23 scaling algorithm](https://developer.nvidia.com/blog/new-scaling-algorithm-and-initialization-with-nvidia-collective-communications-library-2-23/) - Parallel Aggregated Trees for all-gather and reduce-scatter with small and medium messages at scale.
- [Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171) - Meta's collective stack at 100k-GPU scale.
- [MSCCL++](https://arxiv.org/abs/2504.09014) and [repository](https://github.com/microsoft/mscclpp) - GPU-driven communication primitives for inference.
- [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/) - AMD's NCCL-compatible collectives.
- [UCX](https://github.com/openucx/ucx) - The transport framework under many MPI and inference-transfer backends.

## 4. Point-to-point transfer for inference

- [NIXL](https://github.com/ai-dynamo/nixl) - Transfers inference state across GPU, CPU, storage, and network backends.
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - The RDMA transfer engine and KV store behind Kimi serving.
- [fabric-lib: RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656) - One-sided writes with completion counters at 400 Gbps on both ConnectX-7 and EFA, used for KV transfer, RL weight updates, and MoE routing.
- [pplx-kernels](https://github.com/perplexityai/pplx-kernels) - Expert-parallel all-to-all kernels.
- [UCCL](https://github.com/uccl-project/uccl) - Collectives, point-to-point, and expert parallelism across NIC vendors.
- [Insights into DeepSeek-V3](https://arxiv.org/abs/2505.09343) - Multi-plane fat trees and network-hardware co-design for MoE.
- [PD disaggregation and large-scale EP on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/) - A measured multi-node deployment of DeepSeek with KV transfer and expert parallelism.
- [llm-d PD disaggregation guide](https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation) - A deployable prefill/decode split with RDMA KV transfer.
- [llm-d on AMD Instinct over RoCE](https://www.amd.com/en/developer/resources/technical-articles/2026/llm-d-serving-for-amd-instinct-gpus-on-oci.html) - Prefill/decode disaggregation tuned on MI300X bare metal over RoCEv2.

## 5. Transport and congestion control

- [DCQCN](https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p523.pdf) - ECN-based congestion control for RoCEv2.
- [TIMELY](https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p537.pdf) - RTT-based congestion control in the datacenter.
- [HPCC](https://liyuliang001.github.io/publications/hpcc.pdf) - Congestion control from in-network telemetry.
- [Swift](https://research.google/pubs/swift-delay-is-simple-and-effective-for-congestion-control-in-the-datacenter/) - Delay-based congestion control in production.
- [Revisiting Network Support for RDMA](https://arxiv.org/abs/1806.08159) - RDMA without a lossless fabric.
- [SRD](https://doi.org/10.1109/MM.2020.3016891) - AWS's multipath, out-of-order reliable datagram transport behind EFA.
- [Falcon](https://github.com/opencomputeproject/OCP-NET-Falcon) - Google's hardware-offloaded reliable transport, published through OCP.
- [Ultra Ethernet 1.0.3 Specification](https://ultraethernet.org/wp-content/uploads/sites/20/2026/08/UE-Specification-1.0.3.pdf) - The scale-out transport specification.

## 6. Fabric design for AI clusters

- [Building Meta's GenAI Infrastructure](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/) - Two 24k-GPU clusters, one on RoCE and one on InfiniBand.
- [Alibaba HPN](https://ennanzhai.github.io/pub/sigcomm24-hpn.pdf) - A dual-plane, two-tier network for LLM training.
- [Rail-only](https://arxiv.org/abs/2307.12169) - Dropping the spine when traffic stays within rails.
- [Fire-Flyer AI-HPC](https://arxiv.org/abs/2408.14158) - A cost-driven fabric and all-reduce co-design.
- [Jupiter Evolving](https://research.google/pubs/jupiter-evolving-transforming-googles-datacenter-network-via-optical-circuit-switches-and-software-defined-networking/) - Optical circuit switches in a production datacenter network.
- [DGX SuperPOD reference architecture](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/) - Compute, storage, and management fabrics for an H100 SuperPOD.

## 7. Scale-up fabrics

- [Multi-node NVLink tuning guide](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/) - NVLink domains and their InfiniBand boundary in GB200 NVL systems.
- [MI300X acceptance guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html) - Node-level validation for AMD Instinct MI300X systems.
- [UALink 1.0 Specification](https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink200_Specification_v1.0_Evaluation_Copy.pdf) - An open scale-up interconnect.

## 8. Cloud fabrics

- [Elastic Fabric Adapter](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) - AWS's OS-bypass network for AI and HPC.
- [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) - The NCCL plugin that carries collectives over libfabric and EFA.
- [libfabric](https://github.com/ofiwg/libfabric) - The Open Fabric Interfaces library and its providers.
- [Azure InfiniBand setup](https://learn.microsoft.com/en-us/azure/virtual-machines/setup-infiniband) - InfiniBand on Azure HPC and GPU VMs.
- [GPUDirect on Google Cloud A3](https://cloud.google.com/compute/docs/gpus/gpudirect) - GPUDirect-TCPX and TCPXO on A3 instances.
- [OCI cluster networks](https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/managingclusternetworks.htm) - RDMA cluster networking for bare-metal GPU shapes.

## 9. Operating fabrics at scale

- [MegaScale](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng) - Training on more than 10,000 GPUs, with network diagnosis and fault tolerance.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) - The infrastructure section: network, collectives, and interruptions at 16k GPUs.
- [C4](https://arxiv.org/abs/2406.04594) - Real-time detection of communication anomalies in large training jobs.
- [SuperBench](https://www.usenix.org/conference/atc24/presentation/xiong) - Proactive validation of cloud AI infrastructure.
- [Understanding Stragglers in Large Model Training](https://arxiv.org/abs/2505.05713) - What-if analysis of straggler causes from production traces.

## Frontier

Verified on **2026-10-05**. Kept separate from the core list because the evidence changes quickly.

### Watchlist

- Ultra Ethernet NICs and switches in production, pending measured deployments.
- Adaptive routing and packet spraying on Ethernet AI fabrics, pending reproducible measurements.
- Co-packaged optics switches in deployed GPU clusters.
- UALink and Ethernet-based scale-up silicon, pending shipped systems.
- GPU-initiated networking on non-NVIDIA NICs.
- KV-cache transfer across heterogeneous GPU fleets with measured tail latency.

## Footnotes

### Hardware policy

Every resource assumes real NICs, switches, and GPUs. Software RDMA emulation and network simulators are excluded: their numbers do not transfer, and they cannot exercise GPUDirect, NIC offloads, congestion control, or adaptive routing.

### Source policy

A core source must be one of the following:

- the paper that introduced the mechanism;
- the specification or official documentation that defines it;
- the repository that implements it;
- an operator or implementer report with hardware, measurements, and enough detail to reproduce the result.

Performance claims need the NIC, switch, topology, message sizes, software versions, and baseline. Otherwise the number is omitted.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a resource. `python3 scripts/check_links.py` checks every link.
