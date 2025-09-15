# High Performance Computing (HPC) Theory

## Index

+ [Intro to course](#intro-to-course)
+ [HPC Hardware and Software](#hpc-hardware-and-software)
+ [HPC Software Stack](#hpc-software-stack)
+ [Code Optimization Fundamentals](#code-optimization-fundamentals)
+ [Modern CPU Architecture](#modern-cpu-architecture)
+ [Practical Code Optimization](#practical-code-optimization)
+ [Memory Management and Execution Model](#memory-management-and-execution-model)
+ [Cache Optimization and Data Locality](#cache-optimization-and-data-locality)
+ [Loop Optimization and Vectorization](#loop-optimization-and-vectorization)
+ [Advanced Loop Techniques and Prefetching](#advanced-loop-techniques-and-prefetching)
+ [Introduction to Parallel Computing](#introduction-to-parallel-computing)
+ [OpenMP Fundamentals](#openmp-fundamentals)
+ [OpenMP Parallel Regions](#openmp-parallel-regions)
+ [Parallel Algorithm Analysis](#parallel-algorithm-analysis)
+ [OpenMP Parallel Loops](#openmp-parallel-loops)
+ [OpenMP NUMA Awareness](#openmp-numa-awareness)
+ [MPI Fundamentals](#mpi-fundamentals)
+ [MPI Point-to-Point Communication](#mpi-point-to-point-communication)
+ [MPI Collective Communication](#mpi-collective-communication)
+ [HPC Profiling and Analysis](#hpc-profiling-and-analysis)
+ [HPC Debugging Techniques](#hpc-debugging-techniques)
+ [Performance Monitoring and Hardware Counters](#performance-monitoring-and-hardware-counters)
+ [Master's index](https://github.com/DottorBooom/Master-in-Data-Science-and-Artificial-Intelligence) 

## Introduction

+ It will be entirely in English, slides made by the professor on which I will take notes lesson after lesson.
+ The final exam will consist of a project assigned by the professor and a oral exam.
+ This exam will be the second of the two modules that make up the entire course, the second is called "Introduction to Cloud Computning". You can find it on my profile.

The professor have is own repository of the course [here](https://github.com/Foundations-of-HPC/High-Performance-Computing-2024)

Here the repository with my final project [here](https://github.com/DottorBooom/HPC-Project)

### What you will learn

**Core HPC fundamentals and architecture**
+ What is HPC and why it matters for modern computational challenges
+ CPU architecture evolution: from single-core to multi-core, vectorization, and memory hierarchies
+ Performance optimization: compiler techniques, loop optimization, cache-aware programming
+ Memory systems: virtual memory, cache optimization, NUMA architectures

**Parallel programming mastery**
+ Shared-memory programming with OpenMP: threads, synchronization, load balancing
+ Distributed-memory programming with MPI: message passing, collective operations
+ Parallel algorithm design and complexity analysis
+ Performance debugging and profiling with modern tools

**HPC ecosystem and infrastructure**
+ Hardware components: clusters, interconnects, storage systems
+ Software stack: compilers, libraries, resource management (SLURM)
+ Performance analysis: hardware counters, profiling tools (perf, valgrind, gdb)
+ Real-world applications: scientific computing, AI/ML, weather forecasting

**Professional HPC skills**
+ Performance measurement and benchmarking methodologies
+ Debugging parallel applications and avoiding common pitfalls
+ Code optimization strategies for maximum computational efficiency
+ Understanding the complete HPC software and hardware stack

**Attitude**
+ Don't be (only) a user of pre-cooked tools that you consider as black-boxes

### Common rules & principles

+ Every question is **legitimate and useful**, ask what you do not understand
+ Main pourpose it to **learn**, not to grade
+ Learning is a **process**, not a result
+ Nobody is perfect or always right: **errors and mistake are natural**
+ **Learning is a process in our personal brain**, not in other's one. **Clash with your limits** before check the solution

## Intro to course

Introduction to High Performance Computing covering fundamental concepts, performance metrics, and real-world applications.

**Key topics:**
+ Why HPC is essential for complex computational problems
+ HPC applications: weather forecasting, protein folding (AlphaFold), AI training
+ Performance metrics: FLOPS, TOP500 rankings, benchmarking
+ HPC ecosystem: hardware, software, and human resources
+ Current exascale systems and performance vs productivity considerations

Slides are available [here](Lectures/00_Lecture/)

## HPC Hardware and Software

Hardware and software foundations of HPC systems, explaining why HPC is inherently parallel and the evolution from serial to parallel computing.

**Key topics:**
+ End of Dennard scaling and Moore's Law implications
+ Von Neumann architecture limitations and Flynn's Taxonomy
+ Multicore CPUs, hardware accelerators (GPUs, FPGAs, ASICs)
+ HPC cluster components: nodes, interconnects, storage
+ Memory architectures: UMA vs NUMA, cache hierarchies
+ Memory wall problem and parallelism challenges

Slides are available [here](Lectures/01_Lecture/)

## HPC Software Stack

Overview of the complete software stack required for HPC systems, from system management to user applications, with focus on resource management and scheduling.

**Key topics:**
+ Cluster middleware: administration and resource management software
+ Local Resource Management Systems (LRMS): SLURM, LSF, PBS
+ Batch schedulers: job allocation, scheduling policies, resource optimization
+ Scientific software management: modules system, version control
+ Compiler suites: GNU, Intel, PGI - optimization vs portability trade-offs
+ Library types: static vs shared libraries, dependency management

Slides are available [here](Lectures/02_Lecture/)

## Code Optimization Fundamentals

Introduction to code optimization principles and compiler techniques for single-core performance improvement in HPC applications.

**Key topics:**
+ Optimization constraints: time, memory, energy, I/O, robustness considerations
+ Clean code principles: DRY (Don't Repeat Yourself), readability, maintainability
+ Design methodology: algorithm selection, data structures, testing, validation
+ Compiler optimization levels (-O1, -O2, -O3, -Os), architecture-specific compilation (-march=native)
+ Profile-guided optimization (PGO) and automatic profiling techniques
+ Memory aliasing problems and the 'restrict' qualifier in C
+ Storage classes and variable qualifiers (const, volatile, register)

Slides are available [here](Lectures/03_Lecture/)

## Modern CPU Architecture

Detailed exploration of modern single-core CPU architecture evolution, from simple Von Neumann model to complex superscalar, out-of-order, vectorized processors.

**Key topics:**
+ Memory wall problem and cache hierarchy (L1, L2, L3) solutions
+ Locality principles: temporal and spatial locality for cache efficiency
+ Cache types: compulsory, capacity, conflict misses and optimization strategies
+ CPU pipeline evolution: fetch, decode, execute, writeback stages
+ Superscalar architecture: multiple execution ports and instruction-level parallelism (ILP)
+ Vector processing: SIMD capabilities, vector registers (AVX, SSE)
+ Power wall challenges: static vs dynamic power consumption, thermal design
+ Cache coherency problems in multi-socket systems

Slides are available [here](Lectures/04_Lecture/)

## Practical Code Optimization

Hands-on techniques for code optimization with concrete examples, focusing on loop optimization and common performance pitfalls to avoid.

**Key topics:**
+ Avoiding expensive function calls: sqrt(), pow(), floating-point division optimizations
+ Loop hoisting and expression pre-computation for nested loops
+ Variable scope optimization and register usage suggestions
+ Floating-point arithmetic limitations: non-associative operations and compiler constraints
+ Eliminating redundant checks and unnecessary memory references
+ Local accumulator patterns for reduction operations
+ Fast-math compiler flags and IEEE compliance trade-offs

Slides are available [here](Lectures/05_Lecture/)

## Memory Management and Execution Model

Deep dive into program execution model, memory organization, and the fundamental differences between stack and heap allocation in Unix systems.

**Key topics:**
+ Virtual memory model: address space, paging mechanism, Translation Lookaside Buffer (TLB)
+ Stack vs heap: local vs global variables, scope limitations, growth directions
+ Stack frame anatomy: function calls, return addresses, local variables, BP/SP registers
+ Dynamic allocation: alloca() for stack allocation, stack size limits (ulimit)
+ Assembly-level memory access patterns: stack vs heap performance comparison
+ Memory addressing modes and compiler optimization differences
+ Variable scope and lifetime implications for performance

Slides are available [here](Lectures/06_Lecture/)

## Cache Optimization and Data Locality

Advanced techniques for optimizing cache performance through memory access patterns, data organization, and locality-aware programming strategies.

**Key topics:**
+ Cache mapping strategies: direct, associative, n-way associative with set organization
+ Memory access patterns: strided access, matrix transpose optimization, cache conflicts
+ Write policies: write-through vs write-back, write-allocate strategies
+ Cache-associativity conflicts and padding techniques for cache resonance mitigation
+ Data organization: blocking techniques, hot vs cold field separation
+ Space-filling curves: Z-order (Morton), Hilbert curves for locality preservation
+ Memory mountain analysis and cache size detection techniques

Slides are available [here](Lectures/07_Lecture/)

## Loop Optimization and Vectorization

Advanced loop optimization techniques focusing on vectorization, compiler directives, and exploiting SIMD capabilities for maximum performance.

**Key topics:**
+ Loop vectorization fundamentals: data dependencies, vectorizable patterns
+ SIMD instruction sets: SSE, AVX, AVX-512 capabilities and register usage
+ Compiler auto-vectorization: enabling flags, optimization reports, vectorization hints
+ Manual vectorization with intrinsics: Intel/GCC intrinsics for fine-grained control
+ Loop transformations: unrolling, peeling, fusion, distribution, interchange
+ Data alignment requirements and performance implications
+ Vectorization obstacles: complex control flow, function calls, memory aliasing

Slides are available [here](Lectures/08_Lecture/)

## Advanced Loop Techniques and Prefetching

Sophisticated loop optimization strategies with focus on arithmetic intensity classification, multiple accumulator techniques, and hardware/software prefetching methods.

**Key topics:**
+ Loop classification by arithmetic intensity: O(N)/O(N), O(N²)/O(N²), O(N³)/O(N²)
+ Advanced loop unrolling strategies: n×m unrolling patterns, multiple accumulators
+ Critical path analysis and dependency chain optimization
+ Matrix multiplication optimization: loop tiling, blocking algorithms, cache-aware implementations
+ Hardware and software prefetching: explicit prefetch directives, preloading techniques
+ Register spill management and code bloating prevention
+ Performance measurement and analysis: IPC, cycles per element, cache miss patterns

Slides are available [here](Lectures/09_Lecture/)

## Introduction to Parallel Computing

Comprehensive introduction to parallel computing fundamentals, HPC architectures, memory models, and performance analysis in high-performance computing environments.

**Key topics:**
+ HPC rationale: complex problem solving, simulation-based research, data processing challenges
+ Parallelism classification: embarrassingly parallel vs interdependent problems
+ Domain decomposition strategies: data distribution, load balancing, work imbalance management
+ Shared vs distributed memory paradigms: UMA, NUMA architectures, programming models
+ HPC hardware hierarchy: cores, sockets, nodes, clusters, interconnection networks
+ Cache coherency protocols: MESI standard, false sharing, data consistency
+ Parallel performance theory: Amdahl's law, Gustafson's law, strong vs weak scalability
+ MPI and OpenMP programming paradigms introduction

Slides are available [here](Lectures/10_Lecture/)

## OpenMP Fundamentals

Introduction to OpenMP shared-memory programming model, covering basic concepts, thread management, and the fork-join execution paradigm.

**Key topics:**
+ Processes vs threads: memory sharing, resource allocation, creation overhead
+ OpenMP vs MPI: shared-memory vs distributed-memory programming paradigms
+ Fork-join execution model: master thread spawning and synchronization
+ OpenMP evolution: from basic loop parallelization to tasks and accelerator offloading
+ Directive-based approach advantages: abstraction, efficiency, incremental development
+ Static vs dynamic extent: lexical scope and orphan directives
+ Conditional compilation and portability considerations

Slides are available [here](Lectures/11_Lecture/)

## OpenMP Parallel Regions

Comprehensive coverage of OpenMP parallel regions, thread creation, memory management, and synchronization constructs for shared-memory programming.

**Key topics:**
+ Thread creation and stack management: virtual memory layout, stack sizing
+ Variable scope management: shared, private, firstprivate, lastprivate, threadprivate
+ Thread specialization constructs: critical, atomic, single, master/masked directives
+ Synchronization mechanisms: barriers, ordered execution, race condition prevention
+ Work assignment strategies: manual thread ID-based distribution, conditional regions
+ Nested parallelism: multi-level thread creation, environment variable control
+ OpenMP runtime functions and internal control variables (ICVs)

Slides are available [here](Lectures/12_Lecture/)

## Parallel Algorithm Analysis

Analysis of parallel algorithms design and complexity, focusing on prefix sum implementations to demonstrate algorithmic efficiency differences in parallel computing.

**Key topics:**
+ Prefix sum algorithm fundamentals: serial vs parallel approaches
+ Algorithm I: recursive subdivision with n-1 steps, O(N(n+1)/2) complexity
+ Algorithm II: binary tree approach with log₂(n) steps, O(N(1+log₂n)/2) complexity
+ Scalability comparison: speedup analysis and performance characteristics
+ Memory access patterns and their impact on parallel algorithm efficiency
+ Work complexity vs span complexity in parallel algorithm design
+ Trade-offs between algorithmic approaches and practical implementation considerations

Slides are available [here](Lectures/13_Lecture/)

## OpenMP Parallel Loops

Advanced OpenMP loop parallelization techniques, work scheduling strategies, and common pitfalls in shared-memory parallel programming.

**Key topics:**
+ Parallel for construct: basic syntax, variable scoping (shared, private, firstprivate, lastprivate)
+ Work scheduling policies: static, dynamic, guided scheduling with chunk size control
+ Reduction operations: avoiding race conditions, false sharing problems
+ Data race anatomy: memory access conflicts, synchronization requirements
+ False sharing: cache line conflicts, performance degradation, mitigation strategies
+ Loop clauses: collapse, nowait, ordered execution control
+ Nested parallelism within loops: parallel regions vs parallel for differences

Slides are available [here](Lectures/14_Lecture/)

## OpenMP NUMA Awareness

Advanced OpenMP thread affinity and memory allocation strategies for NUMA (Non-Uniform Memory Access) systems optimization.

**Key topics:**
+ NUMA architecture challenges: memory bandwidth, latency variations, remote access costs
+ Thread affinity concepts: places (threads, cores, sockets), binding policies (close, spread, master)
+ Thread placement strategies: hardware thread mapping, SMT considerations
+ OMP_PLACES and OMP_PROC_BIND environment variables configuration
+ Memory allocation policies: touch-first vs touch-by-all strategies
+ Data locality optimization: malloc vs calloc behavior, memory page mapping
+ Performance impact analysis: bandwidth aggregation, cache utilization, synchronization costs
+ Topology discovery tools: numactl, lstopo, hwloc utilities

Slides are available [here](Lectures/15_Lecture/)

## MPI Fundamentals

Introduction to Message Passing Interface (MPI) for distributed-memory parallel programming, covering basic concepts and communication patterns.

**Key topics:**
+ Distributed-memory paradigm: separate address spaces, explicit message passing
+ MPI as a standard: library specification, multiple implementations (OpenMPI, MPICH)
+ Basic MPI structure: initialization, finalization, communicators, ranks
+ Communicators and groups: MPI_COMM_WORLD, context separation, best practices
+ Thread support levels: MPI_THREAD_SINGLE to MPI_THREAD_MULTIPLE
+ Process identification: rank determination, size queries, communication contexts
+ MPI analogy: postal service model for understanding message passing concepts

Slides are available [here](Lectures/16_Lecture/)

## MPI Point-to-Point Communication

Comprehensive coverage of MPI point-to-point communication patterns, including blocking, non-blocking, and specialized communication modes for distributed-memory programming.

**Key topics:**
+ Basic send/receive operations: MPI_Send, MPI_Recv syntax and semantics
+ MPI data types: primitive types mapping, message envelope components
+ Communication safety: deadlock prevention, unsafe code patterns, message ordering
+ Communication modes: standard, synchronous (Ssend), buffered (Bsend), ready (Rsend)
+ Communication protocols: eager vs rendezvous protocols, system buffering implications
+ Non-blocking communications: MPI_Isend/Irecv, request handling, MPI_Test/Wait operations
+ Advanced patterns: MPI_Sendrecv, probe operations, ping-pong performance testing
+ Practical exercises: latency/bandwidth measurement, safe communication design

Slides are available [here](Lectures/17_Lecture/)

## MPI Collective Communication

Advanced MPI collective communication operations for efficient data distribution, synchronization, and computation across process groups in distributed systems.

**Key topics:**
+ Collective operation categories: synchronization, data movement, collective computation
+ Synchronization primitives: MPI_Barrier implementation and performance implications
+ Data distribution patterns: broadcast, scatter/gather, allgather, alltoall operations
+ Variable-size collectives: MPI_Scatterv, MPI_Gatherv for irregular data distribution
+ Collective computation operations: reduce, allreduce, scan operations with predefined operators
+ Algorithm complexity: tree-based vs linear algorithms, hardware-accelerated collectives
+ Groups and communicators: creating custom communicators, MPI_Comm_split operations
+ In-place operations: MPI_IN_PLACE usage for memory-efficient collective operations

Slides are available [here](Lectures/18_Lecture/)

## HPC Profiling and Analysis

Comprehensive guide to HPC application profiling and performance analysis, covering both traditional and modern profiling tools and methodologies.

**Key topics:**
+ Profiling fundamentals: call tree/graph generation, instrumentation vs sampling approaches
+ Dynamic profiling concepts: context-sensitive analysis, calling context trees
+ Classical tools comparison: gprof limitations, google perftools, valgrind capabilities
+ Advanced profiling tools: perf, cachegrind, callgrind for detailed performance analysis
+ Call tree visualization: gprof2dot, kcachegrind GUI tools for performance data interpretation
+ Memory profiling: heap analysis, leak detection, memory access pattern optimization
+ Hardware counter interfaces: PAPI, PMU access, vendor-specific profiling tools
+ Profiler accuracy and limitations: avoiding measurement artifacts and interpretation pitfalls

Slides are available [here](Lectures/19_Lecture/)

## HPC Debugging Techniques

Advanced debugging strategies for parallel HPC applications, covering both serial and parallel debugging approaches with modern tools and methodologies.

**Key topics:**
+ GDB fundamentals: compilation flags, breakpoints, memory examination, stack analysis
+ Multi-threaded debugging: thread management, scheduler control, parallel execution analysis
+ MPI debugging strategies: process attachment, synchronization debugging, deadlock detection
+ Advanced debugging workflows: core file analysis, running process inspection, reverse debugging
+ Debugging parallel applications: coordinated debugging sessions, process synchronization techniques
+ GDB extensions and interfaces: TUI mode, dashboard customization, GUI frontends
+ Memory debugging: valgrind integration, memory error detection, leak analysis
+ Practical debugging patterns: safe debugging code injection, environment-based activation

Slides are available [here](Lectures/20_Lecture/)

## Performance Monitoring and Hardware Counters

Deep dive into Performance Monitoring Units (PMUs) and hardware-based performance analysis using modern Linux tools for detailed HPC performance characterization.

**Key topics:**
+ Performance Monitoring Units (PMUs): architecture, event selection registers, performance counters
+ Hardware counter types: fixed-function counters, programmable counters, architectural vs model-specific events
+ Linux perf framework: kernel infrastructure, user-space tools, event abstraction layer
+ Perf usage patterns: counting mode vs sampling mode, system-wide vs process-specific monitoring
+ Event selection and configuration: mnemonic names, raw event codes, event multiplexing
+ Advanced perf features: call-graph recording, flame graph generation, multi-event profiling
+ Alternative tools integration: PAPI library fundamentals, gperftools integration
+ Performance analysis workflow: data collection strategies, bottleneck identification, optimization guidance

Slides are available [here](Lectures/21_Lecture/)
