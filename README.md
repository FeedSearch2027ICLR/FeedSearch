# FeedSearch

FeedSearch is a header-only graph traversal core with adaptive feedback termination.
This repository also provides a common C++ interface for running the core against
HNSW, Faiss NSG, and DiskANN in-memory Vamana indexes.

## Contents

```text
FeedSearch.cpp                 Core traversal and feedback controller
include/feedsearch/core.hpp    Public core entry point
include/feedsearch/backend.hpp Common graph-index interface
include/feedsearch/metrics.hpp L2, MIPS, and cosine conversions
src/                           HNSW, NSG, Vamana adapters and index factory
tests/                         Core and metric interface tests
```

The repository intentionally excludes datasets, generated indexes, benchmark
outputs, and third-party library source trees.

## Build the core

The core and its tests require a C++17 compiler and CMake 3.18 or newer. They
have no external ANN-library dependency:

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release \
	-DFEEDSEARCH_BUILD_ADAPTERS=OFF \
	-DFEEDSEARCH_BUILD_BENCHMARKS=OFF
cmake --build build-core
ctest --test-dir build-core --output-on-failure
```

Include `feedsearch/core.hpp` and call `feedanns::FeedSearch` with graph-entry
IDs, a neighbor enumerator, and a nonnegative distance callback. The callback
must stop enumerating neighbors when its `visit` call returns `false`.

## Build graph adapters

Set `FEEDSEARCH_BUILD_ADAPTERS=ON` (the default) after installing or building:

- [hnswlib](https://github.com/nmslib/hnswlib), with the directory containing
	`hnswlib.h` discoverable by CMake;
- [Faiss](https://github.com/facebookresearch/faiss), including `IndexNSG` and
	the `faiss` library;
- [DiskANN](https://github.com/microsoft/DiskANN), including the `diskann`
	library and in-memory headers.

For a nonstandard DiskANN location, pass its source or build root with
`-DDISKANN_DIR=/path/to/DiskANN`. Use `DISKANN_EXTRA_LIBRARIES` to provide
additional libraries required by that particular DiskANN build.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
	-DDISKANN_DIR=/path/to/DiskANN
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`FEEDSEARCH_NATIVE_ARCH=OFF` disables `-march=native` for portable builds.

## Unified adapter API

```cpp
#include "feedsearch/backend.hpp"

using namespace feedanns::adapters;

IndexOptions options;
options.graph = Graph::hnsw;  // Graph::nsg or Graph::vamana
options.metric = Metric::l2;  // Metric::mips or Metric::cosine
options.dimension = dimension;

auto index = load_index(index_path, options);
auto workspace = index->workspace();
auto result = index->feed_search(query, 10, 8.0, *workspace);

index->set_width(64);
auto baseline = index->native_search(query, 10, *workspace);
```

Create one workspace per search thread; it may only be used with the index that
created it. Call `set_width()` outside concurrent searches. MIPS FeedSearch
also requires `options.max_base_norm` to be set from database vectors.

The adapters expect static indexes with dense integer IDs. HNSW deleted nodes,
DiskANN frozen nodes, and DiskANN PQ storage are not supported.