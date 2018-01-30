#include "DeviceMemoryStore.h"
#include "cudafuncs.h"

/*
 * FixedSizedChunks
 */

#ifdef _MSC_VER
/* __builtin_ctz() implementations on MSVC
 * https ://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
 */
#include <intrin.h>
uint32_t __inline __builtin_ctz(uint32_t value) {
    unsigned long trailing_zero = 0;
    if (_BitScanForward(&trailing_zero, value))
        return trailing_zero;
    else
        return 32;
}
uint32_t __inline __builtin_clzll(uint64_t value) {
    if (value == 0) return 64;
    unsigned long idx; // bit pos of the first bit set(1) from MSB to LSB.
    _BitScanReverse64(&idx, value);
    return 63 - idx;
}
/* Effective after Haswell */
/* #define __builtin_clzll __lzcnt64 */

#endif

// #define DEBUG_ALLOC

bool HeapBitmap::acquire(uintptr_t *addr) {
    if (freeRegions_.empty())
        return false;
    uintptr_t key = *freeRegions_.begin();
    RegionMap::iterator rit = regions_.find(key);
#ifdef DEBUG_ALLOC
    fprintf(stderr, "Region: %3d, %I64x - %I64x, mask: %2x\n", regionSize_, rit->first - regionSize_, rit->first - 1, rit->second);
#endif
    int iChunk = __builtin_ctz(rit->second);
    rit->second ^= (1 << iChunk);
    *addr = (key - regionSize_) + (iChunk << chunkSizeShift_);
#ifdef DEBUG_ALLOC
    fprintf(stderr, "Acquire Region: %3d, Chunk, %d, addr: %I64x\n", regionSize_, iChunk, *addr);
#endif
    if (isRegionFull(rit->second))
        freeRegions_.erase(key);
    return true;
}

bool HeapBitmap::release(uintptr_t addr) {
#ifdef DEBUG_ALLOC
    fprintf(stderr, "Release: Region: %3d, addr: %I64x\n", regionSize_, addr);
    for (RegionMap::iterator it = regions_.begin(); it != regions_.end(); ++it)
        fprintf(stderr, "Region: %3d, %I64x - %I64x, mask: %2x\n", regionSize_, it->first - regionSize_, it->first - 1, it->second);
#endif
    RegionMap::iterator it = regions_.upper_bound(addr);
    if (it == regions_.end())
        abort_("trying to release a chunk that not allocated.");
#ifdef DEBUG_ALLOC
    fprintf(stderr, "Region: %3d, %I64x - %I64x, mask: %2x\n", regionSize_, it->first - regionSize_, it->first - 1, it->second);
#endif
    if (it->first <= addr)
        abort_("trying to release a chunk that not allocated.");

    uintptr_t key = it->first;
    if (isRegionFull(it->second)) {
        /* will have a vacant chunk, thus, add to freeRegions */
        freeRegions_.insert(key);
    }
    uintptr_t regionBegin = key - regionSize_;
    int chunkIdx = ((addr - regionBegin) >> chunkSizeShift_) & regionMask;
    it->second ^= (1 << chunkIdx);
    if (!isRegionEmpty(it->second))
        return false;
        
    regions_.erase(it);
    size_t nErased = freeRegions_.erase(key);
    assert(nErased == 1);
    return true;
}

void HeapBitmap::addRegion(uintptr_t addr) {
    uintptr_t key = addr + regionSize_;
    freeRegions_.insert(key);
    regions_[key] = regionMask;
}

void FixedSizedChunks::initialize() {
    bitmapLayers_[ 0].set( 2); /*    4 byte / chunk, 128 bytes / region parent : 5 */
    bitmapLayers_[ 1].set( 3); /*    8 byte / chunk, 256 bytes / region parent : 6 */
    bitmapLayers_[ 2].set( 4); /*   16 byte / chunk, 512 bytes / region parent : 7 */
    bitmapLayers_[ 3].set( 5); /*   32 byte / chunk,   1 K     / region parent : 8  */
    bitmapLayers_[ 4].set( 6); /*   64 byte / chunk    2 K     / region parent : 9  */
    bitmapLayers_[ 5].set( 7); /*  128 byte / chunk,   4 K     / region parent : 10 */
    bitmapLayers_[ 6].set( 8); /*  256 byte / chunk,   8 K     / region */
    bitmapLayers_[ 7].set( 9); /*  512 byte / chunk,  16 K     / region */
    bitmapLayers_[ 8].set(10); /*   1k byte / chunk,  32 K     / region */
    bitmapLayers_[ 9].set(11); /*   2k byte / chunk,  64 K     / region */
    bitmapLayers_[10].set(12); /*   4k byte / chunk, 128 K     / region */
}

void FixedSizedChunks::finalize() {
    for (int idx = 0; idx < nBitmapLayers; ++idx)
        bitmapLayers_[idx].clear();
}

size_t FixedSizedChunks::newHeapSize(size_t reqSize) const {
    int layerIdx = layerIdxFromSize(reqSize);
    if (layerIdx <= 5)
        return reqSize * HeapBitmap::nChunksInRegion * HeapBitmap::nChunksInRegion;
    return reqSize * HeapBitmap::nChunksInRegion;
}

void FixedSizedChunks::addFreeHeap(uintptr_t pv, size_t size) {
    assert(size % (1 << 7) == 0);
    int layerIdx = layerIdxFromSize(size / HeapBitmap::nChunksInRegion);
    bitmapLayers_[layerIdx].addRegion(pv);
}

uintptr_t FixedSizedChunks::acquire(size_t *size) {
    /* get the idx for the appropriate layer */
    int layerIdx = layerIdxFromSize(*size);
    *size = 1ull << (layerIdx + 2); /* round up to power of 2. */
    assert((0 <= layerIdx) && (layerIdx < nBitmapLayers));
    
    HeapBitmap &bitmap = bitmapLayers_[layerIdx];
    uintptr_t addr;
    if (bitmap.acquire(&addr))
        return addr;
    if (5 < layerIdx)
        return -1;
    
    const int parentLayerIdx = layerIdx + 5;
    HeapBitmap &parentBitmap = bitmapLayers_[parentLayerIdx];
    if (parentBitmap.acquire(&addr)) {
        bitmap.addRegion(addr);
        bool res = bitmap.acquire(&addr);
        assert(res);
        return addr;
    }

    return (uintptr_t)-1;
}

void FixedSizedChunks::release(uintptr_t addr, size_t size) {
    int layerIdx = layerIdxFromSize(size);
    bool regionReleased = bitmapLayers_[layerIdx].release(addr);
    /* process releasing region */
    if (regionReleased) {
        /* FIXME: bitmap reagion is empty,
         * returning a region to parent memory store. */
    }
}

int FixedSizedChunks::layerIdxFromSize(size_t size) {
    size_t size4 = (size + 3) / 4;
    return int(sizeof(size_t) * 8 - __builtin_clzll(size4 - 1));
}

/*
 * HeapMap
 */

HeapMap::HeapMap() {
}

void HeapMap::finalize() {
    freeRegions_.clear();
}

size_t HeapMap::newHeapSize() {
    return currentHeapSize_;
}

void HeapMap::addFreeHeap(uintptr_t heap, size_t size) {
    freeRegions_[heap] = size;
    currentHeapSize_ += size;
}

uintptr_t HeapMap::acquire(size_t *size) {
    *size += 511;
    *size &= ((size_t)-1) - 511;
    RegionMap::iterator it = freeRegions_.begin();
    for ( ; it != freeRegions_.end(); ++it) {
        if (*size <= it->second)
            break;
    }
    if (it == freeRegions_.end())
        return (uintptr_t)-1;

    uintptr_t addr = it->first;
    uintptr_t newAddr = addr + *size;
    size_t newSize = it->second - *size;
    freeRegions_.erase(it);
    /* Exactly the same size as requested by size found. 
     * No need to register to freeRegions. */
    if (newSize != 0)
        freeRegions_[newAddr] = newSize;

    return addr;
}

void HeapMap::release(uintptr_t addr, size_t size) {
    RegionMap::iterator it = freeRegions_.upper_bound(addr);
    if (it != freeRegions_.end()) {
        /* merge previous region if continuous. */
        if (it->first + it->second == addr) {
            it->second += size;
            /* merge next region if continuous. */
            RegionMap::iterator nextIt = it;
            ++nextIt;
            if (it->first + it->second == nextIt->first) {
                it->second += nextIt->second;
                freeRegions_.erase(nextIt);
            }
            return;
        }
    }
    /* merge next region if continuous. */
    it = freeRegions_.lower_bound(addr);
    if (it != freeRegions_.end()) {
        if (addr + size == it->first) {
            size_t newSize = size + it->second;
            freeRegions_.erase(it);
            freeRegions_[addr] = newSize;
            return;
        }
    }
    /* no continous chunks, just add free region */
    freeRegions_[addr] = size;
}


/* DeviceMemoryStore */
void DeviceMemoryStore::initialize() {
    fixedSizedChunks_.initialize();
    size_t newHeapSize = 512 * (1 << 20); /* 512 M */
    uintptr_t newHeap = cudaMalloc(newHeapSize);
    heapMap_.addFreeHeap(newHeap, newHeapSize);
}

void DeviceMemoryStore::finalize() {
    fixedSizedChunks_.finalize();
    heapMap_.finalize();
    chunkPropSet_.clear();
    for (size_t idx = 0; idx < d_mems_.size(); ++idx)
        cudaFree(d_mems_[idx]);
    d_mems_.clear();
}

void DeviceMemoryStore::useManagedMemory(bool use) {
    useManagedMemory_ = use;
}

void *DeviceMemoryStore::allocate(size_t size) {
    /* FIXME: Parameterize */
    uintptr_t addr;
    if (ChunkSizeToUseMalloc < size) {
        addr = cudaMalloc(size);
#ifdef DEBUG_ALLOC
        assert(chunkPropSet_.find(ChunkProp(addr, 0, fromNone)) == chunkPropSet_.end());
#endif
        chunkPropSet_.insert(ChunkProp(addr, size, fromCudaMalloc));
    }
    else if (SmallChunkSize < size) {
        addr = allocFromHeapMap(&size);
#ifdef DEBUG_ALLOC
        assert(chunkPropSet_.find(ChunkProp(addr, 0, fromNone)) == chunkPropSet_.end());
#endif
        chunkPropSet_.insert(ChunkProp(addr, size, fromHeapMap));
    }
    else {
        addr = allocFromFixedSizedChunks(&size);
#ifdef DEBUG_ALLOC
        assert(chunkPropSet_.find(ChunkProp(addr, 0, fromNone)) == chunkPropSet_.end());
#endif
        chunkPropSet_.insert(ChunkProp(addr, size, fromFixedSizedSeries));
    }
    return reinterpret_cast<void*>(addr);
}

void DeviceMemoryStore::deallocate(void *pv) {
    if (pv == NULL) {
        /* FIXME: Add message. */
        return;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(pv);
    ChunkPropSet::iterator it = chunkPropSet_.find(ChunkProp(addr, 0, fromNone));
    if (it == chunkPropSet_.end())
        abort_("trying to release a chunk that not allocated.");
    // THROW_IF(it == chunkPropSet_.end());
    const ChunkProp &chunk = *it;
    
    switch (chunk.src) {
    case fromFixedSizedSeries:
        deallocateInFixedSizedChunks(chunk.addr, chunk.size);
        break;
    case fromHeapMap:
        deallocateInHeapMap(chunk.addr, chunk.size);
        break;
    case fromCudaMalloc:
        throwOnError(::cudaFree(pv));
        break;
    default:
        abort_("Must not reach."); // Must not reach here.
    }
    chunkPropSet_.erase(it);
}

uintptr_t DeviceMemoryStore::cudaMalloc(size_t size) {
    uintptr_t addr;
    if (!useManagedMemory_) {
        throwOnError(::cudaMalloc(reinterpret_cast<void**>(&addr), size));
    }
    else {
        throwOnError(::cudaMallocManaged(reinterpret_cast<void**>(&addr), size));
    }
    
    d_mems_.pushBack((void*)addr);
    return addr;
}

void DeviceMemoryStore::cudaFree(void *pv) {
    throwOnError(::cudaFree(pv));
}

uintptr_t DeviceMemoryStore::allocFromHeapMap(size_t *size) {
    uintptr_t addr = heapMap_.acquire(size);
    if (addr == (uintptr_t)-1) {
        size_t newHeapSize = heapMap_.newHeapSize();
        uintptr_t newHeap = cudaMalloc(newHeapSize);
        heapMap_.addFreeHeap(newHeap, newHeapSize);
        addr = heapMap_.acquire(size);
    }
#ifdef DEBUG_ALLOC
    fprintf(stderr, "Aqruied Heapmap: %I64x, %zd.\n", addr, *size);
#endif
    return addr;
}

void DeviceMemoryStore::deallocateInHeapMap(uintptr_t addr, size_t size) {
    heapMap_.release(addr, size);
}

uintptr_t DeviceMemoryStore::allocFromFixedSizedChunks(size_t *size) {
    uintptr_t addr = fixedSizedChunks_.acquire(size);
    if (addr == (uintptr_t)-1) {
        /* FIXME: give more appropriate size. */
        size_t newHeapSize = fixedSizedChunks_.newHeapSize(*size);
        uintptr_t newHeap = allocFromHeapMap(&newHeapSize);
        fixedSizedChunks_.addFreeHeap(newHeap, newHeapSize);
        addr = fixedSizedChunks_.acquire(size);
    }
    return addr;
}

void DeviceMemoryStore::deallocateInFixedSizedChunks(uintptr_t addr, size_t size) {
    fixedSizedChunks_.release(addr, size);
}
