#include "DeviceMemoryStore.h"
#include "cudafuncs.h"


/*
 * FixedSizedChunks
 */

#ifdef _MSC_VER
// https ://stackoverflow.com/questions/355967/how-to-use-msvc-intrinsics-to-get-the-equivalent-of-this-gcc-code
#include <intrin.h>
uint32_t __inline __builtin_ctz(uint32_t value)
{	unsigned long trailing_zero = 0;
	if (_BitScanForward(&trailing_zero, value))
		return trailing_zero;
	else
		return 32;
}
uint32_t __inline __builtin_ctzll(uint64_t value)
{
	unsigned long trailing_zero = 0;
	if (_BitScanForward64(&trailing_zero, value))
		return trailing_zero;
	else
		return 64;
}

#endif

bool HeapBitmap::acquire(uintptr_t *addr) {
    if (freeRegions_.empty())
        return false;
    
    RegionMap::iterator regionIt = freeRegions_.begin();
    int iChunk = __builtin_ctz(regionIt->second);
    regionIt->second ^= (1 << iChunk);
    *addr = (regionIt->first << (sizeInPo2_ + nActiveBits_)) + (iChunk << sizeInPo2_);
    if (isRegionFull(regionIt->second))
        freeRegions_.erase(regionIt);
    return true;
}

bool HeapBitmap::tryRelease(uintptr_t addr, bool *regionReleased) {
    uintptr_t regionIdx = addr >> (sizeInPo2_ + nActiveBits_);
    RegionMap::iterator it = regions_.find(regionIdx);
    if (it == regions_.end())
        return false;

	int bitmapIdx = (addr >> sizeInPo2_) & mask_;
    it->second ^= (1 << bitmapIdx);
    if (isRegionEmpty(it->second)) {
        regions_.erase(it);
        size_t nErased = freeRegions_.erase(it->first);
        assert(nErased == 1);
        *regionReleased = true;
    }
    return true;
}

void HeapBitmap::addRegion(uintptr_t addr) {
	uintptr_t regionIdx = addr >> (sizeInPo2_ + nActiveBits_);
	freeRegions_[regionIdx] = mask_;
    regions_[regionIdx] = mask_;
}

void FixedSizedChunks::initialize() {
    bitmapLayers_[ 0].set(5,  2); /*    4 byte / chunk, 128 bytes / region */
    bitmapLayers_[ 1].set(5,  3); /*    8 byte / chunk, 256 bytes / region */
    bitmapLayers_[ 2].set(5,  4); /*   16 byte / chunk, 512 bytes / region */
    bitmapLayers_[ 3].set(5,  5); /*   32 byte / chunk,   1 K / region */
    bitmapLayers_[ 4].set(5,  6); /*   64 byte / chunk    2 K / region */
    bitmapLayers_[ 5].set(5,  7); /*  128 byte / chunk,   4 K / region */
    bitmapLayers_[ 6].set(5,  8); /*  256 byte / chunk,   8 K / region */
    bitmapLayers_[ 7].set(5,  9); /*  512 byte / chunk,  16 K / region */
    bitmapLayers_[ 8].set(5, 10); /*   1k byte / chunk,  32 K / region */
    bitmapLayers_[ 9].set(4, 11); /*   2k byte / chunk,  64 K / region */
    bitmapLayers_[10].set(3, 12); /*   4k byte / chunk, 128 K / region */
    bitmapLayers_[11].set(2, 13); /*   8k byte / chunk, 256 K / region */
    bitmapLayers_[12].set(1, 14); /*  16k byte / chunk, 512 K / region */
    bitmapLayers_[13].set(5, 15); /*  32k byte / chunk,   1 M / region */
}

void FixedSizedChunks::finalize() {
    for (int idx = 0; idx < nBitmapLayers; ++idx)
        bitmapLayers_[idx].clear();
}

void FixedSizedChunks::addHeap(uintptr_t pv, size_t size) {
    const int mega = (1 << 20);
    assert(size % mega == 0);
    
    int nChunks = size / mega;
    for (int idx = 0; idx < nChunks; ++idx)
        bitmapLayers_[nBitmapLayers - 1].addRegion(pv + mega * idx);
}



uintptr_t FixedSizedChunks::allocate(size_t size) {
    size = (size + 3) / 4; /* round up to multiple of 4. */
    /* find appropriate slices */
    int layerIdx = __builtin_ctzll(size);
    assert((0 <= layerIdx) && (layerIdx < nBitmapLayers));
    
    HeapBitmap &bitmap = bitmapLayers_[layerIdx];
    uintptr_t addr;
    if (bitmap.acquire(&addr))
        return addr;
    /* allocate new slice */
    HeapBitmap *newBitmap = allocateRegion(layerIdx);
    if (newBitmap != NULL) {
        newBitmap->acquire(&addr);
        return addr;
    }
    return (uintptr_t)-1;
}

void FixedSizedChunks::deallocate(uintptr_t addr) {
    int layerIdx = 0;
    bool regionReleased = false;
    for ( ; layerIdx < nBitmapLayers; ++layerIdx) {
        if (bitmapLayers_[layerIdx].tryRelease(addr, &regionReleased))
            break;
    }
    if (layerIdx != nBitmapLayers) {
        /* process releasing region */
        if (regionReleased) {
            /* 32-bit region is empty, and will be released. */
            layerIdx += bitmapLayers_[layerIdx].nActiveBits();
            for (; layerIdx < nBitmapLayers; layerIdx += bitmapLayers_[layerIdx].nActiveBits()) {
                regionReleased = false;
                bool released = bitmapLayers_[layerIdx].tryRelease(addr, &regionReleased);
                assert(released);
                if (!regionReleased)
                    break;
            }
        }
    }
}



HeapBitmap *FixedSizedChunks::allocateRegion(int layerIdx) {
	HeapBitmap &thisLayer = bitmapLayers_[layerIdx];
	int parentLayerIdx = layerIdx + thisLayer.nActiveBits();
	if (nBitmapLayers <= parentLayerIdx) {
		/* No parent.  No room to allocate new region. */
		return NULL;
	}

	HeapBitmap &parentLayer = bitmapLayers_[parentLayerIdx];
	uintptr_t addr;
	/* ask parent to provide heap */
	if (!parentLayer.acquire(&addr)) {
		/* allocating region in layer */
		if (allocateRegion(parentLayerIdx) == NULL)
			return NULL;
		parentLayer.acquire(&addr);
	}
	thisLayer.addRegion(addr);
	return &thisLayer;
}



/*
 * FreeHeapMap
 */

FreeHeapMap::FreeHeapMap() {
}

void FreeHeapMap::addFreeHeap(uintptr_t heap, size_t size) {
    freeRegions_[heap] = size;
}

uintptr_t FreeHeapMap::acquire(size_t size) {
    RegionMap::iterator it = freeRegions_.begin();
    for ( ; it != freeRegions_.end(); ++it) {
        if (size <= it->second)
            break;
    }
    if (it == freeRegions_.end())
        return (uintptr_t)-1;

    uintptr_t addr = it->first;
    uintptr_t newAddr = addr + size;
    size_t newSize = it->second - size;
    freeRegions_.erase(it);
    freeRegions_[newAddr] = newSize;

    return addr;
}

void FreeHeapMap::release(uintptr_t addr, size_t size) {
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
            int newSize = size + it->second;
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
    const int giga = 1 << 30;

    fixedSizedChunks_.initialize();
    void *d_pv = allocate(giga);
    d_mems_.pushBack(d_pv);
    uintptr_t addr = reinterpret_cast<uintptr_t>(d_pv);
    freeHeapMap_.addFreeHeap(addr, giga);
}

void DeviceMemoryStore::finalize() {
    fixedSizedChunks_.finalize();
    freeHeapMap_.clearRegions();
    for (size_t idx = 0; idx < d_mems_.size(); ++idx)
        deallocate(d_mems_[idx]);
}


void *DeviceMemoryStore::allocate(size_t size) {
    uintptr_t addr;
    if (ChunkSizeToUseMalloc < size)
        addr = cudaMalloc(size);
    else if (SmallChunkSize < size)
        addr = allocFromFreeHeap(size);
    else
        addr = allocFromFixedSizedChunks(size);
    return reinterpret_cast<void*>(addr);
}

void DeviceMemoryStore::deallocate(void *pv) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(pv);
    HeapMap::iterator it = heapMap_.upper_bound(addr);
    if (it == heapMap_.end())
        abort();
    if (addr < it->first)
        abort();
    
    switch (it->second.src) {
    case fixedSizeSeries:
        deallocateInFixedSizedChunks(addr);
        break;
    case freeHeapMap:
        deallocateToFreeHeap(addr, it->second.size);
        break;
    case fromCudaMalloc:
        throwOnError(cudaFree(pv));
        break;
    }
    !!!!! heapMap_.erase(it);
}

uintptr_t DeviceMemoryStore::cudaMalloc(size_t size) {
    uintptr_t addr;
    throwOnError(::cudaMalloc(reinterpret_cast<void**>(&addr), size));
    heapMap_[addr + size] = HeapProp(addr, fromCudaMalloc);
    return addr;
}

void DeviceMemoryStore::cudaFree(void *pv) {
    throwOnError(::cudaFree(pv));
}

uintptr_t DeviceMemoryStore::allocFromFreeHeap(size_t size) {
	/* FIXME: round up size */
    uintptr_t addr = freeHeapMap_.acquire(size);
    if (addr == (uintptr_t)-1) {
        uintptr_t newHeap = cudaMalloc(size);
        d_mems_.pushBack((void*)newHeap);
        freeHeapMap_.addFreeHeap(newHeap, size);
        addr = fixedSizedChunks_.allocate(size);
		!! heapMap_[addr + size] = HeapProp(addr, freeHeapMap);
	}
    return addr;
}

void DeviceMemoryStore::deallocateToFreeHeap(uintptr_t addr, size_t size) {
    freeHeapMap_.release(addr, size);
}

uintptr_t DeviceMemoryStore::allocFromFixedSizedChunks(size_t size) {
    uintptr_t addr = fixedSizedChunks_.allocate(size);
    if (addr == (uintptr_t)-1) {
		/* FIXME: give more appropriate size. */
        size_t newHeapSize = 16 * (1 << 20);
        uintptr_t newHeap = allocFromFreeHeap(newHeapSize);
		/* register heap map this region to tell which module to deallocate given pointers. */
		heapMap_[newHeap + newHeapSize] = HeapProp(newHeap, fixedSizeSeries);
		fixedSizedChunks_.addHeap(newHeap, newHeapSize);
        addr = fixedSizedChunks_.allocate(size);
	}
    return addr;
}

void DeviceMemoryStore::deallocateInFixedSizedChunks(uintptr_t addr) {
    fixedSizedChunks_.deallocate(addr);
}
