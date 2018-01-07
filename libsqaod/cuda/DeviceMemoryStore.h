#ifndef SQAOD_CUDA_DEVICEMEMORYSTORE_H__
#define SQAOD_CUDA_DEVICEMEMORYSTORE_H__

#include <map>
#include <common/Array.h>
#include <assert.h>

/* ToDo: This implementatin is enough for now.  Will be reconsidered when needed. */

class DeviceMemoryBitmap {
    typedef uint32_t  RegionBitmap;
    typedef std::map<uintptr_t, RegionBitmap> RegionMap;

public:
    DeviceMemoryBitmap() { }

    void set(int nActiveBits, int sizeInPo2)  {
        nActiveBits_ = nActiveBits;
        sizeInPo2_ = sizeInPo2;
        mask_ = RegionBitmap((1ull << nActiveBits) - 1);
    }

    void clear();
    
    bool acquire(uintptr_t *addr);
    bool tryRelease(uintptr_t addr, bool *regionReleased);

    void addRegion(uintptr_t addr);

    int nActiveBits() const { return nActiveBits_; }
    
private:
    bool isRegionFull(RegionBitmap region) const {
        return region == 0;
    }
    bool isRegionEmpty(RegionBitmap region) const {
        return (region & mask_) == mask_;
    }

    RegionMap freeRegions_;
    RegionMap regions_;
    int nActiveBits_;
    int sizeInPo2_;
    RegionBitmap mask_;
};


class DeviceMemoryFixedSizeSeries {
public:
    void initialize();
    void uninitialize();

    void addHeap(uintptr_t pv, size_t size);
    
    uintptr_t allocate(size_t size);
    void deallocate(uintptr_t addr);

private:
    DeviceMemoryBitmap *allocateRegion(int layerIdx);

    enum { nBitmapLayers = 14 };
    DeviceMemoryBitmap bitmapLayers_[nBitmapLayers];
};




class FreeHeapMap {
public:
    FreeHeapMap();

    void addFreeHeap(uintptr_t addr, size_t size);

    void releaseHeaps();
    
    uintptr_t acquire(size_t size);
    
    void release(uintptr_t addr, size_t size);
    
private:
    typedef std::map<uintptr_t, size_t> RegionMap;
    RegionMap freeRegions_;
};



class DeviceMemoryStore {
    /* Memory chunks whose capacity is less than / equals to 256 MBytes are managed by memory store.
     * Chunks larger than 512 MB directly use cudaMalloc()/cudaFree(). */ 
    enum {
        ChunkSizeToUseMalloc = 32 * (1 << 20), /* 32 M */
        SmallChunkSize = 1024,            /* 32 K */
    };
public:
    void initialize();
    void uninitialize();
    
    void *allocate(size_t size);
    void deallocate(void *pv);

private:
    uintptr_t cudaMalloc(size_t size);
    void cudaFree(void *pv);

    uintptr_t allocFromFreeHeap(size_t size);
    void deallocateToFreeHeap(uintptr_t addr);
    
    uintptr_t allocFromFixedSizeSeries(size_t size);
    void deallocateInFixedSizeSeries(uintptr_t addr);
    
    enum MemSource {
        fixedSizeSeries = 0,
        freeHeapMap = 1,
        fromCudaMalloc = 2,
    };
    struct HeapProp {
        HeapProp() { }
        HeapProp(size_t _size, enum MemSource _src)
                : size(_size), src(_src) { }
        size_t size;
        enum MemSource src;
    };

    FreeHeapMap freeHeapMap_;
    DeviceMemoryFixedSizeSeries fixedSizeSeries_;
    
    typedef std::map<uintptr_t, HeapProp> HeapMap;
    HeapMap heapMap_;

    sqaod::ArrayType<void*> d_mems_;
};


#endif
