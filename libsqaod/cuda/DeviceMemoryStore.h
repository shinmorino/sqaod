#ifndef SQAOD_CUDA_DEVICEMEMORYSTORE_H__
#define SQAOD_CUDA_DEVICEMEMORYSTORE_H__

#include <map>
#include <common/Array.h>
#include <assert.h>

/* ToDo: This implementatin is enough for now.  Will be reconsidered when needed. */

class HeapBitmap {
    typedef uint32_t  RegionBitmap;
    typedef std::map<uintptr_t, RegionBitmap> RegionMap;

public:
    HeapBitmap() { }

    void set(int nActiveBits, int sizeInPo2)  {
        nActiveBits_ = nActiveBits;
        sizeInPo2_ = sizeInPo2;
        mask_ = RegionBitmap((1ull << nActiveBits) - 1);
    }

    void clear() {
        freeRegions_.clear();
        regions_.clear();
    }
    
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



class FixedSizedChunks {
public:
    void initialize();
    void finalize();

    void addHeap(uintptr_t pv, size_t size);
    
    uintptr_t allocate(size_t size);
    void deallocate(uintptr_t addr);

private:
    HeapBitmap *allocateRegion(int layerIdx);

    enum { nBitmapLayers = 14 };
    HeapBitmap bitmapLayers_[nBitmapLayers];
};




class FreeHeapMap {
public:
    FreeHeapMap();

    void clearRegions() {
        freeRegions_.clear();
    }

    void addFreeHeap(uintptr_t addr, size_t size);
    
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
    void finalize();
    
    void *allocate(size_t size);
    void deallocate(void *pv);

private:
    uintptr_t cudaMalloc(size_t size);
    void cudaFree(void *pv);

    uintptr_t allocFromFreeHeap(size_t size);
    void deallocateToFreeHeap(uintptr_t addr, size_t size);
    
    uintptr_t allocFromFixedSizedChunks(size_t size);
    void deallocateInFixedSizedChunks(uintptr_t addr);
    
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
    FixedSizedChunks fixedSizedChunks_;
    
    typedef std::map<uintptr_t, HeapProp> HeapMap;
    HeapMap heapMap_;

    sqaod::ArrayType<void*> d_mems_;
};


#endif
