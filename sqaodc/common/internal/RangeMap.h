#pragma once

#include <sqaodc/common/Array.h>
#include <algorithm>


namespace sqaod_internal {

namespace sq = sqaod;

struct RangeMap {
    typedef sq::ArrayType<sq::PackedBitSetPair> PackedBitSetPairArray;
    PackedBitSetPairArray ranges_;

    void clear() {
        ranges_.clear();
    }

    sq::SizeType size() const {
        return ranges_.size();
    }

    const sq::PackedBitSetPair &operator[](sq::IdxType idx) const {
        return ranges_[idx];
    }
    
    void insert(sq::PackedBitSet begin, sq::PackedBitSet end) {

        struct compare_begin {
            compare_begin(sq::PackedBitSet v) : v_(v) { }
            bool operator()(const sq::PackedBitSetPair &pair) const {
                return v_ == pair.bits0;
            }

            sq::PackedBitSet v_;
        };
        struct compare_end {
            compare_end(sq::PackedBitSet v) : v_(v) { }
            bool operator()(const sq::PackedBitSetPair &pair) const {
                return v_ == pair.bits1;
            }

            sq::PackedBitSet v_;
        };
        struct less {
            bool operator()(const sq::PackedBitSetPair &lhs,
                            const sq::PackedBitSetPair &rhs) const {
                return lhs.bits0 < rhs.bits0;
            }
        };

        bool concat = false;

        if (begin == end)
            return; /* empty range */
        
        PackedBitSetPairArray::iterator it;
        it = std::find_if(ranges_.begin(), ranges_.end(), compare_end(begin));
        if (it != ranges_.end()) {
            concat = true;
            it->bits1 = end;
            PackedBitSetPairArray::iterator endIt =
                    std::find_if(ranges_.begin(), ranges_.end(), compare_begin(end));
            if (endIt != ranges_.end()) {
                it->bits1 = endIt->bits1;
                ranges_.erase(endIt);
            }
        }
        else {
            it = std::find_if(ranges_.begin(), ranges_.end(), compare_begin(end));
            if (it != ranges_.end()) {
                it->bits0 = begin;
                concat = true;
            }
        }

        if (!concat) {
            it = std::upper_bound(ranges_.begin(), ranges_.end(),
                                  sq::PackedBitSetPair(begin, 0), less());
            ranges_.insert(it, sq::PackedBitSetPair(begin, end));
        }
    }
};


struct RangeMapArray {
    RangeMapArray() { maps_ = NULL; }
    ~RangeMapArray() { delete [] maps_; }

    void setSize(sq::SizeType size) {
        delete [] maps_;
        maps_ = new RangeMap[size];
        size_ = size;
    }

    RangeMap &operator[](sq::IdxType idx) {
        assert((0 <= idx) && (idx < size_));
        return maps_[idx];
    }

    sq::SizeType size() const {
        return size_;
    }
    

private:    
    RangeMap *maps_;
    sq::SizeType size_;
};

}

