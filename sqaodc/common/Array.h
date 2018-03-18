#pragma once

#include <stdlib.h>
#include <string.h>
#include <utility>
#include <sqaodc/common/defines.h>
#include <sqaodc/common/types.h>


namespace sqaod {

template<class V> struct ValueProp { enum { POD = false }; };
template<> struct ValueProp<char> { enum { POD = true }; };
template<> struct ValueProp<unsigned char> { enum { POD = true }; };
template<> struct ValueProp<short> { enum { POD = true }; };
template<> struct ValueProp<unsigned short> { enum { POD = true }; };
template<> struct ValueProp<int> { enum { POD = true }; };
template<> struct ValueProp<unsigned int> { enum { POD = true }; };
template<> struct ValueProp<long> { enum { POD = true }; };
template<> struct ValueProp<unsigned long> { enum { POD = true }; };
template<> struct ValueProp<long long> { enum { POD = true }; };
template<> struct ValueProp<unsigned long long> { enum { POD = true }; };
template<class V> struct ValueProp<V*> { enum { POD = true }; };

/* FIXME: add move c-tor */

template<class V>
struct ArrayType {
public:
    typedef V* iterator;
    typedef const V* const_iterator;
    typedef V ValueType;
    
    ArrayType(SizeType capacity = 1024) {
        data_ = nullptr;
        size_ = 0;
        allocate(capacity);
    }

    ArrayType(const ArrayType<V> &rhs) {
        data_ = nullptr;
        size_ = 0;
        allocate(rhs.capacity());
        insert(rhs.begin(), rhs.end());
    }

    ~ArrayType() {
        if (data_ != nullptr)
            deallocate();
    }
    
    void reserve(SizeType capacity) {
        if (capacity < capacity_)
            return;
        V *new_data = (V*)malloc(sizeof(V) * capacity);
        if (ValueProp<V>::POD) {
            memcpy(new_data, data_, sizeof(V) * size_);
        }
        else {
            for (SizeType idx = 0; idx < size_; ++idx) {
                new (&new_data[idx]) V(std::move_if_noexcept(data_[idx]));
                data_[idx].~V();
            }
        }
        free(data_);
        data_ = new_data;
        capacity_ = capacity;
    }

    bool empty() const {
        return size() == 0;
    }
    
    SizeType size() const {
        return size_;
    }

    SizeType capacity() const {
        return capacity_;
    }
    
    void clear() {
        erase();
        size_ = 0;
    }

    void erase(iterator it) {
        if (ValueProp<V>::POD) {
            memmove(it, it + 1, sizeof(V) * (size_ - 1));
        }
        else {
            for (; it != end() - 1; ++it) {
                it->~V();
                new (&*it) V(std::move_if_noexcept(*(it + 1)));
            }
            data_[size_ - 1].~V();
        }
        --size_;
    }
    
    void pushBack(const V &v) {
        if (size_ == capacity_)
            reserve(capacity_ * 2);
        if (ValueProp<V>::POD)
            data_[size_] = v;
        else
            new (&data_[size_]) V(v);
        ++size_;
    }

    // FIXME: apply move_if_noexcept
    void pushBack(V &&v) {
        if (size_ == capacity_)
            reserve(capacity_ * 2);
        if (ValueProp<V>::POD)
            data_[size_] = v;
        else
            new (&data_[size_]) V(v);
        ++size_;
    }
    
    iterator begin() {
        return data_;
    }
    iterator end() {
        return data_ + size_;
    }

    const_iterator begin() const {
        return data_;
    }
    const_iterator end() const {
        return data_ + size_;
    }

    V &operator[](IdxType idx) {
        assert(idx < size_); 
        return data_[idx];
    }

    const V &operator[](IdxType idx) const {
        assert(idx < size_); 
        return data_[idx];
    }

    const V *data() const {
        return data_;
    }
    
    const ArrayType &operator=(const ArrayType &rhs) {
        clear();
        reserve(rhs.capacity());
        insert(rhs.begin(), rhs.end());
        return rhs;
    }

    bool operator==(const ArrayType &rhs) const {
        if (size_ != rhs.size_)
            return false;
        if (ValueProp<V>::POD) {
            return memcmp(data_, rhs.data_, sizeof(V) * size_) == 0;
        }
        else {
            for (IdxType idx = 0; idx < size_; ++idx) {
                if (data_[idx] != rhs.data_[idx])
                    return false;
            }
            return true;
        }
    }

    bool operator!=(const ArrayType &rhs) const {
        return !operator==(rhs);
    }

    void insert(const_iterator first, const_iterator last) {
        SizeType nElms = SizeType(last - first);
        if (capacity_ < size_ + nElms)
            reserve(capacity_ * 2);
        if (ValueProp<V>::POD) {
            memcpy(&data_[size_], first, sizeof(V) * nElms);
            size_ += nElms;
        }
        else {
            for (const_iterator it = first; it != last; ++it) {
                new (&data_[size_]) V(*it);
                ++size_;
            }
        }
    }

    void insert(const_iterator pos, const ValueType &v) {
        if (capacity_ < size_ + 1)
            reserve(capacity_ * 2);
        
        /* move */
        SizeType nToMove = SizeType(end() - pos);
        IdxType posIdx = SizeType(pos - begin());
        if (ValueProp<V>::POD) {
            memmove(&data_[posIdx + 1], &data_[posIdx], sizeof(V) * nToMove);
        }
        else {
            IdxType dstIdx  = size();
            IdxType srcIdx  = size() - 1;
            for (SizeType idx = 0; idx < nToMove; ++idx) {
                new (&data_[dstIdx - idx]) V(data_[srcIdx - idx]);
                data_[srcIdx - idx].~V();
            }
        }
        
        if (ValueProp<V>::POD) {
            memcpy(&data_[posIdx], &v, sizeof(V));
        }
        else {
            new (&data_[posIdx]) V(v);
        }
        ++size_;
    }


private:
    void erase() {
        if (!ValueProp<V>::POD) {
            for (IdxType idx = 0; idx < size_; ++idx)
                data_[idx].~V();
        }
    }

    void allocate(SizeType capacity) {
        assert(data_ == nullptr);
        data_ = (V*)malloc(sizeof(V) * capacity);
        capacity_ = capacity;
    }

    void deallocate() {
        erase();
        free(data_);
        data_ = nullptr;
    }
    
    V *data_;
    SizeType capacity_;
    SizeType size_;
};


typedef ArrayType<PackedBitSet> PackedBitSetArray;
typedef ArrayType<PackedBitSetPair> PackedBitSetPairArray;

}
