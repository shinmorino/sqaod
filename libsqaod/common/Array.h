#ifndef SQAOD_COMMON_ARRAY_H__
#define SQAOD_COMMON_ARRAY_H__

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

/* ToDo: add move c-tor */

template<class V>
struct ArrayType {
public:
    typedef V* iterator;
    typedef const V* const_iterator;
    typedef V ValueType;
    
    ArrayType(size_t capacity = 1024) {
        data_ = nullptr;
        size_ = 0;
        allocate(capacity);
    }

    ~ArrayType() {
        if (data_ != nullptr)
            deallocate();
    }
    
    void reserve(size_t capacity) {
        if (capacity < capacity_)
            return;
        V *new_data = (V*)malloc(sizeof(V) * capacity);
        if (ValueProp<V>::POD) {
            memcpy(new_data, data_, sizeof(V) * size_);
        }
        else {
            for (size_t idx = 0; idx < size_; ++idx) {
                new (&new_data[idx]) V(std::move_if_noexcept(data_[idx]));
                data_[idx].~V();
            }
        }
        free(data_);
        data_ = new_data;
        capacity_ = capacity;
    }

    size_t size() const {
        return size_;
    }

    size_t capacity() const {
        return capacity_;
    }
    
    void clear() {
        erase();
        size_ = 0;
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

    V &operator[](size_t idx) {
        assert(idx < size_); 
        return data_[idx];
    }

    const V &operator[](size_t idx) const {
        assert(idx < size_); 
        return data_[idx];
    }

    const V *data() const {
        return data_;
    }
    
private:
    void erase() {
        if (!ValueProp<V>::POD) {
            for (size_t idx = 0; idx < size_; ++idx)
                data_[idx].~V();
        }
    }

    void allocate(size_t capacity) {
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
    size_t capacity_;
    size_t size_;
};


typedef unsigned long long PackedBits;
typedef ArrayType<PackedBits> PackedBitsArray;
typedef ArrayType<std::pair<PackedBits, PackedBits> > PackedBitsPairArray;

}


#endif
