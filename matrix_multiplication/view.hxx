#include <ctime>
#include <cmath>
#include <cstring>
#include <string.h>
#include <cstdlib>
#include <cassert>

#include <iterator>
#include <iostream>
#include <vector>
#include <type_traits>
#include <random>
#include <sstream>
#include <stdexcept>

#include "RAJA/RAJA.hxx"

#if defined(RAJA_ENABLE_CUDA)
#    include <cuda.h>
#    include <cuda_runtime.h>
#endif

template <typename MemSpace>
struct is_GPU_memory {
#if !defined(RAJA_ENABLE_CUDA)
  static_assert(!MemSpace::IsGPU, "GPU memory without CUDA support!");
#endif
  constexpr static bool value = MemSpace::IsGPU;
};

template <typename MemSpace>
class Data {
  template <typename Category, typename T, typename Distance = std::ptrdiff_t, typename Pointer = T*, typename Reference = T&>
  class Iterator;
public:
  using value_type = RAJA::Real_type;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = RAJA::Real_ptr;
  using const_pointer = const RAJA::Real_ptr;
  using iterator = Iterator<std::random_access_iterator_tag, value_type>;
  using const_iterator = Iterator<std::random_access_iterator_tag, const value_type>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type = std::ptrdiff_t;
  using size_type = RAJA::Index_type;

#if defined(RAJA_ENABLE_CUDA)
  template <typename std::enable_if<is_GPU_memory<MemSpace>::value>::type>
  Data(const size_type numElements) 
      : size_(numElements * sizeof(value_type)) {
    cudaErrchk(cudaMallocManaged((void **)&data_, size_, cudaMemAttachGlobal));
  }

  template <typename std::enable_if<is_GPU_memory<MemSpace>::value>::type>
  Data(const size_type numElements, const value_type defaultValue) 
      : Data(numElements) {
    cudaMemset(data_, defaultValue, size_);
  }

  template <typename std::enable_if<is_GPU_memory<MemSpace>::value>::type>
  Data(const Data<Memspace>& rhs) : Data(rhs.numElements / sizeof(value_type)) {
    cudaErrchk(cudaMemcpy((void *)data_, (const void *) rhs.data_, size_, cudaMemcpyDeviceToDevice)); 
  }

  template <typename RhsMemSpace, 
            typename std::enable_if<
              is_GPU_memory<RhsMemSpace>::value 
              && is_GPU_memory<MemSpace>::value
            >::type>
  Data(const Data<RhsMemspace>& rhs) : Data(rhs.numElements / sizeof(value_type)) {
    cudaErrchk(cudaMemcpy((void *)data_, (const void *) rhs.data_, size_, cudaMemcpyDeviceToDevice));
  }

  template <typename RhsMemSpace, 
            typename std::enable_if<
              !is_GPU_memory<RhsMemSpace>::value 
              && is_GPU_memory<MemSpace>::value
            >::type>
  Data(const Data<RhsMemspace>& rhs) : Data(rhs.numElements / sizeof(value_type)) {
    cudaErrchk(cudaMemcpy((void *)data_, (const void *) rhs.data_, size_, cudaMemcpyHostToDevice));
  }

  template <typename RhsMemSpace, 
            typename std::enable_if<
              is_GPU_memory<RhsMemSpace>::value 
              && !is_GPU_memory<MemSpace>::value
            >::type>
  Data(const Data<RhsMemspace>& rhs) : Data(rhs.numElements / sizeof(value_type)) {
    cudaErrchk(cudaMemcpy((void *)data_, (const void *) rhs.data_, size_, cudaMemcpyDeviceToHost));
  }
#endif
  
  template <typename std::enable_if<!is_GPU_memory<MemSpace>::value, int>::type = 0>
  Data(const size_type numElements) 
      : data_(nullptr), size_(numElements * sizeof(value_type)) {
    //auto error = posix_memalign((void**)&data_, RAJA::DATA_ALIGN, size_);

    //if (error) {
    //  throw std::runtime_error(strerror(error));
    //}
    data_ = static_cast<pointer>(malloc(size_));
    if (!data_) {
      throw std::runtime_error("Could not allocate memory.");
    }
  }

  template <typename std::enable_if<!is_GPU_memory<MemSpace>::value, int>::type = 0>
  Data(const size_type numElements, const value_type defaultValue) 
      : Data(numElements) {
    memset(data_, defaultValue, size_);
  }

  template <typename RhsMemSpace, 
            typename std::enable_if<
                !is_GPU_memory<MemSpace>::value
                && !is_GPU_memory<RhsMemSpace>::value
              , int
            >::type = 0>
  Data(const Data<RhsMemSpace>& rhs) : Data(rhs.numElements / sizeof(value_type)) {
    memcpy(data_, rhs.data_, size_);
  }

  ~Data() {
    if (!data_) {
      freer();
    }

    data_ = nullptr;
  }

  Data<MemSpace>& operator=(Data<MemSpace> rhs) {
    using std::swap;
    swap(size_, rhs.size_);
    swap(data_, rhs.data_);
  }

  reference operator[](const size_type index) { return data_[index]; }
  const_reference operator[](const size_type index) const { return data_[index]; }

  reference at(const size_type index) {
    if (index > 0 && index < size_) {
      return data_[index];
    }

    _at_err(index);
  }

  const_reference at(const size_type index) const {
    if (index > 0 && index < size_) {
      return data_[index];
    }
    
    _at_err(index);
  }

  reference front() { return data_[0]; }
  const_reference front() const { return data_[0]; }
  reference back() { return data_[size_/sizeof(value_type) - 1]; }
  const_reference back() const { return data_[size_/sizeof(value_type) - 1]; }
  pointer data() { return data_; }
  const_pointer data() const { return data_; }

  iterator begin() { return iterator(data_); }
  const_iterator begin() const { return const_iterator(data_); }
  const_iterator cbegin() const { return begin(); }
  iterator end() { return iterator(data_ + size_/sizeof(value_type)); }
  const_iterator end() const { return const_iterator(data_ + size_/sizeof(value_type)); }
  const_iterator cend() const { return end(); }

  reverse_iterator rbegin() { return reverse_iterator(data_ + size_/sizeof(value_type) - 1); }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(data_ + size_/sizeof(value_type) - 1); }
  const_reverse_iterator crbegin() const { return rbegin(); }
  reverse_iterator rend() { return reverse_iterator(data_ - 1); }
  const_reverse_iterator rend() const { return const_reverse_iterator(data_ - 1); }
  const_reverse_iterator crend() const { return rend(); }

  size_type size() const { return size_ / sizeof(value_type); }

private:
  pointer data_;
  size_type size_;

  template <typename Category, typename T, typename Distance, typename Pointer, typename Reference>
  class Iterator {
  public:
    using value_type = T;
    using difference_type = Distance;
    using pointer = Pointer;
    using reference = Reference;
    using iterator_category = Category;

    Iterator() : p_(nullptr) { }
    Iterator(pointer p) : p_(p) { }

    Iterator& operator++() { ++p_; return *this; }
    Iterator operator++(int) { auto tmp = *this; operator++(); return tmp; }
    Iterator operator+(const difference_type n) { auto tmp = *this; tmp.p_ += n; return tmp; }
    friend Iterator operator+(const difference_type n, Iterator rhs) { return rhs + n; }
    Iterator& operator+=(const difference_type n) { p_ += n; return *this; }

    Iterator& operator--() { --p_; return *this; }
    Iterator operator--(int) { auto tmp = *this; operator--(); return tmp; }
    Iterator operator-(const difference_type n) { auto tmp = *this; tmp.p_ -= n; return tmp; }
    Iterator& operator-=(const difference_type n) { p_ -= n; return *this; }

    reference operator*() { return *p_; }
    pointer operator->() { return p_; }
    reference operator[](const difference_type n) { return p_[n]; }

    bool operator==(const Iterator& rhs) { return p_ == rhs.p_; }
    bool operator!=(const Iterator& rhs) { return !operator==(rhs); }
    bool operator<(const Iterator rhs) { return p_ < rhs.p_; }
    bool operator>(const Iterator rhs) { return p_ > rhs.p_; }
    bool operator<=(const Iterator rhs) { return p_ <= rhs.p_; }
    bool operator>=(const Iterator rhs) { return p_ >= rhs.p_; }

  private:
    pointer p_;   
  };

#if defined(RAJA_ENABLE_CUDA)
  template <typename std::enable_if<is_GPU_memory<MemSpace>::value>::type>
  void freer() {
    cudaErrchk(cudaFree(data_));
  }
#endif

  template <typename std::enable_if<!is_GPU_memory<MemSpace>::value, int>::type = 0>
  void freer() {
    free(data_);
  }
  
  void _at_err(const size_type index) {
    std::ostringstream os;
    os << "Invalid index " << index << " for size " << size_ << ".\r\n";
    throw std::out_of_range(os.str());
  }
};
