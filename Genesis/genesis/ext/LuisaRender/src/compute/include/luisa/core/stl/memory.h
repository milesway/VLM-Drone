#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <memory>
#include <cstring>

#ifdef LUISA_USE_SYSTEM_STL

#include <span>
#include <bit>

#else

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <EASTL/bit.h>
#include <EASTL/memory.h>
#include <EASTL/shared_array.h>
#include <EASTL/unique_ptr.h>
#include <EASTL/shared_ptr.h>
#include <EASTL/span.h>
#include <EASTL/bonus/compressed_pair.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/hash_fwd.h>

namespace luisa {

inline namespace size_literals {

[[nodiscard]] constexpr auto operator""_k(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u);
}

[[nodiscard]] constexpr auto operator""_M(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u * 1024u);
}

[[nodiscard]] constexpr auto operator""_G(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u * 1024u * 1024u);
}

}// namespace size_literals

namespace detail {
LUISA_EXPORT_API void *allocator_allocate(size_t size, size_t alignment) noexcept;
LUISA_EXPORT_API void allocator_deallocate(void *p, size_t alignment) noexcept;
LUISA_EXPORT_API void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept;
}// namespace detail

[[nodiscard]] inline auto align(size_t s, size_t a) noexcept {
    return (s + (a - 1)) & ~(a - 1);
}

#ifdef LUISA_USE_SYSTEM_STL

template<typename T = std::byte>
using allocator = std::allocator<T>;

#else

template<typename T = std::byte>
struct allocator {
    using value_type = T;
    constexpr allocator() noexcept = default;
    explicit constexpr allocator(const char *) noexcept {}
    template<typename U>
    constexpr allocator(allocator<U>) noexcept {}
    [[nodiscard]] auto allocate(std::size_t n) const noexcept {
        return static_cast<T *>(luisa::detail::allocator_allocate(sizeof(T) * n, alignof(T)));
    }
    [[nodiscard]] auto allocate(std::size_t n, size_t alignment, size_t) const noexcept {
        return static_cast<T *>(luisa::detail::allocator_allocate(sizeof(T) * n, alignment));
    }
    void deallocate(T *p, size_t) const noexcept {
        luisa::detail::allocator_deallocate(p, alignof(T));
    }
    void deallocate(void *p, size_t) const noexcept {
        luisa::detail::allocator_deallocate(p, alignof(T));
    }
    template<typename R>
    [[nodiscard]] constexpr auto operator==(allocator<R>) const noexcept -> bool {
        return std::is_same_v<T, R>;
    }
};

#endif

template<typename T>
[[nodiscard]] inline auto allocate_with_allocator(size_t n = 1u) noexcept {
    return allocator<T>{}.allocate(n);
}

template<typename T>
inline void deallocate_with_allocator(T *p) noexcept {
    allocator<T>{}.deallocate(p, 0u);
}

template<typename T, typename... Args>
[[nodiscard]] inline auto new_with_allocator(Args &&...args) noexcept {
    return std::construct_at(allocate_with_allocator<T>(), std::forward<Args>(args)...);
}

template<typename T>
inline void delete_with_allocator(T *p) noexcept {
    if (p != nullptr) {
        std::destroy_at(p);
        deallocate_with_allocator(p);
    }
}

#ifdef LUISA_USE_SYSTEM_STL

using std::bit_cast;
using std::span;

using std::const_pointer_cast;
using std::dynamic_pointer_cast;
using std::enable_shared_from_this;
using std::make_shared;
using std::make_unique;
using std::reinterpret_pointer_cast;
using std::shared_ptr;
using std::static_pointer_cast;
using std::unique_ptr;
using std::weak_ptr;

using std::make_pair;
using std::aligned_storage_t;

#else

using eastl::bit_cast;
using eastl::span;

using eastl::make_pair;

// smart pointers
using eastl::compressed_pair;
using eastl::const_pointer_cast;
using eastl::dynamic_pointer_cast;
using eastl::enable_shared_from_this;
using eastl::make_shared;
using eastl::make_unique;
using eastl::reinterpret_pointer_cast;
using eastl::shared_array;
using eastl::shared_ptr;
using eastl::static_pointer_cast;
using eastl::unique_ptr;
using eastl::weak_ptr;

using eastl::aligned_storage_t;

#endif

// hash functions
template<typename T>
struct pointer_hash {
    using is_transparent = void;
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(const T *p,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        auto x = reinterpret_cast<uint64_t>(p);
        return hash64(&x, sizeof(x), seed);
    }
    [[nodiscard]] uint64_t operator()(const volatile T *p,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        auto x = reinterpret_cast<uint64_t>(p);
        return hash64(&x, sizeof(x), seed);
    }
    [[nodiscard]] uint64_t operator()(const shared_ptr<T> &ptr,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        return (*this)(ptr.get(), seed);
    }
    template<typename Deleter>
    [[nodiscard]] uint64_t operator()(const unique_ptr<T, Deleter> &ptr,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        return (*this)(ptr.get(), seed);
    }
#ifndef LUISA_USE_SYSTEM_STL
    [[nodiscard]] uint64_t operator()(const std::shared_ptr<T> &ptr,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        return (*this)(ptr.get(), seed);
    }
    template<typename Deleter>
    [[nodiscard]] uint64_t operator()(const std::unique_ptr<T, Deleter> &ptr,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        return (*this)(ptr.get(), seed);
    }
#endif
};

template<>
struct pointer_hash<void> {
    using is_transparent = void;
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(const void *p,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        auto x = reinterpret_cast<uint64_t>(p);
        return hash64(&x, sizeof(x), seed);
    }
    [[nodiscard]] uint64_t operator()(const volatile void *p,
                                      uint64_t seed = hash64_default_seed) const noexcept {
        auto x = reinterpret_cast<uint64_t>(p);
        return hash64(&x, sizeof(x), seed);
    }
};

}// namespace luisa
