#pragma once

#include <mutex>

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/mipmap.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/sparse_image.h>
#include <luisa/runtime/sparse_volume.h>

namespace luisa::compute {

class BindlessTexture2D;
class BindlessTexture3D;
class Command;

namespace detail {
class BindlessArrayExprProxy;
}// namespace detail

template<typename T>
class BindlessBuffer;

template<typename T>
struct Expr;

// BindlessArray is a heap that contain references to buffer, 2d-image and 3d-image
// every element can contain one buffer, one 2d-image and one 3d-image's reference
// see test_bindless.cpp as example
class LC_RUNTIME_API BindlessArray final : public Resource {

public:
    using Modification = BindlessArrayUpdateCommand::Modification;

    struct ModSlotHash {
        using is_transparent = void;
        using is_avalanching = void;
        [[nodiscard]] auto operator()(size_t slot) const noexcept -> uint64_t {
            return hash_value(slot);
        }
        [[nodiscard]] auto operator()(const Modification &m) const noexcept -> uint64_t {
            return (*this)(m.slot);
        }
    };

    struct ModSlotEqual {
        using is_transparent = void;

        template<typename T>
        [[nodiscard]] static auto slot(const T &m) noexcept -> size_t {
            if constexpr (requires(T m) { m.slot; }) {
                return m.slot;
            } else {
                return m;
            }
        }

        template<typename Lhs, typename Rhs>
        [[nodiscard]] auto operator()(const Lhs &lhs, const Rhs &rhs) const noexcept -> bool {
            return slot(lhs) == slot(rhs);
        }
    };

private:
    size_t _size{0u};
    // "emplace" and "remove" operations will be cached under _updates and commit in update() command
    luisa::unordered_set<Modification, ModSlotHash, ModSlotEqual> _updates;
    mutable luisa::spin_mutex _mtx;

private:
    friend class Device;
    friend class ManagedBindless;
    BindlessArray(DeviceInterface *device, size_t size) noexcept;

public:
    BindlessArray() noexcept = default;
    ~BindlessArray() noexcept override;
    using Resource::operator bool;
    BindlessArray(BindlessArray &&) noexcept;
    BindlessArray(BindlessArray const &) noexcept = delete;
    BindlessArray &operator=(BindlessArray &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    BindlessArray &operator=(BindlessArray const &) noexcept = delete;
    // properties
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        std::lock_guard lck{_mtx};
        return _size;
    }
    // whether there are any stashed updates
    [[nodiscard]] auto dirty() const noexcept {
        _check_is_valid();
        std::lock_guard lck{_mtx};
        return !_updates.empty();
    }
    // on-update functions' operations will be committed by update()
    void emplace_buffer_handle_on_update(size_t index, uint64_t handle, size_t offset_bytes) noexcept;
    void emplace_tex2d_handle_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept;
    void emplace_tex3d_handle_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept;

    BindlessArray &remove_buffer_on_update(size_t index) noexcept;
    BindlessArray &remove_tex2d_on_update(size_t index) noexcept;
    BindlessArray &remove_tex3d_on_update(size_t index) noexcept;

    template<typename T>
        requires is_buffer_or_view_v<std::remove_cvref_t<T>>
    auto &emplace_on_update(size_t index, T &&buffer) noexcept {
        size_t offset_bytes;
        if constexpr (is_buffer_view_v<std::remove_cvref_t<T>>) {
            offset_bytes = buffer.offset_bytes();
        } else {
            offset_bytes = 0;
        }
        emplace_buffer_handle_on_update(index, buffer.handle(), offset_bytes);
        return *this;
    }

    auto &emplace_on_update(size_t index, const Image<float> &image, Sampler sampler) noexcept {
        emplace_tex2d_handle_on_update(index, image.handle(), sampler);
        return *this;
    }

    auto &emplace_on_update(size_t index, const Volume<float> &volume, Sampler sampler) noexcept {
        emplace_tex3d_handle_on_update(index, volume.handle(), sampler);
        return *this;
    }

    auto &emplace_on_update(size_t index, const SparseImage<float> &texture, Sampler sampler) noexcept {
        emplace_tex2d_handle_on_update(index, texture.handle(), sampler);
        return *this;
    }

    auto &emplace_on_update(size_t index, const SparseVolume<float> &texture, Sampler sampler) noexcept {
        emplace_tex3d_handle_on_update(index, texture.handle(), sampler);
        return *this;
    }

    [[nodiscard]] luisa::unique_ptr<Command> update() noexcept;

    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::BindlessArrayExprProxy *>(this);
    }
};

}// namespace luisa::compute
