#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/undefined.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/function.h>

namespace luisa::compute {
class Type;
}// namespace luisa::compute

namespace luisa::compute::xir {

class Constant;

class LC_XIR_API Module final : public MetadataListMixin<PoolOwner> {

private:
    FunctionList _function_list;
    ConstantList _constant_list;
    UndefinedList _undefined_list;
    SpecialRegisterList _special_register_list;

    // maps for uniquifying
    luisa::unordered_map<uint64_t, Constant *> _hash_to_constant;
    luisa::unordered_map<const Type *, Undefined *> _type_to_undefined;
    luisa::unordered_map<DerivedSpecialRegisterTag, SpecialRegister *> _tag_to_special_register;

private:
    [[nodiscard]] Constant *_get_or_create_constant(const Constant &temp) noexcept;

public:
    explicit Module(size_t init_pool_cap = 0u) noexcept : Super{init_pool_cap} {}

    [[nodiscard]] KernelFunction *create_kernel() noexcept;
    [[nodiscard]] CallableFunction *create_callable(const Type *ret_type) noexcept;
    [[nodiscard]] ExternalFunction *create_external_function(const Type *ret_type) noexcept;
    [[nodiscard]] auto &function_list() noexcept { return _function_list; }
    [[nodiscard]] auto &function_list() const noexcept { return _function_list; }

    [[nodiscard]] Constant *create_constant(const Type *type, const void *data = nullptr) noexcept;
    [[nodiscard]] Constant *create_constant_zero(const Type *type) noexcept;
    [[nodiscard]] Constant *create_constant_one(const Type *type) noexcept;
    [[nodiscard]] auto &constant_list() noexcept { return _constant_list; }
    [[nodiscard]] auto &constant_list() const noexcept { return _constant_list; }

    [[nodiscard]] Undefined *create_undefined(const Type *type) noexcept;
    [[nodiscard]] auto &undefined_list() noexcept { return _undefined_list; }
    [[nodiscard]] auto &undefined_list() const noexcept { return _undefined_list; }

    [[nodiscard]] SpecialRegister *create_special_register(DerivedSpecialRegisterTag tag) noexcept;
    [[nodiscard]] auto &special_register_list() noexcept { return _special_register_list; }
    [[nodiscard]] auto &special_register_list() const noexcept { return _special_register_list; }

    [[nodiscard]] SPR_ThreadID *create_thread_id() noexcept;
    [[nodiscard]] SPR_BlockID *create_block_id() noexcept;
    [[nodiscard]] SPR_WarpLaneID *create_warp_lane_id() noexcept;
    [[nodiscard]] SPR_DispatchID *create_dispatch_id() noexcept;
    [[nodiscard]] SPR_KernelID *create_kernel_id() noexcept;
    [[nodiscard]] SPR_ObjectID *create_object_id() noexcept;
    [[nodiscard]] SPR_BlockSize *create_block_size() noexcept;
    [[nodiscard]] SPR_WarpSize *create_warp_size() noexcept;
    [[nodiscard]] SPR_DispatchSize *create_dispatch_size() noexcept;
};

}// namespace luisa::compute::xir
