#include <luisa/core/logging.h>
#include <luisa/ast/function_builder.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/backends/ext/raster_cmd.h>
#include <numeric>
namespace luisa::compute {

std::byte *ShaderDispatchCmdEncoder::_make_space(size_t size) noexcept {
    auto offset = _argument_buffer.size();
    _argument_buffer.resize(offset + size);
    return _argument_buffer.data() + offset;
}

ShaderDispatchCmdEncoder::ShaderDispatchCmdEncoder(
    uint64_t handle,
    size_t arg_count,
    size_t uniform_size) noexcept
    : _handle{handle}, _argument_count{arg_count} {
    if (auto arg_size_bytes = arg_count * sizeof(Argument)) {
        _argument_buffer.reserve(arg_size_bytes + uniform_size);
        luisa::enlarge_by(_argument_buffer, arg_size_bytes);
    }
}

ShaderDispatchCmdEncoder::Argument &ShaderDispatchCmdEncoder::_create_argument() noexcept {
    auto idx = _argument_idx;
    _argument_idx++;
    return *std::launder(reinterpret_cast<Argument *>(_argument_buffer.data()) + idx);
}

void ShaderDispatchCmdEncoder::encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::BUFFER;
    arg.buffer = ShaderDispatchCommandBase::Argument::Buffer{handle, offset, size};
}

void ShaderDispatchCmdEncoder::encode_texture(uint64_t handle, uint32_t level) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::TEXTURE;
    arg.texture = ShaderDispatchCommandBase::Argument::Texture{handle, level};
}

void ShaderDispatchCmdEncoder::encode_uniform(const void *data, size_t size) noexcept {
    auto offset = _argument_buffer.size();
    luisa::enlarge_by(_argument_buffer, size);
    std::memcpy(_argument_buffer.data() + offset, data, size);
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::UNIFORM;
    arg.uniform.offset = offset;
    arg.uniform.size = size;
}

void ComputeDispatchCmdEncoder::set_dispatch_size(uint3 launch_size) noexcept {
    _dispatch_size = launch_size;
}
void ComputeDispatchCmdEncoder::set_dispatch_sizes(luisa::span<const uint3> sizes) noexcept {
    luisa::vector<uint3> vec;
    luisa::enlarge_by(vec, sizes.size());
    std::memcpy(vec.data(), sizes.data(), sizes.size_bytes());
    _dispatch_size = std::move(vec);
}

void ComputeDispatchCmdEncoder::set_dispatch_size(IndirectDispatchArg indirect_arg) noexcept {
    _dispatch_size = indirect_arg;
}

void ShaderDispatchCmdEncoder::encode_bindless_array(uint64_t handle) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::BINDLESS_ARRAY;
    arg.bindless_array = Argument::BindlessArray{handle};
}

void ShaderDispatchCmdEncoder::encode_accel(uint64_t handle) noexcept {
    auto &&arg = _create_argument();
    arg.tag = Argument::Tag::ACCEL;
    arg.accel = Argument::Accel{handle};
}

size_t ShaderDispatchCmdEncoder::compute_uniform_size(luisa::span<const Variable> arguments) noexcept {
    return std::accumulate(
        arguments.begin(), arguments.end(),
        static_cast<size_t>(0u), [](auto size, auto arg) noexcept {
            auto arg_type = arg.type();
            // Do not allocate redundant uniform buffer
            return size + (arg_type->is_resource() ? 0u : arg_type->size());
        });
}

size_t ShaderDispatchCmdEncoder::compute_uniform_size(luisa::span<const Type *const> arg_types) noexcept {
    return std::accumulate(
        arg_types.begin(), arg_types.end(),
        static_cast<size_t>(0u), [](auto size, auto arg_type) noexcept {
            LUISA_ASSERT(arg_type != nullptr, "Invalid argument type.");
            // Do not allocate redundant uniform buffer
            return size + (arg_type->is_resource() ? 0u : arg_type->size());
        });
}

ComputeDispatchCmdEncoder::ComputeDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept
    : ShaderDispatchCmdEncoder{handle, arg_count, uniform_size} {}

RasterDispatchCmdEncoder::~RasterDispatchCmdEncoder() noexcept = default;
RasterDispatchCmdEncoder::RasterDispatchCmdEncoder(RasterDispatchCmdEncoder &&) noexcept = default;
RasterDispatchCmdEncoder &RasterDispatchCmdEncoder::operator=(RasterDispatchCmdEncoder &&) noexcept = default;

RasterDispatchCmdEncoder::RasterDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size,
                                                   luisa::span<const Function::Binding> bindings) noexcept
    : ShaderDispatchCmdEncoder{handle, arg_count, uniform_size},
      _bindings{bindings} {
}

luisa::unique_ptr<ShaderDispatchCommand> ComputeDispatchCmdEncoder::build() && noexcept {
    if (_argument_idx != _argument_count) [[unlikely]] {
        LUISA_ERROR("Required argument count {}. "
                    "Actual argument count {}.",
                    _argument_count, _argument_idx);
    }
    return luisa::make_unique<ShaderDispatchCommand>(
        _handle, std::move(_argument_buffer),
        _argument_count, _dispatch_size);
}

void RasterDispatchCmdEncoder::set_raster_state(const RasterState &raster_state) {
    _raster_state = raster_state;
}

void RasterDispatchCmdEncoder::set_mesh_format(MeshFormat const *mesh_format) {
    _mesh_format = mesh_format;
}

luisa::unique_ptr<Command> RasterDispatchCmdEncoder::build() && noexcept {
    if (_argument_idx != _argument_count) [[unlikely]] {
        LUISA_ERROR("Required argument count {}. "
                    "Actual argument count {}.",
                    _argument_count, _argument_idx);
    }
    return luisa::make_unique<DrawRasterSceneCommand>(
        _handle, std::move(_argument_buffer),
        _argument_count, _rtv_texs, _rtv_count,
        _dsv_tex, std::move(_scene), _viewport, _raster_state, _mesh_format);
}

void RasterDispatchCmdEncoder::set_rtv_texs(luisa::span<const ShaderDispatchCommandBase::Argument::Texture> tex) noexcept {
    LUISA_ASSERT(tex.size() <= 8, "Too many render targets: {}.", tex.size());
    _rtv_count = tex.size();
    memcpy(_rtv_texs.data(), tex.data(), tex.size_bytes());
}

void RasterDispatchCmdEncoder::set_dsv_tex(ShaderDispatchCommandBase::Argument::Texture tex) noexcept {
    _dsv_tex = tex;
}

void RasterDispatchCmdEncoder::set_scene(luisa::vector<RasterMesh> &&scene) noexcept {
    _scene = std::move(scene);
}

void RasterDispatchCmdEncoder::set_viewport(Viewport viewport) noexcept {
    _viewport = viewport;
}

}// namespace luisa::compute
