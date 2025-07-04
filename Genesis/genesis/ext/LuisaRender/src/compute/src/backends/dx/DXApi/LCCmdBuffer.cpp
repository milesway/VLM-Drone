#include <DXApi/LCCmdBuffer.h>
#include <DXApi/LCDevice.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/command_list.h>
#include <Shader/ComputeShader.h>
#include <Resource/RenderTexture.h>
#include <Resource/TopAccel.h>
#include <DXApi/LCSwapChain.h>
#include "../../common/command_reorder_visitor.h"
#include <Shader/RasterShader.h>
#include <luisa/core/stl/variant.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/aabb.h>
#include <Resource/DepthBuffer.h>
#include <luisa/backends/ext/raster_cmd.h>
#include <Resource/SparseTexture.h>
#include <luisa/backends/ext/dx_custom_cmd.h>
#include "../../common/shader_print_formatter.h"
#ifdef LCDX_ENABLE_WINPIX
#include <WinPixEventRuntime/pix3.h>
#endif

namespace lc::dx {
using Argument = luisa::compute::Argument;
static bool is_device_buffer(Resource const *res) {
    auto tag = res->GetTag();
    return (tag != Resource::Tag::UploadBuffer) && (tag != Resource::Tag::ReadbackBuffer);
}
template<typename Visitor>
void DecodeCmd(vstd::span<const Argument> args, Visitor &&visitor) {
    using Tag = Argument::Tag;
    for (auto &&i : args) {
        switch (i.tag) {
            case Tag::BUFFER: {
                visitor(i.buffer);
            } break;
            case Tag::TEXTURE: {
                visitor(i.texture);
            } break;
            case Tag::UNIFORM: {
                visitor(i.uniform);
            } break;
            case Tag::BINDLESS_ARRAY: {
                visitor(i.bindless_array);
            } break;
            case Tag::ACCEL: {
                visitor(i.accel);
            } break;
            default: {
                LUISA_ASSUME(false);
            } break;
        }
    }
}
class LCPreProcessVisitor : public CommandVisitor {
public:
    CommandBufferBuilder *bd;
    ResourceStateTracker *stateTracker;
    vstd::vector<std::pair<size_t, size_t>> *argVecs;
    vstd::vector<uint8_t> *argBuffer;
    vstd::vector<BottomAccelData> *bottomAccelDatas;
    vstd::fixed_vector<std::pair<size_t, size_t>, 4> *accelOffset;
    size_t buildAccelSize = 0;
    void AddBuildAccel(size_t size) {
        size = CalcAlign(size, 256);
        accelOffset->emplace_back(buildAccelSize, size);
        buildAccelSize += size;
    }
    void UniformAlign(size_t align) const {
#ifdef LUISA_USE_SYSTEM_STL
        argBuffer->resize(CalcAlign(argBuffer->size(), align));
#else
        argBuffer->resize_uninitialized(CalcAlign(argBuffer->size(), align));
#endif
    }
    template<typename T>
    void EmplaceData(T const &data) {
        size_t sz = argBuffer->size();
        luisa::enlarge_by(*argBuffer, sizeof(T));
        using PlaceHolder = luisa::aligned_storage_t<sizeof(T), 1>;
        *reinterpret_cast<PlaceHolder *>(argBuffer->data() + sz) =
            *reinterpret_cast<PlaceHolder const *>(&data);
    }
    template<typename T>
    void EmplaceData(T const *data, size_t size) {
        size_t sz = argBuffer->size();
        auto byteSize = size * sizeof(T);
        luisa::enlarge_by(*argBuffer, byteSize);
        std::memcpy(argBuffer->data() + sz, data, byteSize);
    }
    struct Visitor {
        LCPreProcessVisitor *self;
        SavedArgument const *arg;
        ShaderDispatchCommandBase const &cmd;
        void operator()(Argument::Buffer const &bf) {
            auto res = reinterpret_cast<Buffer const *>(bf.handle);
            if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                LUISA_ASSERT(is_device_buffer(res), "Unordered access buffer can not be host-buffer.");
                self->stateTracker->RecordState(
                    res,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    true);
            } else {
                if (is_device_buffer(res))
                    self->stateTracker->RecordState(
                        res,
                        self->stateTracker->ReadState(ResourceReadUsage::Srv));
                else {
                    LUISA_ASSERT(res->GetTag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as shader's resource.");
                }
            }
            ++arg;
        }
        void operator()(Argument::Texture const &bf) {
            auto rt = reinterpret_cast<TextureBase *>(bf.handle);
            //UAV
            if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                self->stateTracker->RecordState(
                    rt,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    true);
            }
            // SRV
            else {
                self->stateTracker->RecordState(
                    rt,
                    self->stateTracker->ReadState(ResourceReadUsage::Srv, rt));
            }
            ++arg;
        }
        void operator()(Argument::BindlessArray const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            vstd::fixed_vector<Resource const *, 16> writeMap;
            {
                arr->Lock();
                auto unlocker = vstd::scope_exit([&] {
                    arr->Unlock();
                });
                for (auto &&i : self->stateTracker->WriteStateMap()) {
                    if (arr->IsPtrInBindless(reinterpret_cast<size_t>(i))) {
                        writeMap.emplace_back(i);
                    }
                }
            }
            if (!writeMap.empty()) {
                auto readState = self->stateTracker->ReadState(ResourceReadUsage::Srv);
                for (auto &&i : writeMap) {
                    self->stateTracker->RecordState(i, readState);
                }
            }
            ++arg;
        }
        void operator()(Argument::Uniform const &a) {
            auto bf = cmd.uniform(a);
            if (bf.size() < 4) {
                bool v = (bool)bf[0];
                uint value = v ? std::numeric_limits<uint>::max() : 0;
                self->EmplaceData(value);
            } else {
                self->EmplaceData(bf.data(), arg->structSize);
            }
            ++arg;
        }
        void operator()(Argument::Accel const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            if (accel->GetInstBuffer()) {
                if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                    self->stateTracker->RecordState(
                        accel->GetInstBuffer(),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        true);
                } else {
                    self->stateTracker->RecordState(
                        accel->GetInstBuffer(),
                        self->stateTracker->ReadState(ResourceReadUsage::Srv));
                    self->stateTracker->RecordState(
                        accel->GetAccelBuffer(),
                        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
                }
            }
            ++arg;
        }
    };
    void visit(const DXCustomCmd *cmd) noexcept {
        for (auto &&i : cmd->get_resource_usages()) {
            uint64_t handle =
                luisa::visit(
                    [](auto &&t) -> uint64_t {
                        return t.handle;
                    },
                    i.resource);
            stateTracker->RecordState(reinterpret_cast<Resource const *>(handle), i.required_state,
                                      (i.required_state & (D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE | D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) != 0);
        }
    }
    void visit(const BufferUploadCommand *cmd) noexcept override {
        auto res = reinterpret_cast<Buffer const *>(cmd->handle());
        if (is_device_buffer(res))
            stateTracker->RecordState(res, D3D12_RESOURCE_STATE_COPY_DEST);
        else {
            LUISA_ERROR("Host-buffer should not be used to upload.");
        }
    }
    void visit(const BufferDownloadCommand *cmd) noexcept override {
        auto res = reinterpret_cast<Buffer const *>(cmd->handle());
        if (is_device_buffer(res))
            stateTracker->RecordState(res, stateTracker->ReadState(ResourceReadUsage::CopySource));
        else {
            LUISA_ERROR("Host-buffer should not be used to download.");
        }
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        if (is_device_buffer(srcBf))
            stateTracker->RecordState(srcBf, stateTracker->ReadState(ResourceReadUsage::CopySource));
        else {
            LUISA_ASSERT(srcBf->GetTag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as copy source.");
        }
        if (is_device_buffer(dstBf))
            stateTracker->RecordState(dstBf, D3D12_RESOURCE_STATE_COPY_DEST);
        else {
            LUISA_ASSERT(dstBf->GetTag() == Resource::Tag::ReadbackBuffer, "Only non write-combined-buffer allowed as copy destination.");
        }
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker->RecordState(
            rt,
            D3D12_RESOURCE_STATE_COPY_DEST);
        if (is_device_buffer(bf))
            stateTracker->RecordState(
                bf,
                stateTracker->ReadState(ResourceReadUsage::CopySource));
        else {
            LUISA_ASSERT(bf->GetTag() == Resource::Tag::UploadBuffer, "Only upload-buffer allowed as copy source.");
        }
    }

    void visit(const TextureUploadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        stateTracker->RecordState(
            rt,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const ClearDepthCommand *cmd) noexcept {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        stateTracker->RecordState(
            rt,
            D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        stateTracker->RecordState(
            rt,
            stateTracker->ReadState(ResourceReadUsage::CopySource, rt));
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
        auto src = reinterpret_cast<TextureBase *>(cmd->src_handle());
        auto dst = reinterpret_cast<TextureBase *>(cmd->dst_handle());
        stateTracker->RecordState(
            src,
            stateTracker->ReadState(ResourceReadUsage::CopySource, src));
        stateTracker->RecordState(
            dst,
            D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        stateTracker->RecordState(
            rt,
            stateTracker->ReadState(ResourceReadUsage::CopySource, rt));
        if (is_device_buffer(bf))
            stateTracker->RecordState(
                bf,
                D3D12_RESOURCE_STATE_COPY_DEST);
        else {
            LUISA_ASSERT(bf->GetTag() == Resource::Tag::ReadbackBuffer, "Only non write-combined-buffer allowed as copy destination.");
        }
    }
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
        auto cs = reinterpret_cast<ComputeShader *>(cmd->handle());
        size_t beforeSize = argBuffer->size();
        Visitor visitor{this, cs->Args().data(), *cmd};
        DecodeCmd(cs->ArgBindings(), visitor);
        DecodeCmd(cmd->arguments(), visitor);
        UniformAlign(16);
        size_t afterSize = argBuffer->size();
        argVecs->emplace_back(beforeSize, afterSize - beforeSize);
        if (cmd->is_indirect()) {
            auto buffer = reinterpret_cast<Buffer *>(cmd->indirect_dispatch().handle);
            stateTracker->RecordState(buffer, stateTracker->ReadState(ResourceReadUsage::IndirectArgs));
        }
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        if (!cmd->update_instance_buffer_only()) {
            AddBuildAccel(
                accel->PreProcess(
                    *stateTracker,
                    *bd,
                    cmd->instance_count(),
                    cmd->modifications(),
                    cmd->request() == AccelBuildRequest::PREFER_UPDATE));
        } else {
            accel->PreProcessInst(
                *stateTracker,
                *bd,
                cmd->instance_count(),
                cmd->modifications());
        }
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        BottomAccel::MeshOptions meshOptions{
            .vHandle = reinterpret_cast<Buffer const *>(cmd->vertex_buffer()),
            .vOffset = cmd->vertex_buffer_offset(),
            .vStride = cmd->vertex_stride(),
            .vSize = cmd->vertex_buffer_size(),
            .iHandle = reinterpret_cast<Buffer const *>(cmd->triangle_buffer()),
            .iOffset = cmd->triangle_buffer_offset(),
            .iSize = cmd->triangle_buffer_size()};
        AddBuildAccel(
            accel->PreProcessStates(
                *bd,
                *stateTracker,
                cmd->request() == AccelBuildRequest::PREFER_UPDATE,
                meshOptions,
                bottomAccelDatas->emplace_back()));
    }
    void visit(const MotionInstanceBuildCommand *) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }
    void visit(const ProceduralPrimitiveBuildCommand *cmd) noexcept override {
        auto accel = reinterpret_cast<BottomAccel *>(cmd->handle());
        BottomAccel::AABBOptions aabbOptions{
            .aabbBuffer = reinterpret_cast<Buffer const *>(cmd->aabb_buffer()),
            .offset = cmd->aabb_buffer_offset(),
            .size = cmd->aabb_buffer_size()};
        AddBuildAccel(
            accel->PreProcessStates(
                *bd,
                *stateTracker,
                cmd->request() == AccelBuildRequest::PREFER_UPDATE,
                aabbOptions,
                bottomAccelDatas->emplace_back()));
    }
    void visit(const CurveBuildCommand *) noexcept override { /* TODO */
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        arr->PreProcessStates(
            *bd,
            *stateTracker,
            cmd->modifications());
    };

    void visit(const CustomCommand *cmd) noexcept override {
        switch (cmd->uuid()) {
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH):
                visit(static_cast<ClearDepthCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE):
                visit(static_cast<DrawRasterSceneCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH):
                visit(static_cast<DXCustomCmd const *>(cmd));
                break;
            default:
                LUISA_ERROR("Custom command not supported by this queue.");
        }
    }

    void visit(const DrawRasterSceneCommand *cmd) noexcept {
        auto cs = reinterpret_cast<RasterShader *>(cmd->handle());
        size_t beforeSize = argBuffer->size();
        auto rtvs = cmd->rtv_texs();
        auto dsv = cmd->dsv_tex();
        DecodeCmd(cmd->arguments(), Visitor{this, cs->Args().data(), *cmd});
        UniformAlign(16);
        size_t afterSize = argBuffer->size();
        argVecs->emplace_back(beforeSize, afterSize - beforeSize);

        for (auto &&mesh : cmd->scene()) {
            for (auto &&v : mesh.vertex_buffers()) {
                stateTracker->RecordState(
                    reinterpret_cast<Buffer *>(v.handle()),
                    stateTracker->ReadState(ResourceReadUsage::VertexBufferForGraphics));
            }
            auto &&i = mesh.index();
            if (i.index() == 0) {
                stateTracker->RecordState(
                    reinterpret_cast<Buffer *>(luisa::get<0>(i).handle()),
                    stateTracker->ReadState(ResourceReadUsage::IndexBufferForGraphics));
            }
        }
        for (auto &&i : rtvs) {
            stateTracker->RecordState(
                reinterpret_cast<TextureBase *>(i.handle),
                D3D12_RESOURCE_STATE_RENDER_TARGET);
        }
        if (dsv.handle != ~0ull) {
            stateTracker->RecordState(
                reinterpret_cast<TextureBase *>(dsv.handle),
                D3D12_RESOURCE_STATE_DEPTH_WRITE);
        }
    }
};
#ifdef LCDX_ENABLE_WINPIX
inline DWORD get_pix_color() {
    return ~0;
}
#endif
class LCCmdVisitor : public CommandVisitor {
public:
    Device *device;
    luisa::function<void(luisa::string_view)> *logger;
    CommandBufferBuilder *bd;
    ResourceStateTracker *stateTracker;
    BufferView argBuffer;
    Buffer const *accelScratchBuffer;
    std::pair<size_t, size_t> *accelScratchOffsets;
    std::pair<size_t, size_t> *bufferVec;
    vstd::vector<BindProperty> *bindProps;
    vstd::vector<ButtomCompactCmd> *updateAccel;
    vstd::vector<D3D12_VERTEX_BUFFER_VIEW> *vbv;
    BottomAccelData *bottomAccelData;
    vstd::func_ptr_t<void(Device *, CommandBufferBuilder *)> after_custom_cmd{};

    void visit(const BufferUploadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer upload");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->Upload(bf, cmd->data());
    }

    void visit(const BufferDownloadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer download");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BufferView bf(
            reinterpret_cast<Buffer const *>(cmd->handle()),
            cmd->offset(),
            cmd->size());
        bd->Readback(
            bf,
            cmd->data());
    }
    void visit(const BufferCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer copy");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto srcBf = reinterpret_cast<Buffer const *>(cmd->src_handle());
        auto dstBf = reinterpret_cast<Buffer const *>(cmd->dst_handle());
        bd->CopyBuffer(
            srcBf,
            dstBf,
            cmd->src_offset(),
            cmd->dst_offset(),
            cmd->size());
    }
    void visit(const BufferToTextureCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Buffer copy to texture");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->CopyBufferTexture(
            BufferView{bf, cmd->buffer_offset()},
            rt,
            cmd->texture_offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::BufferToTexture,
            true);
    }

    void visit(const MotionInstanceBuildCommand *) noexcept override {
        LUISA_NOT_IMPLEMENTED();
    }

    struct Visitor {
        LCCmdVisitor *self;
        SavedArgument const *arg;

        void operator()(Argument::Buffer const &bf) {
            auto res = reinterpret_cast<Buffer const *>(bf.handle);

            self->bindProps->emplace_back(
                BufferView(res, bf.offset));
            ++arg;
        }
        void operator()(Argument::Texture const &bf) {
            auto rt = reinterpret_cast<TextureBase *>(bf.handle);
            //UAV
            if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
                self->bindProps->emplace_back(
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalUAVIndex(bf.level)));
            }
            // SRV
            else {
                self->bindProps->emplace_back(
                    DescriptorHeapView(
                        self->device->globalHeap.get(),
                        rt->GetGlobalSRVIndex(bf.level)));
            }
            ++arg;
        }
        void operator()(Argument::BindlessArray const &bf) {
            auto arr = reinterpret_cast<BindlessArray *>(bf.handle);
            auto res = arr->BindlessBuffer();
            self->bindProps->emplace_back(
                BufferView(res, 0));
            ++arg;
        }
        void operator()(Argument::Accel const &bf) {
            auto accel = reinterpret_cast<TopAccel *>(bf.handle);
            if ((static_cast<uint>(arg->varUsage) & static_cast<uint>(Usage::WRITE)) == 0) {
                self->bindProps->emplace_back(
                    accel);
            }
            self->bindProps->emplace_back(
                BufferView(accel->GetInstBuffer()));
            ++arg;
        }
        void operator()(Argument::Uniform const &) {
            ++arg;
        }
    };
    void visit(const ShaderDispatchCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Shader dispatch");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        bindProps->clear();
        auto shader = reinterpret_cast<ComputeShader const *>(cmd->handle());
        auto &&tempBuffer = *bufferVec;
        bufferVec++;
        auto cs = static_cast<ComputeShader const *>(shader);
        BufferView readback_count_buffer;
        BufferView readback_buffer;
        BufferView count_buffer;
        BufferView data_buffer;
        CommandAllocator *alloc{nullptr};
        auto BeforeDispatch = [&]() {
            bindProps->emplace_back(DescriptorHeapView(device->samplerHeap.get()));
            if (tempBuffer.second > 0) {
                bindProps->emplace_back(BufferView(argBuffer.buffer, argBuffer.offset + tempBuffer.first, tempBuffer.second));
            }
            DescriptorHeapView globalHeapView(device->globalHeap.get());
            vstd::push_back_func(*bindProps, shader->BindlessCount(), [&] { return globalHeapView; });
            Visitor visitor{this, cs->Args().data()};
            DecodeCmd(shader->ArgBindings(), visitor);
            DecodeCmd(cmd->arguments(), visitor);
            auto printers = shader->Printers();
            if (!printers.empty()) [[unlikely]] {
                alloc = bd->GetCB()->GetAlloc();
                static const uint zero = 0;
                auto upload_buffer = alloc->GetTempUploadBuffer(sizeof(uint), 16);
                count_buffer = alloc->GetTempDefaultBuffer(sizeof(uint), 16);
                readback_count_buffer = alloc->GetTempReadbackBuffer(sizeof(uint), 16);
                data_buffer = alloc->GetTempDefaultBuffer(1024ull * 1024ull, 16);
                readback_buffer = alloc->GetTempReadbackBuffer(1024ull * 1024ull, 16);
                static_cast<UploadBuffer const *>(upload_buffer.buffer)->CopyData(upload_buffer.offset, {reinterpret_cast<uint8_t const *>(&zero), sizeof(uint)});
                stateTracker->RecordState(count_buffer.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
                stateTracker->UpdateState(*bd);
                bd->CopyBuffer(upload_buffer.buffer, count_buffer.buffer, upload_buffer.offset, count_buffer.offset, sizeof(uint));
                stateTracker->RecordState(count_buffer.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                stateTracker->RecordState(data_buffer.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                stateTracker->UpdateState(*bd);
                bindProps->emplace_back(count_buffer);
                bindProps->emplace_back(data_buffer);
            }
        };
        if (cmd->is_indirect()) {
            auto &&t = cmd->indirect_dispatch();
            auto buffer = reinterpret_cast<Buffer *>(t.handle);
            bindProps->emplace_back();
            BeforeDispatch();
            bd->DispatchComputeIndirect(cs, *buffer, t.offset, t.max_dispatch_size, *bindProps);
        } else if (cmd->is_multiple_dispatch()) {
            size_t bindCount = bindProps->size();
            bindProps->emplace_back();
            BeforeDispatch();
            auto sizes = cmd->dispatch_sizes();
            bd->DispatchCompute(
                cs,
                sizes,
                bindCount,
                *bindProps);
        } else {
            auto &&t = cmd->dispatch_size();
            bindProps->emplace_back(4, make_uint4(t, 0));
            BeforeDispatch();
            bd->DispatchCompute(
                cs,
                t,
                *bindProps);
        }
        if (data_buffer.buffer != nullptr) [[unlikely]] {
            stateTracker->RecordState(count_buffer.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
            stateTracker->RecordState(data_buffer.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
            stateTracker->UpdateState(*bd);
            bd->CopyBuffer(count_buffer.buffer, readback_count_buffer.buffer, count_buffer.offset, readback_count_buffer.offset, sizeof(uint));
            bd->CopyBuffer(data_buffer.buffer, readback_buffer.buffer, data_buffer.offset, readback_buffer.offset, data_buffer.byteSize);
            alloc->ExecuteAfterComplete([logger = this->logger, shader, readback_count_buffer, readback_buffer]() {
                uint size{0};
                static_cast<ReadbackBuffer const *>(readback_count_buffer.buffer)
                    ->CopyData(
                        readback_count_buffer.offset,
                        {reinterpret_cast<uint8_t *>(&size), sizeof(uint)});
                if (size == 0) return;
                vstd::vector<std::byte> data;
                luisa::enlarge_by(data, std::min<size_t>(readback_buffer.byteSize, size));
                static_cast<ReadbackBuffer const *>(readback_buffer.buffer)
                    ->CopyData(
                        readback_buffer.offset,
                        {reinterpret_cast<uint8_t *>(data.data()), data.size()});
                auto printers = shader->Printers();
                size_t offset = 0;
                const auto ptr = data.data();
                const auto end = data.size();
                while (offset < end) {
                    uint flagTypeIdx = *reinterpret_cast<uint32_t *>(ptr + offset);
                    auto &type = printers[flagTypeIdx];
                    ShaderPrintFormatter formatter{type.first, type.second, false};
                    luisa::string result;
                    auto align = std::max<size_t>(4, type.second->alignment());
                    formatter(result, {ptr + offset + align, type.second->size()});
                    size_t ele_size = align + type.second->size();
                    ele_size = ((ele_size + 15ull) & (~15ull));
                    offset += ele_size;
                    if (logger) [[likely]] {
                        (*logger)(result);
                    }
                }
            });
        }
    }
    void visit(const TextureUploadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture upload");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::GetCopyTextureBufferSize(
            rt,
            cmd->size());
        auto bfView = bd->GetCB()->GetAlloc()->GetTempUploadBuffer(copyInfo.alignedBufferSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        auto uploadBuffer = static_cast<UploadBuffer const *>(bfView.buffer);
        if (copyInfo.bufferSize == copyInfo.alignedBufferSize) {
            uploadBuffer->CopyData(
                bfView.offset,
                {reinterpret_cast<uint8_t const *>(cmd->data()),
                 bfView.byteSize});
        } else {
            size_t bufferOffset = bfView.offset;
            size_t leftedSize = copyInfo.bufferSize;
            auto dataPtr = reinterpret_cast<uint8_t const *>(cmd->data());
            while (leftedSize > 0) {
                uploadBuffer->CopyData(
                    bufferOffset,
                    {dataPtr, copyInfo.copySize});
                dataPtr += copyInfo.copySize;
                leftedSize -= copyInfo.copySize;
                bufferOffset += copyInfo.stepSize;
            }
        }
        bd->CopyBufferTexture(
            bfView,
            rt,
            cmd->offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::BufferToTexture,
            false);
    }
    void visit(const ClearDepthCommand *cmd) noexcept {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Clear depth");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto cmdList = bd->GetCB()->CmdList();
        auto alloc = bd->GetCB()->GetAlloc();
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
        auto chunk = alloc->dsvAllocator.allocate(1);
        auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
        dsvHandle = descHeap->hCPU(chunk.offset);
        D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc{
            .Format = static_cast<DXGI_FORMAT>(rt->Format()),
            .ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
            .Flags = D3D12_DSV_FLAG_NONE};
        viewDesc.Texture2D.MipSlice = 0;
        device->device->CreateDepthStencilView(rt->GetResource(), &viewDesc, dsvHandle);
        D3D12_CLEAR_FLAGS clearFlags = D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL;
        RECT rect{0, 0, static_cast<int>(rt->Width()), static_cast<int>(rt->Height())};
        cmdList->ClearDepthStencilView(dsvHandle, clearFlags, cmd->value(), 0, 1, &rect);
    }
    void visit(const TextureDownloadCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture download");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->handle());
        auto copyInfo = CommandBufferBuilder::GetCopyTextureBufferSize(
            rt,
            cmd->size());
        auto alloc = bd->GetCB()->GetAlloc();
        auto bfView = alloc->GetTempReadbackBuffer(copyInfo.alignedBufferSize, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);

        if (copyInfo.alignedBufferSize == copyInfo.bufferSize) {
            alloc->ExecuteAfterComplete(
                [bfView,
                 ptr = cmd->data()] {
                    auto rbBuffer = static_cast<ReadbackBuffer const *>(bfView.buffer);
                    size_t bufferOffset = bfView.offset;
                    rbBuffer->CopyData(
                        bufferOffset,
                        {reinterpret_cast<uint8_t *>(ptr), bfView.byteSize});
                });
        } else {
            auto rbBuffer = static_cast<ReadbackBuffer const *>(bfView.buffer);
            size_t bufferOffset = bfView.offset;
            alloc->ExecuteAfterComplete(
                [rbBuffer,
                 bufferOffset,
                 dataPtr = reinterpret_cast<uint8_t *>(cmd->data()),
                 copyInfo]() mutable {
                    while (copyInfo.bufferSize > 0) {

                        rbBuffer->CopyData(
                            bufferOffset,
                            {dataPtr, copyInfo.copySize});
                        dataPtr += copyInfo.copySize;
                        copyInfo.bufferSize -= copyInfo.copySize;
                        bufferOffset += copyInfo.stepSize;
                    }
                });
        }
        bd->CopyBufferTexture(
            bfView,
            rt,
            cmd->offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::TextureToBuffer,
            false);
    }
    void visit(const TextureCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture copy");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto src = reinterpret_cast<TextureBase *>(cmd->src_handle());
        auto dst = reinterpret_cast<TextureBase *>(cmd->dst_handle());
        bd->CopyTexture(
            src,
            0,
            cmd->src_level(),
            dst,
            0,
            cmd->dst_level());
    }
    void visit(const TextureToBufferCopyCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Texture copy to buffer");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto rt = reinterpret_cast<TextureBase *>(cmd->texture());
        auto bf = reinterpret_cast<Buffer *>(cmd->buffer());
        bd->CopyBufferTexture(
            BufferView{bf, cmd->buffer_offset()},
            rt,
            cmd->texture_offset(),
            cmd->size(),
            cmd->level(),
            CommandBufferBuilder::BufferTextureCopy::TextureToBuffer,
            true);
    }
    void visit(const AccelBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Accel build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto accel = reinterpret_cast<TopAccel *>(cmd->handle());
        vstd::optional<BufferView> scratch;
        if (!cmd->update_instance_buffer_only()) {
            scratch.create(BufferView(accelScratchBuffer, accelScratchOffsets->first, accelScratchOffsets->second));
            if (accel->RequireCompact()) {
                updateAccel->emplace_back(ButtomCompactCmd{
                    .accel = accel,
                    .offset = accelScratchOffsets->first,
                    .size = accelScratchOffsets->second});
            }
            accelScratchOffsets++;
        }
        accel->Build(
            *stateTracker,
            *bd,
            scratch.has_value() ? scratch.ptr() : nullptr);
    }
    void BottomBuild(uint64 handle) {
        auto accel = reinterpret_cast<BottomAccel *>(handle);
        accel->UpdateStates(
            *stateTracker,
            *bd,
            BufferView(accelScratchBuffer, accelScratchOffsets->first, accelScratchOffsets->second),
            *bottomAccelData);
        if (accel->RequireCompact()) {
            updateAccel->emplace_back(ButtomCompactCmd{
                .accel = accel,
                .offset = accelScratchOffsets->first,
                .size = accelScratchOffsets->second});
        }
        accelScratchOffsets++;
        bottomAccelData++;
    }
    void visit(const CurveBuildCommand *) noexcept override { /* TODO */
        LUISA_NOT_IMPLEMENTED();
    }
    void visit(const MeshBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Mesh build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BottomBuild(cmd->handle());
    }
    void visit(const ProceduralPrimitiveBuildCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Procedural build");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        BottomBuild(cmd->handle());
    }
    void visit(const BindlessArrayUpdateCommand *cmd) noexcept override {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Bindless-array update");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        auto arr = reinterpret_cast<BindlessArray *>(cmd->handle());
        arr->UpdateStates(
            *bd,
            *stateTracker,
            cmd->modifications());
    }
    void visit(const DXCustomCmd *cmd) noexcept {
        cmd->execute(
            device->adapter.Get(),
            device->dxgiFactory.Get(),
            device->device.Get(),
            bd->GetCB()->CmdList());
        after_custom_cmd(device, bd);
    }
    void visit(const CustomCommand *cmd) noexcept override {
        switch (cmd->uuid()) {
            case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH):
                visit(static_cast<ClearDepthCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE):
                visit(static_cast<DrawRasterSceneCommand const *>(cmd));
                break;
            case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH):
                visit(static_cast<DXCustomCmd const *>(cmd));
                break;
            default:
                LUISA_ERROR("Custom command not supported by this queue.");
        }
    }
    void visit(const DrawRasterSceneCommand *cmd) noexcept {
#ifdef LCDX_ENABLE_WINPIX
        PIXBeginEvent(bd->GetCB()->CmdList(), get_pix_color(), "Draw raster command");
        auto dispose_pix = vstd::scope_exit([&]() {
            PIXEndEvent(bd->GetCB()->CmdList());
        });
#endif
        bindProps->clear();
        auto cmdList = bd->GetCB()->CmdList();
        auto rtvs = cmd->rtv_texs();
        auto dsv = cmd->dsv_tex();
        DepthFormat dsvFormat{DepthFormat::None};
        // TODO:Set render target
        // Set viewport
        auto alloc = bd->GetCB()->GetAlloc();
        {
            D3D12_VIEWPORT view;
            auto &&viewport = cmd->viewport();
            view.MinDepth = 0;
            view.MaxDepth = 1;
            view.TopLeftX = viewport.start.x;
            view.TopLeftY = viewport.start.y;
            view.Width = viewport.size.x;
            view.Height = viewport.size.y;
            cmdList->RSSetViewports(1, &view);
            RECT rect{
                .left = static_cast<int>(view.TopLeftX + 0.4999f),
                .top = static_cast<int>(view.TopLeftY + 0.4999f),
                .right = static_cast<int>(view.TopLeftX + view.Width + 0.4999f),
                .bottom = static_cast<int>(view.TopLeftY + view.Height + 0.4999f)};
            cmdList->RSSetScissorRects(1, &rect);
        }
        GFXFormat rtvFormats[8];
        {

            D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
            D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle;
            D3D12_CPU_DESCRIPTOR_HANDLE *dsvHandlePtr = nullptr;
            if (!rtvs.empty()) {
                auto chunk = alloc->rtvAllocator.allocate(rtvs.size());
                auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
                rtvHandle = descHeap->hCPU(chunk.offset);
                for (auto i : vstd::range(rtvs.size())) {
                    auto &&rtv = rtvs[i];
                    auto tex = reinterpret_cast<TextureBase *>(rtv.handle);
                    rtvFormats[i] = tex->Format();
                    D3D12_RENDER_TARGET_VIEW_DESC viewDesc{
                        .Format = static_cast<DXGI_FORMAT>(rtvFormats[i]),
                        .ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D};
                    viewDesc.Texture2D = {
                        .MipSlice = rtv.level,
                        .PlaneSlice = 0};
                    descHeap->CreateRTV(tex->GetResource(), viewDesc, chunk.offset + i);
                }
            }
            if (dsv.handle != ~0ull) {
                dsvHandlePtr = &dsvHandle;
                auto chunk = alloc->dsvAllocator.allocate(1);
                auto descHeap = reinterpret_cast<DescriptorHeap *>(chunk.handle);
                dsvHandle = descHeap->hCPU(chunk.offset);
                auto tex = reinterpret_cast<TextureBase *>(dsv.handle);
                D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc{
                    .Format = static_cast<DXGI_FORMAT>(tex->Format()),
                    .ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
                    .Flags = D3D12_DSV_FLAG_NONE};
                viewDesc.Texture2D.MipSlice = 0;
                device->device->CreateDepthStencilView(tex->GetResource(), &viewDesc, dsvHandle);
            }
            cmdList->OMSetRenderTargets(rtvs.size(), &rtvHandle, true, dsvHandlePtr);
        }
        auto shader = reinterpret_cast<RasterShader *>(cmd->handle());
        auto rasterState = cmd->raster_state();
        auto pso = shader->GetPSO({rtvFormats, rtvs.size()}, cmd->mesh_format(), dsvFormat, rasterState);
        auto &&tempBuffer = *bufferVec;
        bufferVec++;
        bindProps->emplace_back(DescriptorHeapView(device->samplerHeap.get()));
        if (tempBuffer.second > 0) {
            bindProps->emplace_back(BufferView(argBuffer.buffer, argBuffer.offset + tempBuffer.first, tempBuffer.second));
        }
        DescriptorHeapView globalHeapView(DescriptorHeapView(device->globalHeap.get()));
        vstd::push_back_func(*bindProps, shader->BindlessCount(), [&] { return globalHeapView; });
        DecodeCmd(cmd->arguments(), Visitor{this, shader->Args().data()});
        bd->SetRasterShader(shader, pso, *bindProps);
        cmdList->IASetPrimitiveTopology([&] {
            switch (rasterState.topology) {
                case TopologyType::Line:
                    return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
                case TopologyType::Point:
                    return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
                default:
                    return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            }
        }());
        auto &&meshes = cmd->scene();
        auto propCount = shader->Properties().size();
        for (auto idx : vstd::range(meshes.size())) {
            auto &&mesh = meshes[idx];
            cmdList->SetGraphicsRoot32BitConstant(propCount, mesh.object_id(), 0);
            vbv->clear();
            auto src = mesh.vertex_buffers();
            vstd::push_back_func(
                *vbv,
                src.size(),
                [&](size_t i) {
                    auto &e = src[i];
                    auto bf = reinterpret_cast<Buffer *>(e.handle());
                    return D3D12_VERTEX_BUFFER_VIEW{
                        .BufferLocation = bf->GetAddress() + e.offset(),
                        .SizeInBytes = static_cast<uint>(e.size()),
                        .StrideInBytes = static_cast<uint>(e.stride())};
                });
            cmdList->IASetVertexBuffers(0, vbv->size(), vbv->data());
            auto const &i = mesh.index();

            luisa::visit(
                [&]<typename T>(T const &i) {
                    if constexpr (std::is_same_v<T, uint>) {
                        cmdList->DrawInstanced(i, mesh.instance_count(), mesh.vertex_offset(), 0);
                    } else {
                        auto bf = reinterpret_cast<Buffer *>(i.handle());
                        D3D12_INDEX_BUFFER_VIEW idx{
                            .BufferLocation = bf->GetAddress() + i.offset_bytes(),
                            .SizeInBytes = static_cast<uint>(i.size_bytes()),
                            .Format = DXGI_FORMAT_R32_UINT};
                        cmdList->IASetIndexBuffer(&idx);
                        cmdList->DrawIndexedInstanced(i.size_bytes() / sizeof(uint), mesh.instance_count(), 0, mesh.vertex_offset(), 0);
                    }
                },
                i);
        }
    }
};

LCCmdBuffer::LCCmdBuffer(
    Device *device,
    GpuAllocator *resourceAllocator,
    D3D12_COMMAND_LIST_TYPE type)
    : CmdQueueBase{device, CmdQueueTag::MainCmd},
      reorder({}),
      queue(
          device,
          resourceAllocator,
          type) {
}
void LCCmdBuffer::Execute(
    CommandList &&cmdList,
    size_t maxAlloc) {
    auto commands = cmdList.commands();
    auto funcs = std::move(cmdList).steal_callbacks();
    auto allocator = queue.CreateAllocator(maxAlloc);
    auto allocType = allocator->Type();
    bool cmdListIsEmpty = true;
    {
        std::unique_lock lck{mtx};
        tracker.listType = allocator->Type();
        LCPreProcessVisitor ppVisitor;
        ppVisitor.stateTracker = &tracker;
        ppVisitor.argVecs = &argVecs;
        ppVisitor.argBuffer = &argBuffer;
        ppVisitor.bottomAccelDatas = &bottomAccelDatas;
        ppVisitor.accelOffset = &accelOffset;
        argVecs.clear();
        argBuffer.clear();
        bottomAccelDatas.clear();
        accelOffset.clear();

        LCCmdVisitor visitor;
        if (logCallback) {
            visitor.logger = &logCallback;
        } else {
            visitor.logger = nullptr;
        }
        visitor.bindProps = &bindProps;
        visitor.updateAccel = &updateAccel;
        visitor.vbv = &vbv;
        visitor.device = device;
        visitor.stateTracker = &tracker;
        visitor.after_custom_cmd = [](Device *device, CommandBufferBuilder *bd) {
            ID3D12DescriptorHeap *h[2] = {
                device->globalHeap->GetHeap(),
                device->samplerHeap->GetHeap()};
            auto cb = bd->GetCB();
            if (cb->GetAlloc()->Type() != D3D12_COMMAND_LIST_TYPE_COPY) {
                cb->CmdList()->SetDescriptorHeaps(vstd::array_count(h), h);
            }
        };
        auto cmdBuffer = allocator->GetBuffer();
        auto cmdBuilder = cmdBuffer->Build();
        visitor.bd = &cmdBuilder;
        ppVisitor.bd = &cmdBuilder;
        for (auto &&command : commands) {
            command->accept(reorder);
        }
        auto cmdLists = reorder.command_lists();
        auto clearReorder = vstd::scope_exit([&] {
            reorder.clear();
        });
        ID3D12DescriptorHeap *h[2] = {
            device->globalHeap->GetHeap(),
            device->samplerHeap->GetHeap()};
        for (auto lst : cmdLists) {
            if (allocType != D3D12_COMMAND_LIST_TYPE_COPY) {
                cmdBuffer->CmdList()->SetDescriptorHeaps(vstd::array_count(h), h);
            }
            cmdListIsEmpty = cmdListIsEmpty && (lst == nullptr);
            // Clear caches
            ppVisitor.argVecs->clear();
            ppVisitor.argBuffer->clear();
            ppVisitor.accelOffset->clear();
            ppVisitor.bottomAccelDatas->clear();
            ppVisitor.buildAccelSize = 0;
            // Preprocess: record resources' states
            for (auto i = lst; i != nullptr; i = i->p_next) {
                i->cmd->accept(ppVisitor);
            }
            visitor.bottomAccelData = ppVisitor.bottomAccelDatas->data();
            DefaultBuffer const *accelScratchBuffer;
            if (ppVisitor.buildAccelSize) {
                accelScratchBuffer = allocator->AllocateScratchBuffer(ppVisitor.buildAccelSize);
                visitor.accelScratchOffsets = ppVisitor.accelOffset->data();
                visitor.accelScratchBuffer = accelScratchBuffer;
            }
            // Upload CBuffers
            if (ppVisitor.argBuffer->empty()) {
                visitor.argBuffer = {};
            } else {
// Use default buffer as arguments buffer
#if false
                auto uploadBuffer = allocator->GetTempDefaultBuffer(ppVisitor.argBuffer->size(), 16);
                tracker.RecordState(
                    uploadBuffer.buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST);
                // Update recorded states
                tracker.UpdateState(
                    cmdBuilder);
                cmdBuilder.Upload(
                    uploadBuffer,
                    ppVisitor.argBuffer->data());
                tracker.RecordState(
                    uploadBuffer.buffer,
                    tracker.ReadState(ResourceReadUsage::Srv));
#else
                // use upload buffer maybe faster?
                auto uploadBuffer = allocator->GetTempUploadBuffer(ppVisitor.argBuffer->size(), 16);
                static_cast<UploadBuffer const *>(uploadBuffer.buffer)
                    ->CopyData(
                        uploadBuffer.offset,
                        {reinterpret_cast<uint8_t const *>(
                             ppVisitor.argBuffer->data()),
                         ppVisitor.argBuffer->size()});
#endif
                visitor.argBuffer = uploadBuffer;
            }
            tracker.UpdateState(
                cmdBuilder);
            visitor.bufferVec = ppVisitor.argVecs->data();
            // Execute commands
            for (auto i = lst; i != nullptr; i = i->p_next) {
                i->cmd->accept(visitor);
            }
            if (!updateAccel.empty()) {
                tracker.ClearFence();
                tracker.RecordState(
                    accelScratchBuffer,
                    D3D12_RESOURCE_STATE_COPY_SOURCE);
                tracker.UpdateState(cmdBuilder);
                for (auto &&i : updateAccel) {
                    i.accel.visit([&](auto &&p) {
                        p->FinalCopy(
                            cmdBuilder,
                            BufferView(
                                accelScratchBuffer,
                                i.offset,
                                i.size));
                    });
                }
                tracker.RestoreState(cmdBuilder);
                auto localUpdateAccel = std::move(updateAccel);
                lck.unlock();
                queue.ForceSync(
                    allocator,
                    *cmdBuffer);
                for (auto &&i : localUpdateAccel) {
                    i.accel.visit([&](auto &&p) {
                        p->CheckAccel(cmdBuilder);
                    });
                }
                lck.lock();
            }
            tracker.ClearFence();
        }
        tracker.RestoreState(cmdBuilder);
    }
    if (funcs.empty()) {
        if (cmdListIsEmpty)
            queue.ExecuteEmpty(std::move(allocator));
        else
            queue.Execute(std::move(allocator));
    } else {
        if (cmdListIsEmpty)
            queue.ExecuteEmptyCallbacks(std::move(allocator), std::move(funcs));
        else
            queue.ExecuteCallbacks(std::move(allocator), std::move(funcs));
    }
}
void LCCmdBuffer::Sync() {
    queue.Complete();
}
void LCCmdBuffer::Present(
    LCSwapChain *swapchain,
    TextureBase *img,
    size_t maxAlloc) {
    auto alloc = queue.CreateAllocator(maxAlloc);
    {
        std::lock_guard lck{mtx};
        tracker.listType = alloc->Type();
        // swapchain->frameIndex = swapchain->swapChain->GetCurrentBackBufferIndex();
        auto &&rt = &swapchain->m_renderTargets[swapchain->frameIndex];
        swapchain->frameIndex += 1;
        swapchain->frameIndex %= swapchain->frameCount;
        auto cb = alloc->GetBuffer();
        auto bd = cb->Build();
        auto cmdList = cb->CmdList();
        tracker.RecordState(
            rt, D3D12_RESOURCE_STATE_COPY_DEST);
        tracker.RecordState(
            img,
            tracker.ReadState(ResourceReadUsage::CopySource, img));
        tracker.UpdateState(bd);
        D3D12_TEXTURE_COPY_LOCATION sourceLocation;
        sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        sourceLocation.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION destLocation;
        destLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destLocation.SubresourceIndex = 0;
        sourceLocation.pResource = img->GetResource();
        destLocation.pResource = rt->GetResource();
        cmdList->CopyTextureRegion(
            &destLocation,
            0, 0, 0,
            &sourceLocation,
            nullptr);
        tracker.RestoreState(bd);
    }
    queue.ExecuteAndPresent(std::move(alloc), swapchain->swapChain.Get(), swapchain->vsync);
}
void LCCmdBuffer::CompressBC(
    TextureBase *rt,
    uint level,
    luisa::compute::BufferView<uint> const &result,
    bool isHDR,
    float alphaImportance,
    GpuAllocator *allocator,
    size_t maxAlloc) {
    alphaImportance = std::max<float>(std::min<float>(alphaImportance, 1), 0);// clamp<float>(alphaImportance, 0, 1);
    struct BCCBuffer {
        uint g_mip_level;
        uint g_tex_width;
        uint g_num_block_x;
        uint g_format;
        uint g_mode_id;
        uint g_start_block_id;
        uint g_num_total_blocks;
        float g_alpha_weight;
    };
    uint width = rt->Width() >> level;
    uint height = rt->Height() >> level;
    uint xBlocks = std::max<uint>(1, (width + 3) >> 2);
    uint yBlocks = std::max<uint>(1, (height + 3) >> 2);
    uint numBlocks = xBlocks * yBlocks;
    uint numTotalBlocks = numBlocks;
    static constexpr size_t BLOCK_SIZE = 16;
    if (result.size_bytes() != BLOCK_SIZE * numBlocks) [[unlikely]] {
        LUISA_ERROR("Texture compress output buffer incorrect size!");
    }
    DefaultBuffer backBuffer(
        device,
        BLOCK_SIZE * numBlocks,
        allocator,
        D3D12_RESOURCE_STATE_COMMON);
    auto outBufferPtr = reinterpret_cast<Buffer *>(result.handle());
    BufferView outBuffer{
        outBufferPtr,
        result.offset_bytes(),
        result.size_bytes()};

    constexpr uint MAX_BATCH = 1024 * 1024;
    int batchNum = (numTotalBlocks + MAX_BATCH - 1) / MAX_BATCH;
    uint startBlockID = 0;
    for (int batch = 0; batch < batchNum; batch++) {
        int target = (batch + 1) * MAX_BATCH;
        auto alloc = queue.CreateAllocator(maxAlloc);
        {
            std::lock_guard lck{mtx};
            tracker.listType = alloc->Type();
            auto bufferReadState = tracker.ReadState(ResourceReadUsage::Srv);
            auto cmdBuffer = alloc->GetBuffer();
            auto cmdBuilder = cmdBuffer->Build();
            ID3D12DescriptorHeap *h[2] = {
                device->globalHeap->GetHeap(),
                device->samplerHeap->GetHeap()};
            cmdBuffer->CmdList()->SetDescriptorHeaps(vstd::array_count(h), h);

            BCCBuffer cbData{
                .g_mip_level = level};
            tracker.RecordState(rt, tracker.ReadState(ResourceReadUsage::Srv, rt));
            auto RunComputeShader = [&](ComputeShader const *cs, uint dispatchCount, BufferView const &inBuffer, BufferView const &outBuffer) {
                auto cbuffer = alloc->GetTempUploadBuffer(sizeof(BCCBuffer), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
                static_cast<UploadBuffer const *>(cbuffer.buffer)->CopyData(cbuffer.offset, {reinterpret_cast<uint8_t const *>(&cbData), sizeof(BCCBuffer)});
                tracker.RecordState(
                    inBuffer.buffer,
                    bufferReadState);
                tracker.RecordState(
                    outBuffer.buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                tracker.UpdateState(cmdBuilder);
                BindProperty prop[4];
                prop[0] = cbuffer;
                prop[1] = DescriptorHeapView(device->globalHeap.get(), rt->GetGlobalSRVIndex());
                prop[2] = inBuffer;
                prop[3] = outBuffer;
                cmdBuilder.DispatchCompute(
                    cs,
                    uint3(dispatchCount, 1, 1),
                    {prop, 4});
            };
            constexpr uint MAX_BLOCK_BATCH = 1024u * 512u;
            if (isHDR)//bc6
            {
                BufferView err1Buffer{&backBuffer};
                BufferView err2Buffer{outBuffer};
                auto bc6TryModeG10 = device->bc6TryModeG10.Get(device);
                auto bc6TryModeLE10 = device->bc6TryModeLE10.Get(device);
                auto bc6Encode = device->bc6EncodeBlock.Get(device);
                while (numBlocks > 0 && startBlockID < target) {
                    uint n = std::min<uint>(numBlocks, MAX_BLOCK_BATCH);
                    uint uThreadGroupCount = n;
                    cbData.g_tex_width = width;
                    cbData.g_num_block_x = xBlocks;
                    cbData.g_format = isHDR ? DXGI_FORMAT_BC6H_UF16 : DXGI_FORMAT_BC7_UNORM;
                    cbData.g_start_block_id = startBlockID;
                    cbData.g_alpha_weight = alphaImportance;
                    cbData.g_num_total_blocks = numTotalBlocks;
                    RunComputeShader(
                        bc6TryModeG10,
                        std::max<uint>((uThreadGroupCount + 3) / 4, 1),
                        err2Buffer,
                        err1Buffer);
                    for (auto i : vstd::range(10)) {
                        cbData.g_mode_id = i;
                        RunComputeShader(
                            bc6TryModeLE10,
                            std::max<uint>((uThreadGroupCount + 1) / 2, 1),
                            ((i & 1) != 0) ? err2Buffer : err1Buffer,
                            ((i & 1) != 0) ? err1Buffer : err2Buffer);
                    }
                    RunComputeShader(
                        bc6Encode,
                        std::max<uint>((uThreadGroupCount + 1) / 2, 1),
                        err1Buffer,
                        err2Buffer);
                    startBlockID += n;
                    numBlocks -= n;
                }

            } else {
                BufferView err1Buffer{outBuffer};
                BufferView err2Buffer{&backBuffer};
                auto bc7Try137Mode = device->bc7TryMode137.Get(device);
                auto bc7Try02Mode = device->bc7TryMode02.Get(device);
                auto bc7Try456Mode = device->bc7TryMode456.Get(device);
                auto bc7Encode = device->bc7EncodeBlock.Get(device);
                while (numBlocks > 0 && startBlockID < target) {
                    uint n = std::min<uint>(numBlocks, MAX_BLOCK_BATCH);
                    uint uThreadGroupCount = n;
                    cbData.g_tex_width = width;
                    cbData.g_num_block_x = xBlocks;
                    cbData.g_format = isHDR ? DXGI_FORMAT_BC6H_UF16 : DXGI_FORMAT_BC7_UNORM;
                    cbData.g_start_block_id = startBlockID;
                    cbData.g_alpha_weight = alphaImportance;
                    cbData.g_num_total_blocks = numTotalBlocks;
                    RunComputeShader(bc7Try456Mode, std::max<uint>((uThreadGroupCount + 3) / 4, 1), err2Buffer, err1Buffer);
                    //137
                    {
                        uint modes[] = {1, 3, 7};
                        for (auto i : vstd::range(vstd::array_count(modes))) {
                            cbData.g_mode_id = modes[i];
                            RunComputeShader(
                                bc7Try137Mode,
                                uThreadGroupCount,
                                ((i & 1) != 0) ? err2Buffer : err1Buffer,
                                ((i & 1) != 0) ? err1Buffer : err2Buffer);
                        }
                    }
                    //02
                    {
                        uint modes[] = {0, 2};
                        for (auto i : vstd::range(vstd::array_count(modes))) {
                            cbData.g_mode_id = modes[i];
                            RunComputeShader(
                                bc7Try02Mode,
                                uThreadGroupCount,
                                ((i & 1) != 0) ? err1Buffer : err2Buffer,
                                ((i & 1) != 0) ? err2Buffer : err1Buffer);
                        }
                    }
                    RunComputeShader(
                        bc7Encode,
                        std::max<uint>((uThreadGroupCount + 3) / 4, 1),
                        err2Buffer,
                        err1Buffer);
                    //TODO
                    startBlockID += n;
                    numBlocks -= n;
                }
            }
            tracker.RecordState(outBufferPtr, D3D12_RESOURCE_STATE_COPY_SOURCE);
            tracker.RestoreState(cmdBuilder);
        }
        if (batch == batchNum - 1) {
            vstd::vector<vstd::function<void()>> callbacks;
            callbacks.emplace_back([backBuffer = std::move(backBuffer)] {});
            queue.ExecuteCallbacks(
                std::move(alloc),
                std::move(callbacks));
        } else {
            queue.Execute(std::move(alloc));
        }
    }
}
}// namespace lc::dx
