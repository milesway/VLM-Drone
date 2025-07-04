#include "shader_compiler.h"
#include <luisa/core/dynamic_module.h>
#include <luisa/vstl/string_utility.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/spin_mutex.h>
namespace lc::hlsl {
#ifndef LC_DXC_THROW_IF_FAILED
#define LC_DXC_THROW_IF_FAILED(x)                  \
    {                                              \
        HRESULT hr_ = (x);                         \
        LUISA_ASSERT(hr_ == S_OK, "bad HRESULT."); \
    }
#endif
static vstd::wstring GetSM(uint shaderModel) {
    vstd::string smStr;
    smStr << vstd::to_string(shaderModel / 10) << '_' << vstd::to_string(shaderModel % 10);
    vstd::wstring wstr;
    wstr.resize(smStr.size());
    for (auto i : vstd::range(smStr.size())) {
        wstr[i] = smStr[i];
    }
    return wstr;
}
IDxcCompiler3 *ShaderCompiler::compiler() {
    return compiler_module.comp;
}
IDxcUtils *ShaderCompiler::utils() {
    return compiler_module.utils;
}
IDxcLibrary *ShaderCompiler::library() {
    return compiler_module.library;
}
ShaderCompiler::~ShaderCompiler() {
}
ShaderCompilerModule::ShaderCompilerModule(std::filesystem::path const &path)
    : dxil(luisa::DynamicModule::load(path, "dxil")),
      dxcCompiler(luisa::DynamicModule::load(path, "dxcompiler")) {
    if (!dxil) {
        LUISA_ERROR("dxil.dll not found.");
    }
    if (!dxcCompiler) {
        LUISA_ERROR("dxcompiler.dll not found.");
    }
    auto voidPtr = dxcCompiler.address("DxcCreateInstance");
    HRESULT(WINAPI * DxcCreateInstance)
    (const IID &, const IID &, LPVOID *) =
        reinterpret_cast<HRESULT(WINAPI *)(const IID &, const IID &, LPVOID *)>(voidPtr);
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&comp));
    DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&library));
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
}
ShaderCompilerModule::~ShaderCompilerModule() {
    // TODO: directx-compiler may crash here
    utils->Release();
    library->Release();
    comp->Release();
}
ShaderCompiler::ShaderCompiler(std::filesystem::path const &path)
    : compiler_module(path) {
}
CompileResult ShaderCompiler::compile(
    vstd::string_view code,
    vstd::span<LPCWSTR> args) {
    DxcBuffer buffer{
        code.data(),
        code.size(),
        CP_ACP};
    ComPtr<IDxcResult> compileResult;

    LC_DXC_THROW_IF_FAILED(compiler()->Compile(
        &buffer,
        args.data(),
        args.size(),
        nullptr,
        IID_PPV_ARGS(compileResult.GetAddressOf())));
    HRESULT status;
    LC_DXC_THROW_IF_FAILED(compileResult->GetStatus(&status));
    if (status == 0) {
        ComPtr<IDxcBlob> resultBlob;
        LC_DXC_THROW_IF_FAILED(compileResult->GetResult(resultBlob.GetAddressOf()));
        return resultBlob;
    } else {
        ComPtr<IDxcBlobEncoding> errBuffer;
        LC_DXC_THROW_IF_FAILED(compileResult->GetErrorBuffer(errBuffer.GetAddressOf()));
        auto errStr = vstd::string_view(
            reinterpret_cast<char const *>(errBuffer->GetBufferPointer()),
            errBuffer->GetBufferSize());
        return vstd::string(errStr);
    }
}
template<typename Vec>
static void AddCompileFlags(Vec &args) {
    vstd::push_back_all(
        args,
        {DXC_ARG_ALL_RESOURCES_BOUND,
         L"-no-warnings",
         L"-enable-16bit-types",
         DXC_ARG_PACK_MATRIX_ROW_MAJOR,
         L"-HV 2021"});
}
template<typename Vec>
static void AddUnsafeMathFlags(Vec &args) {
    // unsafe opt may conflict with dxc
    vstd::push_back_all(
        args,
        {L"-opt-enable",
         L"-funsafe-math-optimizations",
         L"-opt-enable",
         L"-fassociative-math",
         L"-opt-enable",
         L"-freciprocal-math"});
}
CompileResult ShaderCompiler::compile_compute(
    vstd::string_view code,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv) {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 32> args;
    vstd::wstring smStr;
    smStr << L"cs_" << GetSM(shaderModel);
    args.emplace_back(L"/T");
    args.emplace_back(smStr.c_str());
    AddCompileFlags(args);
    if (spirv) {
        args.emplace_back(L"/DSPV");
        args.emplace_back(L"-spirv");
    }
    if (enableUnsafeMath) {
        AddUnsafeMathFlags(args);
    }
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    return compile(code, args);
}
RasterBin ShaderCompiler::compile_raster(
    vstd::string_view code,
    bool optimize,
    uint shaderModel,
    bool enableUnsafeMath,
    bool spirv) {
#ifndef NDEBUG
    if (shaderModel < 10) {
        LUISA_ERROR("Illegal shader model!");
    }
#endif
    vstd::fixed_vector<LPCWSTR, 32> args;
    AddCompileFlags(args);
    if (spirv) {
        args.emplace_back(L"/DSPV");
        args.emplace_back(L"-spirv");
    }
    if (enableUnsafeMath) {
        AddUnsafeMathFlags(args);
    }
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    args.emplace_back(L"/T");
    auto size = args.size();
    vstd::wstring smStr;
    smStr << L"vs_" << GetSM(shaderModel);
    args.emplace_back(smStr.c_str());
    args.emplace_back(L"/DVS");
    RasterBin bin;
    bin.vertex = compile(code, args);
    luisa::enlarge_by(args, size);
    smStr.clear();
    smStr << L"ps_" << GetSM(shaderModel);
    args.emplace_back(smStr.c_str());
    args.emplace_back(L"/DPS");
    bin.pixel = compile(code, args);
    return bin;
}
/*
CompileResult ShaderCompiler::CompileRayTracing(
    vstd::string_view code,
    bool optimize,
    uint shaderModel) {
    if (shaderModel < 10) {
        return "Illegal shader model!"_sv;
    }
    vstd::fixed_vector<LPCWSTR, 32> args;
    vstd::wstring smStr;
    smStr << L"lib_" << GetSM(shaderModel);
    args.emplace_back(L"/T");
    args.emplace_back(smStr.c_str());
    args.push_back_all(
        {L"-Qstrip_debug",
         L"-Qstrip_reflect",
         L"/enable_unbounded_descriptor_tables",
         L"-HV 2021"});
    if (optimize) {
        args.emplace_back(DXC_ARG_OPTIMIZATION_LEVEL3);
    }
    return compile(code, args);
}*/
#undef LC_DXC_THROW_IF_FAILED
}// namespace lc::hlsl
