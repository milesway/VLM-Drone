//
// Created by Mike Smith on 2022/3/21.
//

#include <base/spectrum.h>

namespace luisa::render {

using namespace compute;

struct SRGBSpectrum final : public Spectrum {
    SRGBSpectrum(Scene *scene, const SceneNodeDesc *desc) noexcept : Spectrum{scene, desc} {}
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] bool is_fixed() const noexcept override { return true; }
    [[nodiscard]] uint dimension() const noexcept override { return 3u; }
    [[nodiscard]] luisa::unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    [[nodiscard]] float4 encode_static_srgb_albedo(float3 rgb) const noexcept override { return make_float4(clamp(rgb, 0.f, 1.f), 1.f); }
    [[nodiscard]] float4 encode_static_srgb_unbounded(float3 rgb) const noexcept override { return make_float4(rgb, 1.f); }
    [[nodiscard]] float4 encode_static_srgb_illuminant(float3 rgb) const noexcept override { return make_float4(max(rgb, 0.f), 1.f); }
};

struct SRGBSpectrumInstance final : public Spectrum::Instance {
    SRGBSpectrumInstance(Pipeline &pipeline, CommandBuffer &cb, const Spectrum *spec) noexcept:
        Spectrum::Instance{pipeline, cb, spec} {}
    [[nodiscard]] SampledWavelengths sample(Expr<float>) const noexcept override {
        SampledWavelengths swl{3u};
        auto lambdas = rgb_spectrum_peak_wavelengths;
        for (auto i = 0u; i < 3u; i++) {
            swl.set_lambda(i, lambdas[i]);
            swl.set_pdf(i, 1.f);
        }
        return swl;
    }
    [[nodiscard]] Spectrum::Decode decode_albedo(
        const SampledWavelengths &swl, Expr<float4> v) const noexcept override {
        SampledSpectrum s{node()->dimension()};
        auto sv = saturate(v.xyz());
        for (auto i = 0u; i < 3u; i++) { s[i] = sv[i]; }
        return {.value = s, .strength = linear_srgb_to_cie_y(sv)};
    }
    [[nodiscard]] Spectrum::Decode decode_unbounded(
        const SampledWavelengths &swl, Expr<float4> v) const noexcept override {
        SampledSpectrum s{node()->dimension()};
        auto sv = v.xyz();
        for (auto i = 0u; i < 3u; i++) { s[i] = sv[i]; }
        return {.value = s, .strength = linear_srgb_to_cie_y(sv)};
    }
    [[nodiscard]] Spectrum::Decode decode_illuminant(
        const SampledWavelengths &swl, Expr<float4> v) const noexcept override {
        auto sv = max(v.xyz(), 0.f);
        SampledSpectrum s{node()->dimension()};
        for (auto i = 0u; i < 3u; i++) { s[i] = sv[i]; }
        return {.value = s, .strength = linear_srgb_to_cie_y(sv)};
    }
    [[nodiscard]] Float cie_y(
        const SampledWavelengths &swl,
        const SampledSpectrum &sp) const noexcept override {
        return linear_srgb_to_cie_y(srgb(swl, sp));
    }
    [[nodiscard]] Float3 cie_xyz(
        const SampledWavelengths &swl,
        const SampledSpectrum &sp) const noexcept override {
        return linear_srgb_to_cie_xyz(srgb(swl, sp));
    }
    [[nodiscard]] Float3 srgb(
        const SampledWavelengths &swl,
        const SampledSpectrum &sp) const noexcept override {
        return make_float3(sp[0u], sp[1u], sp[2u]);
    }
    [[nodiscard]] Float4 encode_srgb_albedo(Expr<float3> rgb) const noexcept override {
        return make_float4(clamp(rgb, 0.f, 1.f), 1.f);
    }
    [[nodiscard]] Float4 encode_srgb_unbounded(Expr<float3> rgb) const noexcept override {
        return make_float4(rgb, 1.f);
    }
    [[nodiscard]] Float4 encode_srgb_illuminant(Expr<float3> rgb) const noexcept override {
        return make_float4(max(rgb, 0.f), 1.f);
    }
};

luisa::unique_ptr<Spectrum::Instance> SRGBSpectrum::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<SRGBSpectrumInstance>(pipeline, command_buffer, this);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::SRGBSpectrum)