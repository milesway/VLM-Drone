//
// Created by Mike Smith on 2022/11/23.
//

#include <util/sampling.h>
#include <util/scattering.h>
#include <base/surface.h>
#include <base/interaction.h>
#include <base/pipeline.h>
#include <base/scene.h>

namespace luisa::render {

using namespace luisa::compute;

// Plastic surface from the Tungsten renderer:
// https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/RoughPlasticBsdf.cpp
//
// Original license:
//
// Copyright (c) 2014 Benedikt Bitterli <benedikt.bitterli (at) gmail (dot) com>
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from
// the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute
// it freely, subject to the following restrictions:
//
//     1. The origin of this software must not be misrepresented; you
//        must not claim that you wrote the original software. If you
//        use this software in a product, an acknowledgment in the
//        product documentation would be appreciated but is not required.
//
//     2. Altered source versions must be plainly marked as such, and
//        must not be misrepresented as being the original software.
//
//     3. This notice may not be removed or altered from any source
//        distribution.

class PlasticSurface : public Surface {

private:
    const Texture *_kd;
    const Texture *_ks;
    const Texture *_roughness;
    const Texture *_sigma_a;
    const Texture *_eta;
    const Texture *_thickness;
    bool _remap_roughness;

public:
    PlasticSurface(Scene *scene, const SceneNodeDesc *desc) noexcept:
        Surface{scene, desc},
        _kd{scene->load_texture(desc->property_node_or_default("Kd"))},
        _ks{scene->load_texture(desc->property_node_or_default("Ks"))},
        _roughness{scene->load_texture(desc->property_node_or_default("roughness"))},
        _sigma_a{scene->load_texture(desc->property_node_or_default("sigma_a"))},
        _eta{scene->load_texture(desc->property_node_or_default("eta"))},
        _thickness{scene->load_texture(desc->property_node_or_default("thickness"))},
        _remap_roughness{desc->property_bool_or_default("remap_roughness", true)} {}

    [[nodiscard]] luisa::string info() const noexcept override {
        return luisa::format(
            "{} Kd=[{}] Ks=[{}] roughness=[{}] eta=[{}]", Surface::info(),
            _kd ? _kd->info() : "",
            _ks ? _ks->info() : "",
            _roughness ? _roughness->info() : "",
            _eta ? _eta->info() : ""
        );
    }

    [[nodiscard]] auto remap_roughness() const noexcept { return _remap_roughness; }
    [[nodiscard]] luisa::string_view impl_type() const noexcept override { return LUISA_RENDER_PLUGIN_NAME; }
    [[nodiscard]] uint properties() const noexcept override { return property_reflective; }

protected:
    [[nodiscard]] luisa::unique_ptr<Instance> _build(
        Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
};

class PlasticInstance : public Surface::Instance {

public:
private:
    const Texture::Instance *_kd;
    const Texture::Instance *_ks;
    const Texture::Instance *_roughness;
    const Texture::Instance *_sigma_a;
    const Texture::Instance *_eta;
    const Texture::Instance *_thickness;

public:
    PlasticInstance(Pipeline &pipeline, const Surface *surface,
                    const Texture::Instance *Kd, const Texture::Instance *Ks,
                    const Texture::Instance *roughness, const Texture::Instance *sigma_a,
                    const Texture::Instance *eta, const Texture::Instance *thickness) noexcept:
        Surface::Instance{pipeline, surface},
        _kd{Kd}, _ks{Ks}, _roughness{roughness}, _sigma_a{sigma_a},
        _eta{eta}, _thickness{thickness} {}

public:
    [[nodiscard]] luisa::unique_ptr<Surface::Closure> create_closure(
        const SampledWavelengths &swl, Expr<float> time) const noexcept override;
    void populate_closure(Surface::Closure *closure, const Interaction &it,
                          Expr<float3> wo, Expr<float> eta_i) const noexcept override;
};

luisa::unique_ptr<Surface::Instance> PlasticSurface::_build(
    Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    auto Kd = pipeline.build_texture(command_buffer, _kd);
    auto Ks = pipeline.build_texture(command_buffer, _ks);
    auto roughness = pipeline.build_texture(command_buffer, _roughness);
    auto sigma_a = pipeline.build_texture(command_buffer, _sigma_a);
    auto eta = pipeline.build_texture(command_buffer, _eta);
    auto thickness = pipeline.build_texture(command_buffer, _thickness);
    return luisa::make_unique<PlasticInstance>(
        pipeline, this, Kd, Ks, roughness, sigma_a, eta, thickness);
}

struct PlasticContext {
    Interaction it;
    SampledSpectrum Kd;
    SampledSpectrum Kd_scale;
    Float Kd_weight;
    SampledSpectrum Ks;
    SampledSpectrum sigma_a;
    Float eta;
    Float2 roughness;
};

class PlasticClosureImpl {

private:
    const PlasticContext &_ctx;
    TrowbridgeReitzDistribution _distrib;
    FresnelDielectric _fresnel;
    MicrofacetReflection _coat;
    LambertianReflection _substrate;

private:
    [[nodiscard]] static auto _substrate_weight(Expr<float> Fo, Expr<float> kd_weight) noexcept {
        auto w = kd_weight * (1.0f - Fo);
        return ite(w == 0.f, 0.f, w / (w + Fo));
    }

public:
    explicit PlasticClosureImpl(const PlasticContext &ctx) noexcept:
        _ctx{ctx},
        _distrib{ctx.roughness},
        _fresnel{1.f, ctx.eta},
        _coat{ctx.Ks, &_distrib, &_fresnel},
        _substrate{ctx.Kd} {}

    [[nodiscard]] Surface::Evaluation evaluate(Expr<float3> wo,
                                               Expr<float3> wi,
                                               TransportMode mode) const noexcept {
        auto eval = Surface::Evaluation::zero(_ctx.Kd.dimension());
        $outline {
            auto wo_local = _ctx.it.shading().world_to_local(wo);
            auto sign = ite(cos_theta(wo_local) < 0.f,
                            make_float3(1.f, 1.f, -1.f),
                            make_float3(1.f, 1.f, 1.f));
            wo_local *= sign;
            auto wi_local = sign * _ctx.it.shading().world_to_local(wi);
            // specular
            auto f_coat = _coat.evaluate(wo_local, wi_local, mode);
            auto pdf_coat = _coat.pdf(wo_local, wi_local, mode);
            // diffuse
            auto eta = _fresnel.eta_t();
            auto Fi = fresnel_dielectric(abs_cos_theta(wi_local), 1.f, eta);
            auto Fo = fresnel_dielectric(abs_cos_theta(wo_local), 1.f, eta);
            auto a = exp(-(1.f / abs_cos_theta(wi_local) + 1.f / abs_cos_theta(wo_local)) * _ctx.sigma_a);
            auto f_diffuse = (1.f - Fi) * (1.f - Fo) * sqr(1.f / eta) * a *
                             (_ctx.Kd_scale * _substrate.evaluate(wo_local, wi_local, mode));
            auto pdf_diffuse = _substrate.pdf(wo_local, wi_local, mode);
            auto substrate_weight = _substrate_weight(Fo, _ctx.Kd_weight);
            auto f = (f_coat + f_diffuse) * abs_cos_theta(wi_local);
            auto pdf = lerp(pdf_coat, pdf_diffuse, substrate_weight);
            eval = {.f = f, .pdf = pdf, .f_diffuse = f_diffuse * abs_cos_theta(wi_local), .pdf_diffuse = pdf_diffuse};
        };
        return eval;
    }

    [[nodiscard]] Surface::Sample sample(Expr<float3> wo,
                                         Expr<float> u_lobe, Expr<float2> u,
                                         TransportMode mode) const noexcept {
        auto s = Surface::Sample::zero(_ctx.Kd.dimension());
        $outline {
            auto wo_local = _ctx.it.shading().world_to_local(wo);
            auto sign = ite(cos_theta(wo_local) < 0.f,
                            make_float3(1.f, 1.f, -1.f),
                            make_float3(1.f, 1.f, 1.f));
            wo_local *= sign;
            auto eta = _fresnel.eta_t();
            auto Fo = fresnel_dielectric(abs_cos_theta(wo_local), 1.f, eta);
            auto substrate_weight = _substrate_weight(Fo, _ctx.Kd_weight);
            BxDF::SampledDirection wi_sample;
            $if(u_lobe < substrate_weight) {// samples diffuse
                wi_sample = _substrate.sample_wi(wo_local, u, mode);
            }
            $else {// samples specular
                wi_sample = _coat.sample_wi(wo_local, u, mode);
            };
            SampledSpectrum f{_ctx.Kd.dimension(), 0.f};
            auto pdf = def(0.f);
            SampledSpectrum f_diffuse{_ctx.Kd.dimension(), 0.f};
            auto pdf_diffuse = def(0.f);
            auto wi = def(make_float3(0.f, 0.f, 1.f));
            auto wi_local = def(make_float3(0.f, 0.f, 1.f));
            $if(wi_sample.valid) {
                wi_local = wi_sample.wi;
                wi = _ctx.it.shading().local_to_world(wi_sample.wi * sign);
                auto f_coat = _coat.evaluate(wo_local, wi_local, mode);
                auto pdf_coat = _coat.pdf(wo_local, wi_local, mode);
                // diffuse
                auto Fi = fresnel_dielectric(abs_cos_theta(wi_local), 1.f, eta);
                auto a = exp(-(1.f / abs_cos_theta(wi_local) + 1.f / abs_cos_theta(wo_local)) * _ctx.sigma_a);
                // auto ee = sqr(1.f / _fresnel.eta_t());
                f_diffuse = (1.f - Fi) * (1.f - Fo) * sqr(1.f / eta) * a *
                            (_ctx.Kd_scale * _substrate.evaluate(wo_local, wi_local, mode));
                pdf_diffuse = _substrate.pdf(wo_local, wi_local, mode);
                f = (f_coat + f_diffuse) * abs_cos_theta(wi_local);
                pdf = lerp(pdf_coat, pdf_diffuse, substrate_weight);
            };
            s = {.eval = {.f = f, .pdf = pdf, .f_diffuse = f_diffuse * abs_cos_theta(wi_local), .pdf_diffuse = pdf_diffuse},
                 .wi = wi,
                 .event = Surface::event_reflect};
        };
        return s;
    }
};

class PlasticClosure : public Surface::Closure {

private:
    luisa::unique_ptr<PlasticClosureImpl> _impl;

public:
    using Surface::Closure::Closure;

    [[nodiscard]] SampledSpectrum albedo() const noexcept override { return context<PlasticContext>().Kd; }
    [[nodiscard]] Float2 roughness() const noexcept override {
        auto r = TrowbridgeReitzDistribution::alpha_to_roughness(
            context<PlasticContext>().roughness);
        return lerp(r, 1.f, saturate(context<PlasticContext>().Kd_weight));
    }
    [[nodiscard]] const Interaction &it() const noexcept override { return context<PlasticContext>().it; }

public:
    void pre_eval() noexcept override {
        _impl = luisa::make_unique<PlasticClosureImpl>(
            context<PlasticContext>());
    }
    void post_eval() noexcept override {
        _impl = nullptr;
    }

private:
    [[nodiscard]] Surface::Evaluation _evaluate(Expr<float3> wo, Expr<float3> wi,
                                                TransportMode mode) const noexcept override {
        return _impl->evaluate(wo, wi, mode);
    }
    [[nodiscard]] Surface::Sample _sample(Expr<float3> wo,
                                          Expr<float> u_lobe, Expr<float2> u,
                                          TransportMode mode) const noexcept override {
        return _impl->sample(wo, u_lobe, u, mode);
    }
};

luisa::unique_ptr<Surface::Closure> PlasticInstance::create_closure(
    const SampledWavelengths &swl, Expr<float> time) const noexcept {
    return luisa::make_unique<PlasticClosure>(pipeline(), swl, time);
}

void PlasticInstance::populate_closure(Surface::Closure *closure, const Interaction &it,
                                       Expr<float3> wo, Expr<float> eta_i) const noexcept {
    auto &swl = closure->swl();
    auto time = closure->time();
    auto roughness = def(make_float2(0.f));
    if (_roughness != nullptr) {
        auto r = _roughness->evaluate(it, time);
        auto remap = node<PlasticSurface>()->remap_roughness();
        auto r2a = [](auto &&x) noexcept { return TrowbridgeReitzDistribution::roughness_to_alpha(x); };
        roughness = _roughness->node()->channels() == 1u ?
                        (remap ? make_float2(r2a(r.x)) : r.xx()) :
                        (remap ? r2a(r.xy()) : r.xy());
    }
    auto eta = (_eta ? _eta->evaluate(it, time).x : 1.5f) / eta_i;
    auto [Kd, Kd_lum] = _kd ? _kd->evaluate_albedo_spectrum(it, swl, time) :
                              Spectrum::Decode::one(swl.dimension());
    auto [Ks, Ks_lum] = _ks ? _ks->evaluate_albedo_spectrum(it, swl, time) :
                              Spectrum::Decode::one(swl.dimension());
    auto [sigma_a, sigma_a_lum] = _sigma_a ? _sigma_a->evaluate_albedo_spectrum(it, swl, time) :
                                             Spectrum::Decode::zero(swl.dimension());
    auto thickness = _thickness ? _thickness->evaluate(it, time).x : 1.f;
    auto scaled_sigma_a = sigma_a * thickness;
    auto average_transmittance = exp(-2.f * sigma_a_lum * thickness);
    // Difference from the Tungsten renderer:
    // We use the fitted polynomial to approximate the integrated
    // Fresnel reflectance, rather than compute it on the fly.
    auto diffuse_fresnel = fresnel_dielectric_integral(eta);

    PlasticContext ctx{
        .it = it,
        .Kd = Kd,
        .Kd_scale = 1.f / (1.f - Kd * diffuse_fresnel),
        .Kd_weight = Kd_lum * average_transmittance,
        .Ks = Ks,
        .sigma_a = sigma_a,
        .eta = eta,
        .roughness = roughness,
    };
    closure->bind(std::move(ctx));
}

using NormalMapOpacityPlasticSurface = NormalMapWrapper<OpacitySurfaceWrapper<
    PlasticSurface, PlasticInstance>>;

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::NormalMapOpacityPlasticSurface)
