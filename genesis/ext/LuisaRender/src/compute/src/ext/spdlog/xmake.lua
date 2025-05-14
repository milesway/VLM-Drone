if not _config_project then
    function _config_project(config)
        local batch_size = config["batch_size"]
        if type(batch_size) == "number" and batch_size > 1 and (not _disable_unity_build) then
            add_rules("c.unity_build", {
                batchsize = batch_size
            })
            add_rules("c++.unity_build", {
                batchsize = batch_size
            })
        end
        if type(_config_rules) == "table" then
            add_rules(_config_rules, config)
        end
    end
end

target("spdlog")
_config_project({
    project_kind = "static",
    batch_size = 64,
    no_rtti = true
})
add_includedirs("include", {
    public = true
})
on_load(function(target)
    local function rela(p)
        return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
    end
    if get_config("spdlog_only_fmt") then
        target:add("defines", "FMT_USE_CONSTEVAL=0", "FMT_USE_CONSTEXPR=1", "FMT_UNICODE=0", "FMT_EXCEPTIONS=0", {
            public = true
        })
        target:add("headerfiles", rela("include/spdlog/fmt/**.h"))
        target:add("files", rela("src/bundled_fmtlib_format.cpp"))
    else
        target:add("defines", "SPDLOG_NO_EXCEPTIONS", "SPDLOG_NO_THREAD_ID", "SPDLOG_DISABLE_DEFAULT_LOGGER",
            "FMT_UNICODE=0", "FMT_USE_CONSTEVAL=0", "FMT_USE_CONSTEXPR=1", "FMT_EXCEPTIONS=0", {
                public = true
            })
        target:add("headerfiles", rela("include/**.h"))
        target:add("files", rela("src/*.cpp"))
        if is_plat("windows") then
            target:add("defines", "NOMINMAX", "UNICODE")
        end
    end
    target:add("defines", "SPDLOG_COMPILED_LIB", {
        public = true
    })
end)
target_end()
