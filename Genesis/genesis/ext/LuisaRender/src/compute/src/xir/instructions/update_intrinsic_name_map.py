import os
import functools

SCRIPT_DIR: str = os.path.dirname(os.path.realpath(__file__))


def to_snake_case(name: str) -> str:
    return functools.reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, name).lower()


def find_enum(lines, module_name) -> (int, str):
    import re
    regex = re.compile(f"enum (struct|class) ({module_name}\\w*)Op")
    for i, line in enumerate(lines):
        if matches := regex.match(line):
            enum_name = matches.group(2)
            print(f"Found enum {enum_name}")
            return i, enum_name
    return -1, ""


def process(module_name, file):
    # get current script directory
    src_index = SCRIPT_DIR.rfind("src")
    assert src_index != -1, "src not found in the path"
    header_path = os.path.join(SCRIPT_DIR[:src_index], "include",
                               "luisa", "xir", "instructions",
                               f"{to_snake_case(module_name)}.h")
    with open(header_path, "r") as f:
        lines = [x for line in f.readlines() if (x := line.strip()) and not x.startswith("//")]
        # search for enum (struct|class) {module_name}{infix}Op in lines
        while True:
            enum_index, enum_name = find_enum(lines, module_name)
            if enum_index == -1:
                break
            lines = lines[enum_index + 1:]
            enum_end = lines.index("};")
            ops = [line.split(",")[0].strip() for line in lines[:enum_end]]
            lines = lines[enum_end + 1:]
            file.write(f"""luisa::string_view to_string({enum_name}Op op) noexcept {{
    using namespace std::string_view_literals;
    switch (op) {{
""")
            file.writelines([f"        case {enum_name}Op::{op}: return \"{op.lower()}\"sv;\n" for op in ops])
            file.write(f"""    }}
    LUISA_ERROR_WITH_LOCATION("Unknown {to_snake_case(module_name)} operation (code = {{}}).", static_cast<uint32_t>(op));
}}

{enum_name}Op {to_snake_case(enum_name)}_op_from_string(luisa::string_view name) noexcept {{
    using namespace std::string_view_literals;
    static const luisa::unordered_map<luisa::string_view, {enum_name}Op> m{{
""")
            file.writelines([f"        {{\"{op.lower()}\"sv, {enum_name}Op::{op}}},\n" for op in ops])
            file.write(f"""    }};
    auto iter = m.find(name);
    LUISA_ASSERT(iter != m.end(), "Unknown {to_snake_case(enum_name)} operation: {{}}.", name);
    return iter->second;
}}

""")


if __name__ == "__main__":
    module_names = ["Arithmetic", "Atomic", "Cast", "Autodiff", "ThreadGroup", "RayQuery", "Resource"]
    with open(f"{SCRIPT_DIR}/op_name_map.cpp", "w") as file:
        file.write("// This file is generated by update_intrinsic_name_map.py\n")
        file.write("// Do not edit it manually.\n\n")
        file.write("#include <luisa/core/logging.h>\n")
        file.write("#include <luisa/core/stl/unordered_map.h>\n")
        file.write("\n")
        for module_name in module_names:
            file.write(f"#include <luisa/xir/instructions/{to_snake_case(module_name)}.h>\n")
        file.write("\n")
        file.write("namespace luisa::compute::xir {\n\n")
        for module_name in module_names:
            process(module_name, file)
        file.write("}// namespace luisa::compute::xir\n")
