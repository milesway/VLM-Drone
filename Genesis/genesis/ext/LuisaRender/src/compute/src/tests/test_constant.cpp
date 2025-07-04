#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

struct Foo {
    luisa::uint a;
    float b;
    float c;
    float d;
};
LUISA_STRUCT(Foo, a, b, c, d){};

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    if (argc <= 1) { exit(1); }
    Device device = context.create_device(argv[1]);
    auto buffer = device.create_buffer<uint>(1024);
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());
    Kernel2D kernel = [&]() {
        Foo foo_data[4]{
            {1, 2.0f, 3.0f, 4.0f},
            {5, 6.0f, 7.0f, 8.0f},
            {9, 10.0f, 11.0f, 12.0f},
            {13, 14.0f, 15.0f, 16.0f}
        };
        Constant<Foo> foo(foo_data, 4);
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var<uint> i = coord.x % 4u;
        Var uv = (make_float2(coord) + 0.5f) / make_float2(size) + foo.read(i).b;
        image->write(coord, make_float4(uv, 0.5f, 1.0f));
    };
    auto shader = device.compile(kernel);
    stream << shader().dispatch(resolution)
           << image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_helloworld.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}
