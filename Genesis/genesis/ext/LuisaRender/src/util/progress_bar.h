//
// Created by Mike Smith on 2022/4/7.
//

#pragma once

#include <luisa/core/clock.h>

namespace luisa::render {

// credit: https://github.com/AirGuanZ/agz-utils/blob/master/include/agz-utils/console/pbar.h#L137
class ProgressBar {

public:
    static constexpr auto complete_char = '=';
    static constexpr auto heading_char = '>';
    static constexpr auto incomplete_char = ' ';
    // using clock_type = std::chrono::steady_clock;

private:
    double _progress;
    uint32_t _width;
    Clock _clock;
    bool _silent;

public:
    explicit ProgressBar(bool silent = false, uint32_t width = 50u) noexcept;
    void reset() noexcept;
    void update(double progress) noexcept;
    void done() noexcept;
};

}// namespace luisa::render
