#pragma once
#include <iostream>
#include <vector>
#include "board.h"
#include "weight.h"

class feature : public weight {
public:
    feature() {};
    feature(std::array<int, 5> pos) : weight(1 << 25) {
        board b;
        for (int i = 0; i < 16; ++i)
            b(i) = i;
        for (int i = 0; i < 8; ++i) {
            if (i == 4)
                b.reflect_horizontal();
            b.rotate_right();
            for (int j = 0; j < 5; ++j)
                iso_idxs[i][j] = b(pos[j]);
        }
    }
public:
    virtual void update(const board& b, float v) {
        v /= 8.0;
        for (const auto& iso_idx : iso_idxs)
            value[at(b, iso_idx)] += v;
    }

    virtual float estimate(const board& b) {
        float v = 0.0;
        for (const auto& iso_idx : iso_idxs)
            v += value[at(b, iso_idx)];
        return v;
    }

    virtual int at(const board& b, const std::array<int, 5>& iso_idx) {
        int idx = 0;
        for (const auto p : iso_idx)
            idx = (idx << 5) | b(p);
        return idx;
    }
private:
    std::array<std::array<int, 5>, 8> iso_idxs;
};