#pragma once
#include <iostream>
#include <sstream>
#include <algorithm>
#include "feature.h"
#include "weight.h"

class tuple_netwrok {
public:
    tuple_netwrok() {};
    tuple_netwrok(std::vector<std::array<int, 5>> tuples) {
		for (auto &ps : tuples)
			features.push_back(feature(ps));
    }
public:
	virtual void update(const board& b, float v) {
		v /= features.size();
		for (auto &f : features)
			f.update(b, v);
	}

	virtual float estimate(const board& b) {
		float v = 0.0;
		for (auto &f : features)
			v += f.estimate(b);
		return v;
	}

    virtual void load_weights(const std::string& path) {
		std::ifstream in;
		in.open(path.c_str(), std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		size_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		features.resize(size);
		for (weight& f : features)
			in >> f;
		in.close();
	}

	virtual void save_weights(const std::string& path) {
		std::ofstream out;
		out.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		size_t size = features.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& f : features)
			out << f;
		out.flush();
		out.close();
	}
private:
    std::vector<feature> features;
};
